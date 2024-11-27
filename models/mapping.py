import torch
import torch.nn as nn
import torchvision

import torch.nn.functional as F
from collections import OrderedDict
import math

# Function to print the names of the layers in a model
def print_layers(model):
    for name, module in model.named_modules():
        print(name)

class TaskMapping(nn.Module):
    def __init__(self, task_model, mapping_model, task_layer_name, vlm_dim, mapping_output_size, cutmix_fn=None):
        super(TaskMapping, self).__init__()
        self.task_model = task_model
        self.task_features = None
        self.cutmix_fn = cutmix_fn
        self.task_model.eval()
        self.task_layer_name = task_layer_name
        mapping_layer_name = task_layer_name

        # Freeze all parameters in the task model for inference
        for param in self.task_model.parameters():
            param.requires_grad = False
        if task_layer_name!="model.layer0":
            # Register hook to the task model layer
            task_layer = dict(self.task_model.named_modules())[task_layer_name]
            task_layer.register_forward_hook(self.save_task_features_hook())

            print(f"Hooks registered to layer: {task_layer_name}")
            
            # if mapping_layer_name.startswith('model.'):
            #     if hasattr(mapping_model, 'model'):
            #         mapping_model = getattr(mapping_model, 'model')
            #         mapping_layer_name = mapping_layer_name.replace('model.', '', 1)  # Remove 'model.' prefix from layer name
            #     else:
            #         raise ValueError("The mapping model does not have a 'model' submodule.")

            # If mapping name has nested layers, extract the last layer
            if '.' in mapping_layer_name:
                
                temp_model = mapping_model
                nested_layers = mapping_layer_name.split('.')
                for layer in nested_layers[:-1]:
                    temp_model = getattr(temp_model, layer)
                mapping_model = temp_model
                mapping_layer_name = nested_layers[-1]

            all_layers = OrderedDict(mapping_model.named_children())

            if mapping_layer_name in all_layers:
                start_index = list(all_layers.keys()).index(mapping_layer_name) + 1
                selected_layers = OrderedDict(list(all_layers.items())[start_index:-1])  # Remove the last layer
            else:
                raise ValueError(f"Layer {mapping_layer_name} not found in mapping model.")

        else:
            all_layers = OrderedDict(mapping_model.model.named_children())
            
            
            selected_layers = OrderedDict(list(all_layers.items())[:-1])  # Remove the last layer

        # Calculate intermediate dimension for the MLP
        if vlm_dim > mapping_output_size:
            # Calculate the geometric mean in log space, and find the nearest higher power of 2
            geometric_mean_log2 = (math.log2(mapping_output_size) + math.log2(vlm_dim)) / 2
            inter_dim = 2 ** math.ceil(geometric_mean_log2)
        else:
            # Double the vlm_dim but ensure it does not exceed mapping_output_size
            doubled_projection = min(vlm_dim * 2, mapping_output_size)
            # Adjust to nearest lower or equal power of 2
            inter_dim = 2 ** int(math.log2(doubled_projection))

        print(f"Intermediate dimension: {inter_dim}")
        # Construct the MLP (projection head)
        self.projection_head = nn.Sequential(
            nn.Linear(mapping_output_size, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, vlm_dim),
        )

        # # Add the MLP (projection head) to the pruned layers
        # selected_layers.update({‘projection_head’: mlp_layers})

        # Recreate the mapping model from pruned layers, now including the projection head
        self.mapping_model = nn.Sequential(selected_layers)

    def save_task_features_hook(self):
        def hook(module, input, output):
            self.task_features = output.detach()  # Ensure detached for inference
        return hook

    def forward(self, x, labels=None, return_task_logits=False, use_cutmix=False):
        self.task_features = None  # Reset task features
        
        with torch.no_grad():
            task_model_logits = self.task_model(x)  # Pass input through the task model to extract features through the hook

            if use_cutmix:
                self.task_features, labels = self.cutmix_fn(self.task_features, labels)

        with torch.set_grad_enabled(True):
            if self.task_layer_name == "model.layer0":
                self.task_features = x
            
            output = self.mapping_model(self.task_features)  # Pass task features through the entire mapping model
            
            if len(output.shape) != 4:
                output = output[:, 0, :]
            else:
                output = output.view(output.size(0), -1)  # Flatten the output
            output = self.projection_head(output)

        if return_task_logits:
            return output, task_model_logits, labels
        return output, labels

class WeightedAverage(nn.Module):
    def __init__(self, num_attributes, num_classes):
        super(WeightedAverage, self).__init__()

        self.num_attributes = num_attributes
        self.num_classes = num_classes

        # Initialize raw weights as a uniform distribution 0 and 1
        self.raw_weights = nn.Parameter(torch.rand(num_classes, num_attributes))

    def forward(self, x):
        """
        x: input tensor of shape [batch_size, num_attributes * num_classes]

        Returns:
        weighted_averages: tensor of shape [batch_size, num_classes]
        """
        
        # Ensure weights are normalized correctly across attributes
        weights = F.softmax(self.raw_weights, dim=1)  # This is correct based on your description

        # Correct reshaping and multiplication
        segmented_inputs = x.view(-1, self.num_classes, self.num_attributes)  # Shape: [batch_size, num_classes, num_attributes]
        
        # Expand weights for batch processing
        expanded_weights = weights.unsqueeze(0)  # Adds a batch dimension

        # Element-wise multiplication and sum across attributes
        weighted_sums = torch.sum(segmented_inputs * expanded_weights, dim=2)  # Shape: [batch_size, num_classes]

        return weighted_sums

class AttentionNet(nn.Module):
    def __init__(self, num_classes, num_attributes, in_dim=512):
        super(AttentionNet, self).__init__()

        self.num_classes = num_classes
        self.num_attributes = num_attributes

        # Initialize an network that maps the input dims to the number of attributes*number of classes
        self.att_weights = nn.Sequential(
            nn.Linear(in_dim, self.num_attributes),
            nn.ReLU(),
            nn.Linear(self.num_attributes, self.num_classes * self.num_attributes),
            nn.Sigmoid()
        )
    def forward(self, features, att_logits):
        # segment the input logits for each class
        segmented_inputs = att_logits.view(-1, self.num_classes, self.num_attributes)  # Shape: [batch_size, num_classes, num_attributes]
        # Calculate the attention weights for each class
        att_weights = self.att_weights(features)
        att_weights = att_weights.view(-1, self.num_classes, self.num_attributes) # Shape: [batch_size, num_classes, num_attributes]
        
        # Softmax the attention weights across attributes
        att_weights = F.softmax(att_weights, dim=-1)

        # Element-wise multiplication and sum across attributes
        weighted_sums = torch.sum(segmented_inputs * att_weights, dim=-1)
        
        return weighted_sums

class MeanAggregator(nn.Module):
    def __init__(self, num_classes, num_attributes_per_cls):
        super(MeanAggregator, self).__init__()

        self.num_classes = num_classes
        self.num_attributes_per_cls = num_attributes_per_cls

    def forward(self, cls_attribute_scores_dict):
        """
        x: input tensor of shape [batch_size, num_attributes_per_cls * num_classes]

        Returns:
        mean_values: tensor of shape [batch_size, num_classes]
        """
        class_logits = []
        for class_idx, attribute_score in cls_attribute_scores_dict.items():
            mean_value = torch.mean(attribute_score, dim=-1)
            class_logits.append(mean_value)
        class_logits = torch.stack(class_logits, dim=-1) # Shape: [batch_size, num_classes]
        return class_logits
    
class MaxAggregator(nn.Module):
    def __init__(self, num_classes, num_attributes_per_cls):
        super(MaxAggregator, self).__init__()

        self.num_classes = num_classes
        self.num_attributes_per_cls = num_attributes_per_cls

    def forward(self, cls_attribute_scores_dict):
        """
        x: input tensor of shape [batch_size, num_attributes_per_cls * num_classes]

        Returns:
        max_values: tensor of shape [batch_size, num_classes]
        """
        class_logits = []
        for class_idx, attribute_score in cls_attribute_scores_dict.items():
            max_value, _ = torch.max(attribute_score, dim=-1)
            class_logits.append(max_value)
        class_logits = torch.stack(class_logits, dim=-1) # Shape: [batch_size, num_classes]
        return class_logits

class MultiHeadedAttentionSimilarity(nn.Module):
    def __init__(self, num_classes, num_attributes_per_cls=[100], num_heads=1, out_dim=1):
        super().__init__()

        assert len(num_attributes_per_cls) == num_classes, "Number of attributes must be equal to the number of classes"
        self.multihead_attns = nn.ModuleList([nn.MultiheadAttention(num_attributes_per_cls[i], num_heads) for i in range(num_classes)])
        self.out_linears = nn.ModuleList([nn.Linear(num_attributes_per_cls[i], out_dim) for i in range(num_classes)])
        self.num_classes = num_classes
        self.num_attributes_per_cls = num_attributes_per_cls

    def forward(self, cls_attribute_scores_dict):
        """
        cls_attribute_scores_dict: dictionary of class indices and their corresponding attribute scores
        """
        cls_logits=[]
        for class_idx , attribute_score in cls_attribute_scores_dict.items():
            attribute_score = attribute_score.unsqueeze(1)  # Shape: [batch_size, 1, num_attributes]
            attn_output, _ = self.multihead_attns[class_idx](attribute_score, attribute_score, attribute_score)  # Shape: [batch_size, 1, num_attributes]
            attn_output = self.out_linears[class_idx](attn_output.squeeze(1))  # Shape: [batch_size, out_dim]
            cls_logits.append(attn_output)
        cls_logits = torch.cat(cls_logits, dim=-1) # Shape: [batch_size, num_classes]

        return cls_logits

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_classes, in_dim=512, num_heads=1, out_dim=1):
        super().__init__()
        self.multihead_attns = nn.ModuleList([nn.MultiheadAttention(in_dim, num_heads) for i in range(num_classes)])
        self.out_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for i in range(num_classes)])
        self.num_classes = num_classes

    def forward(self, text_encodings, img_encodings):
        """
        text_encodings: batch of text encodings of shape [num_classes, num_attributes, in_dim]
        img_encodings: batch of image encodings of shape [batch_size, in_dim]
        """
        batch_size, _ = img_encodings.shape
        # Adjust img_encodings to have same format as expected by multihead attention [sequence_len, batch, dim]
        img_encodings = img_encodings.unsqueeze(0)  # Shape: [1, batch_size, in_dim]

        attn_outputs = []
        for i, (attn, linear) in enumerate(zip(self.multihead_attns, self.out_linears)):
            # For each class, use the corresponding text encoding as key and value
            text_encoding = text_encodings[i]  # Shape: [num_attributes, in_dim]

            # Repeat the image encodings to match the number of attributes for the current class
            img_encodings_repeated = img_encodings.repeat(text_encoding.shape[0], 1, 1)  # Shape: [num_attributes, batch_size, in_dim]
            # Repeate the text encoding to match the batch size
            text_encoding = text_encoding.unsqueeze(1).repeat(1, batch_size, 1)  # Shape: [num_attributes, batch_size, in_dim]

            print(img_encodings_repeated.shape)
            # Apply multihead attention using image encodings as query and text encodings as key and value
            attn_output, _ = attn(query=img_encodings_repeated, key=text_encoding, value=text_encoding) # Shape: [num_attributes, batch_size, in_dim]
            print(attn_output.shape)
            attn_output = attn_output[0]  # Shape: [batch_size, in_dim]


            # Apply the linear layer to the attention output
            linear_output = linear(attn_output)  # Shape: [batch_size, out_dim]
            attn_outputs.append(linear_output)

        # Combine outputs for all classes, resulting in shape [batch_size, num_classes]
        logits = torch.cat(attn_outputs, dim=-1)
        return logits


if __name__ == "__main__":
    
    # # Example usage
    # task_model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
    # mapping_model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)

    # task_layer_name = 'layer1'

    # vlm_dim = 512
    # mapping_output_size = 2048

    # task_mapping = TaskMapping(task_model, mapping_model, task_layer_name, vlm_dim, mapping_output_size)

    from resnet import CustomClassifier, CustomVit
    
    task_model = CustomVit("vit_b_16", num_classes=100, use_pretrained=True)
    mapping_model = CustomVit("vit_b_16", num_classes=100, use_pretrained=True)
    
    task_mapping = TaskMapping(task_model, mapping_model, "model.encoder.layers.encoder_layer_1", 512, 768)

    # Example forward pass
    x = torch.randn(1, 3, 224, 224)
    output, _ = task_mapping(x)

    print(output.shape)

    assert False
    # num_attributes_per_cls = [10, 20]
    # num_classes = 2
    # mha = MultiHeadedAttentionSimilarity(num_classes, num_attributes_per_cls, num_heads=1, out_dim=1)
    # cls_attribute_scores_dict = {0: torch.randn(4, 10), 1: torch.randn(4, 20)}

    # cls_logits = mha(cls_attribute_scores_dict)
    # print(cls_logits.shape)

    # mha = MultiHeadedAttention(num_classes=2, in_dim=512, num_heads=1, out_dim=1)
    # text_encodings = torch.randn(2, 10, 512)
    # img_encodings = torch.randn(4, 512)

    # cls_logits = mha(text_encodings, img_encodings)

    # print(cls_logits.shape)

    # assert False
