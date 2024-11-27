import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, swin_b, vit_b_16
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, FCN_ResNet50_Weights, FCN_ResNet101_Weights
import clip

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from torchvision import transforms


from PIL import Image
import timm

class CustomFeatureModel(nn.Module):
    def __init__(self, model_name, use_pretrained=False):
        super(CustomFeatureModel, self).__init__()

        self.train_transform = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])                
                        ])
        
        self.test_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])                
                    ])

        supported_models = ['resnet18', 'resnet50', 'resnet101', 'resnet50_adv_l2_0.1', 'resnet50_adv_l2_0.5', 'resnet50x1_bitm','resnetv2_50x1_bitm', 'resnetv2_101x1_bitm','resnetv2_101x1_bit.goog_in21k']
        if model_name not in supported_models:
            raise ValueError(f"Invalid model_name. Expected one of {supported_models}, but got {model_name}")

        if model_name == 'resnet50_adv_l2_0.1':
            self.model = timm.create_model('resnet50')
            checkpoint = torch.load('./checkpoints/resnet50_l2_eps0.1.ckpt')['model']
            modified_checkpoint = {}
            for k, v in checkpoint.items():
                if 'attacker' in k:
                    continue
                modified_checkpoint[k.replace('module.model.', '')] = v

            self.model.load_state_dict(modified_checkpoint, strict=False) #https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.1.ckpt

            # Get the last FC layer into a seperate variable
            self.last_layer = self.model.fc
            # Remoce the last FC layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        elif model_name == 'resnet50_adv_l2_0.5':
            self.model = timm.create_model('resnet50')
            checkpoint = torch.load('./checkpoints/resnet50_l2_eps0.5.ckpt')['model']
            modified_checkpoint = {}
            for k, v in checkpoint.items():
                if 'attacker' in k:
                    continue
                modified_checkpoint[k.replace('module.model.', '')] = v

            self.model.load_state_dict(modified_checkpoint, strict=False) #https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.5.ckpt

            # Remove the last FC layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])

        else:
            self.model = timm.create_model(model_name, pretrained=use_pretrained, num_classes=0)

        # self.feature_dim = self.model.num_featuress
        self.feature_dim = self.model(torch.zeros(1, 3, 224, 224)).shape[-1]
    
    @torch.no_grad()
    def forward(self, x, return_features=False):

        features = self.model(x)
        if return_features:
            return self.last_layer(features), features
        else:
            return self.last_layer(features)

class CustomSegmentationModel(nn.Module):
    def __init__(self, model_name, use_pretrained=False):
        super(CustomSegmentationModel, self).__init__()

        supported_models = ['deeplabv3_resnet50', 'deeplabv3_resnet101', 'fcn_resnet50', 'fcn_resnet101']
        if model_name not in supported_models:
            raise ValueError(f"Invalid model_name. Expected one of {supported_models}, but got {model_name}")

        if model_name == 'deeplabv3_resnet50':
            weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        elif model_name == 'deeplabv3_resnet101':
            weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        elif model_name == 'fcn_resnet50':
            weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        elif model_name == 'fcn_resnet101':
            weights = FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1

        self.model = torch.hub.load('pytorch/vision:v0.6.0', model_name, weights=weights)

        # self.transform = transforms.Compose([
        #                 transforms.RandomResizedCrop(224),
        #                 transforms.ToTensor(),
        #                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                     std=[0.229, 0.224, 0.225])                
        #             ])
        self.test_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])                
                    ])

        self.train_transform = transforms.Compose([
                            transforms.Resize((520,520)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])                
                        ])
                        
        self.feature_model = self.model.backbone
        # Add a max pooling layer with stride 32 to reduce the dimensionality of the features [2048,2,2]
        self.pool = nn.MaxPool2d(kernel_size=32, stride=32)

        #TODO: add the feature dimension
        self.feature_dim = 8192
        
    def preprocess_pil(self, images):
        # check if images is a list, then preprocess each image
        if isinstance(images, list):
            images_torch = []
            for image in images:
                image_tensor = self.transform(image)
                images_torch.append(image_tensor)
            images_torch = torch.stack(images_torch).to(device) 
        else:
            images_torch = self.transform(images).unsqueeze(0).to(device)
        
        return images_torch

    @torch.no_grad()
    def forward(self, images):
        if not isinstance(images, torch.Tensor):
            images = self.preprocess_pil(images)
        else:
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
        features = self.feature_model(images)['out']
        features = self.pool(features)
        features = features.view(features.shape[0],-1)
        
        return features

class CustomResNet(nn.Module):
    def __init__(self, model_name, num_classes, use_pretrained=False):
        super(CustomResNet, self).__init__()

        # Define available ResNet architectures
        resnets = {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152,
            'resnet50_v1': resnet50,
            'resnet50_v2':resnet50,
        }

        self.train_transform = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])                
                    ])
        
        self.test_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])                
                    ])


        # Check if the provided model_name is valid
        if model_name not in resnets:
            raise ValueError(f"Invalid model_name. Expected one of {list(resnets.keys())}, but got {model_name}")

        # Load the desired ResNet architecture
        if 'v2' not in model_name:
            print('Using V1')
            self.model = resnets[model_name](pretrained=use_pretrained) # V1
        else:
            if 'resnet50' in model_name:
                from torchvision.models import ResNet50_Weights
                self.model = resnets[model_name](weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                raise NotImplementedError(f"{model_name} is not supported")


        # Save the features before the FC layer
        self.features = nn.Sequential(*list(self.model.children())[:-1])

        # if 'v2' not in model_name and 'v1' not in model_name:
        #     print('*************** updating **********')
        #     # Update the final fully connected layer to match the number of desired classes
        #     num_ftrs = self.model.fc.in_features
        #     self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.feature_dim = self.features(torch.zeros(1, 3, 224, 224)).squeeze(-1).squeeze(-1).shape[-1]

    def forward(self, x, return_features=False):
        if return_features:
            features = self.features(x)
            # Flatten features for the FC layer
            flat_features = features.squeeze(-1).squeeze(-1)

            # Compute logits using the model's FC layer
            logits = self.model.fc(flat_features)

            return logits, flat_features
        else:
            return self.model(x)

class CustomVit(nn.Module):
    def __init__(self, model_name, num_classes, use_pretrained=False):
        super(CustomVit, self).__init__()

        # Define available ViT architectures
        vits = {
            'vit_b_16': vit_b_16,
        }

        # Initialize the ViT model
        if model_name in vits:
            self.model = vits[model_name](pretrained=use_pretrained)

            # Replace the classifier head with a new one for the desired number of classes
            self.feature_dim = self.model.heads.head.in_features  # Assuming 'heads' is the classification head module
            self.model.heads = nn.Linear(self.feature_dim, num_classes)
            
            # Register the hook to save the task features
            feature_layer = dict(self.model.named_modules())['encoder.ln']
            feature_layer.register_forward_hook(self.save_task_features_hook())

        else:
            raise ValueError(f"Unsupported model type: {model_name}. Supported types are: {list(vits.keys())}")
    
        self.train_transform = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])                
                    ])
        
        self.test_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])                
                    ])


    def save_task_features_hook(self):
        def hook(module, input, output):
            self.task_features = output.detach()  # Ensure detached for inference
        return hook

    def forward(self, x, return_features=False):
        
        # Apply the classifier head to the features
        logits = self.model(x)
        
        if return_features:
            features = self.task_features[:, 0, :]  # Extract the CLS token features           
            return logits, features
        
        return logits

class CustomClassifier(nn.Module):
    def __init__(self, model_name, use_pretrained=False):
        super(CustomClassifier, self).__init__()

        self.train_transform = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])                
                        ])
        
        self.test_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])                
                    ])

        if model_name == "resnet18-imagenet":
            from torchvision.models import resnet18, ResNet18_Weights
            network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.network_feat_extractor = create_feature_extractor(network,return_nodes=['flatten','fc'])
            self.feature_dim=512
        elif model_name == "resnet50-imagenet":
            from torchvision.models import resnet50, ResNet50_Weights
            network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

            self.network_feat_extractor = create_feature_extractor(network,return_nodes=['flatten','fc'])
            self.feature_dim=2048
        elif model_name == "regnet":
            from torchvision.models import regnet_y_800mf, RegNet_Y_800MF_Weights
            network = torchvision.models.get_model("regnet_y_800mf",weights=RegNet_Y_800MF_Weights.IMAGENET1K_V1 )

        elif model_name == "vit_b_16":
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            network = torchvision.models.get_model("vit_b_16",weights = ViT_B_16_Weights.IMAGENET1K_V1)
            self.network_feat_extractor = create_feature_extractor(network,return_nodes=['getitem_5','heads'])
            self.feature_dim=768
        elif model_name =='swin_v2_t':
            from torchvision.models import swin_v2_t, Swin_V2_T_Weights
            network = torchvision.models.get_model("swin_v2_t",weights= Swin_V2_T_Weights.IMAGENET1K_V1)
            self.network_feat_extractor = create_feature_extractor(network,return_nodes=['flatten','head'])
        elif model_name == "swin_b":
            from torchvision.models import swin_b, Swin_B_Weights
            network = torchvision.models.get_model("swin_b",weights= Swin_B_Weights.IMAGENET1K_V1)
            self.network_feat_extractor = create_feature_extractor(network,return_nodes=['flatten','head'])
            self.feature_dim=1024
        elif model_name == "deit_tiny":
            network = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
            self.network_feat_extractor = create_feature_extractor(network,return_nodes=['flatten','head'])
        elif model_name == "deit_small":
            network = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)
            self.network_feat_extractor = create_feature_extractor(network,return_nodes=['flatten','head'])
        elif model_name =="clip_rn50":
            clip_model, preprocess_clip = clip.load("RN50", device=device)
            self.network_feat_extractor = clip_model
            preprocess= transforms.Compose([transforms.Resize(256),
                    transforms.CenterCrop(224),
                    preprocess_clip])
        else:
            raise NotImplementedError(f"{model_name} is not supported")


    def forward(self, x, return_features=False):

        results = self.network_feat_extractor(x)
        keys = list(results.keys())
        for k in keys:
            if 'head' in k or 'fc' in k:
                logits = results[k]
            else:
                feature = results[k]

        if return_features:
            return logits, feature
        else:
            return logits

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = CustomFeatureModel(model_name='resnet50_adv_l2_0.1', use_pretrained=True)
    # model = CustomClassifier(model_name='resnet18-imagenet', use_pretrained=True)
    # logits, features = model(torch.zeros(1, 3, 224, 224), return_features=True)
    # print(logits.shape, features.shape)
    # print(model.feature_dim)
    # print(model.network_feat_extractor.layer1)

    model = CustomVit(model_name='vit_b_16', num_classes=100, use_pretrained=True)

    logits, features = model(torch.zeros(1, 3, 224, 224), return_features=True)
    print(logits.shape, features.shape)
    print(model.feature_dim)
    # print(model)

    #model = CustomSegmentationModel(model_name='deeplabv3_resnet101', use_pretrained=True)
    #pil_image = Image.open("./data/domainnet_v1.0/real/toothpaste/real_318_000284.jpg")
    #pil_images = [pil_image,pil_image]
    # torch_tensor = torch.zeros(1,3, 224, 224)
    # logits, features = model(torch_tensor, return_features=True)
    # print(logits.shape)
    # print(features.shape)

