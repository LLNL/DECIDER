
import os
import sys
import copy

import argparse
import random
import pickle
import json
import logging
import csv
from tqdm import tqdm
from functools import partial
from datetime import datetime

try:
    del os.environ['OMP_PLACES']
    del os.environ['OMP_PROC_BIND']
except:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from torch.utils.data.dataset import Subset

import lightning as L
from lightning.fabric import Fabric, seed_everything

import clip
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from train_task_distillation import get_dataset, get_CLIP_text_encodings, build_classifier
from models.mapping import TaskMapping, MultiHeadedAttentionSimilarity, MultiHeadedAttention, print_layers, MeanAggregator, MaxAggregator
from utils_proj import SimpleDINOLoss, compute_accuracy, compute_similarities, CutMix, MyAugMix, find_normalization_parameters, get_score, calc_gen_threshold, calc_accuracy_from_scores
from failure_eval import CIFAR100C, load_data, get_save_dir

CLIP_LOGIT_SCALE = 100


def kl_divergence_pytorch(p, q):
    """
    Compute the KL divergence between two probability distributions.
    """
    return (p * (p / q).log()).sum(dim=1)

class PIM_Explanations(nn.Module):
    def __init__(self, attribute_names_per_class, num_attributes_per_class, learning_rate=0.01, num_optimization_steps=8000, aggregation_fn=None):
        super().__init__()

        """
        Args:
        attribute_names_per_class: A dictionary with class names as keys and tensors of shape [batch_size, num_attributes] as values.
        num_attributes_per_class: A list of integers with the number of attributes for each class.
        learning_rate: The learning rate for the optimizer.
        num_optimization_steps: The number of steps for the optimization process.
        aggregation_fn: A function that takes a dictionary of class specific attribute logits and returns a tensor of shape [num_classes].

        """       
       
        self.attribute_names_per_class = attribute_names_per_class
        self.num_attributes_per_class = num_attributes_per_class
        self.learning_rate = learning_rate
        self.num_optimization_steps = num_optimization_steps
        self.aggregation_fn = aggregation_fn

        self.class_names = list(attribute_names_per_class.keys())
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.inverse_normalization = transforms.Normalize(mean=[-m/s for m, s in zip(self.mean, self.std)], std=[1/s for s in self.std])
    
    def optimize_attribute_weights(self, pim_attribute_dict, reference_classes, num_attributes_per_class, learning_rate=0.01, num_optimization_steps=50, aggregation_fn=None):
        """
        Optimize attribute weights for each top-ranked class per instance such that the true class's logit is maximized.
        
        Args:
        :param pim_attribute_dict: A list of dictionaries for each instance in the batch with class indices as keys and tensors of shape [num_attributes] as values.
        :param reference_classes: A tensor of shape [batch_size] with the indices of the reference class for each instance.
        :param num_attributes_per_class: list of number of attributes for each class
        :param learning_rate: The learning rate for the optimizer.
        :param num_optimization_steps: The number of steps for the optimization process.

        Return: A list containing optimized weights for each instance, 
                where each element is a dictionary with class indices as keys and tensors of shape [num_attributes] as values.
        """
        
        batch_size = len(pim_attribute_dict)
        num_classes = len(num_attributes_per_class)
        
        optimized_attribute_weights_batch = []
        # Main loop over the batch
        for i in range(batch_size):
            optimized_attribute_weights = {class_idx: 
                                    torch.ones(num_attributes_per_class[class_idx]) 
                                                for class_idx in range(num_classes)}
            
            # Extract predictions for the current instance
            # dict of class specific attribute logits for the current instance
            instance_predictions = pim_attribute_dict[i]

            # Compute aggregated logits by agreegating the attributes for each class using the aggregation function
            # The function should take a dictionary of class specific attribute logits and return a tensor of shape [num_classes]
            aggregated_logits = aggregation_fn(instance_predictions) 
            
            # Get the ranking of classes based on logits
            _, ranked_classes = aggregated_logits.sort(descending=True) # [num_classes]
            if reference_classes is None:
                # Make the second ranked class the reference class
                reference_class = ranked_classes[1]
            else:
                reference_class = reference_classes[i]

            
            # Find index of the reference class
            reference_class_idx = (ranked_classes == reference_class).nonzero(as_tuple=True)[0].item()

            # convert to numpy
            ranked_classes = ranked_classes.cpu().numpy()

            # print(f"\nRanked classes: {ranked_classes}, Reference class: {reference_class}")

            # Iterate over each class that ranks higher than the true class
            for rank, top_class in enumerate(ranked_classes):
                print(rank, reference_class_idx)
                if rank >= reference_class_idx:  # Skip if not ranked above the true class
                    break

                # Initialize weights for optimization for the current top class
                # Set the weights to a large value to support sigmoid
                weights = nn.Parameter(torch.ones(num_attributes_per_class[top_class], requires_grad=True)*3)
                
                # Set up the optimizer for the weights
                optimizer = torch.optim.Adam([weights], lr=learning_rate)

                pbar = tqdm(range(num_optimization_steps), desc=f'Instance {i}')
                # Optimization loop for adjusting weights for the current top class
                for _ in pbar:

                    optimizer.zero_grad()
                    
                    weights_sig = F.sigmoid(weights)  # Apply sigmoid to ensure weights are between 0 and 1
                    # Adjust logits for the top class based on current weights
                    adjusted_predictions = instance_predictions[top_class] * weights_sig
                    # Re-aggregate the logits for the top class
                    adjusted_logit = aggregation_fn({top_class: adjusted_predictions})
                    
                    # Calculate the loss to ensure true class logit is higher than top class logit using leaky relu
                    loss = F.leaky_relu(adjusted_logit - aggregated_logits[reference_class_idx], negative_slope=0.1) 

                    # Sparse constraint on the weights makes one of the weights close to 0
                    # loss_weights = F.relu(weights_sig.min()) + F.relu(1 - weights_sig.max())
                    loss_weights = torch.norm(weights_sig, p=1)

                    loss += 0.08* loss_weights


                    # Perform gradient descent
                    loss.backward()
                    optimizer.step()
                    
                    # Break if the adjusted logit for the top class is less than the true class logit
                    if loss.item() < 0:
                        break
                    
                    pbar.set_postfix({'Loss': loss.item()})

                # Store the optimized weights for the top class
                optimized_attribute_weights[top_class] = weights_sig.detach()
            optimized_attribute_weights_batch.append(optimized_attribute_weights)
        return optimized_attribute_weights_batch

    def match_probabilities_to_task_model(self, pim_attribute_dict, task_model_logits, num_attributes_per_class, learning_rate=0.01, num_optimization_steps=50, aggregation_fn=None):
        """
        Adjust attribute weights for all classes together for each sample such that the attribute-aggregated 
        probabilities of the PIM match the task model's probabilities.
        
        :param pim_attribute_dict: A list of dictionaries for each instance in the batch with class indices as keys and tensors of shape [num_attributes] as values.
        :param task_model_logits: A tensor of shape [batch_size, num_classes] representing the task model's probabilities.
        :param num_attributes_per_class: list of number of attributes for each class
        :param learning_rate: The learning rate for the optimizer.
        :param num_optimization_steps: The number of steps for the optimization process.
        
        :return:
        A list containing optimized weights for each instance, where each element is a list of tensors corresponding to the optimized weights for each class.
        """

        batch_size = task_model_logits.shape[0]
        num_classes = task_model_logits.shape[1]


        optimized_attribute_weights_batch = []

        # Main loop over the batch
        for i in range(batch_size):
            optimized_attribute_weights = {class_idx: 
                                    torch.ones(num_attributes_per_class[class_idx]) 
                                                for class_idx in range(num_classes)}
            # Initialize list of parameter tensors to weight all attributes for each class
            weight_params = [torch.ones(num_attributes_per_class[class_idx], requires_grad=True)*3 for class_idx in range(num_classes)]
            # Initalize the weights for the attributes for each class; Make it a parameter to be optimized
            weight_params = [nn.Parameter(weights) for weights in weight_params]

            # Set up the optimizer for the weights
            optimizer = torch.optim.Adam(weight_params, lr=learning_rate)

            # Extract predictions for the current instance from PIM
            instance_predictions = pim_attribute_dict[i]
            # Optimization loop for adjusting weights for all classes together
            pbar = tqdm(range(num_optimization_steps), desc=f'Instance {i+1}/{batch_size}')
            for _ in pbar:
                optimizer.zero_grad()

                # Initialize a tensor to store adjusted logits for all classes
                adjusted_logits = torch.zeros(num_classes)

                # Apply sigmoid to ensure weights are between 0 and 1
                weight_params_sig = [F.sigmoid(weights) for weights in weight_params]

                # Adjust logits for all classes based on current weights
                for class_idx in range(num_classes):
                    # Use only the relevant attributes for each class
                    num_attributes = num_attributes_per_class[class_idx]
                    
                    # Adjust logits for the current class based on current weights
                    adjusted_predictions = instance_predictions[class_idx] * weight_params_sig[class_idx]
                    # Re-aggregate the logits for the current class
                    adjusted_logits[class_idx] = aggregation_fn({class_idx: adjusted_predictions})

                adjusted_probs = F.softmax(adjusted_logits, dim=-1).unsqueeze(0)
                task_model_probs = F.softmax(task_model_logits[i].unsqueeze(0), dim=-1)    

                # Calculate the KL divergence to match the task model's probabilities of this instance
                loss = kl_divergence_pytorch(task_model_probs, adjusted_probs)

                loss_weights=0
                for weights_sig in weight_params_sig:
                    loss_weights = torch.norm(weights_sig, p=1)

                # loss_weights=0
                # for weights_sig in weight_params_sig:
                #     loss_weights += torch.abs(weights_sig.sum() - 1)
                    

                loss += 1 * loss_weights

                # Perform gradient descent
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix({'Loss': loss.item()})

            # Update the weights for every class
            for i, weights in enumerate(weight_params_sig):
                optimized_attribute_weights[i] = weights.detach()
            
            optimized_attribute_weights_batch.append(optimized_attribute_weights)
            print(task_model_logits, adjusted_logits)
            print(task_model_probs, adjusted_probs)
        return optimized_attribute_weights_batch

    def identify_topk_candidates(self, attribute_weights, reference_classes, failed_predicted_logits, k):

        """
        Args:
            attribute_weights: A list of dictionaries (len = num of instances) with class indices as keys and tensors of shape [num_attributes] as values.
            reference_classes: A tensor of shape [batch_size] with the indices of the true classes for each instance.
            failed_predicted_logits: A tensor of shape [batch_size, num_classes] with the predicted logits for each instance.
            k: The number of top-k attributes to identify.
        Returns:
            identified_attributes: A list of dictionaries for each instance in the batch with 
                                    class names as keys and a dictionary with 'attribute'(attribute names)  and 'weights' as values.
        """
        
        # for i in range(len(attribute_weights)):
        #     for class_idx, weights in attribute_weights[i].items():
        #         print(f"Instance {i+1}, Class {class_idx}: {weights}")

        batch_size = reference_classes.shape[0]
        identified_attributes_per_instance = []
        # Iterate over every instance in the batch
        for i in range(batch_size):
            identified_attributes = {}
            # Iterate over the attributes weights for each class
            for class_idx, weights in attribute_weights[i].items():
                # If the weights are all ones, skip the class
                if torch.all(weights == 1):
                    continue


                class_name = self.class_names[class_idx]
                identified_attributes[class_name] = {
                    'attributes': self.attribute_names_per_class[class_name],
                    'weights': attribute_weights[i][class_idx]  
                }
    
                # If the current attribute class same the sucefully predicted PIM class, we want the names of the attributes that we killed (least weights)
                if class_idx == reference_classes[i] or args.explanation_type == 'logit_flip':
                    w, idx = torch.topk(weights, k, largest=False, dim=0)
                else:
                    # Else we want the names of the attributes that was left 
                    w, idx = torch.topk(weights, k, largest=True, dim=0)

                # # Get the top-k attributes for the current class
                # w, idx = torch.topk(attribute_weights[i][class_idx], k, largest=False, dim=0)  # w: [k], idx: [k]

                idx = idx.cpu().numpy()
                identified_attributes[class_name] = {
                    'attributes': [self.attribute_names_per_class[class_name][i] for i in idx],  # attribute names with -top-k weights
                    'weights': w  # top-k weights
                }

            identified_attributes_per_instance.append(identified_attributes)
        
        return identified_attributes_per_instance
    
    def plot_explanations(self, failed_images, identified_attributes_per_instance,
                        true_class_names, pim_class_names, predicted_class_names, 
                        max_description_length=800, save_path=None, choice="KLD"):

        failed_images = self.inverse_normalization(failed_images)
        # Convert to uint8
        failed_images = (failed_images * 255.).type(torch.uint8)
        images_np = failed_images.permute(0, 2, 3, 1).cpu().numpy()
        
        # Number of images
        batch_size = failed_images.shape[0]
        
        # Determine the necessary number of columns for each image based on the length of its description
        num_columns = [2] * batch_size  # Start with 2 columns for each: one for the image, one for text
        for i, attributes in enumerate(identified_attributes_per_instance):
            description_text = 'Classes Flipped: \n'
            for class_name, attr_details in attributes.items():
                description_text += f'\nAttributes Weights of {class_name}:\n'
                for attr, weight in zip(attr_details['attributes'], attr_details['weights']):
                    description_text += f"{attr}: {weight:.2f}\n"
            # Determine if additional columns are needed based on the description length
            extra_columns = int(len(description_text) / max_description_length)
            num_columns[i] += extra_columns

        # Set up the plot with a dynamic number of columns
        max_columns = max(num_columns)  # Find the max number of columns needed
        fig, axes = plt.subplots(nrows=batch_size, ncols=max_columns, figsize=(8 * max_columns, batch_size * 6),
                                gridspec_kw={'width_ratios': [3] + [1] * (max_columns - 1)})

        if batch_size == 1:  # Make sure axes is iterable
            axes = [axes]
        axes = np.array(axes)  # Ensure axes is a NumPy array for easy indexing

        # Iterate through each image
        for i, ax_row in enumerate(axes):
            ax_img = ax_row[0]  # Image column is always the first one
            img = images_np[i]
            ax_img.imshow(img)
            ax_img.axis('off')

            # Set the title with class names
            ax_img.set_title(f'Ground Truth: {true_class_names[i]}\nPIM: {pim_class_names[i]}     Task Model: {predicted_class_names[i]}', fontsize=14)

            # Initialize description_text for each image
            description_text = f'Classes Flipped: \n'  # Reset for each image
            identified_attributes = identified_attributes_per_instance[i]
            for class_name, attributes in identified_attributes.items():
                description_text += f'\nTop attributes used by PIM to predict {class_name}:\n'
                for attr, weight in zip(attributes['attributes'], attributes['weights']):
                    # description_text += f"{attr}: {weight:.2f}\n"
                    description_text += f"{attr}\n"
            
            # Splitting text into chunks for each necessary column
            text_chunks = [description_text[j:j + max_description_length] for j in range(0, len(description_text), max_description_length)]
            for j, text_chunk in enumerate(text_chunks):
                anchored_text = AnchoredText(text_chunk, loc="upper left", frameon=False, pad=0.5)
                ax_row[j + 1].add_artist(anchored_text)  # j+1 since the first column is for the image
                ax_row[j + 1].axis('off')  # Turn off axis for the text columns

        plt.tight_layout(pad=3, w_pad=0.5, h_pad=0.5)
        plt.subplots_adjust(left=0.5, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.4)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()


    def get_explanations(self, images, task_model_logits, pim_attribute_dict, pim_class_logits, true_classes, choice='kld', save_path=None):
        """
        Args:
            images: A torch image [batch_size, C, H, W] with the images for which we want to generate explanations.
            task_model_logits: [batch_size, num_classes]
            pim_attribute_dict: A list of dictionaries for each instance in the batch with class indices as keys and tensors of shape [num_attributes] as values.
            pim_class_logits: A tensor of shape [batch_size, num_classes] with the aggregated attribute logits for each instance.
            true_classes: A tensor of shape [batch_size] with the indices of the true classes for each instance.
            choice: 'kld' or 'logit_flip' (In KLD, we match the probabilities of the PIM to the task model. In logit_flip, we optimize the attribute weights for each top-ranked class per instance such that the true class's logit is maximized.)
        """

        # Convert pim_logits_dict to a list of dictionaries and get pim predictions
        num_classes = len(self.num_attributes_per_class)

        # pim_attribute_dict = []
        # for i in range(batch_size):
        #     pim_attribute_dict.append({class_idx: pim_logits_dict[class_idx][i]
        #                                    for class_idx in range(num_classes)})
        
        task_model_predictions = torch.argmax(task_model_logits, dim=1)
        pim_predictions = torch.argmax(pim_class_logits, dim=1)

        # print(task_model_logits, pim_class_logits, true_classes)
        if choice == 'logit_flip':

            # NOTE: We flip the top prediction of the pim model to understand what attributes are important for the top prediction of task model
            # We use the task models predictions as reference classes
            attribute_weights = self.optimize_attribute_weights(pim_attribute_dict, task_model_predictions, 
                                                            self.num_attributes_per_class, self.learning_rate, 
                                                            self.num_optimization_steps, self.aggregation_fn)
        elif choice == 'kld':
            attribute_weights = self.match_probabilities_to_task_model(pim_attribute_dict, task_model_logits, 
                                                                    self.num_attributes_per_class, self.learning_rate, 
                                                                    self.num_optimization_steps, self.aggregation_fn)
        else:
            raise ValueError(f"Invalid choice: {choice}. Please choose 'kld' or 'logit_flip'.")
        
        print(attribute_weights)
        # Get True class names and failed class names
        true_class_names = [self.class_names[i] for i in true_classes]
        pim_class_names = [self.class_names[i] for i in pim_predictions]
        predicted_class_names = [self.class_names[i] for i in task_model_predictions]

        identified_attributes_per_instance = self.identify_topk_candidates(attribute_weights, pim_predictions, 
                                                                           task_model_logits, k=5)
        
        self.plot_explanations(images, identified_attributes_per_instance, 
                               true_class_names, pim_class_names, predicted_class_names, 
                               save_path=save_path, choice=choice)
        
        return identified_attributes_per_instance

@torch.no_grad()
def evaluate_pim(data_loader, class_attributes_embeddings, class_attribute_prompt_list,
                    clip_model, classifier, pim_model, aggregator, device): 
    
    # Set the model to eval mode
    pim_model.eval()
    aggregator.eval()
    classifier.eval()
    
    gt_labels, task_model_logit_list, pim_attribute_logits_list, pim_class_logits  = [], [], [], []
    task_model_failure_indices, pim_model_failure_indices = [], []
    
    for i, (images_batch, labels, images_clip_batch) in enumerate(data_loader):
        
        images_batch = images_batch.to(device)
        labels = labels

        pim_image_embeddings, _, _ = pim_model(images_batch, return_task_logits=True)
        task_model_logits = classifier(images_batch)

        task_model_logits = task_model_logits.detach().cpu()

        # Cosine similarity between the pim image embeddings and the class_attributes_embeddings
        normalized_pim_image_embeddings = F.normalize(pim_image_embeddings, dim=-1)
        normalized_class_attributes_embeddings = F.normalize(class_attributes_embeddings, dim=-1)
        normalized_pim_image_embeddings = normalized_pim_image_embeddings.to(normalized_class_attributes_embeddings.dtype)
        pim_similarities = CLIP_LOGIT_SCALE*(normalized_pim_image_embeddings @ normalized_class_attributes_embeddings.t()) # (batch_size, num_classes*num_attributes_perclass)

        # Split the similarities into class specific dictionary
        pim_similarities = pim_similarities.to(torch.float32)
        pim_logits = []
        # Make a dictionary for each instance with class indices as keys and tensors of shape [num_attributes] as values and append to the list
        for i in range(pim_similarities.shape[0]):
            pim_similarities_dict = {}
            start = 0
            for j, class_prompts in enumerate(class_attribute_prompt_list):
                num_attributes = len(class_prompts)
                pim_similarities_dict[j] = pim_similarities[i, start:start+num_attributes].detach().cpu()
                start += num_attributes
            
            # Compute the pim logits using the multiheaded attention
            pim_logits.append(aggregator(pim_similarities_dict))
            pim_attribute_logits_list.append(pim_similarities_dict)
        pim_logits = torch.stack(pim_logits, dim=0)
        # Get the indices of the failed predictions
        task_model_failed_indices = (task_model_logits.argmax(dim=-1) != labels).nonzero(as_tuple=True)[0]
        pim_model_failed_indices = (pim_logits.argmax(dim=-1) != labels).nonzero(as_tuple=True)[0]


        gt_labels.append(labels)
        task_model_logit_list.append(task_model_logits)
        task_model_failure_indices.append(task_model_failed_indices)
        pim_model_failure_indices.append(pim_model_failed_indices)
        pim_class_logits.append(pim_logits)


    gt_labels = torch.cat(gt_labels, dim=0).detach().cpu()
    task_model_logit_list = torch.cat(task_model_logit_list, dim=0).detach().cpu()
    pim_class_logits = torch.cat(pim_class_logits, dim=0).detach().cpu()

    # Compute the task model and pim model failure indices
    task_model_failure_indices = (task_model_logit_list.argmax(dim=-1) != gt_labels).nonzero(as_tuple=True)[0].detach().cpu().tolist()
    pim_model_failure_indices = (pim_class_logits.argmax(dim=-1) != gt_labels).nonzero(as_tuple=True)[0].detach().cpu().tolist()

    # task_model_failure_indices = torch.cat(task_model_failure_indices, dim=0).detach().cpu().tolist()
    # pim_model_failure_indices = torch.cat(pim_model_failure_indices, dim=0).detach().cpu().tolist()

    return gt_labels, task_model_logit_list, pim_attribute_logits_list, pim_class_logits, task_model_failure_indices, pim_model_failure_indices
        
def main(args):

    ########################### Create the model ############################
    clip_model, clip_transform = clip.load(args.clip_model_name, device=args.device)
    clip_model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    

    classifier, train_transform, test_transform = build_classifier(args.classifier_name, num_classes=args.num_classes, 
                                                                    pretrained=args.use_imagenet_pretrained, 
                                                                    checkpoint_path=args.classifier_checkpoint_path)

    mapper,_, _ = build_classifier(args.classifier_name, num_classes=args.num_classes, pretrained=True, checkpoint_path=None)
    
    cutmix = CutMix(args.cutmix_alpha, args.num_classes)
    pim_model = TaskMapping(task_model=classifier, mapping_model=mapper, 
                              task_layer_name=args.task_layer_name, vlm_dim=args.vlm_dim, 
                              mapping_output_size=mapper.feature_dim, cutmix_fn=cutmix)
    

    print(f"Loaded Classifier checkpoint from {args.classifier_checkpoint_path}")
    

    train_loader, val_loader, test_loader, class_names = load_data(args, train_transform, test_transform, clip_transform)

    class_attributes_embeddings_prompts = torch.load(args.attributes_embeddings_path)
    # Get the class attribute prompts and their embeddings
    class_attribute_prompts = class_attributes_embeddings_prompts["class_attribute_prompts"] # List of list of prompts
    class_attributes_embeddings = class_attributes_embeddings_prompts["class_attributes_embeddings"]

    assert len(class_attribute_prompts) == args.num_classes, "Number of classes does not match the number of class attributes"

    num_attributes_per_cls = [len(attributes) for attributes in class_attribute_prompts]
    # Extract the attribute names from the prompts
    attribute_names_per_class = {}
    for i in range(len(class_attribute_prompts)):

        class_name = class_names[i].lower()
        attribute_names_per_class[class_names[i]] = [prompt.replace(f"This is a photo of {class_name} with ", "") for prompt in class_attribute_prompts[i]]
    
    # Create the PIM model
    if args.attribute_aggregation == "mha":
        aggregator = MultiHeadedAttentionSimilarity(args.num_classes, num_attributes_per_cls=num_attributes_per_cls, num_heads=1, out_dim=1)
    elif args.attribute_aggregation == "mean":
        aggregator = MeanAggregator(num_classes=args.num_classes, num_attributes_per_cls=num_attributes_per_cls)
    elif args.attribute_aggregation == "max":
        aggregator = MaxAggregator(num_classes=args.num_classes, num_attributes_per_cls=num_attributes_per_cls)
    else:
        raise Exception("Invalid attribute aggregation method")

    if args.resume_checkpoint_path:
        state = torch.load(args.resume_checkpoint_path)
        epoch = state["epoch"]
        classifier.load_state_dict(state["classifier"])
        pim_model.load_state_dict(state["pim_model"])
        aggregator.load_state_dict(state[f"aggregator"])

        print(f"Loaded checkpoint from {args.resume_checkpoint_path}")


    print(f"Built {args.classifier_name} classifier with checkpoint path: {args.classifier_checkpoint_path}")
    print(f"Built {args.classifier_name} mapper with checkpoint path: {args.classifier_checkpoint_path} from layer {args.task_layer_name} epoch {epoch}")
    print(f"Built MultiHeadedAttention with {args.num_classes} classes and {num_attributes_per_cls} attributes per class")

    clip_model.to(device)
    classifier.to(device)
    pim_model.to(device)
    aggregator.to(device)

    # Evaluate the PIM model
    gt_labels, task_model_logit_list, pim_attribute_logits_list, pim_class_logits, task_model_failure_indices, pim_model_failure_indices = evaluate_pim(test_loader, class_attributes_embeddings, class_attribute_prompts,
                                                                                                                            clip_model, classifier, pim_model, aggregator, device)
    
    # Find the indices where the task model failed but the PIM model succeeded
    pim_sucess_task_fail_indices = [i for i in task_model_failure_indices if i not in pim_model_failure_indices]
    print(f"Total number of task model failed images: {len(task_model_failure_indices)}")
    print(f"Total number of PIM failed images: {len(pim_model_failure_indices)}")
    print(f"Num of task failed pim succeded images: {len(pim_sucess_task_fail_indices)}")
    # Create an instance of the PIM_Explanations class
    pim_explanations = PIM_Explanations(attribute_names_per_class, num_attributes_per_cls, aggregation_fn=aggregator)


    # Randomly select a few images from the pim_sucess_task_fail_indices
    picked_indices = np.random.choice(pim_sucess_task_fail_indices, 50, replace=False)

    # attribute frequencies dictionary
    attribute_frequencies = {}
    # Ierate over the failed images and get the explanations
    for i in picked_indices:
        indices = [i]
        # Subselect the task model failed (TF) images
        task_model_failed_images = torch.stack([test_loader.dataset[i][0] for i in indices])
        TF_gt_labels = gt_labels[indices]
        TF_task_model_logits = task_model_logit_list[indices]
        TF_pim_attribute_logits_list = [pim_attribute_logits_list[i] for i in indices]
        TF_pim_class_logits = pim_class_logits[indices]

        # Get the explanations
        img_path = os.path.join(args.save_dir, f"explanations_{i}.png")
        print(img_path)
        identified_attributes_per_instance = pim_explanations.get_explanations(task_model_failed_images, TF_task_model_logits, TF_pim_attribute_logits_list, TF_pim_class_logits,
                                                TF_gt_labels, choice=args.explanation_type, save_path=img_path)
        
        # # Add the identified attributes to the attribute_frequencies dictionary
        # for i, attributes in enumerate(identified_attributes_per_instance):
        #     for class_name, attr_details in attributes.items():
        #         for attr in attr_details['attributes']:
        #             if attr in attribute_frequencies:
        #                 attribute_frequencies[attr] += 1
        #             else:
        #                 attribute_frequencies[attr] = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet on WILDS Dataset')

    parser.add_argument('--data_dir', type=str, default='./data', help='Path to the data directory')
    parser.add_argument('--domain_name', type=str, default='clipart', help='Domain to use for training')
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='Name of the dataset')
    parser.add_argument('--attributes', nargs='+', type=int, default=None, help='Attributes to use for training')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes in the dataset')
    parser.add_argument('--method', type=str, default='baseline', help='Baseline or PIM for failure detection')
    parser.add_argument('--score', type=str, default='msp', help='Failure detection score - msp/energy/pe')
    parser.add_argument('--eval_dataset', type=str, default='cifar100', help='Evaluation dataset')
    parser.add_argument('--filename', type=str, default='cifar100c.log', help='Filename')
    parser.add_argument('--cifar100c_corruption', default="gaussian_blur", type=str, help='Corruption type')
    parser.add_argument('--severity', default=5, type=int, help='Severity of corruption')
    
    parser.add_argument('--use_saved_features',action = 'store_true', help='Whether to use saved features or not')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--img_size', type=int, default=75, help='Image size for the celebA dataloader only')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')

    parser.add_argument('--task_layer_name', type=str, default='model.layer2', help='Name of the layer to use for the task model')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='Alpha value for the beta distribution for cutmix')
    parser.add_argument('--augmix_severity', type=int, default=3, help='Severity of the augmix')
    parser.add_argument('--augmix_alpha', type=float, default=1.0, help='Alpha value for the beta distribution for augmix')
    parser.add_argument('--augmix_prob', type=float, default=0.2, help='Probability of using augmix')
    parser.add_argument('--cutmix_prob', type=float, default=0.2, help='Probability of using cutmix')

    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs before using cutmix')
    parser.add_argument('--task_failure_discrepancy_weight', type=float, default=2.0, help='Weight for the discrepancy loss')
    parser.add_argument('--task_success_discrepancy_weight', type=float, default=1.5, help='Weight for the discrepancy loss')

    parser.add_argument('--attributes_path', type=str, help='Path to the attributes file')
    parser.add_argument('--attributes_embeddings_path', type=str, help='Path to the attributes embeddings file')

    parser.add_argument('--attribute_aggregation', default='mha', choices=['mha', 'mean', 'max'], help='Type of aggregation of the attribute scores')

    parser.add_argument('--classifier_name', required=True,  help='Name of the classifier to use sam_vit_h, mae_vit_large_patch16, dino_vits16, resnet50, resnet50_adv_l2_0.1, resnet50_adv_l2_0.5, resnet50x1_bitm, resnetv2_101x1_bit.goog_in21k, deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50, fcn_resnet101')
    parser.add_argument('--classifier_checkpoint_path', type=str, help='Path to checkpoint to load the classifier from')
    parser.add_argument('--classifier_dim', type=int, default=None, help='Dimension of the classifier output')

    parser.add_argument('--use_imagenet_pretrained', action='store_true', help='Whether to use imagenet pretrained weights or not')
    parser.add_argument('--clip_model_name', default='ViT-B/32', help='Name of the CLIP model to use.')
    parser.add_argument('--prompt_path', type=str, help='Path to the prompt file')

    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, choices=['adam','adamw', 'sgd'], default='adamw', help='Type of optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--aggregator_learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--scheduler', type=str, choices=['MultiStepLR', 'cosine'], default='cosine', help='Type of learning rate scheduler to use')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation frequency')
    parser.add_argument('--save_dir', type=str, default='./logs', help='Directory to save the results')
    parser.add_argument('--prefix', type=str, default='', help='prefix to add to the save directory')

    parser.add_argument('--vlm_dim', type=int, default=512, help='Dimension of the VLM embeddings')
    parser.add_argument('--resume_checkpoint_path', type=str, help='Path to checkpoint to resume training from')
    
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of gpus for DDP per node')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for DDP')

    parser.add_argument('--explanation_type', type=str, default='kld', help='Type of explanation to generate - kld or logit_flip')
    parser.add_argument('--sparsity_weight', type=float, default=0.0, help='Weight for the sparsity loss')


    args = parser.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    # Print the arguments
    print(args)
    sys.stdout.flush()
    
    # Make directory for saving results
    args.save_dir = get_save_dir(args)    
    args.save_dir = os.path.join(args.save_dir, f'explanations_{args.explanation_type}')
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\nResults will be saved to {args.save_dir}")
    
    corruption_list = ['brightness', 'defocus_blur', 'fog', 'gaussian_blur', 'glass_blur', 'jpeg_compression', 'motion_blur', 'saturate','snow','speckle_noise', 'contrast', 'elastic_transform', 'frost', 'gaussian_noise', 'impulse_noise', 'pixelate','shot_noise', 'spatter','zoom_blur']
    severity = [4]
    if args.eval_dataset == 'cifar100c':
        for c in corruption_list:
            for s in severity:
                args.cifar100c_corruption = c
                args.severity = s
                print(f'Corruption = {args.cifar100c_corruption}, Severity = {args.severity}')
                seed_everything(args.seed)
                main(args)
    elif args.eval_dataset =='pacs':
        if args.method=='baseline':
            scores_all = ['msp', 'energy', 'pe']
            for score in scores_all:
                args.score = score
                for dn in ['art_painting', 'cartoon', 'photo', 'sketch']:
                    args.domain_name = dn
                    seed_everything(args.seed)
                    main(args)
        else:
            for dn in ['art_painting', 'cartoon', 'photo', 'sketch']:
                    args.domain_name = dn
                    seed_everything(args.seed)
                    main(args)

    else:
        seed_everything(args.seed)
        main(args)