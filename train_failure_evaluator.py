import os
import sys
import copy

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from torch.utils.data.dataset import Subset

from torchmetrics.classification import Accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning as L
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger
import argparse
from tqdm import tqdm
from functools import partial
from datetime import datetime

import clip
import csv
from tqdm import tqdm
import numpy as np
import random
import pickle
import json

from train_task_distillation import get_dataset, get_CLIP_text_encodings, build_classifier

from models.mapping import TaskMapping, MultiHeadedAttentionSimilarity, MultiHeadedAttention, print_layers, MaxAggregator, MeanAggregator
from utils_proj import SimpleDINOLoss, compute_accuracy, compute_similarities, CutMix, MyAugMix, find_normalization_parameters

CLIP_LOGIT_SCALE = 100

def get_save_dir(args):
    if args.resume_checkpoint_path:
        return os.path.dirname(args.resume_checkpoint_path)

    projector_name = "mapper"

    att_name = ""
    if args.dataset_name == "NICOpp":
        if args.attributes:
            att_name = "".join([str(att) for att in args.attributes])
            att_name = f"att_{att_name}"
        else:
            att_name = "att_all"

    if args.dataset_name == 'domainnet' and args.domain_name:
        att_name = f"{args.domain_name}"

    if args.classifier_checkpoint_path:
        save_dir = os.path.dirname(os.path.dirname(args.classifier_checkpoint_path))
        save_dir = os.path.join(save_dir, att_name, projector_name)
    else:
        save_dir = os.path.join(args.save_dir, args.dataset_name, args.classifier_name, att_name, projector_name)
    
    save_dir_details = f"{args.prefix}_agg_{args.attribute_aggregation}_bs_{args.batch_size}_lr_{args.learning_rate}_augmix_prob_{args.augmix_prob}_cutmix_prob_{args.cutmix_prob}_scheduler_warmup_epoch_{args.warmup_epochs}_layer_{args.task_layer_name}"

    return os.path.join(save_dir, save_dir_details)

def progbar_wrapper(iterable, total, **kwargs):
    """Wraps the iterable with tqdm for global rank zero.

    Args:
        iterable: the iterable to wrap with tqdm
        total: the total length of the iterable, necessary in case the number of batches was limited.

    """
    if fabric.is_global_zero:
        return tqdm(iterable, total=total, **kwargs)
    return iterable
   
def train_one_epoch(data_loader, class_attributes_embeddings, class_attribute_prompt_list,
                    augmix,cutmix, clip_model, classifier, pim_model, aggregator, optimizer, epoch): 

    # Set the models to train mode
    pim_model.train()
    aggregator.train()
    clip_model.eval()
    classifier.eval()

    total_loss = 0
    total_task_model_acc = 0
    total_pim_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Training Epoch {epoch+1}"
    )
    class_attributes_embeddings = fabric.to_device(class_attributes_embeddings)
    for i, (images_batch, labels, images_clip_batch) in enumerate(pbar):
        labels_orig = labels.clone()
        augmix_flag =bool(torch.bernoulli(torch.tensor(args.augmix_prob)))
        if epoch >= args.warmup_epochs and augmix_flag :
            images_batch = augmix (images_batch)
        cutmix_flag = bool(torch.bernoulli(torch.tensor(args.cutmix_prob)))
        
        # After certain epoch, select_cutmix with probability 0.5
        if epoch >= args.warmup_epochs and cutmix_flag:
            images_batch, labels = cutmix(images_batch, labels) # Cutmix the images and labels, labels are not one hot encoded anymore

        images_batch = fabric.to_device(images_batch)
        images_clip_batch = fabric.to_device(images_clip_batch)
        labels = fabric.to_device(labels)

        optimizer.zero_grad()

        pim_image_embeddings, task_model_logits, _ = pim_model(images_batch, labels, return_task_logits=True)

        # Cosine similarity between the pim image embeddings and the class_attributes_embeddings
        normalized_pim_image_embeddings = F.normalize(pim_image_embeddings, dim=-1)
        normalized_class_attributes_embeddings = F.normalize(class_attributes_embeddings, dim=-1)

        # convertthe normalized_pim_image_embeddings to data type of normalized_class_attributes_embeddings
        normalized_pim_image_embeddings = normalized_pim_image_embeddings.to(normalized_class_attributes_embeddings.dtype)
        pim_similarities = CLIP_LOGIT_SCALE*(normalized_pim_image_embeddings @ normalized_class_attributes_embeddings.t()) # (batch_size, num_classes*num_attributes_perclass)

        # convert the pim_similarities to the data type of the data type of aggregator
        pim_similarities = pim_similarities.to(torch.float32)

        # Split the similarities into class specific dictionary
        pim_similarities_dict = {}
        start = 0
        for i, class_prompts in enumerate(class_attribute_prompt_list):
            num_attributes = len(class_prompts)
            pim_similarities_dict[i] = pim_similarities[:, start:start+num_attributes]
            start += num_attributes
            
        pim_logits = aggregator(pim_similarities_dict)

        loss = F.cross_entropy(pim_logits, labels, reduction = 'none') # Can accomodate the non one hot encoded labels as well # loss is of shape (batch_size,)

        # Check if in this batch if for any samples the task model is correct and the pim model is incorrect
        # if so update a count of such samples
        if (not augmix_flag and not cutmix_flag) and epoch >= args.warmup_epochs:
            task_prediction = torch.argmax(task_model_logits, dim=-1)
            pim_prediction = torch.argmax(pim_logits, dim=-1)
            correct_task_pim_incorrect_idx= torch.where((task_prediction == labels) & (pim_prediction != labels))[0]
            loss[correct_task_pim_incorrect_idx]= loss[correct_task_pim_incorrect_idx]*args.task_success_discrepancy_weight # Weight the loss by the number of such samples
            print(f"Correct task, incorrect pim: {len(correct_task_pim_incorrect_idx)}")

            incorrect_task_pim_incorrect_idx= torch.where((task_prediction != labels) & (pim_prediction != labels))[0]
            loss[incorrect_task_pim_incorrect_idx]= loss[incorrect_task_pim_incorrect_idx]*args.task_failure_discrepancy_weight # Weight the loss by the number of such samples
            print(f"incorrect task, incorrect pim: {len(incorrect_task_pim_incorrect_idx)}")
        
        loss= loss.mean()

        fabric.backward(loss)
        
        optimizer.step()

        task_model_probs = F.softmax(task_model_logits, dim=-1)
        pim_probs = F.softmax(pim_logits, dim=-1)
        
        task_model_acc = compute_accuracy(task_model_probs, labels_orig)
        pim_acc = compute_accuracy(pim_probs, labels_orig)

        total_task_model_acc += task_model_acc
        total_pim_acc += pim_acc

        total_loss += loss.item()

    total_loss = fabric.all_gather(total_loss).mean() / len(data_loader)
    total_task_model_acc = fabric.all_gather(total_task_model_acc).mean() / len(data_loader)
    total_pim_acc = fabric.all_gather(total_pim_acc).mean() / len(data_loader)

    performance_dict = {"total_loss": total_loss, "task_model_acc":total_task_model_acc *100., "pim_acc": total_pim_acc *100.}


    return performance_dict

@torch.no_grad()
def validate(data_loader, class_attributes_embeddings, class_attribute_prompt_list,
                    clip_model, classifier, pim_model, aggregator, optimizer, epoch): 
    
    # Set the model to eval mode
    pim_model.eval()
    aggregator.eval()
    classifier.eval()
    total_loss = 0
    total_task_model_acc = 0
    total_pim_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Test Epoch {epoch+1}"
    )
    correct_pim =0.
    correct_task = 0.
    total = 0.

    class_attributes_embeddings = fabric.to_device(class_attributes_embeddings)
    
    for i, (images_batch, labels, images_clip_batch) in enumerate(pbar):
        
        images_batch = fabric.to_device(images_batch)
        images_clip_batch = fabric.to_device(images_clip_batch)
        labels = fabric.to_device(labels)

        pim_image_embeddings, task_model_logits, _ = pim_model(images_batch, return_task_logits=True)

        # Cosine similarity between the pim image embeddings and the class_attributes_embeddings
        normalized_pim_image_embeddings = F.normalize(pim_image_embeddings, dim=-1)
        normalized_class_attributes_embeddings = F.normalize(class_attributes_embeddings, dim=-1)
        normalized_pim_image_embeddings = normalized_pim_image_embeddings.to(normalized_class_attributes_embeddings.dtype)
        normalized_pim_image_embeddings = fabric.to_device(normalized_pim_image_embeddings)
        pim_similarities = CLIP_LOGIT_SCALE*(normalized_pim_image_embeddings @ normalized_class_attributes_embeddings.t()) # (batch_size, num_classes*num_attributes_perclass)

        # Split the similarities into class specific dictionary
        pim_similarities = pim_similarities.to(torch.float32)
        pim_similarities_dict = {}
        start = 0
        for i, class_prompts in enumerate(class_attribute_prompt_list):
            num_attributes = len(class_prompts)
            pim_similarities_dict[i] = pim_similarities[:, start:start+num_attributes]
            start += num_attributes
        
        # Compute the pim logits using the multiheaded attention
        pim_logits = aggregator(pim_similarities_dict)

        loss = F.cross_entropy(pim_logits, labels)

        task_model_probs = F.softmax(task_model_logits, dim=-1)
        pim_probs = F.softmax(pim_logits, dim=-1)
        
        
        _, pim_predicted = pim_probs.max(1)
        total += labels.size(0)
        correct_pim += pim_predicted.eq(labels).sum().item()

        _, task_predicted = task_model_probs.max(1)
        correct_task += task_predicted.eq(labels).sum().item()

        task_model_acc = compute_accuracy(task_model_probs, labels)
        pim_acc = compute_accuracy(pim_probs, labels)

        total_task_model_acc += task_model_acc
        total_pim_acc += pim_acc

        total_loss += loss.item()

    total_loss = fabric.all_gather(total_loss).mean() / len(data_loader)
    # total_task_model_acc = fabric.all_gather(total_task_model_acc).mean() / len(data_loader)
    # total_pim_acc = fabric.all_gather(total_pim_acc).mean() / len(data_loader)
    
    total_correct_pim = fabric.all_gather(correct_pim).mean() 
    total_correct_task = fabric.all_gather(correct_task).mean() 
    total = fabric.all_gather(total).mean()
    total_pim_acc = total_correct_pim/total
    total_task_model_acc = total_correct_task/total

    performance_dict = {"total_loss": total_loss, "task_model_acc":total_task_model_acc *100., "pim_acc": total_pim_acc *100.}

    return performance_dict

def main(args):
    
    ########################### Create the model ############################
    clip_model, clip_transform = clip.load(args.clip_model_name, device=args.device)
    clip_model.eval()    
    print (f"args.classifier_name: {args.classifier_name}")

    classifier, train_transform, test_transform = build_classifier(args.classifier_name, num_classes=args.num_classes, 
                                                                    pretrained=args.use_imagenet_pretrained, 
                                                                    checkpoint_path=args.classifier_checkpoint_path)
    mapping_name = args.classifier_name
    if args.dataset_name =="imagenet":
        mapping_name= f"{mapping_name}_v2"
    print(f"mapping_name: {mapping_name}")
    mapper,_, _ = build_classifier(mapping_name, num_classes=args.num_classes, pretrained=True, checkpoint_path=None)
    
    cutmix = CutMix(args.cutmix_alpha, args.num_classes)
    pim_model = TaskMapping(task_model=classifier, mapping_model=mapper, 
                              task_layer_name=args.task_layer_name, vlm_dim=args.vlm_dim, 
                              mapping_output_size=mapper.feature_dim, cutmix_fn=cutmix)
    
    # This is for the cross modal attention between PIM and the class attributes
    # mha = MultiHeadedAttention(args.num_classes, in_dim=args.vlm_dim, num_heads=1)

    if not os.path.exists(args.attributes_embeddings_path):
        with open(args.attributes_path, 'r') as f:
            class_attributes_dict = json.load(f)
        
        # For each class, get the attributes clip text embeddings
        class_attributes_embeddings = []
        class_attribute_prompts = []
        for class_names, attributes_dict in class_attributes_dict.items():
            
            # attributes has two lists, one for core attributes and one for non core additional attributes
            #attributes = attributes_dict['core_attributes']
            attributes = attributes_dict
            prompt = [f"This is a photo of {class_names} with {attribute}" for attribute in attributes]
            with torch.no_grad():
                tokenized_prompt = clip.tokenize(prompt).to(args.device)
                class_attributes_embeddings.append(clip_model.encode_text(tokenized_prompt))
                class_attribute_prompts.append(prompt)

        class_attributes_embeddings = torch.cat(class_attributes_embeddings, dim=0) # Shape: [num_classes*num_attributes_perclass, embedding_dim]

        # Make the parent directory if it does not exist
        os.makedirs(os.path.dirname(args.attributes_embeddings_path), exist_ok=True)
        # Save the class attributes embeddings and the class prompts in a single file
        torch.save({"class_attributes_embeddings": class_attributes_embeddings, 
                    "class_attribute_prompts": class_attribute_prompts}, args.attributes_embeddings_path)
        
    else:
        class_attributes_embeddings_prompts = torch.load(args.attributes_embeddings_path)
        class_attribute_prompts = class_attributes_embeddings_prompts["class_attribute_prompts"]
        class_attributes_embeddings = class_attributes_embeddings_prompts["class_attributes_embeddings"]

        assert len(class_attribute_prompts) == args.num_classes, "Number of classes does not match the number of class attributes"

    num_attributes_per_cls = [len(attributes) for attributes in class_attribute_prompts]
    
    if args.attribute_aggregation == "mha":
        aggregator = MultiHeadedAttentionSimilarity(args.num_classes, num_attributes_per_cls=num_attributes_per_cls, num_heads=1, out_dim=1)
    elif args.attribute_aggregation == "mean":
        aggregator = MeanAggregator(num_classes=args.num_classes, num_attributes_per_cls=num_attributes_per_cls)

    elif args.attribute_aggregation == "max":
        aggregator = MaxAggregator(num_classes=args.num_classes, num_attributes_per_cls=num_attributes_per_cls)

    else:
        raise Exception("Invalid attribute aggregation method")
    fabric.print(f"Built {args.classifier_name} classifier with checkpoint path: {args.classifier_checkpoint_path}")
    fabric.print(f"Built {args.classifier_name} mapper")
    fabric.print(f"Built MultiHeadedAttention with {args.num_classes} classes and {num_attributes_per_cls} attributes per class")


    clip_model = fabric.to_device(clip_model)
    fabric.to_device(classifier)
    fabric.to_device(pim_model)
    fabric.to_device(aggregator)

    ########################### Load the dataset ############################

    # Create the data loader and wrap them with Fabric
    train_dataset, val_dataset, test_dataset, failure_dataset, class_names = get_dataset(args.dataset_name, train_transform, test_transform, 
                                                            data_dir=args.data_dir, clip_transform=clip_transform, 
                                                            img_size=args.img_size, domain_name=args.domain_name, 
                                                            return_failure_set=True)#FIXME: for PACS, clip_transform=clip_transform give this error AttributeError: 'Tensor' object has no attribute 'convert' return image.convert("RGB")

    try:
        transform_pipeline =train_dataset.dataset.transform1 
    except:
        transform_pipeline =train_dataset.transform1
    mean, std = find_normalization_parameters (transform_pipeline)
    augmix = MyAugMix(severity=args.augmix_severity, alpha=args.augmix_alpha, mean=mean, std=std) 
    

    if args.dataset_name in ['cifar100']:
        # Merge falure dataset with train dataset
        train_dataset = ConcatDataset([train_dataset, val_dataset])

    fabric.print(f"Using {args.dataset_name} dataset")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader)

    fabric.print(f"Number of training examples: {len(train_loader.dataset)}")
    fabric.print(f"Number of validation examples: {len(val_loader.dataset)}")
    fabric.print(f"Number of test examples: {len(test_loader.dataset)}")

    ########################### Create the optimizer ############################

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(pim_model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(pim_model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "sgd":

        optimizer = torch.optim.SGD(pim_model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    else:
        raise Exception("Invalid optimizer")
    
    # Add the aggregator parameters to the optimizer with a different learning rate
    optimizer.add_param_group({"params": aggregator.parameters(), "lr": args.aggregator_learning_rate})

    # Learning rate scheduler
    if args.scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    else:
        scheduler = None

    start_epoch = 0

    state = {"clip_model": clip_model, 
            "classifier": classifier,
            "pim_model": pim_model,
            "aggregator": aggregator,
            "optimizer": optimizer, 
            "scheduler": scheduler,
            "epoch": start_epoch}

    if args.resume_checkpoint_path:
        state = fabric.load(args.resume_checkpoint_path)
        start_epoch = state["epoch"]

        # Load the model and optimizer
        clip_model = state["clip_model"]
        classifier = state["classifier"]
        pim_model = state["pim_model"]
        aggregator = state[f"{args.aggregator}"]
        optimizer = state["optimizer"]
        scheduler = state["scheduler"]

        fabric.print(f"Loaded checkpoint from {args.resume_checkpoint_path} at epoch {start_epoch}")
    if start_epoch >= args.num_epochs:
        fabric.print(f"Already finished training for {args.num_epochs} epochs. Exiting...")
        return

    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.num_epochs):
        
        train_performance_dict = train_one_epoch(
                                            train_loader, class_attributes_embeddings, class_attribute_prompts,augmix,
                                            cutmix, clip_model, classifier, pim_model, aggregator, optimizer, epoch)
        
        if epoch % args.val_freq == 0:
            val_performance_dict = validate( 
                val_loader, class_attributes_embeddings, class_attribute_prompts, 
                                             clip_model, classifier, pim_model, aggregator, optimizer, epoch)

        if scheduler is not None:
            scheduler.step()
        
        # Print the losses
        fabric.print(f"Epoch: {epoch+1}/{args.num_epochs} | Train Loss: {train_performance_dict['total_loss']:.4f} | Val Loss: {val_performance_dict['total_loss']:.4f} | Train task model Acc: {train_performance_dict['task_model_acc']:.4f} | Val task model Acc: {val_performance_dict['task_model_acc']:.4f} | Train pim Acc: {train_performance_dict['pim_acc']:.4f} | Val pim Acc: {val_performance_dict['pim_acc']:.4f}")
        # Add train_ to all the keys
        train_performance_dict = {f"train_{key}": value for key, value in train_performance_dict.items()}
             
        # Add test_ to all the keys
        val_performance_dict = {f"test_{key}": value for key, value in val_performance_dict.items()}
        losses_dict = {**train_performance_dict, **val_performance_dict}


        fabric.log_dict(losses_dict, step=epoch)
        
        # Save best model based on validation loss
        if val_performance_dict["test_total_loss"] < best_val_loss:
            
            state.update(epoch=epoch)
            fabric.save(os.path.join(args.save_dir, "pim_weights_best.pth"), state)

            best_val_loss = val_performance_dict["test_total_loss"]
        
        if epoch % 5 == 0:
            state.update(epoch=epoch)
            fabric.save(os.path.join(args.save_dir, f"pim_weights_{epoch+1}.pth"), state)
    
    state.update(epoch=epoch)
    fabric.save(os.path.join(args.save_dir, "pim_weights_final.pth"), state)
    fabric.print(f"Finished training for {args.num_epochs} epochs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet on WILDS Dataset')

    parser.add_argument('--data_dir', type=str, default='./data', help='Path to the data directory')
    parser.add_argument('--domain_name', type=str, default='clipart', help='Domain to use for training')
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='Name of the dataset')
    parser.add_argument('--attributes', nargs='+', type=int, default=None, help='Attributes to use for training')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes in the dataset')
    
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


    args = parser.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    # Print the arguments
    print(args)
    sys.stdout.flush()
    
    # Make directory for saving results
    args.save_dir = get_save_dir(args)    
    os.makedirs(os.path.join(args.save_dir, 'lightning_logs'), exist_ok=True)
    
    print(f"\nResults will be saved to {args.save_dir}")
    
    with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    tb_logger = TensorBoardLogger(args.save_dir)
    csv_logger = CSVLogger(args.save_dir, flush_logs_every_n_steps=1)

    #accelarator = args.device
    fabric = L.Fabric(accelerator="gpu",num_nodes=args.num_nodes, devices=args.num_gpus, strategy="auto", loggers=[tb_logger, csv_logger])
    
    fabric.launch()
    print = fabric.print
    
    # The total number of processes running across all devices and nodes
    fabric.print(f"World size: {fabric.world_size}")  # 2 * 3 = 6
    
    seed_everything(args.seed)

    main(args)
