import os
import sys
import copy
try:
    del os.environ['OMP_PLACES']
    del os.environ['OMP_PROC_BIND']
except:
    pass

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
import logging
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, roc_curve

from train_task_distillation import get_dataset, get_CLIP_text_encodings, build_classifier

from models.mapping import TaskMapping, MultiHeadedAttentionSimilarity, MultiHeadedAttention, print_layers, MeanAggregator, MaxAggregator
from utils_proj import SimpleDINOLoss, compute_accuracy, compute_similarities, CutMix, MyAugMix, find_normalization_parameters, get_score, calc_gen_threshold, calc_accuracy_from_scores, compute_gde_scores, compute_ts, compute_auroc_fpr
from models.cluster import ClusterCreater


CLIP_LOGIT_SCALE = 100


class CIFAR100C(torch.utils.data.Dataset):
    def __init__(self, corruption='gaussian_blur', transform=None,clip_transform=None, level=0):
        numpy_path = f'data/CIFAR-100-C/{corruption}.npy'
        t = 10000 # We choose 10000 because, every numpy array has 50000 images, where the first 10000 images belong to severity 0 and so on. t is just an index for that
        self.transform = transform # Standard CIFAR100 test transform
        self.clip_transform = clip_transform
        self.data_ = np.load(numpy_path)[level*10000:(level+1)*10000,:,:,:] # Choosing 10000 images of a given severity
        self.data = self.data_[:t,:,:,:] # Actually redundant, I don't want to disturb the code structure
        self.targets_ = np.load('data/CIFAR-100-C/labels.npy')
        self.targets = self.targets_[:t] # We select the first 10000. The next 10000 is identical to the first 10000 and so on
        self.np_PIL = transforms.Compose([transforms.ToPILImage()])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image_ = self.data[idx,:,:,:]
        if self.transform:
            image = self.transform(image_)
            image_to_clip = self.clip_transform(self.np_PIL(image_))
        targets = self.targets[idx]
        return image, targets, image_to_clip


def get_failure_results(val_task_logits_list, val_labels_list, val_scores, 
                         test_task_logits_list, test_labels_list, test_scores, 
                         val_task_model_acc, test_task_model_acc, 
                         test_pim_acc, clip_class_level_acc, clip_attribute_level_acc,
                         threshold=None):
    """
    Evaluate model predictions against the true labels and compute various performance metrics.

    Args:
    - val_task_logits_list: Logits from the validation dataset.
    - val_labels_list: True labels for the validation dataset.
    - val_scores: Confidence or probability scores for the validation dataset.
    - test_task_logits_list: Logits from the test dataset.
    - test_labels_list: True labels for the test dataset.
    - test_scores: Confidence or probability scores for the test dataset.
    - threshold: Threshold for calculating accuracy based on scores.
    - val_task_model_acc: The true validation accuracy.
    - test_task_model_acc: The true test accuracy.
    - test_pim_acc: Test accuracy from another metric/model for comparison.
    - clip_class_level_acc: Class level accuracy from CLIP.
    - clip_attribute_level_acc: Attribute level accuracy from CLIP.

    Returns:
    - A dictionary containing various evaluated performance metrics.
    """

    if not threshold:
        threshold = calc_gen_threshold(val_scores, val_task_logits_list, val_labels_list, name='pim')

    # Calculate estimated accuracy based on the provided scores and threshold
    estimated_val_acc, val_estimated_success_failure_idx = calc_accuracy_from_scores(val_scores, threshold)
    estimated_test_acc, test_estimated_success_failure_idx = calc_accuracy_from_scores(test_scores, threshold)

    # Calculate the true success and failure indices
    val_true_success_failure_idx = torch.argmax(val_task_logits_list, 1) == val_labels_list
    test_true_success_failure_idx = torch.argmax(test_task_logits_list, 1) == test_labels_list

    # Compute AUROC and FPR at TPR95 for validation and test sets
    val_fpr_at_tpr, val_auroc = compute_auroc_fpr(val_true_success_failure_idx.cpu().numpy(), val_scores.cpu().numpy())
    test_fpr_at_tpr, test_auroc = compute_auroc_fpr(test_true_success_failure_idx.cpu().numpy(), test_scores.cpu().numpy())

    # Compute confusion matrices and recall for validation and test sets
    cm_val = confusion_matrix(val_true_success_failure_idx.cpu().numpy(), val_estimated_success_failure_idx.cpu().numpy())
    cm_test = confusion_matrix(test_true_success_failure_idx.cpu().numpy(), test_estimated_success_failure_idx.cpu().numpy())
    failure_recall_val = cm_val[0,0]/(cm_val[0,0]+cm_val[0,1])
    success_recall_val = cm_val[1,1]/(cm_val[1,0]+cm_val[1,1])
    failure_recall_test = cm_test[0,0]/(cm_test[0,0]+cm_test[0,1])
    success_recall_test = cm_test[1,1]/(cm_test[1,0]+cm_test[1,1])

    # Compute Matthews correlation coefficient for validation and test sets
    mathews_corr_val = matthews_corrcoef(val_true_success_failure_idx.cpu().numpy(), val_estimated_success_failure_idx.cpu().numpy())
    mathews_corr_test = matthews_corrcoef(test_true_success_failure_idx.cpu().numpy(), test_estimated_success_failure_idx.cpu().numpy())

    # Organize results into a dictionary
    results = {
        "true_val_acc": val_task_model_acc.item() if isinstance(val_task_model_acc, torch.Tensor) else val_task_model_acc,
        "estimated_val_acc": estimated_val_acc.item() if isinstance(estimated_val_acc, torch.Tensor) else estimated_val_acc,
        "true_test_acc": test_task_model_acc.item() if isinstance(test_task_model_acc, torch.Tensor) else test_task_model_acc,
        "estimated_test_acc": estimated_test_acc.item() if isinstance(estimated_test_acc, torch.Tensor) else estimated_test_acc,
        "val_cm": cm_val.tolist(), 
        "test_cm": cm_test.tolist(),
        "val_failure_recall": failure_recall_val,
        "val_success_recall": success_recall_val,
        "test_failure_recall": failure_recall_test,
        "test_success_recall": success_recall_test,
        "val_mathews_corr": mathews_corr_val,
        "test_mathews_corr": mathews_corr_test,
        "val_fpr_at_tpr": val_fpr_at_tpr,
        "val_auroc": val_auroc,
        "test_fpr_at_tpr": test_fpr_at_tpr,
        "test_auroc": test_auroc,
        "pim_model_test_acc": test_pim_acc,
        "clip_class_level_acc": clip_class_level_acc,
        "clip_attribute_level_acc": clip_attribute_level_acc,
    }

    return results

@torch.no_grad()
def clip_attribute_classifier(data_loader, class_attributes_embeddings, class_attribute_prompt_list,
                                clip_model, classifier, pim_model, aggregator, class_names): 
    
    # Set the model to eval mode
    clip_model.eval()
    classifier.eval()
    pim_model.eval()
    aggregator.eval()

    total_loss = 0
    total_task_model_acc = 0
    total_clip_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Eval"
    )

    # Construct CLIP text embeddings
    class_level_prompts = ["This is a photo of a " + class_name for class_name in class_names]
    class_level_prompts = clip.tokenize(class_level_prompts).to(device)
    class_text_embeddings = clip_model.encode_text(class_level_prompts)

    
    labels_list, clip_logits_list, clip_probs_list = [], [], []
    task_model_logits_list, task_model_probs_list = [], []

    clip_class_level_probs_list = []

    for i, (images_batch, labels, images_clip_batch) in enumerate(pbar):
        
        images_batch = images_batch.to(device)
        images_clip_batch = images_clip_batch.to(device)
        labels = labels.to(device)

        pim_image_embeddings, task_model_logits, _ = pim_model(images_batch, return_task_logits=True)

        clip_image_embeddings = clip_model.encode_image(images_clip_batch)

        # Cosine similarity between the pim image embeddings and the class_attributes_embeddings
        normalized_clip_image_embeddings = F.normalize(clip_image_embeddings, dim=-1)
        normalized_class_attributes_embeddings = F.normalize(class_attributes_embeddings, dim=-1)

        normalized_class_text_embeddings = F.normalize(class_text_embeddings, dim=-1)

        normalized_clip_image_embeddings = normalized_clip_image_embeddings.to(normalized_class_attributes_embeddings.dtype)
        clip_similarities = CLIP_LOGIT_SCALE*(normalized_clip_image_embeddings @ normalized_class_attributes_embeddings.t()) # (batch_size, num_classes*num_attributes_perclass)

        clip_class_level_logits = CLIP_LOGIT_SCALE*(normalized_clip_image_embeddings @ normalized_class_text_embeddings.t()) # (batch_size, num_classes)

        # Split the similarities into class specific dictionary
        clip_similarities = clip_similarities.to(torch.float32)
        clip_similarities_dict = {}
        start = 0
        for i, class_prompts in enumerate(class_attribute_prompt_list):
            num_attributes = len(class_prompts)
            clip_similarities_dict[i] = clip_similarities[:, start:start+num_attributes]
            start += num_attributes
        
    
        # Compute the pim logits using the multiheaded attention
        clip_logits = aggregator(clip_similarities_dict)

        loss = F.cross_entropy(clip_logits, labels)

        task_model_probs = F.softmax(task_model_logits, dim=-1)
        clip_probs = F.softmax(clip_logits, dim=-1)

        clip_class_level_probs = F.softmax(clip_class_level_logits, dim=-1)
        clip_class_level_probs_list.append(clip_class_level_probs)
        
        task_model_acc = compute_accuracy(task_model_probs, labels)
        clip_acc = compute_accuracy(clip_probs, labels)

        total_task_model_acc += task_model_acc
        total_clip_acc += clip_acc

        total_loss += loss.item()

        labels_list.append(labels)
        clip_logits_list.append(clip_logits)
        clip_probs_list.append(clip_probs)
        task_model_logits_list.append(task_model_logits)
        task_model_probs_list.append(task_model_probs)

    labels_list = torch.cat(labels_list, dim=0)
    clip_logits_list = torch.cat(clip_logits_list, dim=0)
    clip_probs_list = torch.cat(clip_probs_list, dim=0)
    task_model_logits_list = torch.cat(task_model_logits_list, dim=0)
    task_model_probs_list = torch.cat(task_model_probs_list, dim=0)
    
    clip_class_level_probs_list = torch.cat(clip_class_level_probs_list, dim=0)

    clip_acc = compute_accuracy(clip_probs_list, labels_list)

    clip_class_level_acc = compute_accuracy(clip_class_level_probs_list, labels_list)

    task_model_acc = compute_accuracy(task_model_probs_list, labels_list)
    print(f'clip Attribute level Accuracy on {args.dataset_name} = {clip_acc} Clip Class Level Accuracy = {clip_class_level_acc} and Task Model Accuracy = {task_model_acc}')

    return clip_class_level_acc, clip_acc

def get_save_dir(args):
    
    if args.method == 'pim':
        if args.resume_checkpoint_path is not None and os.path.exists(args.resume_checkpoint_path):
            save_dir = os.path.dirname(args.resume_checkpoint_path)
        else:
            raise Exception("Checkpoint path not found")
    else:
        # use classifer checkpoint path
        if args.classifier_checkpoint_path is not None and os.path.exists(args.classifier_checkpoint_path):
            save_dir = os.path.dirname(args.classifier_checkpoint_path)
        elif 'imagenet' in args.classifier_checkpoint_path:
            save_dir = os.path.dirname(args.classifier_checkpoint_path)
            save_dir = os.path.join(save_dir, args.dataset_name)
        else:
            raise Exception("Checkpoint path not found")

    save_dir = os.path.join(save_dir, 'failure_results')

    return f"{save_dir}"

def progbar_wrapper(iterable, total, **kwargs):
    """Wraps the iterable with tqdm for global rank zero.

    Args:
        iterable: the iterable to wrap with tqdm
        total: the total length of the iterable, necessary in case the number of batches was limited.

    """
    return tqdm(iterable, total=total, **kwargs)

@torch.no_grad()
def evaluate_classifier(data_loader, classifier, device='cpu'): 
    
    # Set the model to eval mode
    classifier.eval()
    total_loss = 0
    total_task_model_acc = 0
    total_pim_acc = 0
    labels_list, logits_list, probs_list = [], [], []
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Evaluation"
    )
    
    for i, (images_batch, labels, _) in enumerate(pbar):
        
        images_batch = images_batch.to(device)
        labels = labels.to(device)
        labels_list.append(labels)

        logits = classifier(images_batch)
        logits_list.append(logits)

        probs = F.softmax(logits, dim=-1)
        probs_list.append(probs)
    
    labels_list = torch.cat(labels_list, dim=0)
    logits_list = torch.cat(logits_list, dim=0)
    probs_list = torch.cat(probs_list, dim=0)

    classifier_acc = compute_accuracy(probs_list, labels_list)
    print(f'Classifier Accuracy on {args.dataset_name} = {classifier_acc}')
    print(labels_list.shape, logits_list.shape, probs_list.shape)
    return classifier_acc, labels_list, logits_list, probs_list


@torch.no_grad()
def get_features_logits(data_loader, class_attributes_embeddings, class_attribute_prompt_list,
                    clip_model, classifier, pim_model, aggregator): 
    
    # Set the model to eval mode
    pim_model.eval()
    aggregator.eval()
    classifier.eval()
    clip_model.eval()
    total_loss = 0
    total_task_model_acc = 0
    total_pim_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Feature Evaluation"
    )
    
    labels_list, pim_logits_list, pim_probs_list = [], [], []
    task_model_logits_list, task_model_probs_list = [], []
    pim_attribute_logits_list = []

    task_model_features_all, clip_model_features_all, pim_features_all = [], [], []
    
    for i, (images_batch, labels, images_clip_batch) in enumerate(pbar):
        
        images_batch = images_batch.to(device)
        images_clip_batch = images_clip_batch.to(device)
        labels = labels.to(device)

        pim_image_embeddings, task_model_logits, _ = pim_model(images_batch, return_task_logits=True)
        
        _, task_model_embeddings = classifier(images_batch, return_features=True)

        clip_image_embeddings = clip_model.encode_image(images_clip_batch).detach().cpu()

        task_model_features_all.append(task_model_embeddings.detach().cpu())
        clip_model_features_all.append(clip_image_embeddings.detach().cpu())
        pim_features_all.append(pim_image_embeddings.detach().cpu())

        # Cosine similarity between the pim image embeddings and the class_attributes_embeddings
        normalized_pim_image_embeddings = F.normalize(pim_image_embeddings, dim=-1)
        normalized_class_attributes_embeddings = F.normalize(class_attributes_embeddings, dim=-1)
        normalized_pim_image_embeddings = normalized_pim_image_embeddings.to(normalized_class_attributes_embeddings.dtype)
        pim_similarities = CLIP_LOGIT_SCALE*(normalized_pim_image_embeddings @ normalized_class_attributes_embeddings.t()) # (batch_size, num_classes*num_attributes_perclass)

        # Split the similarities into class specific dictionary
        pim_similarities = pim_similarities.to(torch.float32)
        pim_similarities_dict = {}
        start = 0
        for i, class_prompts in enumerate(class_attribute_prompt_list):
            num_attributes = len(class_prompts)
            pim_similarities_dict[i] = pim_similarities[:, start:start+num_attributes]
            start += num_attributes
        
        pim_attribute_logits_list.append(pim_similarities_dict)
        
        # Compute the pim logits using the multiheaded attention
        pim_logits = aggregator(pim_similarities_dict)
    
        loss = F.cross_entropy(pim_logits, labels)

        task_model_probs = F.softmax(task_model_logits, dim=-1)
        pim_probs = F.softmax(pim_logits, dim=-1)
        
        task_model_acc = compute_accuracy(task_model_probs, labels)
        pim_acc = compute_accuracy(pim_probs, labels)

        total_task_model_acc += task_model_acc
        total_pim_acc += pim_acc

        total_loss += loss.item()

        labels_list.append(labels)
        pim_logits_list.append(pim_logits)
        pim_probs_list.append(pim_probs)
        task_model_logits_list.append(task_model_logits)
        task_model_probs_list.append(task_model_probs)

    labels_list = torch.cat(labels_list, dim=0)
    pim_logits_list = torch.cat(pim_logits_list, dim=0)
    pim_probs_list = torch.cat(pim_probs_list, dim=0)
    task_model_logits_list = torch.cat(task_model_logits_list, dim=0)
    task_model_probs_list = torch.cat(task_model_probs_list, dim=0)

    task_model_features_all = torch.cat(task_model_features_all, dim=0).numpy()
    clip_model_features_all = torch.cat(clip_model_features_all, dim=0).numpy()
    pim_features_all = torch.cat(pim_features_all, dim=0).numpy()
    
    pim_acc = compute_accuracy(pim_probs_list, labels_list)
    task_model_acc = compute_accuracy(task_model_probs_list, labels_list)
    print(f'PIM Accuracy on {args.dataset_name} = {pim_acc} and Task Model Accuracy = {task_model_acc}')
    
    features_dict = {
        'task_model_features': task_model_features_all,
        'clip_model_features': clip_model_features_all,
        'pim_features': pim_features_all
    }
    logits_dict = {
        'gt_labels': labels_list,
        'task_model_logits': task_model_logits_list,
        'pim_logits': pim_logits_list
    }

    probs_dict = {
        'task_model_probs':task_model_probs_list,
        'pim_probs': pim_probs_list
    }

    accuracies = {
        'task_model_acc': task_model_acc,
        'pim_acc': pim_acc
    }

    return features_dict, logits_dict, probs_dict, accuracies

@torch.no_grad()
def evaluate_pim(data_loader, class_attributes_embeddings, class_attribute_prompt_list,
                    clip_model, classifier, pim_model, aggregator): 
    
    # Set the model to eval mode
    pim_model.eval()
    aggregator.eval()
    classifier.eval()
    clip_model.eval()
    total_loss = 0
    total_task_model_acc = 0
    total_pim_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Eval"
    )
    
    labels_list, pim_logits_list, pim_probs_list = [], [], []
    task_model_logits_list, task_model_probs_list = [], []
    pim_attribute_logits_list = []



    for i, (images_batch, labels, images_clip_batch) in enumerate(pbar):
        
        images_batch = images_batch.to(device)
        labels = labels.to(device)

        pim_image_embeddings, task_model_logits, _ = pim_model(images_batch, return_task_logits=True)

        # Cosine similarity between the pim image embeddings and the class_attributes_embeddings
        normalized_pim_image_embeddings = F.normalize(pim_image_embeddings, dim=-1)
        normalized_class_attributes_embeddings = F.normalize(class_attributes_embeddings, dim=-1)
        normalized_pim_image_embeddings = normalized_pim_image_embeddings.to(normalized_class_attributes_embeddings.dtype)
        pim_similarities = CLIP_LOGIT_SCALE*(normalized_pim_image_embeddings @ normalized_class_attributes_embeddings.t()) # (batch_size, num_classes*num_attributes_perclass)

        # Split the similarities into class specific dictionary
        pim_similarities = pim_similarities.to(torch.float32)
        pim_similarities_dict = {}
        start = 0
        for i, class_prompts in enumerate(class_attribute_prompt_list):
            num_attributes = len(class_prompts)
            pim_similarities_dict[i] = pim_similarities[:, start:start+num_attributes]
            start += num_attributes
        
        pim_attribute_logits_list.append(pim_similarities_dict)
        
        # Compute the pim logits using the multiheaded attention
        pim_logits = aggregator(pim_similarities_dict)
    
        loss = F.cross_entropy(pim_logits, labels)

        task_model_probs = F.softmax(task_model_logits, dim=-1)
        pim_probs = F.softmax(pim_logits, dim=-1)
        
        task_model_acc = compute_accuracy(task_model_probs, labels)
        pim_acc = compute_accuracy(pim_probs, labels)

        total_task_model_acc += task_model_acc
        total_pim_acc += pim_acc

        total_loss += loss.item()

        labels_list.append(labels)
        pim_logits_list.append(pim_logits)
        pim_probs_list.append(pim_probs)
        task_model_logits_list.append(task_model_logits)
        task_model_probs_list.append(task_model_probs)

    labels_list = torch.cat(labels_list, dim=0)
    pim_logits_list = torch.cat(pim_logits_list, dim=0)
    pim_probs_list = torch.cat(pim_probs_list, dim=0)
    task_model_logits_list = torch.cat(task_model_logits_list, dim=0)
    task_model_probs_list = torch.cat(task_model_probs_list, dim=0)
    

    pim_acc = compute_accuracy(pim_probs_list, labels_list)
    task_model_acc = compute_accuracy(task_model_probs_list, labels_list)
    print(f'PIM Accuracy on {args.dataset_name} = {pim_acc} and Task Model Accuracy = {task_model_acc}')
    return pim_acc, task_model_acc, labels_list, pim_logits_list, pim_probs_list, task_model_logits_list, task_model_probs_list, pim_attribute_logits_list

def load_data(args, train_transform, test_transform, clip_transform):

    # This will be in train domain
    train_dataset, val_dataset, test_dataset, failure_dataset, class_names = get_dataset(args.dataset_name, train_transform, test_transform, 
                                                            data_dir=args.data_dir, clip_transform=clip_transform, 
                                                            img_size=args.img_size, domain_name=args.domain_name, 
                                                            return_failure_set=True)
 
    if args.dataset_name in ['cifar100']:
        # Merge falure dataset with train dataset
        train_dataset = ConcatDataset([train_dataset, val_dataset])

    print(f"Using {args.dataset_name} dataset")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    if args.eval_dataset == 'cifar100c':
        transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        testset = CIFAR100C(corruption=args.cifar100c_corruption, transform=transform_test,clip_transform=clip_transform, level=args.severity)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    elif args.eval_dataset == 'pacs':
        _, val_dataset, _, failure_dataset, class_names = get_dataset(args.dataset_name, train_transform, test_transform, 
                                                            data_dir=args.data_dir, clip_transform=clip_transform, 
                                                            img_size=args.img_size, domain_name='sketch', 
                                                            return_failure_set=True,use_real=False)
        #concat val and failure
        val_dataset = ConcatDataset([val_dataset, failure_dataset])
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    elif args.eval_dataset == 'NICOpp':
        d = args.classifier_checkpoint_path.split('/')[3]
        print(f'Classifier trained on {d}')
        _, val_dataset, _, failure_dataset, class_names = get_dataset(args.dataset_name, train_transform, test_transform, 
                                                            data_dir=args.data_dir, clip_transform=clip_transform, 
                                                            img_size=args.img_size, domain_name=d, 
                                                            return_failure_set=True,use_real=False)
        #concat val and failure
        val_dataset = ConcatDataset([val_dataset, failure_dataset])
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)  # This is from the "evaluation" domain

    print(f"Number of validation examples: {len(val_loader.dataset)}")
    print(f"Number of test examples: {len(test_loader.dataset)}")

    return train_loader, val_loader, test_loader, class_names


def main(args):

    # PIM_core_non_core(args)
    # assert False

    ########################### Create the model ############################
    clip_model, clip_transform = clip.load(args.clip_model_name, device=args.device)
    clip_model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    

    classifier, train_transform, test_transform = build_classifier(args.classifier_name, num_classes=args.num_classes, 
                                                                    pretrained=True, 
                                                                    checkpoint_path=args.classifier_checkpoint_path)
    # classifier_cloned = copy.deepcopy(classifier)
    mapping_name = args.classifier_name
    if args.dataset_name =="imagenet":
        mapping_name= f"{mapping_name}_v2"
    print(f"mapping_name: {mapping_name}")
    mapper,_, _ = build_classifier(mapping_name, num_classes=args.num_classes, pretrained=True, checkpoint_path=None)
    
    cutmix = CutMix(args.cutmix_alpha, args.num_classes)
    pim_model = TaskMapping(task_model=classifier, mapping_model=mapper, 
                              task_layer_name=args.task_layer_name, vlm_dim=args.vlm_dim, 
                              mapping_output_size=mapper.feature_dim, cutmix_fn=cutmix)
    

    print(f"Loaded Classifier checkpoint from {args.classifier_checkpoint_path}")
    
    ########################### Load the dataset ############################
    
    train_loader, val_loader, test_loader, class_names = load_data(args, train_transform, test_transform, clip_transform)
    
    if args.method == 'baseline':
        classifier.to(device)
        classifier.eval()
        
        # Evaluating task model
        print('Evaluating on Validation Data')
        val_task_model_acc, val_labels_list, val_logits_list, val_probs_list = evaluate_classifier(val_loader, classifier, device=device)
        if args.score in ['msp', 'pe', 'energy', 'max_logit']:
            val_scores = get_score(args.score, val_logits_list)
        elif args.score == 'gde':
            seed_list = [11,21,31,41]
            models = []
            for s in seed_list:
                models.append(copy.deepcopy(classifier))
                models[-1].load_state_dict(torch.load(f'logs/CelebA/failure_estimation/None/resnet50/classifier_seed{s}/best_checkpoint.pth',map_location=device)['model_state_dict'])
                models[-1].eval()
            val_scores = compute_gde_scores(models, val_loader, device)
        elif args.score == 'ts':
            val_scores, tscaler = compute_ts(val_logits_list, val_labels_list, val_logits_list, tscaler=None, mode='val')
        threshold = calc_gen_threshold(val_scores, val_logits_list, val_labels_list, name='classifier')

        # Just for verification
        estimated_val_acc, val_estimated_success_failure_idx = calc_accuracy_from_scores(val_scores, threshold)

        # Repeating this for test data
        print('Evaluating on Test Data')
        test_task_model_acc, test_labels_list, test_logits_list, test_probs_list = evaluate_classifier(test_loader, classifier, device=device)
        if args.score in ['msp', 'pe', 'energy', 'max_logit']:
            test_scores = get_score(args.score, test_logits_list)
        elif args.score == 'gde':
            test_scores = compute_gde_scores(models, test_loader, device)
        elif args.score == 'ts':
            test_scores = compute_ts(test_logits_list, test_labels_list, test_logits_list, tscaler, mode='test')
        estimated_test_acc, test_estimated_success_failure_idx = calc_accuracy_from_scores(test_scores, threshold)

        print(f'Score = {args.score}')
        print(f'True Validation Accuracy = {val_task_model_acc}, Estimated Validation Accuracy = {estimated_val_acc}, True Test Accuracy = {test_task_model_acc}, Estimated Test Accuracy = {estimated_test_acc}')
        val_true_success_failure_idx = torch.argmax(val_probs_list, 1) == val_labels_list
        test_true_success_failure_idx = torch.argmax(test_probs_list, 1) == test_labels_list

        val_fpr_at_tpr, val_auroc = compute_auroc_fpr(val_true_success_failure_idx.cpu().numpy(), val_scores.cpu().numpy())
        print(f'Validation FPR@TPR95 = {val_fpr_at_tpr}, Validation AUROC = {val_auroc}')

        test_fpr_at_tpr, test_auroc = compute_auroc_fpr(test_true_success_failure_idx.cpu().numpy(), test_scores.cpu().numpy())
        print(f'Test FPR@TPR95 = {test_fpr_at_tpr}, Test AUROC = {test_auroc}')

        print('Confusion Matrices')
        cm_val = confusion_matrix(val_true_success_failure_idx.cpu().numpy(), val_estimated_success_failure_idx.cpu().numpy())
        cm_test = confusion_matrix(test_true_success_failure_idx.cpu().numpy(), test_estimated_success_failure_idx.cpu().numpy())

        failure_recall_val = cm_val[0,0]/(cm_val[0,0]+cm_val[0,1])
        success_recall_val = cm_val[1,1]/(cm_val[1,0]+cm_val[1,1])

        failure_recall_test = cm_test[0,0]/(cm_test[0,0]+cm_test[0,1])
        success_recall_test = cm_test[1,1]/(cm_test[1,0]+cm_test[1,1])

        mathews_corr_val = matthews_corrcoef(val_true_success_failure_idx.cpu().numpy(), val_estimated_success_failure_idx.cpu().numpy())
        mathews_corr_test = matthews_corrcoef(test_true_success_failure_idx.cpu().numpy(), test_estimated_success_failure_idx.cpu().numpy())

        print('Validation Data')
        print(cm_val)
        print(f'Gen Gap = {torch.abs(val_task_model_acc-estimated_val_acc)}')
        print(f'Failure Recall = {failure_recall_val}')
        print(f'Success Recall = {success_recall_val}')
        print(f'Mathews Correlation = {mathews_corr_val}')

        print('Test Data')
        print(cm_test)
        print(f'Gen Gap = {torch.abs(test_task_model_acc-estimated_test_acc)}')
        print(f'Failure Recall = {failure_recall_test}')
        print(f'Success Recall = {success_recall_test}')
        print(f'Mathews Correlation = {mathews_corr_test}')

        # convert confusion matrix to list
        cm_val_list = cm_val.tolist()
        cm_test_list = cm_test.tolist()
        # Convert the results to a dictionary
        results = {
            "domain_name": args.domain_name,
            "true_val_acc": val_task_model_acc,
            "estimated_val_acc": estimated_val_acc.item(),
            "true_test_acc": test_task_model_acc,
            "estimated_test_acc": estimated_test_acc.item(),
            "val_cm": cm_val_list, # Confusion matrix for validation data as a list
            "test_cm": cm_test_list,
            "val_failure_recall": failure_recall_val,
            "val_success_recall": success_recall_val,
            "test_failure_recall": failure_recall_test,
            "test_success_recall": success_recall_test,
            "val_mathews_corr": mathews_corr_val,
            "test_mathews_corr": mathews_corr_test,
            "val_fpr_at_tpr":val_fpr_at_tpr,
            "val_auroc":val_auroc,
            "test_fpr_at_tpr":test_fpr_at_tpr,
            "test_auroc":test_auroc
        }
        if args.eval_dataset == 'NICOpp':
            d = args.classifier_checkpoint_path.split('/')[3]
            results['train_domain_name'] = d

        # Save it as a CSV file
        results_file = f'{args.save_dir}/{args.score}_results.json'
        print (f'************* Saving results to {results_file} *************')
        if args.eval_dataset == 'cifar100c':
            # update the results dictionary
            results["cifar100c_corruption"] = args.cifar100c_corruption
            results["severity"] = args.severity

            results_file = f'{args.save_dir}/{args.score}_cifar100c_results.json'

        with open(results_file, 'a') as f:
            json.dump(results, f)
            f.write('\n')
    
    elif args.method == 'pim':
        class_attributes_embeddings_prompts = torch.load(args.attributes_embeddings_path)
        # Get the class attribute prompts and their embeddings
        class_attribute_prompts = class_attributes_embeddings_prompts["class_attribute_prompts"] # List of list of prompts
        class_attributes_embeddings = class_attributes_embeddings_prompts["class_attributes_embeddings"]

        assert len(class_attribute_prompts) == args.num_classes, "Number of classes does not match the number of class attributes"

        num_attributes_per_cls = [len(attributes) for attributes in class_attribute_prompts]
        # # Extract the attribute names from the prompts
        # attribute_names_per_class = {}
        # for i in range(len(class_attribute_prompts)):
        #     attribute_names_per_class[class_names[i]] = [prompt.replace(f"This is a photo of {class_names[i]} with ", "") for prompt in class_attribute_prompts[i]]
        

        
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
            # classifier.load_state_dict(state["classifier"])
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

        clip_model.eval()
        classifier.eval()
        pim_model.eval()
        aggregator.eval()



        # # This evaluates CLIP attribute classifier, NOTE: use only with mean and max aggregators
        clip_class_level_acc, clip_attribute_level_acc = clip_attribute_classifier(test_loader, class_attributes_embeddings, class_attribute_prompts, clip_model, classifier, pim_model, aggregator, class_names)

        # Evaluating task model
        print('\n\nEvaluating on Validation Data')


        val_features_dict, val_logits_dict, val_probs_dict, val_accuracies_dict =  get_features_logits(val_loader, class_attributes_embeddings, class_attribute_prompts,
                                                                                        clip_model, classifier, pim_model, aggregator)


        print('\nEvaluating on Test Data')

        test_features_dict, test_logits_dict, test_probs_dict, test_accuracies_dict =  get_features_logits(test_loader, class_attributes_embeddings, class_attribute_prompts,
                                                                                            clip_model, classifier, pim_model, aggregator)

        if args.score == 'cross_entropy':
            val_scores = get_score(args.score, val_logits_dict['task_model_logits'], val_logits_dict['pim_logits'])
            test_scores = get_score(args.score, test_logits_dict['task_model_logits'], test_logits_dict['pim_logits'])

        elif args.score == 'lds_task':

            from latent_dissagrement import get_latent_disaggrement
            # Latent Dissagrement where the task model features are evaluated


            train_features_dict, train_logits_dict, train_probs_dict, train_accuracies_dict =  get_features_logits(train_loader, class_attributes_embeddings, class_attribute_prompts,
                                                                                            clip_model, classifier, pim_model, aggregator)

            train_features_pair = [train_features_dict['clip_model_features'][:10000], train_features_dict['task_model_features'][:10000]]
            val_features_pair = [val_features_dict['clip_model_features'], val_features_dict['task_model_features']]
            test_features_pair = [test_features_dict['clip_model_features'], test_features_dict['task_model_features']]
            
            val_scores = get_latent_disaggrement(n_neighbours=200, test_features=val_features_pair, anchor_features=train_features_pair, metric='ndcg_rank', verbose=True)
            test_scores = get_latent_disaggrement(n_neighbours=200, test_features=test_features_pair, anchor_features=train_features_pair, metric='ndcg_rank', verbose=True)
            
            # if scores are in numy convert them to torch
            if isinstance(val_scores, np.ndarray):
                val_scores = torch.from_numpy(val_scores)
            if isinstance(test_scores, np.ndarray):
                test_scores = torch.from_numpy(test_scores)

        elif args.score == 'lds_pim':

            from latent_dissagrement import get_latent_disaggrement
            # Latent Dissagrement where the task model features are evaluated

            train_features_dict, train_logits_dict, train_probs_dict, train_accuracies_dict =  get_features_logits(train_loader, class_attributes_embeddings, class_attribute_prompts,
                                                                                            clip_model, classifier, pim_model, aggregator)

            train_features_pair = [train_features_dict['clip_model_features'][:10000], train_features_dict['pim_features'][:10000]]
            val_features_pair = [val_features_dict['clip_model_features'], val_features_dict['pim_features']]
            test_features_pair = [test_features_dict['clip_model_features'], test_features_dict['pim_features']]
            
            val_scores = get_latent_disaggrement(n_neighbours=200, test_features=val_features_pair, anchor_features=train_features_pair, metric='ndcg_rank', verbose=True)
            test_scores = get_latent_disaggrement(n_neighbours=200, test_features=test_features_pair, anchor_features=train_features_pair, metric='ndcg_rank', verbose=True)

            # if scores are in numy convert them to torch
            if isinstance(val_scores, np.ndarray):
                val_scores = torch.from_numpy(val_scores)
            if isinstance(test_scores, np.ndarray):
                test_scores = torch.from_numpy(test_scores)

        results_dict = get_failure_results(val_logits_dict['task_model_logits'], val_logits_dict['gt_labels'], val_scores, 
                                            test_logits_dict['task_model_logits'], test_logits_dict['gt_labels'], test_scores, 
                                            val_accuracies_dict['task_model_acc'], test_accuracies_dict['task_model_acc'],  
                                            test_accuracies_dict['pim_acc'], clip_class_level_acc, clip_attribute_level_acc,
                                            threshold = None)

        if args.eval_dataset == 'NICOpp':
            results_dict['train_domain_name'] = d

        # Save it as a CSV file
        results_file = f'{args.save_dir}/{args.score}_results.json'

        if args.eval_dataset == 'cifar100c':
            # update the results dictionary
            results_dict["cifar100c_corruption"] = args.cifar100c_corruption
            results_dict["severity"] = args.severity

            results_file = f'{args.save_dir}/{args.score}_cifar100c_results.json'

        print(results_dict)

        with open(results_file, 'a') as f:
            json.dump(results_dict, f)
            f.write('\n')

        plot = False
        if plot:
            fig, ax = plt.subplots(1,2, figsize=(8,5))
            plt.subplot(121)
            plt.hist(val_scores.cpu().numpy()[val_true_success_failure_idx.cpu().numpy()], label='True Success', alpha=0.2)
            plt.hist(val_scores.cpu().numpy()[~val_true_success_failure_idx.cpu().numpy()], label='True Failure')
            plt.vlines(threshold, plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], linestyles ="dotted", colors ="k")
            plt.legend()
            plt.xlabel('Negative Cross Entropy')
            plt.title('Val Data')

            plt.subplot(122)
            plt.hist(test_scores.cpu().numpy()[test_true_success_failure_idx.cpu().numpy()], label='True Success', alpha=0.2)
            plt.hist(test_scores.cpu().numpy()[~test_true_success_failure_idx.cpu().numpy()], label='True Failure')
            plt.vlines(threshold, plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], linestyles ="dotted", colors ="k")
            plt.legend()
            plt.xlabel('Negative Cross Entropy')
            plt.title('Test Data')

            plt.savefig(f'{args.save_dir}/{args.score}_ce_comparison_true_classes.png', bbox_inches='tight')
    
    else:
        raise NotImplementedError


def PIM_core_non_core(args):

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
    
    ########################### Load the dataset ############################
    
    train_loader, val_loader, test_loader, class_names = load_data(args, train_transform, test_transform, clip_transform)


    class_attributes_embeddings_prompts = torch.load(args.attributes_embeddings_path)
    # Get the class attribute prompts and their embeddings
    class_attribute_prompts = class_attributes_embeddings_prompts["class_attribute_prompts"] # List of list of prompts
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

    if args.resume_checkpoint_path:
        state = torch.load(args.resume_checkpoint_path)
        epoch = state["epoch"]
        # classifier.load_state_dict(state["classifier"])
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



    land_phrase = "a photo of land."
    water_phrase = "A photo of water."
    core_landbird = "A photo of a bird"
    # core_waterbird = "A photo of a bird"

    
    # prompts=[land_phrase, core_landbird, water_phrase, core_waterbird]
    prompts=[land_phrase, water_phrase, core_landbird]

    # Tokenize the prompts
    prompts_embeddings = clip_model.encode_text(clip.tokenize(prompts).to(device))


    # Set the model to eval mode
    pim_model.eval()
    aggregator.eval()
    classifier.eval()
    clip_model.eval()
    
    pbar = progbar_wrapper(
        test_loader, total=len(test_loader), desc=f"Feature Evaluation"
    )
    
    task_model_features_all, clip_model_features_all, pim_features_all = [], [], []
    
    for i, (images_batch, labels, images_clip_batch, attr) in enumerate(pbar):
        
        images_batch = images_batch.to(device)
        images_clip_batch = images_clip_batch.to(device)
        labels = labels.to(device)

        pim_image_embeddings, task_model_logits, _ = pim_model(images_batch, return_task_logits=True)
        
        # Compute cosine similarity between the pim image embeddings and the class_attributes_embeddings
        normalized_pim_image_embeddings = F.normalize(pim_image_embeddings, dim=-1)
        normalized_prompts_embeddings = F.normalize(prompts_embeddings, dim=-1)
        normalized_class_attributes_embeddings = F.normalize(class_attributes_embeddings, dim=-1)

        normalized_pim_image_embeddings = normalized_pim_image_embeddings.to(normalized_prompts_embeddings.dtype)

        pim_non_core_logits = CLIP_LOGIT_SCALE*(normalized_pim_image_embeddings @ normalized_prompts_embeddings.t()) # (batch_size, num_prompts)
        pim_similarities = CLIP_LOGIT_SCALE*(normalized_pim_image_embeddings @ normalized_class_attributes_embeddings.t()) # (batch_size, num_prompts)

        # Split the similarities into class specific dictionary
        pim_similarities = pim_similarities.to(torch.float32)
        pim_similarities_dict = {}
        start = 0
        for i, class_prompts in enumerate(class_attribute_prompts):
            num_attributes = len(class_prompts)
            pim_similarities_dict[i] = pim_similarities[:, start:start+num_attributes]
            start += num_attributes
        # Compute the pim logits using the multiheaded attention
        pim_logits = aggregator(pim_similarities_dict)


        pim_non_core_pred = torch.argmax(pim_non_core_logits, dim=1)
        pim_class_pred = torch.argmax(pim_logits, dim=1)

        print(f"GT Labels: {labels[:15]}\tClass Predictions: {pim_class_pred[:15]}\nAttr: {attr[:15]}\tNon Core Predictions: {pim_non_core_pred[:15]}\n")

        import pdb; pdb.set_trace()

        # Split the similarities into class specific dictionary
        for i in range(images_batch.shape[0]):
            pim_similarities_dict = {}
            for j, prompt in enumerate(prompts):
                pass



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


    args = parser.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    # Print the arguments
    print(args)
    sys.stdout.flush()
    
    # Make directory for saving results
    args.save_dir = get_save_dir(args)    
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

    elif args.eval_dataset =='NICOpp':
        if args.method=='baseline':
            scores_all = ['msp', 'energy', 'pe']
            for score in scores_all:
                args.score = score
                for dn in ['autumn', 'dim', 'grass', 'outdoor', 'rock', 'water']:
                    args.domain_name = dn
                    seed_everything(args.seed)
                    main(args)
        else:
            for dn in ['autumn', 'dim', 'grass', 'outdoor', 'rock', 'water']:
                args.domain_name = dn
                seed_everything(args.seed)
                main(args)

    else:
        seed_everything(args.seed)
        main(args)
