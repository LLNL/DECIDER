import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from models.resnet import CustomResNet
from data_utils.domainnet_data import DomainNetDataset, get_domainnet_loaders
from data_utils.cifar100_data import CIFAR100TwoTransforms, CIFAR100C, get_CIFAR100_dataloader
from data_utils.cifar10_data import get_CIFAR10_dataloader
from data_utils.celebA_dataset import get_celebA_dataloader
from data_utils.pacs_dataset import get_pacs_dataloader
from data_utils.office_home_dataset import OfficeHomeDataset, get_office_home_dataloader
from data_utils.cats_dogs_dataset import CatsDogsTwoTransforms, get_cats_dogs_loaders

from train_task_distillation import get_dataset, build_classifier
from data_utils import subpop_bench

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_images(loader, title, n_rows=2, n_cols=5, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Extracts the first batch of images from the given data loader and plots them in a grid with their labels.
    Adjusts the image contrast if necessary.
    """
    # Get the first batch
    if args.dataset_name in subpop_bench.DATASETS:
        _, images, labels, _ = next(iter(loader))
    else:
        images, labels = next(iter(loader))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))
    axes = axes.flatten()

    # Function to denormalize the image
    def denormalize(image):
        image = image.numpy().transpose(1, 2, 0)
        image = std * image + mean
        image = np.clip(image, 0, 1)
        return image

    for i in range(n_rows * n_cols):
        image = denormalize(images[i])
        label = labels[i]

        axes[i].imshow(image)
        axes[i].set_title(f"Label: {label}", fontsize=10)
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(args.save_dir, f"{title}.png")
    plt.savefig(save_path)
    # plt.show()

def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, len(train_loader.dataset)
    
    # Wrap the train_loader with tqdm for progress bar
    pbar = tqdm(train_loader, desc=f'Training epoch: {epoch+1}')
    for data in pbar:
        if args.dataset_name in subpop_bench.DATASETS:
            inputs, labels = data[1], data[2]
        else:
            inputs, labels = data[0], data[1]

        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item() * inputs.size(0)
        total_loss += batch_loss
        _, predicted = outputs.max(1)
        batch_correct = (predicted == labels).sum().item()
        total_correct += batch_correct

        # Set metrics for tqdm
        pbar.set_postfix({"Epoch Loss": total_loss/total_samples, "Epoch Acc": total_correct/total_samples})

    return total_loss/total_samples, total_correct/total_samples

def validate(val_loader, model, criterion, device, epoch):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, len(val_loader.dataset)
    
    # Wrap the val_loader with tqdm for progress bar
    pbar = tqdm(val_loader, desc=f'Validating epoch: {epoch+1}')
    with torch.no_grad():
        for data in pbar:
            if args.dataset_name in subpop_bench.DATASETS:
                inputs, labels = data[1], data[2]
            else:
                inputs, labels = data[0], data[1]

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_loss = loss.item() * inputs.size(0)
            total_loss += batch_loss
            _, predicted = outputs.max(1)
            batch_correct = (predicted == labels).sum().item()
            total_correct += batch_correct
            # Set metrics for tqdm
            pbar.set_postfix({"Epoch Loss": total_loss/total_samples, "Epoch Acc": total_correct/total_samples})

    return total_loss/total_samples, total_correct/total_samples

def get_dataloaders(dataset_name, domain_name=None,
                    batch_size=512, data_dir='./data', image_size=224, 
                    train_transform=None, test_transform=None, clip_transform=None, 
                    subsample_trainset=False, return_dataset=False):
    
    if dataset_name == 'domainnet':
        loaders, class_names = get_domainnet_loaders(domain_name, batch_size=batch_size, data_dir=data_dir,
                                                     train_transform=None, test_transform=None, clip_transform=None, 
                                                     subsample_trainset=False, return_dataset=False)
    elif dataset_name == "pacs":
        loaders, class_names = get_pacs_dataloader(domain_name, batch_size=batch_size, data_dir=data_dir, 
                                                train_transform=None, test_transform=None, clip_transform=None, 
                                                return_dataset=False, use_real=False)
        
    elif dataset_name == "cats_dogs":
        loaders, class_names = get_cats_dogs_loaders(batch_size=batch_size, data_dir=data_dir, 
                                                   train_transform=None, test_transform=None, clip_transform=None, 
                                                   return_dataset=False)
    elif dataset_name == "office_home":
        loaders, class_names = get_office_home_dataloader(domain_name, batch_size=batch_size, data_dir=data_dir, 
                                                          train_transform=None, test_transform=None, clip_transform=None, 
                                                          return_dataset=False, use_real=False)

    elif dataset_name in subpop_bench.DATASETS:
        hparams = {
            'batch_size': batch_size,
            'image_size': image_size,
            'num_workers': 4,
            'group_balanced': None,
        }

        attribute_names = ["autumn", "dim", "grass", "outdoor", "rock", "water"]

        if domain_name in attribute_names:
            # Get a list of domain indices from the domain names
            domain_idx = [attribute_names.index(domain_name)]
        else:
            domain_idx = None
        loaders, class_names = subpop_bench.get_dataloader(dataset_name, data_dir, hparams, 
                                                           train_attr='yes', sample_by_attributes=domain_idx)
    # elif dataset_name == 'CelebA':
    #     class_attr = 'Young' # attribute for binary classification
    #     imbalance_attr = ['Male']
    #     imbalance_percent = {1: [20], 0:[80]} # 1 = Young, 0 = Not Young; 20% of the Young data will be Male
    #     ignore_attrs = []  # Example: ignore samples that are 'Bald' or 'Wearing_Earrings'

    #     loaders, class_names = get_celebA_dataloader(batch_size, class_attr, imbalance_attr, imbalance_percent, 
    #                                                  ignore_attrs, img_size=image_size, mask=False, mask_region=None)
    elif dataset_name == 'cifar10-limited':
        loaders, class_names = get_CIFAR10_dataloader(batch_size=batch_size, data_dir=data_dir, 
                                                      subsample_trainset=True)
    elif dataset_name == 'cifar10':
        loaders, class_names = get_CIFAR10_dataloader(batch_size=batch_size, data_dir=data_dir, subsample_trainset=False)
    elif dataset_name == 'cifar100':
        loaders, class_names = get_CIFAR100_dataloader(batch_size=batch_size, data_dir=data_dir, 
                                                       selected_classes=None, retain_orig_ids=False,    
                                                       train_transform=None, test_transform=None, clip_transform=None, 
                                                       subsample_trainset=False, return_dataset=False)
        
        # concatenate the train and the failure dataset
        train_loader, val_loader, test_loader, failure_loader = loaders['train'], loaders['val'], loaders['test'], loaders['failure']
        train_dataset = torch.utils.data.ConcatDataset([train_loader.dataset, failure_loader.dataset])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    elif dataset_name == 'cifar100-90cls':
        # Randomly select 90 numbers from 0-99 without replacement use seed
        random.seed(42)
        selected_classes = random.sample(range(100), 90)
        loaders, class_names = get_CIFAR100_dataloader(batch_size=batch_size, data_dir=data_dir, 
                                                       selected_classes=selected_classes, retain_orig_ids=True,    
                                                       train_transform=None, test_transform=None, clip_transform=None, 
                                                       subsample_trainset=False, return_dataset=False)
    # elif dataset_name in ["food101", "sun397", "eurosat", "ucf101", 
    #                       "stanfordcars", "flowers102", "dtd", "oxfordpets", "svhn", "gtsrb"]:
        
    #     loaders, class_names = get_zsl_dataloaders(dataset_name, batch_size=batch_size, data_path=data_dir, 
    #                                                preprocess=None, clip_transform=None)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return loaders, class_names

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ########################### Load Dataset ###########################
       
    loaders, class_names = get_dataloaders(args.dataset_name, args.domain, args.batch_size, args.data_path, args.image_size, 
                                           train_transform=None, test_transform=None, clip_transform=None, 
                                           subsample_trainset=False, return_dataset=False)
    
    train_loader, val_loader, test_loader = loaders['train'], loaders['val'], loaders['test']

    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    print(f"Number of classes: {len(class_names)}")


    model, _, _ = build_classifier(args.classifier_model, len(class_names), pretrained=args.use_pretrained)
    model.to(device)

    print(f"Classifier model: {args.classifier_model}")
    print(f"Using pretrained weights: {args.use_pretrained}")

    print(model)
    # Dataparallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)    
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    
    # Learning rate scheduler
    if args.scheduler == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 180], gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    else:
        scheduler = None
    
    # Make directory for saving results
    if args.dataset_name in ['domainnet', 'pacs', 'office_home']:
        args.save_dir = f"logs/{args.dataset_name}-{args.domain}/{args.classifier_model}/classifier"
    elif args.dataset_name in subpop_bench.DATASETS:   
        args.save_dir = f"logs/{args.dataset_name}/failure_estimation/{args.domain}/{args.classifier_model}/classifier"
    else:
        args.save_dir = f"logs/{args.dataset_name}/{args.classifier_model}/classifier"
    
    # Add seed to save_dir
    args.save_dir += f"_seed{args.seed}"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    plot_images(train_loader, title="Training Image")
    plot_images(val_loader, title="Validation Image")

    # Save arguments if not resuming
    if not args.resume:
        with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
        
        start_epoch = 0
        
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []
        best_val_accuracy = 0.0

    else:
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint_path)
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        # Load epoch
        start_epoch = checkpoint['epoch'] + 1
        best_val_accuracy = checkpoint['best_val_accuracy']
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # Load training and validation metrics
        train_losses = checkpoint['train_losses']
        train_accuracies = checkpoint['train_accuracies']
        val_losses = checkpoint['val_losses']
        val_accuracies = checkpoint['val_accuracies']

        print(f"Resuming training from epoch {start_epoch}")

    
    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(test_loader, model, criterion, device, epoch)
        
        if scheduler is not None:
            scheduler.step()

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # Update and save training plots
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, '-o', label='Training')
        plt.plot(val_losses, '-o', label='Validation')
        plt.title('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, '-o', label='Training')
        plt.plot(val_accuracies, '-o', label='Validation')
        plt.title('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'training_validation_plots.png'))
        plt.close()

        # if data parallel, save the model without the module
        if torch.cuda.device_count() > 1:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        # Save best model based on validation accuracy
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
                
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'best_val_accuracy': best_val_accuracy,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(args.save_dir, f"best_checkpoint.pth"))

        if epoch % 10 == 0:

            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'best_val_accuracy': best_val_accuracy,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
            }, os.path.join(args.save_dir, f"checkpoint_{epoch}.pth"))

    torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
        }, os.path.join(args.save_dir, f"checkpoint_{epoch}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train desired classifier model on the desired Dataset')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--domain', type=str, help='Name of the domain if data is from DomainNet dataset')
    parser.add_argument('--data_path', default='./data' ,type=str, help='Path to the dataset')
    parser.add_argument('--image_size', type=int, default=224, help='Size to resize images to (assumes square images)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam', help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, choices=['MultiStepLR', 'cosine', 'No'], default='MultiStepLR', help='Scheduler to use')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--classifier_model', type=str, choices=['resnet18', 'resnet50', 'vit_b_16', 'swin_b', 'SimpleCNN', 'resnext50_32x4d'], default='resnet18', help='Type of classifier model to use')
    parser.add_argument('--use_pretrained', action='store_true', help='Use pretrained weights for ResNet')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint to resume training from')

    args = parser.parse_args()

    # Set seed
    seed_everything(args.seed)
    # torch.manual_seed(42)
    
    main(args)
