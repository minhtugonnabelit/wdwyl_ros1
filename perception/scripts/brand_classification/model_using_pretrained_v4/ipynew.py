# %% [markdown]
# Model Generation

# %%
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from resnet_model import *


# %%
resize_shape = (256, 256)

data_dir = '/home/selimon/Desktop/AI/wdwyl_ros1/src/perception/brand_classification/data/training_validation'
print(os.listdir(data_dir))

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_tfms = tt.Compose([
    tt.Resize(resize_shape),
    tt.RandomHorizontalFlip(),
    tt.RandomRotation(10),
    tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    tt.ToTensor(),
    tt.Normalize(*stats, inplace=True)
])

valid_tfms = tt.Compose([
    tt.Resize(resize_shape),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

# Load dataset
full_dataset = ImageFolder(data_dir)

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# %%
batch_size = 80

def denormalize(images, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        denorm_images = denormalize(images, *stats)
        ax.imshow(make_grid(denorm_images[:64], nrow=8).permute(1, 2, 0).clamp(0,1))
        break

# %%
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        print('cuda')
        return torch.device('cuda')
    else:
        print('cpu')
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = get_default_device()

# %% [markdown]
# ## Plot Functions

# %%
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')

def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')

# %% [markdown]
# ## Training Model

# %%
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history

# %%
# Variable to keep track of the best model
best_val_acc = 0
best_model_wts = None

for fold, (train_idx, valid_idx) in enumerate(kf.split(full_dataset)):
    print(f'FOLD NUMBER: {fold + 1}/{k_folds}')

    # Create subset data for train and validation using indices from KFold
    train_ds = Subset(full_dataset, train_idx)
    valid_ds = Subset(full_dataset, valid_idx)

    # Apply transformations to train and valid sets
    train_ds.dataset.transform = train_tfms
    valid_ds.dataset.transform = valid_tfms

    # Create DataLoader objects
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size, shuffle=False, num_workers=3, pin_memory=True)

    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)

    model = to_device(ResNetPretrained(4), device)
    print(model)

    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    epochs = 35
    max_lr = 0.0125
    grad_clip = 0.085
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    # Train the model and record history for this fold
    history = fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl,
                            grad_clip=grad_clip,
                            weight_decay=weight_decay,
                            opt_func=opt_func)

    # Plotting
    plot_accuracies(history)
    plt.show()
    plot_losses(history)
    plt.show()
    plot_lrs(history)
    plt.show()

    # Save the model if it has the best validation accuracy so far
    val_acc = history[-1]['val_acc']
    if val_acc > best_val_acc:
        print(f'SAVING FOLD NUMBER: {fold + 1}/{k_folds}')
        best_val_acc = val_acc
        best_model_wts = model.state_dict()


# Save the best model weights
torch.save(best_model_wts, 'resnet_model.pth')