import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    """Base class for image classification models."""
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


class ResNetPretrained(ImageClassificationBase):
    def __init__(self, num_classes, device='cpu'):
        super().__init__()
        self.resnet = models.resnet152(pretrained=True)
        
        # Freeze all parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
                
        # Add dropout layer
        self.dropout = nn.Dropout(0.5)
        
        self.to(device)
       
        
    def forward(self, xb):
        
        # Forward pass through ResNet
        out = self.resnet(xb)
        
        # Apply dropout
        out = self.dropout(out)
        
        
        return out


# ## Trying v2 Model:
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models


# def accuracy(outputs, labels):
#     _, preds = torch.max(outputs, dim=1)
#     return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# class ImageClassificationBase(nn.Module):
#     """Base class for image classification models."""
#     def training_step(self, batch):
#         images, labels = batch 
#         out = self(images)                  # Generate predictions
#         loss = F.cross_entropy(out, labels) # Calculate loss
#         return loss
    
#     def validation_step(self, batch):
#         images, labels = batch 
#         out = self(images)                    # Generate predictions
#         loss = F.cross_entropy(out, labels)   # Calculate loss
#         acc = accuracy(out, labels)           # Calculate accuracy
#         return {'val_loss': loss.detach(), 'val_acc': acc}
    
#     def validation_epoch_end(self, outputs):
#         batch_losses = [x['val_loss'] for x in outputs]
#         epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
#         batch_accs = [x['val_acc'] for x in outputs]
#         epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
#         return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
#     def epoch_end(self, epoch, result):
#         print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
#             epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# class ResNetPretrained(ImageClassificationBase):
#     def __init__(self, num_classes, device='cpu'):
#         super().__init__()
#         self.resnet = models.resnet152(pretrained=True)
        
#         # Freeze all parameters
#         for param in self.resnet.parameters():
#             param.requires_grad = False
        
#         # Replace the final fully connected layer
#         num_ftrs = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_ftrs, num_classes)
                
#         # Add dropout layer
#         self.dropout = nn.Dropout(0.5)
        
#         self.to(device)
       
        
#     def forward(self, xb):
        
#         # Forward pass through ResNet
#         out = self.resnet(xb)
        
#         # Apply dropout
#         out = self.dropout(out)
        
#         return out