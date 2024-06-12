# Build CNN based on ResNet9 architecture

### Data Augmentation
* Random Flipping
* Random Rotation
* Colour Jittering
* Normalising

### Learning Rate Scheduling using ‘One Cycle Learning Rate Policy’
Start with a low learning rate, gradually increasing it batch-by-batch to a high learning rate for about 30% of epochs, then gradually decreasing it to a very low value for remaining epochs

### Weight Decay
Prevents the weights from becoming too large by adding an additional term to the loss function

### Gradient Clipping
limit the values of gradients to a small range to prevent undesirable changes in parameters due to large gradient values
