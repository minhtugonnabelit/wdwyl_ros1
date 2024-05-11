import torch
import torchvision.transforms as tt

from resnet_9_model import ResNet9

from PIL import Image


def get_default_device():
    """Pick GPU if available, else CPU"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def predict_image(img, model, device):
    """Make a prediction for the image using the model"""
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item(), yb[0]

def image_classify(image_path):
    """Classify an image using a trained model"""
    resize_shape = (48, 48)
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    valid_tfms = tt.Compose([
        tt.Resize(resize_shape),
        tt.ToTensor(), 
        tt.Normalize(*stats)
    ])
    
    device = get_default_device()
    model = ResNet9(3, 4).to(device)
    model.load_state_dict(torch.load('resnet9_model.pth', map_location=device))
    model.eval()

    # Open the image file
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found.")
        return None, None

    # Apply transformations to the image
    img_tensor = valid_tfms(image).to(device)

    # Make a prediction for the image using the model
    predicted_class, yb = predict_image(img_tensor, model, device)

    # Calculate softmax probabilities
    probs = torch.softmax(yb, dim=0)

    return predicted_class, probs


if __name__ == "__main__":
    image_path = '/home/selimon/Desktop/AI/wdwyl_ros1/src/perception/brand_classification/data/testing/heineken/1.jpg'
    predicted_class, probs = image_classify(image_path)
    
    if predicted_class is not None:
        brands = ('4_pines', 'crown', 'great_northern', 'heineken')
        print(f'Predicted: {brands[predicted_class]}')
        print('Probabilities:')
        print(probs)