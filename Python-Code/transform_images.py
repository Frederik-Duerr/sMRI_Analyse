from torchvision import transforms
from PIL import Image


def Image_transforms_train_val(X_train_image_paths,X_val_image_paths):
    transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225] # Same here
                )
    ])
    X_train_tensors = []
    X_val_tensors = []

    for Image_path in X_train_image_paths:
        image = Image.open(Image_path).convert("L")
        tensor = transformer(image)
        X_train_tensors.append(tensor)

    for Image_path in X_val_image_paths:
        image = Image.open(Image_path).convert("L")
        tensor = transformer(image)
        X_val_tensors.append(tensor)
    
    return X_train_tensors, X_val_tensors

