from torchvision import transforms
from PIL import Image


def Image_transforms_train_val(X_train_image_paths):
    transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225] # Same here
                )
    ])
    X_train_axial = []
    X_train_sagittal = []
    X_train_coronal = []

    X_train_tensors_axial = []
    X_train_tensors_sagittal = []
    X_train_tensors_coronal = []
  
    print(X_train_image_paths['Orientation'].value_counts())

    X_train_sagittal = X_train_image_paths.loc[X_train_image_paths['Orientation'] =='sagittal','image_path']
    X_train_axial = X_train_image_paths.loc[X_train_image_paths['Orientation'] =='axial','image_path']
    X_train_coronal = X_train_image_paths.loc[X_train_image_paths['Orientation'] == 'coronal','image_path']


    for Image_path in X_train_axial:
              
        image = Image.open(Image_path).convert("L")
        tensor = transformer(image)
        X_train_tensors_axial.append(tensor)

    for Image_path in X_train_sagittal:
              
        image = Image.open(Image_path).convert("L")
        tensor = transformer(image)
        X_train_tensors_sagittal.append(tensor)

    for Image_path in X_train_coronal:
              
        image = Image.open(Image_path).convert("L")
        tensor = transformer(image)
        X_train_tensors_coronal.append(tensor)   
    
    return X_train_tensors_axial, X_train_tensors_sagittal,X_train_tensors_coronal

