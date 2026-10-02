from torchvision import transforms
def Image_transforms(X_train_images,X_val_images):
    Image_list = [X_train_images,X_val_images,]
    for list in Image_list: 
        for image in list: 
            image = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            Image_list.append(image)

    return X_train_images,X_val_images