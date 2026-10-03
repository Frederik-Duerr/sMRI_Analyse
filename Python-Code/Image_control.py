from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np

def image_control(dataframe):
    size = []
    mode = []
    format = []
    max_int = []
    min_int = []
    for path in dataframe['image_path']:
        # print(path)
        image = Image.open(path)
        size.append(image.size)
        mode.append(image.mode)
        format.append(image.format)
        # max_int.append(np.max(image))
        # min_int.append(np.min(image))

    dataframe['Image_Size'] = size
    dataframe['Image_Mode'] = mode
    dataframe['Image_Format'] = format
    # dataframe['Image_Max_Int'] = max_int
    # dataframe['Image_Min_Int'] = min_int
    return dataframe
        #plt.imshow(image,cmap='gray')
        #plt.show()

def Image_histogram(dataframe):
    path = dataframe['image_path'][100]
    
    image = Image.open(path)
    image = np.array(image)
    plt.hist(image)
    plt.show()