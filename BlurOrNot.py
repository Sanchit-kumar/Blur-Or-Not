import numpy as np 
from skimage.transform import resize
from scipy.ndimage import variance 
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import laplace #using the laplacian variance to predict the blur image
import matplotlib.pyplot as plt

IMG_WIDTH=600 
IMG_HEIGHT=400 
IMG_SIZE=(IMG_HEIGHT,IMG_WIDTH) #taking image size because I have tested threshod value w.r.t this size

def BlurOrNot(path):             #input will be path of the image
                                # Loading the image
    img = io.imread(path)
    img_org=img
    img = resize(img, IMG_SIZE) # Resizing the image

    img = rgb2gray(img)         # Using grayscale for prediction

    # Using laplacian for the gradient information  with kernel size=3 for making predictions
    
    laplace_value = laplace(img, ksize=3)
    
    var=variance(laplace_value)

    if var>0.003:               # This threshold value I have alloted by testing my results on the a blur image data
                                # available in kaggle i.e https://www.kaggle.com/kwentar/blur-dataset/version/2
                                # Link of my work which shows how it take threshold is available in report.
        print("Output: Not blurred")
    else:
        print("Output: Blurred Image")
    plt.imshow(img_org)
    plt.show()
