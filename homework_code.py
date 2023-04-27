#Lets get even fancier and take a look at the images that our model is classifying. 
#This will display the images that have been classified as well as the name and confidence score. 
%matplotlib inline
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np

# load the model
model = load_model('goodmodel1.h5')

# define a list of image paths
image_paths = ['/mnt/c/Users/kenda/OneDrive/Desktop/ecol 346/Independent Project/Images/Test/Cancer/Test.png', '/mnt/c/Users/kenda/OneDrive/Desktop/ecol 346/Independent Project/Images/Test/Normal/Test2.png', '/mnt/c/Users/kenda/OneDrive/Desktop/ecol 346/Independent Project/Images/Test/Cancer/Test3.png','/mnt/c/Users/kenda/OneDrive/Desktop/ecol 346/Independent Project/Images/Test/Normal/Test4.png']

def predict(model, image_paths):
    for i, img_path in enumerate(image_paths):
        # load the image
        img = load_img(img_path, target_size=(150, 150))
        
        # display the image
        plt.imshow(np.uint8(img))
        plt.show()

        # preprocess the image
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.

        # make prediction
        pred = model.predict(x)[0]
        
        # print the prediction result
        if pred <= 0.5:
            print(f"Image {i+1} is predicted as normal with confidence {100-pred.item()*100:.2f}%")
        else:
            print(f"Image {i+1} is predicted as cancer with confidence {pred.item()*100:.2f}%")
        print('\n')

# call the predict function
predict(model, image_paths)
