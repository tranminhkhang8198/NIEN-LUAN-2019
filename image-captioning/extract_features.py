import string
import numpy as np
from numpy import array
import os
from PIL import Image
import glob
from pickle import dump, load
import pickle
from time import time

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.applications.inception_v3 import preprocess_input


# Load the inception v3 model
model = InceptionV3(weights="imagenet")
# Create a new model, by removing the last layer (output layer) from the inception v3
model_new = Model(model.input, model.layers[-2].output)

# Path contains all the images
images = "./dataset_images/"
# Create a list of all image names in the directory
img = glob.glob(images + "*.jpg")


def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x


# Function to encode a given image into a vector of size (2048, )
def encode(image):
    image = preprocess(image)  # preprocess the image
    fea_vec = model_new.predict(image)  # Get the encoding vector for the image
    fea_vec = np.reshape(
        fea_vec, fea_vec.shape[1]
    )  # reshape from (1, 2048) to (2048, )
    return fea_vec


def get_path_images(filename):
    # Read the validation image names in set# Read the images names in a set
    input_images = set(open(filename, "r").read().strip().split("\n"))
    # create a list of all the images with their full path names
    input_img = []
    for i in img:  # img is list of full path names of all images
        if i[len(images) :] in input_images:  # Check if the image belongs to input set
            input_img.append(i)  # Add it to the list of input images
    return input_img


def encoding_images(input_img, save_path):
    start = time()
    encoding_images = {}
    for img in input_img:
        encoding_images[img[len(images) :]] = encode(img)
    print("Time taken in seconds =", time() - start)
    # Save the bottleneck test features to disk
    with open(save_path, "wb") as encoded_pickle:
        pickle.dump(encoding_images, encoded_pickle)


# encoding train images
train_images_file = "./dataset_text/train_images.txt"
save_train_img_path = "./pickle/encoded_train_images.pkl"
train_img = get_path_images(train_images_file)
encoding_images(train_img, save_train_img_path)
