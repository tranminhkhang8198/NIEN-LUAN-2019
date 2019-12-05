import string
import numpy as np
from numpy import array
import pandas as pd
import os
from PIL import Image
import glob
from pickle import dump, load
import pickle
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import (
    LSTM,
    Embedding,
    TimeDistributed,
    Dense,
    RepeatVector,
    Activation,
    Flatten,
    Reshape,
    concatenate,
    Dropout,
    BatchNormalization,
)
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image as keras_image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random

import requests
import shutil
from datetime import datetime

from gensim.models import KeyedVectors

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from Speaker import Speaker
from threading import Thread


# Load the inception v3 model
model = InceptionV3(weights="imagenet")
# Create a new model, by removing the last layer (output layer) from the inception v3
model_new = Model(model.input, model.layers[-2].output)


# load doc into memory
def load_doc(filename):
    file = open(filename, "r")
    text = file.read()
    file.close()
    return text


def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split("\n"):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split(".")[0]
        dataset.append(identifier)
    return set(dataset)


# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split("\n"):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = "startseq " + " ".join(image_desc) + " endseq"
            # store
            descriptions[image_id].append(desc)
    return descriptions


# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


def process_vocab(train_descriptions):
    # Create a list of all the training captions
    all_train_captions = []
    for key, val in train_descriptions.items():
        for cap in val:
            all_train_captions.append(cap)

    # Consider only words which occur at least 10 times in the corpus
    word_count_threshold = 10
    word_counts = {}
    nsents = 0
    for sent in all_train_captions:
        nsents += 1
        for w in sent.split(" "):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

    ixtoword = {}
    wordtoix = {}
    ix = 1

    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    vocab_size = len(ixtoword) + 1  # one for appended 0's

    return vocab_size, ixtoword, wordtoix


def process_embedding(vocab_size, wordtoix):
    # define embedding dimension
    embedding_dim = 400

    # Get 200-dim dense vector for each of the 10000 words in out vocabulary
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # model = fasttext.load_model("./word2vec/wiki.vi.model.bin")
    model = KeyedVectors.load_word2vec_format(
        "./word2vec/wiki.vi.model.bin", binary=True
    )

    for word, i in wordtoix.items():
        if word in model:
            embedding_matrix[i] = model[word]

    return embedding_matrix, embedding_dim


def define_model(vocab_size, max_length, embedding_matrix, embedding_dim):
    # feature extractor model
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation="relu")(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation="relu")(decoder1)
    outputs = Dense(vocab_size, activation="softmax")(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # summarize model
    print(model.summary())

    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = False
    # compile model
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    return model


def greedySearch(photo):
    # seed the generation process
    in_text = "startseq"
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = ixtoword[yhat]
        # append as input for generating the next word
        in_text += " " + word
        # stop if we predict the end of the sequence
        if word == "endseq":
            break

    # ignore 'startseq' and 'endseq' from the final caption
    final = in_text.split()
    final = final[1:-1]
    final = " ".join(final)
    return final


def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = keras_image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = keras_image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x


# Function to encode a given image into a vector of size (2048, )
def encode(image):
    image = preprocess(image)  # preprocess the image
    fea_vec = model_new.predict(image)  # Get the encoding vector for the image
    return fea_vec


if __name__ == "__main__":
    # Load the inception v3 model
    model = InceptionV3(weights="imagenet")
    # Create a new model, by removing the last layer (output layer) from the inception v3
    model_new = Model(model.input, model.layers[-2].output)

    # Get max length of the descriptions
    filename = "./dataset_text/train_images.txt"
    train = load_set(filename)
    train_descriptions = load_clean_descriptions("descriptions.txt", train)
    max_length = max_length(train_descriptions)

    # Get params for model
    vocab_size, ixtoword, wordtoix = process_vocab(train_descriptions)
    embedding_matrix, embedding_dim = process_embedding(vocab_size, wordtoix)

    # Define model and load weight
    model = define_model(vocab_size, max_length, embedding_matrix, embedding_dim)
    model_path = "./model_weights/model_20.h5"
    model.load_weights(model_path)

    while True:
        # Save image from url
        print("*" * 100)
        image_url = input("Please enter image url: ")
        resp = requests.get(image_url, stream=True)
        filename = "local_image" + str(datetime.timestamp(datetime.now())) + ".jpg"
        test_images_dir = "./test_images/"
        local_file = open(test_images_dir + filename, "wb")
        resp.raw.decode_content = True
        shutil.copyfileobj(resp.raw, local_file)
        del resp

        # Get image path after saving
        image_path = test_images_dir + filename

        # Preprocess image
        image = encode(image_path)
        image = image.reshape((1, 2048))

        # Get caption
        text = greedySearch(image)

        # Show the result
        x = plt.imread(image_path)
        plt.title(text + "\n")
        plt.imshow(x)
        plt.show()

        tf.reset_default_graph()
