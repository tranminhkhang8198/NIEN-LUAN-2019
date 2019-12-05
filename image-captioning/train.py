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
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# import fasttext
from gensim.models import KeyedVectors

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


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
    print(len(all_train_captions))

    # Consider only words which occur at least 10 times in the corpus
    word_count_threshold = 10
    word_counts = {}
    nsents = 0
    for sent in all_train_captions:
        nsents += 1
        for w in sent.split(" "):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print("preprocessed words %d -> %d" % (len(word_counts), len(vocab)))

    ixtoword = {}
    wordtoix = {}
    ix = 1

    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    vocab_size = len(ixtoword) + 1  # one for appended 0's
    print("Vocabulary size: ", vocab_size)
    return vocab_size, ixtoword, wordtoix


def process_embedding(vocab_size, wordtoix):
    # define embedding dimension
    embedding_dim = 400
    # Get 400-dim dense vector for each in out vocabulary
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # Load pre-trained language model
    model = KeyedVectors.load_word2vec_format(
        "./word2vec/wiki.vi.model.bin", binary=True
    )

    for word, i in wordtoix.items():
        if word in model:
            embedding_matrix[i] = model[word]
    print("Embedding matrix shape: ", embedding_matrix.shape)

    return embedding_matrix, embedding_dim


# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n = 0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n += 1
            # retrieve the photo feature
            photo = photos[key + ".jpg"]
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(" ") if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n == num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n = 0


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


# get train images
filename = "./dataset_text/train_images.txt"
train = load_set(filename)
print("Train len pics: ", len(train))

# get train images descriptions
train_descriptions = load_clean_descriptions("descriptions.txt", train)
print("Desc len: ", len(train_descriptions))

# get train images encoded
train_features = load(open("./pickle/encoded_train_images.pkl", "rb"))
print("Photos: train=%d" % len(train_features))

# determine the maximum sequence length
max_length = max_length(train_descriptions)
print("Description Length: %d" % max_length)

# get params for model
vocab_size, ixtoword, wordtoix = process_vocab(train_descriptions)
embedding_matrix, embedding_dim = process_embedding(vocab_size, wordtoix)

# define model
model = define_model(vocab_size, max_length, embedding_matrix, embedding_dim)

model_path = "./model_weights/model_20.h5"

if not os.path.exists(model_path):
    epochs = 20
    number_pics_per_bath = 3
    steps = len(train_descriptions) // number_pics_per_bath

    for i in range(epochs):
        generator = data_generator(
            train_descriptions,
            train_features,
            wordtoix,
            max_length,
            number_pics_per_bath,
        )
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        model.save("./model_weights/model_" + str(i) + ".h5")

    for i in range(epochs):
        generator = data_generator(
            train_descriptions,
            train_features,
            wordtoix,
            max_length,
            number_pics_per_bath,
        )
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        model.save("./model_weights/model_" + str(i) + ".h5")

    model.optimizer.lr = 0.0001
    epochs = 20
    number_pics_per_bath = 6
    steps = len(train_descriptions) // number_pics_per_bath

    for i in range(epochs):
        generator = data_generator(
            train_descriptions,
            train_features,
            wordtoix,
            max_length,
            number_pics_per_bath,
        )
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)

    model.save_weights("./model_weights/model_20.h5")
