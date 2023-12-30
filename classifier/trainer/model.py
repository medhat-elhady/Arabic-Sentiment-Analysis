
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import datetime
import shutil

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from tensorflow.keras import callbacks
#from google.cloud import aiplatform
from official.nlp import optimization  # to create AdamW optmizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


#'./data/ar_reviews_100k.tsv'
def prepare_data(train_data_path):
    
    df = pd.read_csv(train_data_path, sep='\t')
    # split the data to train and test
    # encode labels to be 0 & 1
    

    df_train, df_test = train_test_split(df, shuffle=True, stratify=df['label'], test_size=0.2)


    #labels = tf.keras.utils.to_categorical(df['label'].values, 3)
    oneencoder = OneHotEncoder()
    labels_train = oneencoder.fit_transform(df_train['label'].values.reshape(-1, 1)).toarray()
    features_train = df_train['text'].values


    labels_test = oneencoder.transform(df_test['label'].values.reshape(-1, 1)).toarray()
    features_test = df_test['text'].values
    
    return features_train, labels_train, features_test, labels_test


def build_classifier_model(dropout_rate=0.1):
    
    # defining the URL of the smallBERT model to use
    tfhub_handle_encoder = (
        "https://www.kaggle.com/models/jeongukjae/distilbert/frameworks/TensorFlow2/variations/multi-cased-l-6-h-768-a-12/versions/1"
    )

    # defining the corresponding preprocessing model for the BERT model above
    tfhub_handle_preprocess = (
        "https://kaggle.com/models/jeongukjae/distilbert/frameworks/TensorFlow2/variations/multi-cased-preprocess/versions/2"
    )
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='precessing_layer')

    encoder_inputs = preprocessing_layer(text_input)
    
    encoder = hub.KerasLayer(tfhub_handle_encoder,
                            trainable=True, name ='encoder_layer')
   
    outputs = encoder(encoder_inputs)
    net = outputs["pooled_output"]
    net = tf.keras.layers.Dropout(dropout_rate)(net)
    #net = tf.keras.layers.Dense(260, activation='relu')(net)
    net = tf.keras.layers.Dense(3, activation="softmax", name="classifier")(net)
    return tf.keras.Model(text_input, net)


# Let's check that the model runs with the output of the preprocessing model.

def train_and_evaluate(hparam):
    
    batch_size = hparam['batch_size']
    validation_split = float(hparam['validation_split'])
    init_lr = hparam['lr']
    output_dir = hparam['output_dir']
    
    train_data_path = hparam['train_data_path']
    dropout_rate = hparam['dropout_rate']
    epochs = hparam['epochs']
    
    if tf.io.gfile.exists(output_dir):
        tf.io.gfile.rmtree(output_dir)
    
    model_export_path = os.path.join(output_dir, 'SavedModel')
    checkpoint_path = os.path.join(output_dir, "checkpoints")
    tensorboard_path = os.path.join(output_dir, "tensorboard")
    
    checkpoint_cb = callbacks.ModelCheckpoint(
        checkpoint_path, save_weights_only=True, verbose=1
    )
    
    tensorboard_cb = callbacks.TensorBoard(tensorboard_path, histogram_freq=1)

    classifier_model = build_classifier_model(dropout_rate)

    features, labels, features_test, labels_test = prepare_data(train_data_path)
    
    steps_per_epoch = len(features) // batch_size
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)
    
    
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = [tf.metrics.CategoricalAccuracy()]
        


    optimizer = optimization.create_optimizer(
        init_lr=init_lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type="adamw",
    )

    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)



    history = classifier_model.fit(
        x=features, 
        y = labels, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data= (features_test, labels_test), 
        callbacks=[tensorboard_cb]
    )


    # Exporting the model with default serving function.
    classifier_model.save(model_export_path)
    return history
