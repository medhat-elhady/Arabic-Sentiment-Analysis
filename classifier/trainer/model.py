
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

import pandas as pd
import numpy as np


#'./data/ar_reviews_100k.tsv'
def prepare_data(train_data_path):
    df = pd.read_csv(train_data_path, sep='\t')
    df = df[df['label']!='Mixed']
    
    msk = np.random.rand(len(df)) < 0.8
    df_train = df[msk]
    df_test = df[~msk]
    
    labels_train = df_train['label'].map({'Positive':1, 'Negative':0}).values
    features_train = df_train['text'].values
    
    
    labels_test = df_test['label'].map({'Positive':1, 'Negative':0}).values
    features_test = df_test['text'].values
    
    return features_train, labels_train, features_test, labels_test


def build_classifier_model(dropout_rate=0.1):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    processing_link = "https://kaggle.com/models/jeongukjae/distilbert/frameworks/TensorFlow2/variations/multi-cased-preprocess/versions/2"
    encoder_link = "https://www.kaggle.com/models/jeongukjae/distilbert/frameworks/TensorFlow2/variations/multi-cased-l-6-h-768-a-12/versions/1"
    preprocessing_layer = hub.KerasLayer(processing_link)

    encoder_inputs = preprocessing_layer(text_input)
    
    encoder = hub.KerasLayer(encoder_link,trainable=True)
   
    outputs = encoder(encoder_inputs)
    net = outputs["pooled_output"]
    net = tf.keras.layers.Dropout(dropout_rate)(net)
    #net = tf.keras.layers.Dense(260, activation='relu')(net)
    net = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(net)
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
    
    
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = tf.keras.metrics.BinaryAccuracy()


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
        callbacks=[checkpoint_cb, tensorboard_cb]
    )


    # Exporting the model with default serving function.
    classifier_model.save(model_export_path)
    return history
