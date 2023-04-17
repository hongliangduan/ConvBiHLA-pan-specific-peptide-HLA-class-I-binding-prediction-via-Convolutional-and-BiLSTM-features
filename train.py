import numpy as np
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization, Activation,Input,Attention
from tensorflow.keras.layers import Convolution1D, MaxPooling1D,concatenate,Flatten
from tensorflow.keras.layers import LSTM,Bidirectional,GRU
import pandas as pd
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,f1_score,matthews_corrcoef
from tensorflow.keras import initializers, regularizers, constraints


import time
def get_now_time():
    now =  time.localtime()
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", now)
    return now_time

def binary_focal_loss(gamma=2, alpha=0.5):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


class LossHistory(callbacks.Callback):
    def __init__(self,file_name):
        self.file=file_name
    def on_train_begin(self, logs={}):
        self.losses, self.acc, self.val_losses, self.val_acc= [],[],[],[]
        self.loss_f = open(self.file, 'a')
        self.loss_f.write('train_begin:'+get_now_time()+'\n')


    def on_epoch_end(self, epoch, logs={}):
        self.loss_f.write('epoch: '+ str(epoch) + ', loss:' + str(logs.get('loss'))+
                          ', acc:' + str(logs.get('accuracy')) +
                          ', val_loss:' + str(logs.get('val_loss'))
                          + ', val_acc:' + str(logs.get('val_accuracy'))+'\n')

    def on_train_end(self, logs=None):
        self.loss_f.write('train_end:'+get_now_time()+'\n')
        self.loss_f.close()

# 构建模型
def CNN_lstm_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test,ckpt):
    main_input = Input(shape=(max_len,), dtype='float64')
    embedder = Embedding(len(word_index) + 1, 128, input_length=max_len, trainable=True,mask_zero=True)
    embed = embedder(main_input)
    lstm = (Bidirectional(LSTM(64, return_sequences=True)))(embed)
    cnn1 = Convolution1D(64, 2, padding='same', strides=1, activation='relu')(lstm)
    cnn1 = MaxPooling1D(pool_size=49)(cnn1)
    cnn2 = Convolution1D(64, 3, padding='same', strides=1, activation='relu')(lstm)
    cnn2 = MaxPooling1D(pool_size=48)(cnn2)
    cnn3 = Convolution1D(64, 4, padding='same', strides=1, activation='relu')(lstm)
    cnn3 = MaxPooling1D(pool_size=47)(cnn3)
    cnn= concatenate([cnn1,cnn2,cnn3], axis=-1)
    cnn = Flatten()(cnn)
    lstm = Flatten()(lstm)
    last_emb = concatenate([cnn,lstm], axis=-1)
    last_emb = Dropout(0.5)(last_emb)
    last_emb = Dense(1024, activation='relu')(last_emb)
    last_emb = Dropout(0.5)(last_emb)
    last_emb = Dense(512, activation='relu')(last_emb)
    last_emb = Dropout(0.5)(last_emb)
    main_output = Dense(2, activation='softmax')(last_emb)
    model = Model(inputs=main_input, outputs=main_output)
    model.summary()
    model.compile(oss=[binary_focal_loss(alpha=.25, gamma=2)],
                  optimizer='adam',
                  metrics=['accuracy'])
    history = LossHistory(f'{task}_record.txt')
    callbacks_list=[callbacks.ModelCheckpoint(filepath=ckpt,
                    monitor='val_accuracy',
                    save_best_only=True),
                    TensorBoard(log_dir='dir'),history]
    model.fit(x_train_padded_seqs, y_train, batch_size=1024, epochs=300,
              validation_data=(x_test_padded_seqs, y_test),callbacks=callbacks_list)


def cut(x):
    a=' '.join(x)
    return a

def process_data(data):
    x_data = tokenizer.texts_to_sequences([cut(x) for x in list(data[cloumn])])
    x_data_padded_seqs = pad_sequences(x_data, maxlen=max_len,padding='post',truncating='post')  # padding
    y_data = to_categorical(list(data['label']), num_classes=2)
    return x_data_padded_seqs,y_data
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='fold0', required=False)
    parser.add_argument('--test_data', required=True)
    args = parser.parse_args()
    task = args.task
    test_data = args.test_data
    train = pd.read_csv(f"train_data_{task}.csv")
    test= pd.read_csv(f"{test_data}.csv")
    train = train.sample(frac=1)
    max_len=50
    cloumn='comb'
    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts([cut(x) for x in list(train[cloumn])])
    tokenizer_json = tokenizer.to_json()
    with open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    word_index = tokenizer.word_index
    x_train_padded_seqs,y_train = process_data(train)
    x_test_padded_seqs,y_test = process_data(test)
    CNN_lstm_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, f'{task}_{test_data}.h5')