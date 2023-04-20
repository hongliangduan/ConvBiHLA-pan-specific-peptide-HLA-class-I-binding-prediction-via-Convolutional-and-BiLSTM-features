import pandas as pd
from tensorflow.python.keras.preprocessing.text import text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,f1_score,matthews_corrcoef
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
def binary_focal_loss(gamma=2, alpha=0.5):
    """
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
def cut(x):
    a=' '.join(x)
    return a# #

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_file', required=True)
parser.add_argument('--ckpt_path', required=True)
args = parser.parse_args()
task = args.task
test_file = args.test_file
ckpt_path = args.ckpt_path
model = load_model(ckpt_path,custom_objects={'binary_focal_loss_fixed': binary_focal_loss()})
with open('tokenizer.json') as f:
    token = json.load(f)
    tokenizer =text.tokenizer_from_json(token)
    word_index = tokenizer.word_index
    print(word_index)

test = pd.read_csv(test_file)
x_test = tokenizer.texts_to_sequences([cut(x) for x in list(test['comb'])])
x_test_padded_seqs = pad_sequences(x_test, maxlen=50,padding='post',truncating='post')  # padding
y_test = to_categorical(list(test['label']), num_classes=2)
result = model.predict(x_test_padded_seqs) # 预测样本属于每个类别的概率
result_labels = np.argmax(result, axis=1)
y_predict = list(map(int, result_labels))
y_predict_onehot = to_categorical(y_predict)
accuracy= accuracy_score(y_test, y_predict_onehot)
print('准确率', accuracy)
roc_auc_score=roc_auc_score(y_test,result)
print('roc_auc_score',roc_auc_score1)
f1_s=f1_score(list(test['label']),y_predict)
print('f1_score',f1_s)
MCC=matthews_corrcoef(list(test['label']),y_predict)
print('mcc',MCC)


