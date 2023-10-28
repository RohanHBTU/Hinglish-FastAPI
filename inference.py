import pickle
import pandas as pd
import numpy
#import logging, os
#logging.disable(logging.WARNING)
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow_text as tf_text
from metaphone import doublemetaphone
import re

with open('vocab_data.pkl', 'rb') as fp:
    hin_vocab = pickle.load(fp)
vocab_keys=[l for l in hin_vocab]
#all_data_vocab_53k_mixed_batch_v2
reloaded = tf.saved_model.load("translator")

def t_text(line):
    line=re.sub("[.!?\\-\'\"]", "",line).lower().strip()
    string=''
    for j in line.split(' '):
        if doublemetaphone(j)[0]+'*'+doublemetaphone(j[::-1])[0]+'*'+j[:2]+'*'+j[len(j)-1:] in vocab_keys:
            string=string+list(hin_vocab[doublemetaphone(j)[0]+'*'+doublemetaphone(j[::-1])[0]+'*'+j[:2]+'*'+j[len(j)-1:]])[0]+' '
        else:
            string=string+j+' '
    return string.lower().strip()

def outcome(input):
    trans_text=t_text(input)
    result=reloaded.tf_translate(tf.constant([trans_text]))['text'][0].numpy().decode()
    return result

#print(outcome("Please timer ko rokey"))