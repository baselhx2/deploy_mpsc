from django.shortcuts import render
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers, Model
import joblib
        

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    
class CustomOutputLayer(layers.Layer):
    def __init__(self, min_val, max_val, **kwargs):
        super(CustomOutputLayer, self).__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val
        
    def call(self, inputs):
        outputs = tf.sigmoid(inputs)* (self.max_val- self.min_val)+ self.min_val
        output = tf.reduce_mean(outputs, axis=1, keepdims=True)
        
        return output
        
def get_model(seq_1_len, seq_2_len, feats_len, seq_1_max, seq_2_max, out_min_val, out_max_val, use_cuslayer=True):
    seq_1 = layers.Input((seq_1_len,))
    s_1 = layers.Embedding(seq_1_max, 128)(seq_1)
    s_1 = TransformerBlock(128, 4, 128)(s_1)
    s_1 = layers.GlobalAvgPool1D()(s_1)
    
    seq_2 = layers.Input((seq_2_len,))
    s_2 = layers.Embedding(seq_2_max, 128)(seq_2)
    s_2 = TransformerBlock(128, 4, 128)(s_2)
    s_2 = layers.GlobalAvgPool1D()(s_2)
    
    rest_ = layers.Input((feats_len,))
    
    vec = layers.Concatenate(axis=1)([s_1, s_2, layers.Dense(64)(rest_)])
    vec = layers.Dense(64)(vec)
    if use_cuslayer:
        out = CustomOutputLayer(out_min_val, out_max_val)(vec)
    else:
        out = layers.Dense(1)(vec)
        
    model = Model(inputs=[seq_1, seq_2, rest_], outputs=[out])
    
    return model

MODEL_PATH = '../../models/'
MAX_SEQ_LEN_NAME = 20
MAX_SEQ_LEN_DESC = 68
FEATS_LEN = 5249
MAX_NUM_WORDS_NAME = 40000
MAX_NUM_WORDS_DESC = 80000
PIPES = joblib.load(MODEL_PATH+ 'pipes.pkl')
MODEL = get_model(MAX_SEQ_LEN_NAME,
                  MAX_SEQ_LEN_DESC,
                  FEATS_LEN,
                  MAX_NUM_WORDS_NAME,
                  MAX_NUM_WORDS_DESC,
                  0,
                  8,
                  )
MODEL.load_weights(MODEL_PATH+ 'model_1_weights.hdf')

##
def index(request):
    return render(request, 'index.html')

def predict(request):
     if request.method=='POST':
        temp = {}
        temp['name'] = str(request.POST.get('name'))
        temp['item_description'] = str(request.POST.get('item_description'))
        temp['item_condition_id'] = int(request.POST.get('item_condition_id'))
        temp['category_name'] = '/'.join([str(request.POST.get('main_category')),
                                          str(request.POST.get('sub1_category')),
                                          str(request.POST.get('sub2_category'))])
        temp['brand_name'] = str(request.POST.get('brand_name'))
        temp['shipping'] = str(request.POST.get('shipping'))
        
        sub_df = pd.DataFrame({'0': temp}).transpose()
        sub_df['shipping'] = sub_df['shipping'].map({0: 'Free shipping', 1: 'Shipping fees'})
        sub_df['item_condition_id'] = sub_df['item_condition_id'].map(
            {1: 'New', 2: 'Like new', 3: 'Good', 4: 'Fair', 5: 'Poor'})

        sub_pp = [pipe.transform(sub_df) for pipe in PIPES]
        sub_pred = np.expm1(MODEL.predict(sub_pp))
        
        return render(request, 'result.html', {'result': sub_pred[0]})

