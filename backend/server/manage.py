#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

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


class Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self


class ColumnSelector(Transformer):
    def __init__(self, column=None, columns=None, fillna_value=None, fillna_from_column=None):
        self.column = column
        self.columns = columns
        self.fillna_value = fillna_value
        self.fillna_from_column = fillna_from_column

    def transform(self, df):
        if self.column and self.columns:
            raise Exception('You can not use both option [column, columns]')
        elif self.column:
            series = df[self.column]
        elif self.columns:
            series = df.apply(lambda x: ' '.join([str(x[f]) for f in self.columns]), axis=1)
        else:
            raise Exception('You have to chose one option [column, columns]')

        if self.fillna_from_column:
            series = series.fillna(df[self.fillna_from_column])

        if self.fillna_value:
            series = series.fillna(self.fillna_value)

        return series


class SubCategorySpliter(Transformer):
    def __init__(self, delimiter='/', maxsplit=2, nth_split=0):
        self.delimiter = delimiter
        self.nth_split = nth_split
        self.maxsplit = maxsplit

    def transform(self, series):
        return series.apply(lambda x: x.split(self.delimiter, self.maxsplit)[self.nth_split])


class SeriesToArray(Transformer):
    def __init__(self, shape=(-1, 1)):
        self.shape = shape

    def transform(self, series):
        return series.values.reshape(self.shape)


class TextToSeq(Transformer):
    def __init__(
            self,
            num_words=None,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            maxlen=100,
            sparse=True,
            dtype=np.uint32
    ):
        self.num_words = num_words
        self.filters = filters
        self.lower = lower
        self.maxlen = maxlen
        self.sparse = sparse
        self.dtype = dtype

    def fit(self, series, y=None):
        self.tokenizer = Tokenizer(num_words=self.num_words, filters=self.filters, lower=self.lower)
        self.tokenizer.fit_on_texts(series)

        return self

    def transform(self, series):
        seqs = self.tokenizer.texts_to_sequences(series)
        pad_seqs = pad_sequences(seqs, maxlen=self.maxlen, dtype=self.dtype)
        if self.sparse:
            pad_seqs = csr_matrix(pad_seqs)

        return pad_seqs


class DataGenerator(keras.utils.Sequence):
    def __init__(self, df, pipes, target, batch_size=1024, train=True, shuffle=True):
        self.df = df
        self.pipes = pipes
        self.target = target
        self.train = train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        sub_idxs = self.idxs[idx * self.batch_size:(idx + 1) * self.batch_size]
        sub_df = self.df.iloc[sub_idxs]
        X, y = self.__data_generation(sub_df)

        return X, y

    def on_epoch_end(self):
        self.idxs = np.arange(len(self.df))
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __data_generation(self, sub_df):
        X = [pipe.transform(sub_df) for pipe in self.pipes]
        if self.train:
            return X, np.log1p(sub_df[self.target].values)

        return X

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
