#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:42:09 2017
Simple neural networks using tensorflow

@author: rhdzmota
"""

# Imports
import numpy as np
import pandas as pd


class Dataset(object):
    """Dataset class for handling dataset normalization and splits."""

    def __init__(self, input_data, output_data, normalize=None, datatypes={}):
        """Dataset class for handling dataset normalization and splits."""
        self.input_data = input_data.values if self._isPandasOrSeries(
                                                input_data) else input_data
        self.output_data = output_data.values if self._isPandasOrSeries(
                                                output_data) else output_data
        self.input_data = self._addOneDim(self.input_data)
        self.output_data = self._addOneDim(self.output_data)
        self.norm_input_data, self.norm_output_data = None, None
        self.normalized = normalize
        if normalize:
            if len(datatypes.keys()) == 0:  # create datatypes dict
                datatypes["input_data"] = [1]*np.shape(input_data)[-1]
                datatypes["output_data"] = [1]*np.shape(output_data)[-1]
            if "mean" in normalize:
                df_input = pd.DataFrame(self.input_data)
                self.norm_input_data, self.mu_x, self.std_x = \
                    self._normalizeWithMean(df_input,
                                            datatypes["input_data"])
                df_output = pd.DataFrame(self.output_data)
                self.norm_output_data, self.mu_y, self.std_y = \
                    self._normalizeWithMean(df_output,
                                            datatypes["output_data"])
            if "minmax" in normalize:
                df_input = pd.DataFrame(self.input_data)
                self.norm_input_data, self.min_x, self.max_x = \
                    self._normalizeWithMinMax(df_input,
                                              datatypes["input_data"])
                df_output = pd.DataFrame(self.output_data)
                self.norm_output_data, self.min_y, self.max_y = \
                    self._normalizeWithMinMax(df_output,
                                              datatypes["output_data"])
        datasets = self._splitDataset(0.7)
        self.test, self.train = datasets["test"], datasets["train"]

    def _normalizeWithMean(self, df, typelist):
        """Normalize data using (x-mu)/sigma."""
        vector_mu, vector_std = [], []
        for col, is_numeric in zip(df.columns, typelist):
            _mean = np.mean(df[col].values) if is_numeric else 0
            _std = np.std(df[col].values) if is_numeric else 1
            vector_mu.append(_mean)
            vector_std.append(_std)
        return (((df-vector_mu)/vector_std).values, np.array(vector_mu),
                np.array(vector_std))

    def _normalizeWithMinMax(self, df, typelist):
        """Normalize with minmax."""
        _min, _max = [], []
        for col in df.columns:
            _min.append(np.min(df[col].values))
            _max.append(np.max(df[col].values))
        _min, _max = np.array(_min), np.array(_max)
        return (df-_min)/(_max-_min), _min, _max

    def _isPandasOrSeries(self, df):
        return (type(df) == pd.core.frame.DataFrame) or \
                                (type(df) == pd.core.series.Series)

    def _addOneDim(self, data):
        shape = len(np.shape(data))
        return np.asarray([[i] for i in data]) if shape == 1 \
            else np.asarray(data)

    def _splitDataset(self, cut=0.7):
        # Length and random index
        n = len(self.input_data)
        _index = np.arange(n)
        np.random.shuffle(_index)
        # Retreive data
        x_data = self.norm_input_data if self.normalized else self.input_data
        y_data = self.norm_output_data if self.normalized else self.output_data
        # convert to dataframe
        x_data, y_data = pd.DataFrame(x_data), pd.DataFrame(y_data)
        # separate into test and train
        _split = np.int(np.asscalar(np.round(n*cut)))
        train_set = (x_data.iloc[_index[:_split]].values,
                     y_data.iloc[_index[:_split]].values)
        test_set = (x_data.iloc[_index[_split:]].values,
                    y_data.iloc[_index[_split:]].values)
        return {'test': test_set, 'train': train_set}

    def nextBatch(self, batch_size=None):
        """Generate next data batch."""
        n = len(self.train[0])
        if not batch_size:
            batch_size = np.int(n*0.2) if n > 500 else 100
        if batch_size > n:
            print("Error: No enought data (min. batch_size = 100).")
            return None
        sample = np.random.choice(np.arange(n), batch_size)
        return self.train[0][sample], self.train[1][sample]


def testData():
    """Test Object."""
    x_data = pd.DataFrame({"x1": np.random.rand(5000),
                           "x2": np.arange(5000)**2})
    y_data = x_data.apply(lambda x: np.power(np.sqrt(x[1]), x[0]) + 5 *
                          np.sin(np.pi * x[0]), 1)
    datasets = Dataset(x_data, y_data, normalize="mean")
    return x_data, y_data, datasets
