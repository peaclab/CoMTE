#!/usr/bin/env python3
"""Contains the code for ICAPAI'21 paper "Counterfactual Explanations for Multivariate Time Series"

Authors:
    Emre Ates (1), Burak Aksar (1), Vitus J. Leung (2), Ayse K. Coskun (1)
Affiliations:
    (1) Department of Electrical and Computer Engineering, Boston University
    (2) Sandia National Laboratories

This work has been partially funded by Sandia National Laboratories. Sandia
National Laboratories is a multimission laboratory managed and operated by
National Technology and Engineering Solutions of Sandia, LLC., a wholly owned
subsidiary of Honeywell International, Inc., for the U.S. Department of
Energy’s National Nuclear Security Administration under Contract DENA0003525.
"""

import sys
import logging
import random
from pathlib import Path
import os
from glob import glob
import re

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_dataset(set_name, binary=False, window_size = 10, exclude_columns=[], **kwargs):
    if set_name in ['cstr', 'pronto']:
        kwargs['window_size'] = window_size
        kwargs['windowize'] = False
        kwargs['skip'] = 1
        kwargs['trim'] = 1
        # kwargs['noise_scale'] = 1
        # kwargs['test_split_size'] = 1
        return get_dataset_csv(set_name, exclude_columns, **kwargs)
    if set_name not in ['taxonomist', 'hpas', 'test', 'natops']:
        raise ValueError("Wrong set_name")
    if binary:
        if set_name in ['taxonomist', 'test', 'natops']:
            kwargs['make_binary'] = True
        elif set_name == 'hpas':
            kwargs['classes'] = ['none', 'dcopy']
    rootdir = Path(kwargs.get('rootdir', './data'))
    if set_name == 'taxonomist':
        kwargs['window'] = 45
        kwargs['skip'] = 45
    if set_name == 'test':
        set_name = 'taxonomist'
        kwargs['window'] = 60
        kwargs['skip'] = 60
        kwargs['test'] = True
    if set_name == 'natops':
        kwargs['windowize'] = False
    return load_hpc_data(rootdir / set_name, **kwargs)


def windowize(timeseries, labels, window=45, trim=60, skip=15, test=False):
    result_labels = []
    result_timeseries = []
    node_ids = labels.index.get_level_values('node_id').unique()
    if test:
        node_ids = random.sample(list(node_ids), 100)
    for node_id in tqdm(node_ids):
        subset = timeseries.loc[[node_id], :, :].dropna()
        temp = []
        temp_labels = []
        for i in range(len(subset.iloc[trim:-trim])):
            if i < window:
                continue
            elif i % skip != 0:
                continue
            data = subset.iloc[i - window: i]
            data.index = data.index.set_levels(
                data.index.levels[0] + '_{}'.format(i), level=0)
            temp.append(data)
            label = labels.loc[[node_id], :, :]
            label.index = label.index.set_levels(
                label.index.levels[0] + '_{}'.format(i), level=0)
            temp_labels.append(label)
        result_timeseries.append(pd.concat(temp, axis=0))
        result_labels.append(pd.concat(temp_labels, axis=0))
    return (pd.concat(result_timeseries, axis=0),
            pd.concat(result_labels, axis=0))


def drop_columns(timeseries):
    return timeseries.drop(
        [x for x in timeseries.columns
         if x.endswith('HASW') or 'per_core' in x], axis=1)


def select_classes(timeseries, labels, classes):
    if classes is None:
        return timeseries, labels
    labels = labels[labels['label'].isin(classes)]
    timeseries = timeseries.loc[labels.index.get_level_values('node_id'), :, :]
    return timeseries, labels


def process_data(timeseries, labels, classes=None, **kwargs):
    timeseries = drop_columns(timeseries)
    timeseries, labels = select_classes(timeseries, labels, classes=classes)
    timeseries = timeseries.dropna(axis=0)
    assert(not timeseries.isnull().any().any())
    # if kwargs.get('windowize', True):
    #     return windowize(timeseries, labels, **kwargs)
    return timeseries, labels


def load_hpc_data(data_folder, make_binary=False, for_autoencoder=False, **kwargs):
    if for_autoencoder:
        # Only get data from a single hardware node
        if 'none' not in kwargs.get('classes'):
            raise ValueError("Autoencoder has to train with healthy class")
        nodeid_df = pd.read_csv(data_folder / 'nids.csv')
        labels = pd.concat([pd.read_hdf(data_folder / 'train_labels.hdf'),
                            pd.read_hdf(data_folder / 'test_labels.hdf')])
        labels = labels[labels['label'].isin(kwargs.get('classes'))]
        best_nid = 0
        best_count = 0
        for nid in nodeid_df['nid'].unique():
            node_ids = nodeid_df[nodeid_df['nid'] == nid]['node_id']
            if len(labels.loc[node_ids, :, :]['label'].unique()) == 1:
                continue
            min_count = labels.loc[node_ids, :, :]['label'].value_counts().min()
            if min_count > best_count:
                best_nid = nid
                best_count = min_count

        node_ids = nodeid_df[nodeid_df['nid'] == best_nid]['node_id']
        labels = labels.loc[node_ids, :, :]
        logging.info("Returning runs from nid000%d, counts: %s",
                     best_nid, labels['label'].value_counts().to_dict())
        timeseries = pd.concat([pd.read_hdf(data_folder / 'train.hdf'),
                                pd.read_hdf(data_folder / 'test.hdf')])

        train_nodeids, test_nodeids = train_test_split(
            labels.index.get_level_values('node_id').unique(), test_size=0.2, random_state=0)
        test_timeseries = timeseries.loc[test_nodeids, :, :]
        test_labels = labels.loc[test_nodeids, :, :]
        timeseries = timeseries.loc[train_nodeids, :, :]
        labels = labels.loc[train_nodeids, :, :]
    else:

        timeseries = pd.read_hdf(data_folder / 'train.hdf')
        labels = pd.read_hdf(data_folder / 'train_labels.hdf')
        labels['label'] = labels['label'].astype(str)
        if make_binary:
            label_to_keep = labels.mode()['label'][0]
            labels[labels['label'] != label_to_keep] = 'other'

        test_timeseries = pd.read_hdf(data_folder / 'test.hdf')
        test_labels = pd.read_hdf(data_folder / 'test_labels.hdf')
        test_labels['label'] = test_labels['label'].astype(str)
        if make_binary:
            test_labels[test_labels['label'] != label_to_keep] = 'other'

    timeseries, labels = process_data(timeseries, labels, **kwargs)
    assert(not timeseries.isnull().any().any())
    test_timeseries, test_labels = process_data(
        test_timeseries, test_labels, **kwargs)
    assert(not test_timeseries.isnull().any().any())

    # Divide test data
    if kwargs.get('test', False):
        test_node_ids = [
            test_labels[test_labels['label'] == label].index.get_level_values('node_id')[0]
            for label in test_labels['label'].unique()]
        test_labels = test_labels.loc[test_node_ids, :]
        test_timeseries = test_timeseries.loc[test_node_ids, :]

    return timeseries, labels, test_timeseries, test_labels

def read_csv_wrapper(root_dir, pattern="F*.csv", **kwargs):
    data = {'train': [], 'test': [], 'labels_train': [], 'labels_test': []}

    for file_path in glob(os.path.join(root_dir, pattern)):
        file_name = os.path.basename(file_path)
        label = int(file_name.split('_')[0][1])  # Extracting the label (F1, F2, ..., F9) as integer
        data_type = 'train' if 'train' in file_name else 'test'

        df = pd.read_csv(file_path)

        df.index = pd.MultiIndex.from_product([['node_{}'.format(label)], df.index], names=['node_id', 'timestamp'])
        data[data_type].append(df)

    train_df = pd.concat(data['train'])
    test_df = pd.concat(data['test'])

    train_labels_df = pd.read_csv('./data/CSTR1/labels.csv')
    train_labels_df['label'].map(str)
    train_labels_df = train_labels_df.set_index('node_id')
    train_labels_df.index = pd.MultiIndex.from_product([train_labels_df.index, [0]], names=['node_id', 'timestamp'])

    test_labels_df = pd.read_csv('./data/CSTR1/labels.csv')
    test_labels_df['label'].map(str)
    test_labels_df = test_labels_df.set_index('node_id')
    test_labels_df.index = pd.MultiIndex.from_product([test_labels_df.index, [0]], names=['node_id', 'timestamp'])


    timeseries, labels = process_data(train_df, train_labels_df, **kwargs)
    test_timeseries, test_labels = process_data(test_df, test_labels_df, **kwargs)
    
    return timeseries, labels, test_timeseries, test_labels

def get_dataset_csv(set_name, exclude_columns, **kwargs):
    if set_name == 'cstr':
        pattern = kwargs.get('pattern', "F*.csv")
        root_dir = kwargs.get('rootdir', './data/CSTR1')
        # train_data, train_labels, test_data, test_labels = read_csv_wrapper(root_dir, pattern, **kwargs)
        train_data, train_labels, test_data, test_labels = windowize_csv(root_dir, pattern, **kwargs)
        return train_data, train_labels, test_data, test_labels
    else:
        pattern = kwargs.get('pattern', "test_*.csv")
        root_dir = kwargs.get('rootdir', './data/' + set_name)
        # train_data, train_labels, test_data, test_labels = read_csv_wrapper(root_dir, pattern, **kwargs)
        train_data, train_labels, test_data, test_labels = windowize_and_split(root_dir, pattern, exclude_columns, **kwargs)
        return train_data, train_labels, test_data, test_labels

def add_noise(ts, noise_scale):

    if noise_scale == 0:
        return ts
    
    noisy_df = ts.copy()
    for column in ts.columns:
        if column not in ['Timestamp', 'label']:
            # Add noise to non-excluded columns
            std_dev = ts[column].mean()
            noise = np.random.normal(0, std_dev*noise_scale, ts.shape[0])  # Example of Gaussian noise
            noisy_df[column] += noise
    return noisy_df

def windowize_and_split(root_dir, pattern="test_*.csv",  exclude_columns = [], window_size=5, noise_scale=0, train_split_size=0.8, test_split_size=0.2, use_classes=None,
                  **kwargs):
    
    ts_windowized = []

    for file_path in glob(os.path.join(root_dir, pattern)):
        file_name = os.path.basename(file_path)
        file_number = int(re.split('\_|\.',file_name)[1])  # Extracting the label (F1, F2, ..., F9) as integer

        ts = pd.read_csv(file_path)

        ts.drop(columns=exclude_columns, inplace=True)
        ts = add_noise(ts, noise_scale)
        
        windows_train = []
        ts_window_temp = []

        for i in range(window_size, len(ts)):
            windows_train.append(ts.iloc[i - window_size:i].values.flatten())
            ts_window_temp = ts.iloc[i - window_size:i].copy()
            new_node_id = 'node_{}_{}'.format(file_number,i)
            ts_window_temp.index = pd.MultiIndex.from_product([[new_node_id], ts_window_temp['Timestamp']], names=['node_id', 'timestamp'])
            ts_windowized.append(ts_window_temp)

    ts_windowized_df = pd.concat(ts_windowized)


    ts_windowized_df_labels = ts_windowized_df.groupby(level='node_id').agg({'label': 'last'})
    ts_windowized_df_labels.index = pd.MultiIndex.from_product([ts_windowized_df_labels.index, [0]], names=['node_id', 'timestamp'])

    ts_windowized_df.drop(columns=['Timestamp', 'label'], inplace=True)


    train_nodeids, test_nodeids = train_test_split(
                ts_windowized_df.index.get_level_values('node_id').unique(), train_size=train_split_size, test_size=test_split_size, random_state=0)
    test_timeseries = ts_windowized_df.loc[test_nodeids, :, :]
    test_labels = ts_windowized_df_labels.loc[test_nodeids, :, :]
    timeseries = ts_windowized_df.loc[train_nodeids, :, :]
    labels = ts_windowized_df_labels.loc[train_nodeids, :, :]


    train_ts, train_labels = process_data(timeseries, labels, use_classes)
    test_ts, test_labels = process_data(test_timeseries, test_labels, use_classes)

    return train_ts, train_labels, test_ts, test_labels



def windowize_csv(root_dir, pattern="F*.csv", window=5, noise_scale=0.1, exclude_columns = ['Timestamp', 'label'], use_classes=None,
                  **kwargs):
    window_size = window
    data = {'train': [], 'test': [], 'labels_train': [], 'labels_test': []}

    for file_path in glob(os.path.join(root_dir, pattern)):
        file_name = os.path.basename(file_path)
        file_number = int(file_name.split('_')[0][1])  # Extracting the label (F1, F2, ..., F9) as integer
        data_type = 'train' if 'train' in file_name else 'test'

        ts = pd.read_csv(file_path)

        # Add noise to DataFrame
        noisy_df = ts.copy()
        for column in ts.columns:
            if column not in exclude_columns:
                # Add noise to non-excluded columns
                std_dev = ts[column].mean()
                noise = np.random.normal(0, std_dev*noise_scale, ts.shape[0])  # Example of Gaussian noise
                noisy_df[column] += noise

        ts = noisy_df
        windows_train = []
        ts_window = []

        for i in range(window_size, len(ts)):
            windows_train.append(ts.iloc[i - window_size:i].values.flatten())
            ts_window = ts.iloc[i - window_size:i].copy()
            new_node_id = 'node_{}_{}'.format(file_number,i)
            ts_window.index = pd.MultiIndex.from_product([[new_node_id], ts_window['Timestamp']], names=['node_id', 'timestamp'])
            data[data_type].append(ts_window)

    train_df = pd.concat(data['train'])
    test_df = pd.concat(data['test'])
    train_labels = train_df.groupby(level='node_id').agg({'label': 'last'})
    test_labels = test_df.groupby(level='node_id').agg({'label': 'last'})
    train_labels.index = pd.MultiIndex.from_product([train_labels.index, [0]], names=['node_id', 'timestamp'])
    test_labels.index = pd.MultiIndex.from_product([test_labels.index, [0]], names=['node_id', 'timestamp'])

    train_ts = train_df.drop(columns=exclude_columns, inplace=False)
    test_ts = test_df.drop(columns=exclude_columns, inplace=False)

    train_ts, train_labels = process_data(train_ts, train_labels, use_classes)
    test_ts, test_labels = process_data(test_ts, test_labels, use_classes)
    return train_ts, train_labels, test_ts, test_labels
