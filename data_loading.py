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

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_dataset(set_name, binary=False, **kwargs):
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


def windowize(timeseries, labels, window=45, trim=60, skip=15, test=False, **kwargs):
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
    if kwargs.get('windowize', True):
        return windowize(timeseries, labels, **kwargs)
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
