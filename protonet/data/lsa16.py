import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import handshape_datasets as hd


class DataLoader(object):
    def __init__(self, data, n_classes, n_way, n_support, n_query):
        self.data = data
        self.n_way = n_way
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query

    def get_next_episode(self):
        n_examples = 20
        support = np.zeros([self.n_way, self.n_support, 32, 32, 3], dtype=np.float32)
        query = np.zeros([self.n_way, self.n_query, 32, 32, 3], dtype=np.float32)
        classes_ep = np.random.permutation(self.n_classes)[:self.n_way]

        for i, i_class in enumerate(classes_ep):
            selected = np.random.permutation(n_examples)[:self.n_support + self.n_query]
            support[i] = self.data[i_class, selected[:self.n_support]]
            query[i] = self.data[i_class, selected[self.n_support:]]

        return support, query

def load_lsa16(data_dir, config, splits):
    """
    Load lsa16 dataset.

    Args:
        data_dir (str): path of the directory with 'splits', 'data' subdirs.
        config (dict): general dict with program settings.
        splits (list): list of strings 'train'|'val'|'test'

    Returns (dict): dictionary with keys as splits and values as tf.Dataset

    """

    DATASET_NAME = "lsa16"
    DATASET_PATH = "/develop/data/lsa16/data"

    loadedData = hd.load(DATASET_NAME, DATASET_PATH)

    features = loadedData[0]
    classes = loadedData[1]['y']
    uniqueClasses, imgsPerClass = np.unique(classes, return_counts=True)

    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        classes,
                                                        test_size=0.5,
                                                        random_state=0,
                                                        stratify=classes)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    trainClasses, amountPerTrain = np.unique(y_train, return_counts=True)
    testClasses, amountPerTest = np.unique(y_test, return_counts=True)

    ret = {}
    for split in splits:
        # n_way (number of classes per episode)
        if split in ['val', 'test']:
            n_way = config['data.test_way']
        else:
            n_way = config['data.train_way']

        # n_support (number of support examples per class)
        if split in ['val', 'test']:
            n_support = config['data.test_support']
        else:
            n_support = config['data.train_support']

        # n_query (number of query examples per class)
        if split in ['val', 'test']:
            n_query = config['data.test_query']
        else:
            n_query = config['data.train_query']

        if split in ['val', 'test']:
            y = y_test
            x = x_test
        else:
            y = y_train
            x = x_train

        amountPerClass = amountPerTest if split in ['val', 'test'] else amountPerTrain

        i = np.argsort(y)
        x = x[i, :, :, :]
        data = np.reshape(x, (len(uniqueClasses), amountPerClass[0], 32, 32, 3))

        data_loader = DataLoader(data,
                                 n_classes=len(uniqueClasses),
                                 n_way=n_way,
                                 n_support=n_support,
                                 n_query=n_query)

        ret[split] = data_loader

    return ret
