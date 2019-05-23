import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import handshape_datasets as hd
from PIL import Image


class DataLoader(object):
    def __init__(self, data, n_classes, n_way, n_support, n_query):
        self.data = data
        self.n_way = n_way
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query

    def get_next_episode(self):
        n_examples = 20
        support = np.zeros([self.n_way, self.n_support, 28, 28, 3], dtype=np.float32)
        query = np.zeros([self.n_way, self.n_query, 28, 28, 3], dtype=np.float32)
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
    DATASET_OUTPUT_PATH = "/develop/data/lsa16/data"

    loadedData = hd.load(DATASET_NAME, DATASET_OUTPUT_PATH)

    features = loadedData[0]
    classes = loadedData[1]['y']

    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        classes,
                                                        test_size=0.33,
                                                        random_state=42)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    split_dir = os.path.join(data_dir, 'splits', config['data.split'])
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

        # Convert original data to format [n_classes, n_img, w, h, c]
        data = np.zeros((len(classes), len(features[0]), 28, 28, 3))

        for i_class in range(len(classes)):
            for i_img in range(len(features[i_class])):
                # data[i_class, i_img, :, :, :] = ????????

        data /= 255.0
        data[:, :, :, 0] = (data[:, :, :, 0] - 0.485) / 0.229
        data[:, :, :, 1] = (data[:, :, :, 1] - 0.456) / 0.224
        data[:, :, :, 2] = (data[:, :, :, 2] - 0.406) / 0.225

        data_loader = DataLoader(data,
                                 n_classes=len(classes),
                                 n_way=n_way,
                                 n_support=n_support,
                                 n_query=n_query)

        ret[split] = data_loader

    return ret
