import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

class DataLoader(object):
    def __init__(self, data, n_classes, n_way, n_support, n_query):
        self.data = data
        self.n_way = n_way
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query

    def get_next_episode(self):
        n_examples = 20
        support = np.zeros([self.n_way, self.n_support, 28, 28, 1], dtype=np.float32)
        query = np.zeros([self.n_way, self.n_query, 28, 28, 1], dtype=np.float32)
        classes_ep = np.random.permutation(self.n_classes)[:self.n_way]

        for i, i_class in enumerate(classes_ep):
            selected = np.random.permutation(n_examples)[:self.n_support + self.n_query]
            support[i] = self.data[i_class, selected[:self.n_support]]
            query[i] = self.data[i_class, selected[self.n_support:]]

        return support, query

def load_cifar10(data_dir, config, splits):
    """
    Load cifar10 dataset.

    Args:
        data_dir (str): path of the directory with 'splits', 'data' subdirs.
        config (dict): general dict with program settings.
        splits (list): list of strings 'train'|'val'|'test'

    Returns (dict): dictionary with keys as splits and values as tf.Dataset

    """

    num_classes = 10

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    print(y_train, 'y_train')
    print(y_test, 'y_test')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print(x_train, 'x_train')

    ret = {}
    """
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

        # Get all class names
        class_names = []
        with open(os.path.join(split_dir, f"{split}.txt"), 'r') as f:
            for class_name in f.readlines():
                class_names.append(class_name.rstrip('\n'))

        # Get class names, images paths and rotation angles per each class
        class_paths, rotates = class_names_to_paths(data_dir,
                                                    class_names)
        classes, img_paths, rotates = get_class_images_paths(
            class_paths,
            rotates)

        data = np.zeros([len(classes), len(img_paths[0]), 28, 28, 1])
        for i_class in range(len(classes)):
            for i_img in range(len(img_paths[i_class])):
                data[i_class, i_img, :, :,:] = load_and_preprocess_image(
                            img_paths[i_class][i_img], rotates[i_class])

        data_loader = DataLoader(data,
                                 n_classes=,
                                 n_way=n_way,
                                 n_support=n_support,
                                 n_query=n_query)

        ret[split] = data_loader
    """
    return ret