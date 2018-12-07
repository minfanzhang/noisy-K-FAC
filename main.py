from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import os

from misc.utils import get_logger, get_args, makedirs
from misc.config import process_config
from misc.data_loader import load_pytorch
from core.model import Model
from core.train import Trainer


_INPUT_DIM = {
    'fmnist': [784],
    'mnist': [784],
    'cifar10': [32, 32, 3],
    'cifar100': [32, 32, 3]
}


def main():
    tf.set_random_seed(1231)
    np.random.seed(1231)

    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("Add a config file using \'--config file_name.json\'")
        exit(1)

    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    # set logger
    path = os.path.dirname(os.path.abspath(__file__))
    path1 = os.path.join(path, 'core/model.py')
    path2 = os.path.join(path, 'core/train.py')
    logger = get_logger('log', logpath=config.summary_dir+'/',
                        filepath=os.path.abspath(__file__), package_files=[path1, path2])

    logger.info(config)

    # load data
    train_loader, test_loader = load_pytorch(config)

    # define computational graph
    sess = tf.Session()

    model_ = Model(config, _INPUT_DIM[config.dataset], len(train_loader.dataset))
    trainer = Trainer(sess, model_, train_loader, test_loader, config, logger)

    trainer.train()

def gradient_check():
    tf.set_random_seed(1231)
    np.random.seed(1231)

    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("Add a config file using \'--config file_name.json\'")
        exit(1)

    # set logger
    path = os.path.dirname(os.path.abspath(__file__))
    path1 = os.path.join(path, 'core/model.py')
    path2 = os.path.join(path, 'core/train.py')
    logger = get_logger('log', logpath=config.summary_dir+'/',
                        filepath=os.path.abspath(__file__), package_files=[path1, path2])

    logger.info(config)

    batch_sizes = [1,4,16,64,128,256,512]

    precon = False
    for bs in batch_sizes:
        start_time = time.time()
        print("processing batch size {}".format(bs))

        # load data
        train_loader, test_loader = load_pytorch(config)

        # define computational graph
        sess = tf.Session()

        model_ = Model(config, _INPUT_DIM[config.dataset], len(train_loader.dataset))
        trainer = Trainer(sess, model_, train_loader, test_loader, config, logger)

        trainer.grad_check(sess, bs, precon)
        print('batch size {} takes {} secs to finish'.format(
            bs, time.time()-start_time))
        tf.reset_default_graph()

    precon = True
    for bs in batch_sizes:
        start_time = time.time()
        print("processing batch size {}".format(bs))

        # load data
        train_loader, test_loader = load_pytorch(config)

        # define computational graph
        sess = tf.Session()

        model_ = Model(config, _INPUT_DIM[config.dataset], len(train_loader.dataset))
        trainer = Trainer(sess, model_, train_loader, test_loader, config, logger)

        trainer.grad_check(sess, bs, precon)
        print('batch size {} takes {} secs to finish'.format(
            bs, time.time()-start_time))
        tf.reset_default_graph()

if __name__ == "__main__":
    gradient_check()
    #main()
