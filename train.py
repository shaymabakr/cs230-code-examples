"""Train the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.input_data import input_fn
from model.utils import Params
from model.utils import set_logger
from model.model import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/SIGNS',
                    help="Directory containing the dataset")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # -----------------------------------------------
    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, "train_signs")

    # Get the filenames and shuffle them
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f) for f in filenames]
    train_filenames = filenames

    # Specify the train and eval datasets size
    params.train_size = len(train_filenames)

    # Create the two iterators over the two datasets
    train_inputs = input_fn(train_filenames, params)

    # -----------------------------------------------
    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('train', train_inputs, params)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))

    with tf.Session() as sess:
        # Initialize model variables
        sess.run(model_spec['variable_init_op'])

        for epoch in range(params.num_epochs):
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
            # compute number of batches in one epoch (one full pass over the training set)
            num_steps = (params.train_size + 1) // params.batch_size

            for step in range(num_steps):
                _, loss_val, acc_val = sess.run([model_spec['train_op'], model_spec['loss'],
                                        model_spec['accuracy']])
                logging.info("Epoch {}, step {}, loss: {}, accuracy: {}"
                             .format(epoch, step, loss_val, acc_val))
