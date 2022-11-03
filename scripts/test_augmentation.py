####################################################
# Benchmark: A script to test the RL-based data cleaning method
# Authors: Mohamed Abdelaal
# Date: April 2022
# Software AG
# All Rights Reserved
###################################################

import os
import sys
import numpy as np
import argparse
import pandas as pd
from aug2clean.ensemble import aug2clean
from aug2clean.setup.utils import create_detections_path
from aug2clean.model.train import train_model
from aug2clean.setup.detectors.detect_method import DetectMethod, EXP_PATH, DATA_PATH
from aug2clean.model.utils import ExperimentName, ExperimentType


if __name__ == '__main__':
    # Initialize an argument parser
    parser = argparse.ArgumentParser()
    # Add the parser's options
    parser.add_argument('--dataset_name', nargs='+', default=None, required=True)
    parser.add_argument('--nb_iterations', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--tune_hyperparams', action='store_true')

    args = parser.parse_args()

    # Retrieve the input arguments
    dataset_names = args.dataset_name
    nb_iterations = args.nb_iterations
    epochs = args.epochs
    verbose = args.verbose
    tune_hyperparams = args.tune_hyperparams

    method = DetectMethod.AUG2CLEAN

    for dataset_name in dataset_names:

        # Prepare data paths
        dataset_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name))
        # Retrieve the dirty and clean data
        clean_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'clean.csv'))
        dirty_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'dirty.csv'))

        # Load the dirty data and its ground truth
        dirty_df = pd.read_csv(dirty_path, header="infer", encoding="utf-8", low_memory=False)
        # Get the number of samples in the dirty data set
        nb_samples_dirty = dirty_df.shape[0]
        # Create a list of values corresponding to increasing percentages of the data size
        nb_samples_list = [round(nb_samples_dirty * x) for x in np.arange(0.1, 2.1, 0.1)]

        # Create a path to the detections.csv file
        detections_path = create_detections_path(EXP_PATH, dataset_name, method.__str__())

        for nb_samples in nb_samples_list:

            for _ in range(nb_iterations):

                generated_data, encoded_dirty_df = aug2clean(dirty_path,
                                                             clean_path,
                                                             detections_path,
                                                             dataset_path,
                                                             dataset_name,
                                                             nb_samples=nb_samples,
                                                             detectors_list=[],
                                                             threshold=0.2,
                                                             verbose=True)

                # Merge the different parts of the data set
                # final_df = dirty_df.merge(generated_data, left_index=True, right_index=True)
                final_df = pd.concat([encoded_dirty_df, generated_data], ignore_index=True)

                # Train a model

                train_model(final_df,
                            dataset_name,
                            tune_params=tune_hyperparams,
                            exp_name=ExperimentName.AUGMENTATION.__str__(),
                            exp_type=ExperimentType.AUG2CLEAN.__str__(),
                            nb_generated_samples=nb_samples,
                            verbose=verbose,
                            epochs=epochs)
