####################################################
# Benchmark: A script to test the RL-based data cleaning method
# Authors: Mohamed Abdelaal
# Date: April 2022
# Software AG
# All Rights Reserved
###################################################

import os
import sys
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
    parser.add_argument('--dataset_name',  nargs='+', default=None, required=True)
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

    for dataset_name in dataset_names:

        dataset_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name))
        # Retrieve the dirty and clean data
        clean_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'clean.csv'))
        dirty_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'dirty.csv'))

        # Load the dirty data and its ground truth
        data_df = pd.read_csv(clean_path, header="infer", encoding="utf-8", low_memory=False)

        # Train a model
        experiment_name = ExperimentName.MODELING.__str__()
        experiment_type = ExperimentType.GROUND_TRUTH.__str__()

        for _ in range(nb_iterations):

            train_model(data_df,
                        dataset_name,
                        tune_params=tune_hyperparams,
                        exp_name=experiment_name,
                        exp_type=experiment_type,
                        verbose=verbose,
                        epochs=epochs)



