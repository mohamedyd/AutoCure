#####################################################################################
# Benchmark: A script to test the impact of error rate on the performance of AutoCure
# Authors: Mohamed Abdelaal
# Date: April 2022
# Software AG
# All Rights Reserved
#####################################################################################

import os
import sys
import argparse
import numpy as np
import pandas as pd
import shutil
from aug2clean.ensemble import aug2clean
from aug2clean.setup.utils import create_detections_path, create_target_path
from aug2clean.model.train import train_model
from aug2clean.setup.detectors.detect_method import DetectMethod, EXP_PATH, DATA_PATH
from aug2clean.setup.repairs.repair import RepairMethod
from aug2clean.model.utils import ExperimentName, ExperimentType, create_results_path

from aug2clean.setup.create_dirty import ErrorType, inject_errors
from aug2clean.dataset.dataset import Dataset
from baseline.baseline import train_e2e_baseline


if __name__ == '__main__':
    # Initialize an argument parser
    parser = argparse.ArgumentParser()
    # Add the parser's options
    parser.add_argument('--dataset_name', nargs='+', default=None, required=True)
    parser.add_argument('--nb_iterations', type=int, default=10)
    parser.add_argument('--experiment_type', type=ExperimentType, choices=list(ExperimentType), default=None)
    parser.add_argument('--detection_method',  nargs='+', type=DetectMethod, choices=list(DetectMethod), default=None)
    parser.add_argument('--repair_method',  nargs='+', type=RepairMethod, choices=list(RepairMethod), default=None)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--nb_generated_samples', type=int, default=6000)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--tune_hyperparams', action='store_true')

    args = parser.parse_args()

    # Retrieve the input arguments
    dataset_names = args.dataset_name
    nb_iterations = args.nb_iterations
    epochs = args.epochs
    verbose = args.verbose
    nb_generated_samples = args.nb_generated_samples
    tune_hyperparams = args.tune_hyperparams
    exper_type = args.experiment_type
    detectors_list = args.detection_method
    repair_list = args.repair_method

    # Create a list of all available detectors and repair methods
    available_detectors = list(DetectMethod)
    available_repair = list(RepairMethod)

    # Use all available base detectors if no specific detectors are selected
    involved_detectors = available_detectors if not detectors_list else detectors_list
    involved_repairs = available_repair if not repair_list else repair_list


    print("============================================")
    print("Dataset name(s): ", dataset_names)
    print("Experiment type: ", exper_type)
    print("Number of iterations: ", nb_iterations)
    print('Involved detection methods: ', involved_detectors)
    print("involved Repair methods: ",involved_repairs)
    print(ExperimentType.AUG2CLEAN.__str__())
    print(ExperimentType.E2E_PIPELINE.__str__())
    print("============================================")


    for dataset_name in dataset_names:

        # Create a data object
        data_obj = Dataset(dataset_name)
        # Get the labels to avoid injecting errors into them
        muted_columns = data_obj.cfg.labels

        dataset_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name))
        # Retrieve the dirty and clean data
        clean_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'clean.csv'))
        dirty_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'dirty.csv'))

        # Load the dirty data and its ground truth
        clean_df = pd.read_csv(clean_path, header="infer", encoding="utf-8", low_memory=False)

        # Define the error types
        err_method = [ErrorType.OUTLIERS, [ErrorType.EXPLICIT_MV.func]]

        # Create a temp path to store the generated dirty data sets
        _, temp_dirty_dir = create_results_path(dataset_name=dataset_name,
                                                 experiment_name=ExperimentName.ROBUSTNESS.__str__(),
                                                 filename='dirty.csv',
                                                 return_results_dir=True)

        temp_dirty_path = os.path.abspath(os.path.join(temp_dirty_dir, "dirty.csv"))

        for error_rate in np.arange(0.1, 0.9, 0.1):

            # Generate a new dirty data set
            inject_errors(clean_df=clean_df, error_types=err_method, params=[error_rate, 5],
                          muted_columns=[muted_columns], data_path=temp_dirty_dir)

            for _ in range(nb_iterations):

                if exper_type == ExperimentType.AUG2CLEAN:

                    # Create a path to the detections.csv file
                    detections_path = create_detections_path(EXP_PATH, dataset_name, DetectMethod.AUG2CLEAN.__str__())

                    generated_data, encoded_dirty_df = aug2clean(temp_dirty_path,
                                                                 clean_path,
                                                                 detections_path,
                                                                 dataset_path,
                                                                 dataset_name,
                                                                 detectors_list=[],
                                                                 threshold=0.2,
                                                                 prevent_data_exclusion=True,
                                                                 verbose=verbose)

                    # Merge the different parts of the data set
                    # final_df = dirty_df.merge(generated_data, left_index=True, right_index=True)
                    final_df = pd.concat([encoded_dirty_df, generated_data], ignore_index=True)

                    # Train a model
                    train_model(final_df,
                                dataset_name,
                                tune_params=tune_hyperparams,
                                exp_name=ExperimentName.ROBUSTNESS.__str__(),
                                exp_type=ExperimentType.AUG2CLEAN.__str__(),
                                nb_generated_samples=nb_generated_samples,
                                verbose=verbose,
                                error_rate=error_rate,
                                epochs=epochs)

                elif exper_type == ExperimentType.E2E_PIPELINE:

                    for repair_method in involved_repairs:

                        for detect_method in involved_detectors:

                            target_path = create_target_path(EXP_PATH, dataset_name, detect_method.__str__(),
                                                             repair_method.__str__())
                            # Create a path to the detections.csv file
                            detections_path = create_detections_path(EXP_PATH, dataset_name, detect_method.__str__())

                            try:
                                train_e2e_baseline(clean_path=clean_path,
                                                   dirty_path=temp_dirty_path,
                                                   detections_path=detections_path,
                                                   target_path=target_path,
                                                   detect_method=detect_method,
                                                   repair_method=repair_method,
                                                   dataset_name=dataset_name,
                                                   dataset_path=dataset_path,
                                                   verbose=verbose,
                                                   tune_params=tune_hyperparams,
                                                   epochs=epochs,
                                                   exp_name=ExperimentName.ROBUSTNESS.__str__(),
                                                   error_rate=error_rate)
                                
                            except:
                                print("[ERROR] Failed to run the pipeline")
                                break
                else:
                    raise NotImplementedError

            # Remove the detections and repair
            shutil.rmtree(os.path.abspath(os.path.join(detections_path, os.pardir, os.pardir)))
            if exper_type == ExperimentType.E2E_PIPELINE:
                shutil.rmtree(os.path.abspath(os.path.join(target_path, os.pardir, os.pardir, os.pardir)))
