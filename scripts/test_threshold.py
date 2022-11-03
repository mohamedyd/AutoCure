####################################################
# Benchmark: A script to test the detection performance of min-k versus the value of k
# Authors: Mohamed Abdelaal
# Date: April 2022
# Software AG
# All Rights Reserved
###################################################

import os
import argparse
import numpy as np
from aug2clean.setup.detectors.detect_method import DetectMethod, DATA_PATH, EXP_PATH
from aug2clean.setup.utils import create_target_path, create_detections_path
from aug2clean.setup.detectors.detect import detect


if __name__ == '__main__':
    # Initialize an argument parser
    parser = argparse.ArgumentParser()
    # Add the parser's options
    parser.add_argument('--dataset_name',  nargs='+', default=None, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Retrieve the input arguments
    dataset_names = args.dataset_name
    verbose = args.verbose

    detect_method = DetectMethod.MIN_K

    for dataset_name in dataset_names:

        # Prepare the paths
        dataset_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name))
        clean_path = os.path.join(dataset_path, 'clean.csv')
        dirty_path = os.path.join(dataset_path, 'dirty.csv')
        detections_path = create_detections_path(EXP_PATH, dataset_name, detect_method.__str__())

        for threshold in np.arange(0.1, 1, 0.1):

            print("[INFO] Detecting errors using min-k with threshold {}".format(threshold))
            # Remove the detections CSV file, if it already exists
            if os.path.exists(detections_path):
                os.remove(detections_path)

            # Execute the min-k detector
            detect(clean_path,
                   dirty_path,
                   detections_path,
                   dataset_path,
                   dataset_name,
                   detect_method=detect_method,
                   mink_threshold=threshold)
