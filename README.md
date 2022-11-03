AutoCure: Automated Tabular Data Curation for ML Pipelines
=========================================================
The repository comprises the source code of the AutoCure method which implements an adaptive ensemble-based error 
detection method followed by a data augmentation approach to automatically curate tabular data for predictive tasks.  

## Setup

clone with submodules

```
https://audio.digitalbusinessplatform.de/gitlab/kompaki/norepair4ml/augmentation-data-cleaning.git
```

Install requirements

```
python3 -m venv venv 
source venv/bin/activate
pip3 install --upgrade setuptools
pip3 install --upgrade pip
pip3 install -e .
```

Install error detection and repair methods

##### RAHA and BARAN

To install these methods, you can do so in two different ways:

Option 1: through pip3
```
pip3 install raha
```
Option 2: through the setup.py script which exists in the raha main directory
```
python3 setup.py install
```

##### Katara

For this method, we do not need to install packages, but we need to download the knowledge base:
Download the knowledge base ([link](https://bit.ly/3hPkpWX)) and unzip the file. The files of the knowledge base should 
be placed in the following path.
```
cd detectors/katara/knowedge-base
``` 

##### FAHES

To install FAHES, navigate to the src directory and run make to compile the source code
``` shell script
cd FAHES/src/
``` 

``` shell script
make clean && make
``` 

##### HoloClean
To install HoloClean, read te installation part of its README file.

## Usage

The executable python scripts are placed in the scripts folder. 

### Run the threshold experiment
This command runs an experiment to examine the impact of changing the voting threshold on the detection recall and precision. In this experiment, we use the traditional Min-K detection method to find errors in dirty data sets. The results are stored in the directory `experiments/evaluation/data/test_threshold/{data set name}`. Several data sets van be employed at the same time, while the option `verbose` can be used to print intermediate results and to show the execution progress.  
```shell script
 python3 scripts/test_threshold.py 
        --dataset_name adult housing 
        --verbose
```

### Run the pipeline with the clean data set
This command trains a neural network on the clean versions of the data sets, i.e., the ground truth. The option `tune_hyperparams` activates the Optuna module to optimize the hyperparameters. While, the option `epochs` defines the number of epochs while training the neural network. To estimate the variance and average values, the option `nb_iterations` is used to define how many times the experiment will be repeated. The results are stored in the directory `experiments/evaluation/data/modeling/{data set name}`. For each run, a figure of the learning curves is stored in the directory `experiments/evaluation/plots/learning_curve/{data set name}`. It is important to highlight that this experiment can be executed, even in the lack of the ground truth version of the data set.
```shell script
 python3 scripts/test_clean.py 
        --dataset_name adult housing 
        --nb_iterations 10
        --epochs 500
        --tune_hyperparams
        --verbose
```

### Run AutoCue 
This command runs a pipeline composed of the AutoCure method followed by a neural network with regression, binary classification, or multi-class classification, according to the data set. The option `nb_generated_samples` defines the amount of clean data to be augmented by AutoCure. The results are stored in the directory `experiments/evaluation/data/modeling/{data set name}`.
```shell script
 python3 scripts/test_augclean.py 
        --dataset_name adult housing 
        --nb_iterations 10
        --nb_generated_samples 2000
        --epochs 500
        --tune_hyperparams
        --verbose
```

### Run the baseline methods
This command runs an end-to-end pipeline consists of an error detection method, a repair method, and a neural network. The list of available error detection methods include: `IF`, `SD`, `IQR`, `mvdetector`, `raha`, `ed2`, `dBoost`, `min_k`, `holoclean`, `fahes`, `katara`, `nadeef`. While, the list of repair methods involve `baran`, `cleanwithGroundTruth`, `standardImputer`, `mlImputer`, `activeClean`. The results are stored in the directory `experiments/evaluation/data/modeling/{data set name}`.
```shell script
 python3 scripts/test_e2e.py 
        --dataset_name adult
        --nb_iterations 10
        --detection_method raha ed2 katara 
        --repair_method mlImputer baran
        --epochs 500
        --tune_hyperparams
        --verbose
```

### Run the augmentation experiment
This command runs the AutoCure experiment with gradually increasing the amount of augmented clean data. The results are stored in the directory `experiments/evaluation/data/augmentation/{data set name}`.
```shell script
 python3 scripts/test_augmentation.py 
        --dataset_name adult housing 
        --nb_iterations 10
        --epochs 500
        --tune_hyperparams
        --verbose
```

### Run the robustness experiment
This command runs the AutoCure method or the baseline methods while increasing the amount errors in the dirty data set. The option `experiment_type` is used to select between running AutoCure or the baseline methods. Specifically, it has two valid values, namely `e2e` and `aug2clean`. The results are stored in the directory `experiments/evaluation/data/robustness/{data set name}`.
```shell script
 python3 scripts/test_error_rates.py 
        --dataset_name adult
        --nb_iterations 10
        --experiment_type e2e
        --detection_method raha ed2 katara 
        --repair_method mlImputer baran
        --nb_generated_samples 2000
        --epochs 500
        --tune_hyperparams
        --verbose
```

### Adding a dataset
To add a new dataset, follow the steps below:  
* Add the CSV files of the dirty and the ground truth in the directory `experiments/data/{data set name}/`. Rename the CSV file of the ground truth to `clean.csv` and the CSV file of the dirty version to `dirty.csv`.
* Create a YML file to store metadata about the newly-added data set. The YML file has the following fields which have to be defined. The field `name` defines the name of the data set, while the field `ml_task` is used to define the machine learning task associated with this data set. Possible ML tasks include `binary_classification`, `multiclass_classification`, and `regression`.  
```
name: adult
ml_task: binary_classification
labels: income
fd_constraints:
```

* To run the rule-based baseline methods, the following cleaning signals have to be provided for the new dataset:
	- Functional dependency (FD rules) and patterns to run NADEEF
	- Denial constraints (DC rules) for HoloClean

## Experiments

In this section, we list all experiments planned in this project.

- [x] test the performance of autoCure in terms of
    - [x] execution time
    - [x] performance  of downstream ML models
- [x] compare AutoCure with a set of baselines
- [x] test robustness against different error types
- [x] test the amount of augmented data

## Todos

- [x] implement an ensemble-based error detection
    * [x] implement a dataset class for loading, preparing, storing data, etc.
    * [x] implement baseline to show how selecting data repair requires expertise
        - [x] implement an ensemble detector which will be used for AutoCure and for the baselines
        - [x] implement repair methods
- [x] implement the adaptive data sampler 
- [x] implement the data augmentation method

## Follow-up Ideas

- integrate a data valuation method
- integrate an automatic tool for the generation of functional dependency rules

