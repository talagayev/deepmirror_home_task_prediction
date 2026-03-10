# deepmirror home away task

The task for deepmirror consisted in the building a workflow that would predict human liver microsome intrinsic clearance within. For this use of external public data was advised.

One of the main benchmarks would consist in predicting the test set presented in the [OpenADMET ExpansionRx ADMET Challenge](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge)

A full detailed report can be found at: [Report.md](Report.md)

## Installation

To start with the installation of the package you first need to download it. for this do the following:

    git clone https://github.com/talagayev/deepmirror_home_task_prediction.git
    cd deepmirror_home_task_prediction

To install the package you need to create a `conda` environment with all of the used packages. This can be done via:

    conda env create -f environment.yml

This creates the environment `deepmirror_ml_task`. To install the entry points you do the following:

    conda activate deepmirror_ml_task
    pip install -e .

## Command line interface

Currently the CLI is done through entrypoints. The following are available:

    deepmirror-cli preprocess-smiles
    deepmirror-cli deduplicate
    deepmirror-cli applicability-domain
    deepmirror-cli model-crossvalidation
    deepmirror-cli predict-model

The `deepmirror-cli preprocess-smiles` is responsible for preprocessing the `SMILES`. The command `deepmirror-cli deduplicate` deduplicates the duplicate `SMILES` after preprocession. The command `deepmirror-cli applicability-domain` is required to see if the test molecules are in the applicability domain of your model features. The command `deepmirror-cli model-crossvalidation` does cross-validation as well as training a final model with the best parameteres from the CV. The command `deepmirror-cli predict-model` is used for predictions.

All the commands can be done via command-line. The inputs can be parsed via CLi or via config files. The config file examples are located at the `example_config_files` folder.


## Trained models

The trained models that were used for the predictions are located in the folder `trained_models`. To use them in the prediction use the command

    deepmirror-cli predict-model --config prediction_config.yml

With the config requiring you to select the correct model refit folder and also the columns you want to predict.
