# deepmirror home away task

The task for deepmirror consisted in the building a workflow that would predict human liver microsome intrinsic clearance within. For this use of external public data was advised.

One of the main benchmarks would consist in predicting the test set presented in the [OpenADMET ExpansionRx ADMET Challenge](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge)


Here are the results of the trained models that I was able to train during the timeline. Sadly due to hardware restrictions I was only able to focus on `Chemeleon` and was only able to run for all of the training datasets `Autogluon` with the `Expansion` training data set being the only one that I was able to run with `Chemprop` due to it having less data points compared to the remainder of the data sets

| Name | Endpoint | mean_MAE | mean_RAE | mean_R2 | mean_Spearman R | mean_Kendall's Tau | std_MAE | std_RAE | std_R2 | std_Spearman R | std_Kendall's Tau |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Expansion_Chemprop_Chemeleon | HLM | 0.287312 | 0.783742 | 0.337835 | 0.626342 | 0.445753 | 0.008625 | 0.023417 | 0.035346 | 0.023421 | 0.018854 |
| Expansion_Autogluon_Chemeleon | HLM | 0.305677 | 0.833865 | 0.287409 | 0.601983 | 0.422438 | 0.008584 | 0.024212 | 0.037064 | 0.024118 | 0.018819 |
| No_scale_Autogluon_Chemeleon | HLM | 0.312914 | 0.853602 | 0.228064 | 0.586194 | 0.417628 | 0.009238 | 0.025866 | 0.042334 | 0.025499 | 0.019976 |
| All_Autogluon_Chemeleon | HLM | 0.324737 | 0.885854 | 0.175903 | 0.552629 | 0.395399 | 0.009453 | 0.026394 | 0.047395 | 0.027840 | 0.021445 |
| All_Novartis_Autogluon_Chemeleon | HLM | 0.330205 | 0.900845 | 0.185863 | 0.503487 | 0.356561 | 0.008884 | 0.027546 | 0.049547 | 0.030463 | 0.022984 |


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
