Report for OpenADMET HLM CLint Prediction
==============================

In this workflow a model was built for the prediction of the human liver microsome (HLM) intrinsic clearance [mg/min/kg] with the benchmarking data used being the test set of the [OpenADMET ExpansionRx challenge](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data).

Due to not having particiapated in the challenge itself all of the steps of the workflow and the building of the workflow were performed during the home task timeline.

## Data Gathering & Data Curation

The first step consisted in the gathering of all the training and test data, that would be used for training the model and then the predictions.

For the Training data the following datasets were used:
1. [ChEMBL Data](https://www.ebi.ac.uk/chembl/) containing as well the [AZ dataset](https://www.ebi.ac.uk/chembl/explore/assay/CHEMBL3301370)
2. [OpenADMET Polaris](https://polarishub.io/datasets/asap-discovery/antiviral-admet-2025-unblinded)
3. [Biogen/Fang dataset](https://github.com/molecularinformatics/Computational-ADME)
4. [OpenADMET ExpansionRx Training data](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-train-data)
5. [Novartis dataset](https://www.nature.com/articles/s41467-024-49979-3)

For the ChEBML dataset the following scrapping of the [ChEMBL database was performed](https://github.com/talagayev/deepmirror_home_task_prediction/blob/main/OpenADMET_Data_Workflow/Data_Gathering/OpenADMET_training_data/ChEMBL_data_scrapping.ipynb).
With ChEMBL containing various assays for HLM and with the task being to identify the HLM CLint, this being the intrinsic clearance and ChEBML containing also information about the hepatic clearance and assays where the clearance is not specified only the datapoints mentioning intrinsic clearance were retained.

### Absolute `0` value conversion
For each of the datasets the following procedure was performed:
1. The values were converted in `log1p` values, due to the test set and certain training sets containing absolute `0` values. While an alternative of handling this is described in the following [OpenADMET Notebook](https://github.com/OpenADMET/ExpansionRx-Challenge-Tutorial/blob/main/expansion_tutorial.ipynb), where a `1` is added to avoid the error with the `log` in this approach I was curious of testing how the prediction will behave when everything will be conveted to `log1p` retaining thus the distribution. This will also have an affect on the CV values, with the `log1p` values being usually higher then the `log10` ones.


### Scaling of in vitro to in vivo
2. An important point during this home away task was to try to establish and identify a way to align external data to use it for HLM CLint predictions. Here one thing that I noticed during the data curation consisted in the variance of units that are used to display `HLM CLint`. Here mainly two units are often used to describe it, those being `uM/min/mg` and `mg/min/kg` with one being the value obtained from assays, while the second one is the extrapolated value using the average human liver microsome mass and human body weight. Here is a (short explanation)[https://www.sciencedirect.com/topics/chemistry/intrinsic-clearance].

