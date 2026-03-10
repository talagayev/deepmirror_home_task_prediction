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

