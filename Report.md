Report for OpenADMET HLM CLint Prediction
==============================

In this workflow a model was built for the prediction of the human liver microsome (HLM) intrinsic clearance `mg/min/kg` with the benchmarking data used being the test set of the [OpenADMET ExpansionRx challenge](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data).

Due to not having particiapated in the challenge itself all of the steps of the workflow and the building of the workflow were performed during the home task timeline.

## Data Gathering & Data Curation

The first step consisted in the gathering of all the training and test data, that would be used for training the model and then the predictions.

For the Training data the following datasets were used:
1. [ChEMBL Data](https://www.ebi.ac.uk/chembl/) containing as well the [AZ dataset](https://www.ebi.ac.uk/chembl/explore/assay/CHEMBL3301370) --> 11702 HLM CLint data points
2. [OpenADMET Polaris](https://polarishub.io/datasets/asap-discovery/antiviral-admet-2025-unblinded) --> 403 data points.
3. [Biogen/Fang dataset](https://github.com/molecularinformatics/Computational-ADME) --> 3087 data points.
4. [OpenADMET ExpansionRx Training data](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-train-data) --> 3759 data points.
5. [Novartis dataset](https://www.nature.com/articles/s41467-024-49979-3) --> 273638 data points.

For the ChEBML dataset the following scrapping of the [ChEMBL database was performed](https://github.com/talagayev/deepmirror_home_task_prediction/blob/main/OpenADMET_Data_Workflow/Data_Gathering/OpenADMET_training_data/ChEMBL_data_scrapping.ipynb).
With ChEMBL containing various assays for HLM and with the task being to identify the HLM CLint, this being the intrinsic clearance and ChEBML containing also information about the hepatic clearance and assays where the clearance is not specified only the datapoints mentioning intrinsic clearance were retained.

### Absolute `0` value conversion
For each of the datasets the following procedure was performed:
1. The values were converted in `log1p` values, due to the test set and certain training sets containing absolute `0` values. While an alternative of handling this is described in the following [OpenADMET Notebook](https://github.com/OpenADMET/ExpansionRx-Challenge-Tutorial/blob/main/expansion_tutorial.ipynb), where a `1` is added to avoid the error with the `log` in this approach I was curious of testing how the prediction will behave when everything will be conveted to `log1p` retaining thus the distribution. This will also have an affect on the CV values, with the `log1p` values being usually higher then the `log10` ones.

x 
### Scaling of in vitro to in vivo
2. An important point during this home away task was to try to establish and identify a way to align external data to use it for HLM CLint predictions. Here one thing that I noticed during the data curation consisted in the variance of units that are used to display `HLM CLint`. Here mainly two units are often used to describe it, those being `uM/min/mg` and `mg/min/kg` with one being the value obtained from assays, while the second one is the extrapolated value using the average human liver microsome mass and human body weight. Here is a (short explanation)[https://www.sciencedirect.com/topics/chemistry/intrinsic-clearance].

<img src="https://github.com/talagayev/deepmirror_home_task_prediction/blob/main/OpenADMET_Data_Workflow/Figures/HLM_scaling.png" height="250">

While [certain sources](https://www.epa.gov/system/files/documents/2021-07/exhibit_a_supp_mat_c_ivive_literature_review_07152021.pdf) claim that the correct values to be used are `40 mg/g liver` and `27 g of liver/kg body weight` [other sources](https://doi.org/10.1016/B978-0-12-820018-6.00022-3) claim that for the extrapolation for the human microsomes `40 mg/g liver` and `21 g of liver/kg body weight` should be used.
Depending on what values are used this means that all the data points containing `uM/min/mg` need to be multiplied by a factor of `0.840` or `1.028`. While for this study I used the latter factor, since I relied on it coming from a reliable source I would also look into the `0.840` factor as well, to understand if the predicted values would correlate more with the true values. 

With the test set that we need to predict containing `mg/min/kg` values an extrapolation was done. While the `1.028` does not sound significant for a mouse the extrapolation values can range around a factor of `2-3` depending on what values are used.

This affected the datasets in a following way:
1. ChEMBL contained `8859` molecules which needed to be extrapolated, while only `2843` contained the extrapolated valules. The AZ dataset values needed to be extrapolated
2. Fang dataset contained already `mg/min/kg` values and thus no extrapolation was required.
3. OpenADMET Polaris dataset contained `uM/min/mg` values so extrapolation was required.
4. ExpansionRx training dataset contained `mg/min/kg` data points and thus no extrapolation was required
5. Novartis dataset didn't contain any information about the units, so here due to the small factor and PC power restrictions no extrapolation was done.

## Training data sets preprocessing and dedpuplication
With the training data coming from various sources with each having it's own caveats, OpenADMET ExpansionRx being the one that has the same conditions like the test set, certain data sets requiring extrapolation/scaling, while others don't require them 4 separate training data sets were used. Those were the following:

1. Expansion --> Dataset only containing the Training data from ExpansionRx due to it having the closest conditions --> 3759 data points
2. No Scale --> Dataset that contains only data points that did not require scaling, excluding Novartis --> 9690 data points
3. All --> Dataset that contains all the training data, excluding Novartis --> 18952 data points
4. All + Novartis --> All data sets found  --> x data points.

These data sets underwent preprocessing. This consisted of SMILES standardization, salt stripping + protonation. For the protonation the package `dimorphite_dl` was used, where for the protonation the `ph = 7.4` was used with the most common variant being saved. The test set was also preprocessed in an identical way to have them aligned.

The preprocessed training data sets, containing the preprocessed SMILES underwent deduplication, where if identical SMILES were identified only one was kept with the average value. While there could be a discussion about having higher priority for values from certain data sets, for example the OpenADMET ExpansionRx having higher priority and weight due to it having identical condiditions like the test set due to time limit restrictions it was not possible to experiment with various deduplication approaches.

## Applicability Domain
To estimate the uncertainty of the predictions of the models Applicability domain of the training data with various features were calculated. The following features were used for this caclulation:
1. Chemeleon
2. Morgan (radius = 2, 3; bits = 512, 1024, 2048)
3. RDKit Path (min_path = [3, 4] max_path = [5, 6, 7, 8] bits = 1024, 2048, 4096)
4. Chemeleon + Morgan
5. Chemeleon + RDKit Path

The applicability domain showed, that Chemeleon performed well and had only a certain amount test molecules outside of the AD, while also the Chemeleon + RDKit Path combination performed unexpectedly well. Also during this step it was evaluated if it would be beneficial to fine tune the OpenADMET ExpansionRx Training set data in data sets, from the AD side of things it was reasonable to do it in some cases.
This information will be later used for the evaluation to identify the results for the prediction of the molecules outside AD

## Crossvalidation & Model Training

Here sadly it is the case, that due to me having a slow PC and having only a certain timeline for the predictions I was at this step limited for the selection of Model + Feature combinations.

I used Autogluon and Chemprop for model Crossvalidation/Training and building.

I was able to run Autogluon with the ExpansionRx dataset for all features, while for the remainder I focused on Chemeleon, due to me requiring to do crossvalidation only for chemeleon itself, while for the FPs, I would need to test out the most preferable combination with Chemprop taking too long on my PC and thus I was able to get the final predictions for the `No Scale` and `All` datasets for Chemprop on the final day with only some hours left before the deadline.

For the CV a 5x5 nested CV was used. The primary metric for model performance was MAE. Due to the previously mentioned fact that in this approach `log1p` values were used the MAE values in the CV were higher then in the final evalutaion, where they were converted to `log10` values to recreate the OpenADMET model performance evaluation.

For the Autogluon prediction the `best_quality` setting was used with `time_limit = 600s`. [Here is an example yaml file used](https://github.com/talagayev/deepmirror_home_task_prediction/blob/main/OpenADMET_Data_Workflow/Crossvalidation/expansion/crossvalidation_expansion.yml)

For Chemprop nested 5x5 CV was performed with 10 trials and 100 max epochs. [Here is one yml containing more details](https://github.com/talagayev/deepmirror_home_task_prediction/blob/main/OpenADMET_Data_Workflow/Crossvalidation/expansion/crossvalidation_expansion_chemprop.yml)

After the CV the best performing model was obtained after refitting.

## Prediction & Evaluation

The models were used for predicting the test set and finally evaluated using the [evaluation metrics provided by OpenADMET](https://github.com/OpenADMET/ExpansionRx-Challenge-Tutorial/blob/main/eval/eval.py)


| Dataset | mean MAE | std_MAE | mean RAE | std_RAE | mean R2 | std_R2 |
|-----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Expansion_Chemprop_Chemeleon     | 0.303279    | Chemeleon    | 0.303279    | Chemeleon    | 0.303279    | Chemeleon    |
| Expansion_Chemprop_Chemeleon     | 0.303279    | Chemeleon    | 0.303279    | Chemeleon    | 0.303279    | Chemeleon    |
| Expansion_Chemprop_Chemeleon     | 0.303279    | Chemeleon    | 0.303279    | Chemeleon    | 0.303279    | Chemeleon    |
