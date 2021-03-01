# Data processing steps

## 1. Raw data
Raw data was downloaded from onlinesurveys.ac.uk and is stored in `data/01 raw data/`.

## 2. Data processing
Data was processed in a number of ways:

- meaningful column names
- manually convert all heights into meters
- manually convert all weights into kg
- manually convert all waist measurements into meters
- cells with ambiguous height, weight, or waist measurement were set as missing values
- calculate BMI as a new variable
- calculate height/waist ratio as a new variable


Results are saved in `data/02 processed data/`.

## 3. Scoring
We then scored the raw delay discounting data using Bayesian methods. Results were saved in `data/03 scored data/`. This utilised lists of the delay discounting questions posed to participants, which are contained in `data/discounting questions`. The Python code to run the Bayesian scoring is in the Jupyter notebook (`*.ipynb`) files.

## 3. Final data
Finally, we applied exclusion criteria to the data (see `exclusion.ipynb`). Results were saved in `data/04 final data/`.
