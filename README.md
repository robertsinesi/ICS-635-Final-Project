# Robert Sinesi ICS 625 Final Project
## Project Goal
This project attempts to make predictions on band gap sizes of semiconductor materials.

To run this, use the following:

```bash
cd "Project Code"
conda env create -f Sinesi_ICS_635_FP_Conda_Environment.yml
conda activate bandgap-prediction

python Final_Project_2.py
```

## Dataset
The data was taken from Materials Project, an open-access database. There are 68,339 materials pulled from the database. Materials that are not considered semiconductors or had missing data were excluded. There are 123 features after featurization. The data is split into 60% training, 16% validation, and 20% testing. 
## Models
The models considered for this project are:
Random Forest
XGBoost
6 Neural Networks with varying architectures
An ensemble method that uses the three best performing models on the validation set
## Hyperparameter Optimization
The hyperparameters for all models were optimized using a grid search approach.
## Results
### Validation Set
| Model | MAE | R<sup>2</sup> |
| ------------- | ------------- | ------------- |
| XGBoost | 0.38 | 0.86 |
| Random Forest | 0.41 | 0.84 |
| Deep MLP | 0.42 | 0.83 |
| Schindler MLP | 0.45 | 0.82 |
| Very Deep MLP | 0.51 | 0.78 |
| Wide MLP | 0.51 | 0.78 |
| Baseline | 0.52 | 0.78 |
| Shallow MLP | 0.52 | 0.78 |
### Test Set
| Model | MAE | R<sup>2</sup> |
| ------------- | ------------- | ------------- |
| XGBoost | 0.38 | 0.86 |
## New Predictions
| Formula | Predicted Band Gap (eV) | Uncertainty |
| ------------- | ------------- | ------------- |
| XGBoost | 0.38 | 0.86 |
| Random Forest | 0.41 | 0.84 |
| Deep MLP | 0.42 | 0.83 |
| Schindler MLP | 0.45 | 0.82 |
| Very Deep MLP | 0.51 | 0.78 |
