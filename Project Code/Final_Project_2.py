#%% Imports
import pandas as pd
import numpy as np
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from pymatgen.core import Composition
import joblib
import random
import scipy.stats as stats

#%% Neural Network Architectures

class BasicMLP(nn.Module):
    def __init__(self, input_dim, hidden_size=128, dropout=0.0):
        super(BasicMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_size=256, dropout=0.0):
        super(DeepMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, hidden_size//4)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size//4, 1)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        return self.out(x)

class WideMLP(nn.Module):
    def __init__(self, input_dim, hidden_size=512, dropout=0.0):
        super(WideMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

class ShallowMLP(nn.Module):
    def __init__(self, input_dim, hidden_size=64, dropout=0.0):
        super(ShallowMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

class VeryDeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_size=512, dropout=0.0):
        super(VeryDeepMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//4, hidden_size//8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//8, 1)
        )
    
    def forward(self, x):
        return self.layers(x)
    
class SchindlerMLP(nn.Module):
    def __init__(self, input_dim, hidden_size=128, dropout=0.0):
        super(SchindlerMLP, self).__init__()
        # You can ignore hidden_size and dropout if you want exactly Peter Schindler's structure.
        # We'll use his structure manually.
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

#%% MLPipeline

class MLPipeline:
    
    def __init__(self, data_path, metric_interval=5):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metric_interval = metric_interval
        self.results = []
        self.load_and_prepare_data()
        os.makedirs('saved_models', exist_ok=True)
        os.makedirs('Plots', exist_ok=True)
        
    def load_and_prepare_data(self):
# =============================================================================
#         df = pd.read_csv(self.data_path)
#         df = df[['formula_pretty', 'band_gap', 'density', 'formation_energy_per_atom', 'energy_above_hull', 'structure']]
#         df = df.dropna(subset=['band_gap'])
# 
#         def is_valid_formula(formula):
#             try:
#                 _ = Composition(formula)
#                 return True
#             except:
#                 return False
# 
#         df = df[df['formula_pretty'].apply(is_valid_formula)]
#         print(f"Number of valid chemical formulas: {len(df)}")
# 
#         df = StrToComposition().featurize_dataframe(df, 'formula_pretty')
#         ep_feat = ElementProperty.from_preset("magpie")
#         df = ep_feat.featurize_dataframe(df, col_id='composition')
#         df.to_csv('materials_project_full_featurized.csv', index=False)
# =============================================================================

        # Load pre-featurized dataset
        df = pd.read_csv('materials_project_full_featurized.csv')
    
        df = df.drop(columns=['formula_pretty', 'structure', 'composition'])
        self.X = df.drop(columns=['band_gap'])
        self.y = df['band_gap']
        
        self.X = self.X.dropna(axis=1)
    
        print(f"Number of features after featurization: {self.X.shape[1]}")
    
        # First split: Train+Val vs Test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
    
        # Second split: Train vs Validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42
        )
    
        # Now scale using only training set
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
        # Save mean values for missing physical properties
        self.default_physical_properties = {
            'density': self.X_train['density'].mean(),
            'formation_energy_per_atom': self.X_train['formation_energy_per_atom'].mean(),
            'energy_above_hull': self.X_train['energy_above_hull'].mean()
        }

    def tune_random_forest(self):
        param_grid = {
            'n_estimators': [100],  # For tuning only
            'max_depth': [5, 10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf = RandomForestRegressor(random_state=42)
        grid = GridSearchCV(rf, param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=3)
        grid.fit(self.X_train, self.y_train)
        best_params = grid.best_params_
    
        # Retrain with n_estimators=800 and best other params
        best_rf = RandomForestRegressor(
            n_estimators=3000,
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            random_state=42
        )
        best_rf.fit(self.X_train, self.y_train)
    
        # Validation evaluation (not test yet)
        preds = best_rf.predict(self.X_val)
        self.evaluate_model(preds, 'Random Forest', self.y_val)
        joblib.dump(best_rf, 'saved_models/RandomForest_best.pkl')
        
    def load_and_evaluate_rf(self):
        print("Loading Random Forest model...")
        rf_model = joblib.load('saved_models/RandomForest_best.pkl')
        preds = rf_model.predict(self.X_test)
        self.evaluate_model(preds, 'Random Forest', self.y_test)

    def tune_xgboost(self):
        param_grid = {
            'n_estimators': [100],  # For tuning only
            'max_depth': [3, 6, 9, 12],
            'learning_rate': [0.1, 0.05, 0.01, 0.005],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        best_mae = float('inf')
        best_params = None
    
        for n in param_grid['n_estimators']:
            for depth in param_grid['max_depth']:
                for lr in param_grid['learning_rate']:
                    for subsample in param_grid['subsample']:
                        for colsample in param_grid['colsample_bytree']:
                            model = xgb.XGBRegressor(
                                n_estimators=n, max_depth=depth, learning_rate=lr,
                                subsample=subsample, colsample_bytree=colsample,
                                random_state=42, verbosity=0
                            )
                            model.fit(self.X_train, self.y_train)
                            preds = model.predict(self.X_val)  # <-- use validation
                            mae = mean_absolute_error(self.y_val, preds)
    
                            if mae < best_mae:
                                best_mae = mae
                                best_params = (depth, lr, subsample, colsample)
    
        print(f"Best XGBoost params: {best_params}")
        depth, lr, subsample, colsample = best_params
    
        # Retrain with n_estimators=800
        best_model = xgb.XGBRegressor(
            n_estimators=3000,
            max_depth=depth,
            learning_rate=lr,
            subsample=subsample,
            colsample_bytree=colsample,
            random_state=42,
            verbosity=0
        )
        best_model.fit(self.X_train, self.y_train)
    
        preds = best_model.predict(self.X_val)
        self.evaluate_model(preds, 'XGBoost', self.y_val)
        joblib.dump(best_model, 'saved_models/XGBoost_best.pkl')
        
    def load_and_evaluate_xgb(self):
        print("Loading XGBoost model...")
        xgb_model = joblib.load('saved_models/XGBoost_best.pkl')
        preds = xgb_model.predict(self.X_test)
        self.evaluate_model(preds, 'XGBoost', self.y_test)

    def tune_and_train_nn(self, model_class):
        model_save_path = f'saved_models/{model_class.__name__}.pth'

        if os.path.exists(model_save_path):
            print(f"Model for {model_class.__name__} already fully trained. Loading and evaluating...")
            checkpoint = torch.load(model_save_path)
            hyperparams = checkpoint['hyperparams']
            model = model_class(
                hyperparams['input_dim'],
                hidden_size=hyperparams['hidden_size'],
                dropout=hyperparams['dropout']
            ).to(self.device)
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            with torch.no_grad():
                preds_val = model(torch.tensor(self.X_val_scaled, dtype=torch.float32).to(self.device)).cpu().numpy().flatten()
            self.evaluate_model(preds_val, model_class.__name__, self.y_val)  # ✅ Fix here
            return
    
        # Hyperparameter grid
        param_grid = {
            'optimizer': ['Adam', 'SGD'],
            'learning_rate': [0.01, 0.002, 0.001, 0.0008, 0.0005, 0.0001],
            'weight_decay': [0, 1e-10, 1e-8, 1e-6, 1e-4],
            'hidden_size_multiplier': [1, 2, 4, 8, 16, 32],
            'dropout': [0, 0.05, 0.1, 0.15, 0.2]
        }
        
        best_mae = float('inf')
        best_params = None
        input_dim = self.X_train_scaled.shape[1]
    
        # Grid search
        for optimizer in param_grid['optimizer']:
            for lr in param_grid['learning_rate']:
                for wd in param_grid['weight_decay']:
                    for mult in param_grid['hidden_size_multiplier']:
                        for drop in param_grid['dropout']:
                            model = model_class(input_dim, hidden_size=128*mult, dropout=drop).to(self.device)
                            crit = nn.MSELoss()
                            if optimizer == 'Adam':
                                opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                            else:
                                opt = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    
                            X_train_tensor = torch.tensor(self.X_train_scaled, dtype=torch.float32).to(self.device)
                            y_train_tensor = torch.tensor(self.y_train.values, dtype=torch.float32).view(-1, 1).to(self.device)
                            model.train()
                            for epoch in range(20):  # Short tuning run
                                opt.zero_grad()
                                loss = crit(model(X_train_tensor), y_train_tensor)
                                loss.backward()
                                opt.step()
    
                            model.eval()
                            with torch.no_grad():
                                preds_val = model(torch.tensor(self.X_val_scaled, dtype=torch.float32).to(self.device)).cpu().numpy().flatten()
                            if np.isnan(preds_val).any():
                                continue
                            mae = mean_absolute_error(self.y_val, preds_val)
    
                            if mae < best_mae:
                                best_mae = mae
                                best_params = (optimizer, lr, wd, mult, drop)
    
        print(f"Best params for {model_class.__name__}: {best_params}")
        self.full_train_nn(model_class, best_params)

    def full_train_nn(self, model_class, best_params):
        optimizer_choice, lr, wd, mult, drop = best_params
        input_dim = self.X_train_scaled.shape[1]
        model_save_path = f"saved_models/{model_class.__name__}.pth"
    
        # Check if already exists
        if os.path.exists(model_save_path):
            print(f"Model for {model_class.__name__} already fully trained. Loading and evaluating...")
            checkpoint = torch.load(model_save_path)
            hyperparams = checkpoint['hyperparams']
            model = model_class(
                hyperparams['input_dim'],
                hidden_size=hyperparams['hidden_size'],
                dropout=hyperparams['dropout']
            ).to(self.device)
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
    
            with torch.no_grad():
                preds_val = model(torch.tensor(self.X_val_scaled, dtype=torch.float32).to(self.device)).cpu().numpy().flatten()
            self.evaluate_model(preds_val, model_class.__name__)
            return
    
        # Start training
        model = model_class(input_dim, hidden_size=128*mult, dropout=drop).to(self.device)
        criterion = nn.MSELoss()
        if optimizer_choice == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    
        X_train_tensor = torch.tensor(self.X_train_scaled, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(self.y_train.values, dtype=torch.float32).view(-1, 1).to(self.device)
        X_val_tensor = torch.tensor(self.X_val_scaled, dtype=torch.float32).to(self.device)
    
        losses = []
        for epoch in range(30000):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
    
            if epoch % self.metric_interval == 0:
                model.eval()
                with torch.no_grad():
                    preds_val = model(X_val_tensor).cpu().numpy().flatten()
                mae = mean_absolute_error(self.y_val, preds_val)
                losses.append(mae)
    
                if len(losses) > 10 and abs(losses[-1] - losses[-self.metric_interval]) < 1e-10:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
        # Save final model
        torch.save({
            'model_state': model.state_dict(),
            'hyperparams': {
                'input_dim': input_dim,
                'hidden_size': 128*mult,
                'dropout': drop
            }
        }, model_save_path)
    
        # Evaluate on Validation
        with torch.no_grad():
            preds_val = model(X_val_tensor).cpu().numpy().flatten()
        self.evaluate_model(preds_val, model_class.__name__, self.y_val)
    
        # Plot training curve
        epochs_recorded = [1] + list(np.arange(self.metric_interval, (len(losses)) * self.metric_interval, self.metric_interval))
        
        name_mapping = {
            'BasicMLP': 'Baseline MLP',
            'DeepMLP': 'Deep MLP',
            'WideMLP': 'Wide MLP',
            'ShallowMLP': 'Shallow MLP',
            'VeryDeepMLP': 'Very Deep MLP',
            'SchindlerMLP': 'Schindler MLP'
        }
        display_name = name_mapping.get(model_class.__name__, model_class.__name__)
    
        plt.figure(figsize=(8,6))
        plt.plot(epochs_recorded, losses, linewidth=1.5)
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Mean Absolute Error (MAE)', fontsize=18)
        plt.title(f'{display_name} Training Curve', fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Plots/{model_class.__name__}_training_curve.png', bbox_inches='tight')
        plt.close()
                    
    def generate_regression_plots(self, y_true, y_pred, model_name):
        plt.rcParams.update({'font.size': 16})  # Bigger text globally
    
        # 1. Scatter Plot: True vs Predicted
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
        plt.xlabel('True Band Gap (eV)', fontsize=18)
        plt.ylabel('Predicted Band Gap (eV)', fontsize=18)
        plt.title(f'{model_name}: True vs Predicted', fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Plots/{model_name}_scatter.png', bbox_inches='tight')
        plt.close()
    
        # 2. Residuals Plot
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predicted Band Gap (eV)', fontsize=18)
        plt.ylabel('Residual (True - Predicted) (eV)', fontsize=18)
        plt.title(f'{model_name}: Residuals', fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Plots/{model_name}_residuals.png', bbox_inches='tight')
        plt.close()
    
        # 3. Residuals Histogram with true Gaussian fit
        plt.figure(figsize=(8, 6))
        
        # Plot the histogram with default color and correct normalization
        n, bins, _ = plt.hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
        
        # Fit and plot the Gaussian curve
        mu, std = stats.norm.fit(residuals)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        
        # Add statistics box
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        plt.text(0.95, 0.95, f'Mean = {mu:.3f} eV\nStd = {std:.3f} eV',
                 transform=plt.gca().transAxes, fontsize=14,
                 verticalalignment='top', horizontalalignment='right', bbox=props)
        
        # Axis labels and title
        plt.xlabel('Residual (eV)', fontsize=18)
        plt.ylabel('Density', fontsize=18)
        plt.title(f'{model_name}: Residuals Histogram with Gaussian Fit', fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(f'Plots/{model_name}_residuals_hist.png', bbox_inches='tight')
        plt.close()

    def evaluate_model(self, preds, model_name, y_true):
        name_mapping = {
            'BasicMLP': 'Baseline MLP',
            'DeepMLP': 'Deep MLP',
            'WideMLP': 'Wide MLP',
            'ShallowMLP': 'Shallow MLP',
            'VeryDeepMLP': 'Very Deep MLP',
            'SchindlerMLP': 'Schindler MLP',
            'RandomForest': 'Random Forest',
            'XGBoost': 'XGBoost',
            'Ensemble': 'Ensemble'
        }
        model_name = name_mapping.get(model_name.replace(' ', ''), model_name)

        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true, preds)
        print(f"{model_name}: MAE = {mae:.4f}, R² = {r2:.4f}")
        self.results.append({'Model': model_name, 'MAE': mae, 'R2': r2})
        self.generate_regression_plots(y_true, preds, model_name)
        
    def build_ensemble_model(self):
        print("\nBuilding ensemble model from top 3 models...")
    
        # Sort by lowest MAE and pick top 3
        top_models = sorted(self.results, key=lambda x: x['MAE'])[:3]
        model_names = [m['Model'] for m in top_models]
        print(f"Top 3 models: {model_names}")
    
        # Extract inverse-MAE weights
        maes = np.array([m['MAE'] for m in top_models])
        inv_maes = 1 / maes
        weights = inv_maes / inv_maes.sum()
        print(f"Model weights based on 1/MAE: {weights}")
    
        preds_list = []
    
        for model_name in model_names:
            if model_name in ['Random Forest', 'XGBoost']:
                model_path = f"saved_models/{model_name.replace(' ', '')}_best.pkl"
                model = joblib.load(model_path)
                preds = model.predict(self.X_test)
            else:
                model_path = f"saved_models/{model_name.replace(' ', '')}.pth"
                checkpoint = torch.load(model_path)
                hyperparams = checkpoint['hyperparams']
                model = globals()[model_name.replace(' ', '')](
                    hyperparams['input_dim'],
                    hidden_size=hyperparams['hidden_size'],
                    dropout=hyperparams['dropout']
                ).to(self.device)
                model.load_state_dict(checkpoint['model_state'])
                model.eval()
                with torch.no_grad():
                    preds = model(torch.tensor(self.X_test_scaled, dtype=torch.float32).to(self.device)).cpu().numpy().flatten()
            
            preds_list.append(preds)
    
        # Weighted ensemble
        preds_array = np.vstack(preds_list)
        uncertainty = np.std(preds_array, axis=0)
        ensemble_preds = np.average(preds_array, axis=0, weights=weights)
        self.ensemble_predictions = ensemble_preds
        self.ensemble_uncertainty = uncertainty
    
        self.evaluate_model(ensemble_preds, 'Ensemble', self.y_test)
        
    def export_results(self):
        self.results = sorted(self.results, key=lambda x: x['MAE'])
        df_results = pd.DataFrame(self.results)
        df_results.to_csv('model_comparison_results.csv', index=False)
        print(df_results)
        
    def export_test_results(self, model_name, mae, r2):
        test_results = pd.DataFrame([{
            'BestModel': model_name,
            'TestMAE': mae,
            'TestR2': r2
        }])
        test_results.to_csv('test_results_summary.csv', index=False)
        print("\nTest results exported to 'test_results_summary.csv'.")
        
    def evaluate_best_model_on_test(self, model_name):
        print(f"\nEvaluating {model_name} on TEST set...")
    
        if model_name == "Ensemble":
            print("Using ensemble model for final test evaluation...")
            preds = self.ensemble_predictions
        elif model_name in ['Random Forest', 'XGBoost']:
            model_path = f"saved_models/{model_name.replace(' ', '')}_best.pkl"
            model = joblib.load(model_path)
            preds = model.predict(self.X_test)
        else:
            model_path = f"saved_models/{model_name.replace(' ', '')}.pth"
            checkpoint = torch.load(model_path)
            hyperparams = checkpoint['hyperparams']
            model = globals()[model_name.replace(' ', '')](
                hyperparams['input_dim'],
                hidden_size=hyperparams['hidden_size'],
                dropout=hyperparams['dropout']
            ).to(self.device)
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(self.X_test_scaled, dtype=torch.float32).to(self.device)).cpu().numpy().flatten()
    
        self.evaluate_model(preds, model_name + ' Test', self.y_test)
    
        # Now export the result separately
        mae = mean_absolute_error(self.y_test, preds)
        r2 = r2_score(self.y_test, preds)
        self.export_test_results(model_name, mae, r2)
        
    def generate_and_predict_new_materials(self, model_name, num_new=5):
        print("\nGenerating new plausible semiconductor materials...")
    
        # Load the original dataset to compare formulas
        original_df = pd.read_csv(self.data_path)
        original_formulas = set(original_df['formula_pretty'])
    
        # Define semiconductor-relevant elements
        metals = ["Al", "Ga", "In", "Zn", "Cd"]
        semi_metals = ["Si", "Ge", "Sn"]
        pnictogens = ["N", "P", "As", "Sb"]
        chalcogens = ["O", "S", "Se", "Te"]
    
        # Generate plausible formulas
        new_formulas = []
        attempts = 0
        while len(new_formulas) < num_new and attempts < 500:  # avoid infinite loop
            metal = random.choice(metals)
            semi_metal = random.choice(semi_metals)
            anion = random.choice(pnictogens + chalcogens)
            formula = f"{metal}1{semi_metal}1{anion}2"
            try:
                comp = Composition(formula)
                reduced_formula = comp.reduced_formula
                if reduced_formula not in original_formulas and reduced_formula not in new_formulas:
                    new_formulas.append(reduced_formula)
            except:
                continue
            attempts += 1
    
        if len(new_formulas) < num_new:
            print(f"Warning: Only {len(new_formulas)} unique new formulas could be generated.")
    
        print(f"Generated formulas: {new_formulas}")
    
        # Build dataframe
        new_df = pd.DataFrame(new_formulas, columns=['formula_pretty'])
    
        # Featurize
        new_df = StrToComposition().featurize_dataframe(new_df, 'formula_pretty')
        ep_feat = ElementProperty.from_preset("magpie")
        new_df = ep_feat.featurize_dataframe(new_df, col_id='composition')
    
        if 'composition' in new_df.columns:
            new_df = new_df.drop(columns=['composition'])
    
        # Fill missing physical features
        for col, mean_value in self.default_physical_properties.items():
            if col not in new_df.columns:
                new_df[col] = mean_value
    
        # Reorder columns
        new_df = new_df[self.X.columns]
    
        # Scale
        X_new_scaled = self.scaler.transform(new_df)
    
        # Predict
        if model_name == 'Ensemble':
            # --- Ensemble prediction with uncertainty ---
            preds_list = []
            model_list = ['Random Forest', 'XGBoost', 'DeepMLP', 'SchindlerMLP', 'WideMLP', 'VeryDeepMLP', 'ShallowMLP', 'BasicMLP']
            model_list = [m for m in model_list if os.path.exists(f'saved_models/{m.replace(" ", "")}_best.pkl') or os.path.exists(f'saved_models/{m}.pth')]
            
            for sub_model_name in model_list[:3]:  # top 3 models
                if sub_model_name in ['Random Forest', 'XGBoost']:
                    model_path = f"saved_models/{sub_model_name.replace(' ', '')}_best.pkl"
                    model = joblib.load(model_path)
                    preds = model.predict(X_new_scaled)
                else:
                    model_path = f"saved_models/{sub_model_name}.pth"
                    checkpoint = torch.load(model_path)
                    hyperparams = checkpoint['hyperparams']
                    model = globals()[sub_model_name](
                        hyperparams['input_dim'],
                        hidden_size=hyperparams['hidden_size'],
                        dropout=hyperparams['dropout']
                    ).to(self.device)
                    model.load_state_dict(checkpoint['model_state'])
                    model.eval()
                    with torch.no_grad():
                        preds = model(torch.tensor(X_new_scaled, dtype=torch.float32).to(self.device)).cpu().numpy().flatten()
                preds_list.append(preds)
    
            preds_array = np.vstack(preds_list)
            preds = np.mean(preds_array, axis=0)
            uncertainty = np.std(preds_array, axis=0)
    
            # Save results
            predictions_df = pd.DataFrame({
                'Formula': new_formulas,
                'PredictedBandGap': preds,
                'Uncertainty': uncertainty
            })
            predictions_df.to_csv('new_material_predictions_with_uncertainty.csv', index=False)
    
            # Plot
            plt.figure(figsize=(10, 6))
            plt.bar(new_formulas, preds, yerr=uncertainty, capsize=5)
            plt.xlabel('Material Formula', fontsize=16)
            plt.ylabel('Predicted Band Gap (eV)', fontsize=16)
            plt.title('Predicted Band Gaps with Uncertainty for New Materials', fontsize=18)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig('Plots/new_materials_bandgap_with_uncertainty.png', bbox_inches='tight')
            plt.close()
    
        else:
            # --- Single model prediction (no uncertainty) ---
            if model_name in ['Random Forest', 'XGBoost']:
                model_path = f"saved_models/{model_name.replace(' ', '')}_best.pkl"
                model = joblib.load(model_path)
                preds = model.predict(X_new_scaled)
            else:
                model_path = f"saved_models/{model_name}.pth"
                checkpoint = torch.load(model_path)
                hyperparams = checkpoint['hyperparams']
                model = globals()[model_name](
                    hyperparams['input_dim'],
                    hidden_size=hyperparams['hidden_size'],
                    dropout=hyperparams['dropout']
                ).to(self.device)
                model.load_state_dict(checkpoint['model_state'])
                model.eval()
                with torch.no_grad():
                    preds = model(torch.tensor(X_new_scaled, dtype=torch.float32).to(self.device)).cpu().numpy().flatten()
    
            # Save results
            predictions_df = pd.DataFrame({
                'Formula': new_formulas,
                'PredictedBandGap': preds
            })
            predictions_df.to_csv('new_material_predictions.csv', index=False)
    
        # Show results
        print("\nPredictions on Newly Generated Materials:")
        for idx, formula in enumerate(new_formulas):
            if model_name == 'Ensemble':
                print(f"{formula}: Predicted Band Gap = {preds[idx]:.3f} eV ± {uncertainty[idx]:.3f} eV")
            else:
                print(f"{formula}: Predicted Band Gap = {preds[idx]:.3f} eV")

#%% Main Run Block

if __name__ == "__main__":
    pipeline = MLPipeline('materials_project_semiconductors.csv', metric_interval=5)

    # Classical Models
    if os.path.exists('saved_models/RandomForest_best.pkl'):
        pipeline.load_and_evaluate_rf()
    else:
        pipeline.tune_random_forest()

    if os.path.exists('saved_models/XGBoost_best.pkl'):
        pipeline.load_and_evaluate_xgb()
    else:
        pipeline.tune_xgboost()

    # Neural Networks
    for model_class in [BasicMLP, DeepMLP, WideMLP, ShallowMLP, VeryDeepMLP, SchindlerMLP]:
        pipeline.tune_and_train_nn(model_class)

    pipeline.build_ensemble_model()
    
    pipeline.export_results()

    # Evaluate best model on the untouched test set
    best_model_name = pipeline.results[0]['Model']
    pipeline.evaluate_best_model_on_test(best_model_name)
    
    pipeline.generate_and_predict_new_materials(best_model_name, num_new=5)