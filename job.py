# %% [markdown]
# # Implementing "Fundamental Analysis via Machine Learning" (Ceo et al. 2024)
# This paper can be found at https://doi.org/10.1080/0015198X.2024.2313692

# %% [markdown]
# ## 1. Imports

# %%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# %% [markdown]
# ## 2 Data Preprocessing

# %% [markdown]
# ### 2.1 Load dataset

# %%
# Compustat dataset downloaded from WRDS with only the necessary columns
df = pd.read_csv('datasets/compustat_crsp.csv')
 
# print fields 
print(df.columns)


# %% [markdown]
# ### 2.2 Clean, Filter and Format the data
# The below is from the paper
# We further impose the following data requirements:
# 1. The following financial statement items must be non-missing: total assets, sales revenue, income before extraordinary items, and common shares outstanding
# 2. The stocks must be ordinary common shares listed on the NYSE, AMEX, or NASDAQ
# 3. The firms cannot be in the financial (SIC 6000–6999) or regulated utilities (SIC 4900–4999) industries
# 4. The stock prices at the end of the third month after the end of the fiscal year must be greater than US$1.

# %%
# 1) 
df = df.dropna(subset=['at', 'sale', 'ib', 'csho'])    # Filter out records with missing data in these columns

# 2)
df = df[(df['SecurityType'] == 'EQTY') & (df['SecuritySubType'] == 'COM')] # Common equity
df = df[df['PrimaryExch'].isin(['N', 'A', 'Q'])] # NYSE ('N'), AMEX ('A'), NASDAQ ('Q')

# 3)
df = df[~df['sic'].isin(range(6000, 7000))]  # Exclude financial firms
df = df[~df['sic'].isin(range(4900, 5000))]  # Exclude regulated utilities

# 4)
df = df[df['MthPrc'] > 1]

# Sort values based on GVKEY (unique firm identifier) and year 
df = df.sort_values(by=['gvkey', 'datadate'])

# Convert into time series
df['datadate'] = pd.to_datetime(df['datadate'])

# %% [markdown]
# ### 2.3 Clean and format features
# Creating new features as needed for the model. Created as per the formulas in the appendix of the paper

# %%
# Calculate iva (inventory and assets) as per the paper
# Add inventory and assets equties and inventory and assets other to get total inventory and assets
df['iva'] = df['ivaeq'].fillna(0) + df['ivao'].fillna(0)

# Calculate CFO (Cash flow from operations) as per the cited paper - NEED TO GO THROUGH
def calculate_cfo(df_in):
    """
    Calculates the final CFO feature by first calculating CFO from the
    balance sheet, then combining it with the reported CFO.
    
    Balance Sheet CFO = ib - Accruals
    Where Accruals = Δ(act - che) - Δ(lct - dlc - txp) - dp
    """
    df = df_in.copy()
    
    # 1. Calculate Balance Sheet CFO
    
    # Fill NAs with 0 for this calculation, as it's an imputation step
    ib = df['ib'].fillna(0)
    act = df['act'].fillna(0)
    che = df['che'].fillna(0)
    lct = df['lct'].fillna(0)
    dlc = df['dlc'].fillna(0)
    txp = df['txp'].fillna(0)
    dp = df['dp'].fillna(0)

    # Calculate non-cash current assets
    df['non_cash_ca'] = act - che
    
    # Calculate adjusted current liabilities 
    df['adj_cl'] = lct - dlc - txp
    
    # Calculate year-over-year changes (Δ)
    # Assumes df is already sorted by gvkey, datadate
    df['change_non_cash_ca'] = df.groupby('gvkey')['non_cash_ca'].diff()
    df['change_adj_cl'] = df.groupby('gvkey')['adj_cl'].diff()
    
    # Calculate Total Accruals
    df['total_accruals'] = (
        df['change_non_cash_ca'].fillna(0) - 
        df['change_adj_cl'].fillna(0) - 
        dp
    )
    
    # Calculate CFO from balance sheet
    df['cfo_bs'] = ib - df['total_accruals']
    
    # 2. Combine with Reported CFO
    
    # Calculate reported CFO
    df['cfo_reported'] = df['oancf'].fillna(0) - df['xidoc'].fillna(0)
    
    # Use reported CFO if available (and not 0), otherwise use balance sheet CFO
    df['cfo'] = df['cfo_reported'].replace(0, np.nan).fillna(df['cfo_bs'])
    
    # Return just the final 'cfo' column
    return df['cfo']

df['cfo'] = calculate_cfo(df)

# %% [markdown]
# ### 2.4 Scale data
# Scale by common shares outstanding to get per-share vaules. This means all vars will be comparable across companies. Careful about dividing by 0 errors.

# %%
# Define base features to be used in the model
base_features = [
        'ib', 'sale', 'cogs', 'xsga', 'dp', 'xint', 'nopi', 'txt', # From Category I
        'xad', 'xrd', 'spi', 'xido', 'dvc',                      # From Category II
        'at', 'act', 'lct', 'lt', 'ceq', 'che', 'invt', 'rect',  # From Category III
        'ppent', 'iva', 'intan', 'ap', 'dlc', 'txp', 'dltt', 're',
        'cfo'                                                    # From Category IV
]

# Fill missing values with 0
df[base_features] = df[base_features].fillna(0)

# Will store the names of the new per-share features
per_share_features = []

# Scaling
for col in base_features:
    col_per_share = f"{col}_per_share"
    df[col_per_share] = df[col] / df['csho']
    per_share_features.append(col_per_share)

# Handling infs
df = df.replace([np.inf, -np.inf], np.nan)

# Dropping rows with NaN values
df = df.dropna()

# %% [markdown]
# ### 2.5 Target Variable
# The target is one-year-forward earnings per share. The equation is below:
# $$y_t = \frac{E_{t+1}}{csho_{t+1}}$$

# %%
# Get E_t+1 (next year's 'ib')
df['ib_t1'] = df.groupby('gvkey')['ib'].shift(-1)

# Get csho_t+1 (next year's 'csho')
df['csho_t1'] = df.groupby('gvkey')['csho'].shift(-1)

# Calculate the target variable
df['y'] = df['ib_t1'] / df['csho_t1']

# %% [markdown]
# ### 2.6 Create difference features
# Calculate year-over-year change for all 30 per-share features

# %%
# Will store the names of the new difference features
diff_features = []

# Group by gvkey so that the diff is computed for the same company
df_diffs = df.groupby('gvkey')[per_share_features].diff()

# Compute diffs
for col in per_share_features:
    diff_col = f"{col}_diff"
    df[diff_col] = df_diffs[col]
    diff_features.append(diff_col)

# Preserves data from early years by filling the difference features with 0
df[diff_features] = df[diff_features].fillna(0)

# %% [markdown]
# ### 2.7 Final Cleanup

# %%
X_cols = per_share_features + diff_features  # List of 60 features
y_col = 'y'

# Only keep required cols
# Drop intermediate cols used for computation
df = df[['gvkey', 'datadate'] + X_cols + [y_col]]

df = df.dropna(subset=[y_col])

# %% [markdown]
# ## 3 ML Train Test Split

# %%
from sklearn.preprocessing import MinMaxScaler

def get_train_test_split(df, X_cols, y_col, prediction_year):
    """"
    Applies a train test split to the data
    Applies a rolling-window, winsoriaation and normalisation
    Returns X_train, X_test, y_train, y_test
    """
    
    # Range of years to use for training
    train_start_year = prediction_year - 10
    train_end_year = prediction_year - 1

    df['year'] = df['datadate'].dt.year

    # Splitting data into train and test set
    train_df = df[(df['year'] >= train_start_year) & (df['year'] <= train_end_year)]
    test_df = df[df['year'] == prediction_year]

    # Not enough data for this year
    if train_df.empty or test_df.empty:
        return None, None, None, None

    X_train = train_df[X_cols]
    y_train = train_df[y_col].values.ravel() # .ravel() is for sklearn
    X_test = test_df[X_cols]
    y_test = test_df[y_col].values.ravel() # .ravel() is for sklearn

    # Winsorisation
    lb = X_train.quantile(0.01)
    ub = X_train.quantile(0.99)
    
    X_train = X_train.clip(lb, ub, axis=1)
    X_test = X_test.clip(lb, ub, axis=1)

    # BEGIN
    # Remove constant features (zero variance) before normalization
    # This prevents "Weights sum to zero" errors in MLPRegressor
    train_var = X_train.var()
    non_constant_cols = train_var[train_var > 1e-8].index.tolist()
    
    if len(non_constant_cols) == 0:
        # All features are constant, can't train
        return None, None, None, None
    
    # Filter to only non-constant features
    X_train = X_train[non_constant_cols]
    X_test = X_test[non_constant_cols]
    
    # Check for minimum number of samples
    if len(X_train) < 10:
        return None, None, None, None
    # END

    # Normalisation
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # BEGIN
    # Convert to DataFrame for easier manipulation
    X_train = pd.DataFrame(X_train, columns=non_constant_cols)
    X_test = pd.DataFrame(X_test, columns=non_constant_cols)
    
    # Check for NaN or inf values after scaling
    if X_train.isna().any().any() or X_test.isna().any().any():
        return None, None, None, None
    
    if np.isinf(X_train).any().any() or np.isinf(X_test).any().any():
        return None, None, None, None
    
    # Check for constant features AFTER scaling (due to numerical precision)
    train_var_scaled = X_train.var()
    non_constant_cols_scaled = train_var_scaled[train_var_scaled > 1e-10].index.tolist()
    
    if len(non_constant_cols_scaled) == 0:
        return None, None, None, None
    
    # Filter again to remove any features that became constant after scaling
    X_train = X_train[non_constant_cols_scaled]
    X_test = X_test[non_constant_cols_scaled]
    
    # Convert back to numpy arrays
    X_train = X_train.values
    X_test = X_test.values
    # END

    return X_train, X_test, y_train, y_test
    

# %% [markdown]
# ## 5 GPU-Accelerated Training (PyTorch)
# Using PyTorch for GPU acceleration - significantly faster than scikit-learn for neural networks
# 

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import time

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  GPU will be used for training - expect 5-50x speedup!")
else:
    print("⚠ No GPU detected. Install PyTorch with CUDA support for GPU acceleration:")
    print("  Visit: https://pytorch.org/get-started/locally/")
    print("  Or run: pip install torch --index-url https://download.pytorch.org/whl/cu118")

# Quick GPU test
if torch.cuda.is_available():
    test_tensor = torch.randn(100, 100).to(device)
    result = torch.matmul(test_tensor, test_tensor)
    print(f"✓ GPU test passed - tensor operations working on GPU")
    del test_tensor, result  # Free memory
    torch.cuda.empty_cache()

# Define PyTorch Neural Network
class PyTorchMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation='relu', dropout=0.0):
        super(PyTorchMLP, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()

# Dataset class for PyTorch
class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_pytorch_model(X_train, y_train, hidden_sizes=(32, 16), activation='relu', 
                       lr=0.001, epochs=1000, batch_size=None, alpha=1e-5, 
                       device=device, early_stopping=True, patience=10):
    """Train a single PyTorch neural network"""
    # Use larger batch size for GPU
    if batch_size is None:
        batch_size = 128 if device.type == 'cuda' else 32
    
    model = PyTorchMLP(X_train.shape[1], hidden_sizes, activation).to(device)
    
    # Verify model is on correct device
    if device.type == 'cuda':
        assert next(model.parameters()).is_cuda, "Model not on GPU!"
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=alpha)
    
    dataset = RegressionDataset(X_train, y_train)
    # Optimize DataLoader for GPU: pin_memory speeds up GPU transfer
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True if device.type == 'cuda' else False,
        num_workers=0  # Set to 0 to avoid multiprocessing issues in notebooks
    )
    
    best_loss = float('inf')
    patience_counter = 0
    training_history = []  # Track loss per epoch
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            # Use non_blocking for faster GPU transfer when pin_memory is enabled
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        training_history.append(avg_loss)  # Store loss for this epoch
        
        # Early stopping
        if early_stopping:
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
    
    return model, training_history

def train_model_gpu(df, X_cols, y_col, start_year=1950, end_year=2025, 
                   n_estimators=10, max_samples=0.6, cv_folds=5):
    """
    GPU-accelerated training using PyTorch
    10 year rolling window for training 
    Bagging ensemble with PyTorch neural networks
    
    Returns:
        results_df: DataFrame with predictions
        metrics_dict: Dictionary with training metrics per year
        models_dict: Dictionary with trained models per year
    """
    
    # Hyperparameter grid for grid search
    param_grid = {
        'hidden_sizes': [(32, 16), (16, 8, 4)],
        'activation': ['relu'],
        'alpha': [1e-5]
    }
    
    all_results = []
    metrics_dict = {}  # Store metrics per year
    models_dict = {}  # Store models per year
    
    for year in range(start_year, end_year+1):
        start_time = time.time()
        
        print(f"\n--- Processing Prediction Year: {year} ---")
        
        X_train, X_test, y_train, y_test = get_train_test_split(df, X_cols, y_col, year)
        
        if X_train is None or X_test is None:
            print(f"Skipping year {year}: Not enough data for 10 year training window")
            continue
        
        print(f"Training on {X_train.shape[0]} samples (Years {year-10}-{year-1})...")
        print(f"Testing on {X_test.shape[0]} samples (Year {year})...")
        
        # Grid search with cross-validation
        best_score = float('inf')
        best_params = None
        best_models = None
        best_training_histories = None  # Store training histories
        
        for hidden_sizes in param_grid['hidden_sizes']:
            for activation in param_grid['activation']:
                for alpha in param_grid['alpha']:
                    # Cross-validation
                    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    cv_scores = []
                    cv_models = []
                    cv_histories = []  # Store training histories for each fold
                    cv_val_losses = []  # Store validation losses
                    
                    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                        X_train_fold = X_train[train_idx]
                        y_train_fold = y_train[train_idx]
                        X_val_fold = X_train[val_idx]
                        y_val_fold = y_train[val_idx]
                        
                        # Train model
                        model, training_history = train_pytorch_model(
                            X_train_fold, y_train_fold,
                            hidden_sizes=hidden_sizes,
                            activation=activation,
                            alpha=alpha,
                            device=device
                        )
                        
                        # Validate
                        model.eval()
                        with torch.no_grad():
                            X_val_tensor = torch.FloatTensor(X_val_fold).to(device)
                            y_pred_val = model(X_val_tensor).cpu().numpy()
                            val_mse = mean_squared_error(y_val_fold, y_pred_val)
                            cv_scores.append(val_mse)
                            cv_models.append(model)
                            cv_histories.append(training_history)
                            cv_val_losses.append(val_mse)
                    
                    avg_cv_score = np.mean(cv_scores)
                    
                    if avg_cv_score < best_score:
                        best_score = avg_cv_score
                        best_params = {'hidden_sizes': hidden_sizes, 'activation': activation, 'alpha': alpha}
                        best_models = cv_models
                        best_training_histories = cv_histories
        
        print(f"Best CV score: {best_score:.6f}, Best params: {best_params}")
        
        # Train final bagging ensemble with best params
        print(f"Training {n_estimators} models for bagging ensemble...")
        ensemble_models = []
        ensemble_histories = []  # Store training histories for ensemble models
        
        for i in range(n_estimators):
            # Bootstrap sampling (60% of data)
            n_samples = int(len(X_train) * max_samples)
            indices = np.random.choice(len(X_train), size=n_samples, replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]
            
            model, training_history = train_pytorch_model(
                X_boot, y_boot,
                hidden_sizes=best_params['hidden_sizes'],
                activation=best_params['activation'],
                alpha=best_params['alpha'],
                device=device
            )
            ensemble_models.append(model)
            ensemble_histories.append(training_history)
        
        # Predict using ensemble (average predictions)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        predictions = []
        
        for model in ensemble_models:
            model.eval()
            with torch.no_grad():
                pred = model(X_test_tensor).cpu().numpy()
                predictions.append(pred)
        
        y_pred = np.mean(predictions, axis=0)
        
        # Calculate test metrics
        test_mse = mean_squared_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        test_mae = np.mean(np.abs(y_test - y_pred))
        test_rmse = np.sqrt(test_mse)
        
        end_time = time.time()
        print(f"Year {year} complete in {end_time - start_time:.2f} seconds.")
        print(f"  Test MSE: {test_mse:.6f}, Test R²: {test_r2:.4f}, Test MAE: {test_mae:.6f}, Test RMSE: {test_rmse:.6f}")
        
        # Store metrics for this year
        metrics_dict[year] = {
            'best_cv_score': best_score,
            'best_params': best_params,
            'cv_training_histories': best_training_histories,  # List of training histories from CV folds
            'ensemble_training_histories': ensemble_histories,  # List of training histories from ensemble models
            'test_mse': test_mse,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'training_time': end_time - start_time
        }
        
        # Store models for this year
        models_dict[year] = {
            'ensemble_models': ensemble_models,
            'best_params': best_params,
            'input_size': X_train.shape[1]
        }
        
        # Store results
        test_identifiers = df[df['year'] == year][['gvkey', 'datadate']]
        
        year_results = pd.DataFrame({
            'gvkey': test_identifiers['gvkey'],
            'datadate': test_identifiers['datadate'],
            'y_actual': y_test,
            'y_predicted_ann': y_pred
        })
        
        all_results.append(year_results)
    
    print("\n--- Backtest Complete ---")
    
    if not all_results:
        print("No results to compile. The backtest did not run (check data and year range).")
        return None, None, None
    
    # Compile results
    final_results_df = pd.concat(all_results, ignore_index=True)
    
    mse = mean_squared_error(
        final_results_df['y_actual'], 
        final_results_df['y_predicted_ann']
    )
    
    r2 = r2_score(
        final_results_df['y_actual'], 
        final_results_df['y_predicted_ann']
    )
    
    mae = np.mean(np.abs(final_results_df['y_actual'] - final_results_df['y_predicted_ann']))
    rmse = np.sqrt(mse)
    
    print(f"Overall Out-of-Sample Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    
    # Add overall metrics to metrics_dict
    metrics_dict['overall'] = {
        'mse': mse,
        'r2': r2,
        'mae': mae,
        'rmse': rmse
    }
    
    return final_results_df, metrics_dict, models_dict

def save_best_model(models_dict, metrics_dict, base_path='models'):
    """
    Save only the best model per year (the one with best validation performance)
    
    Args:
        models_dict: Dictionary returned from train_model_gpu
        metrics_dict: Dictionary with metrics returned from train_model_gpu
        base_path: Base directory to save models
    """
    import os
    
    os.makedirs(base_path, exist_ok=True)
    
    for year, model_info in models_dict.items():
        if year == 'overall':  # Skip overall metrics
            continue
        
        # Get the best model - use the first model from CV (best performing during grid search)
        # Since we use ensemble averaging for predictions, we'll save the first ensemble model
        # which represents the best hyperparameters
        best_model = model_info['ensemble_models'][0]  # First model uses best hyperparameters
        
        model_path = os.path.join(base_path, f'best_model_year_{year}.pth')
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'best_params': model_info['best_params'],
            'input_size': model_info['input_size'],
            'year': year,
            'test_mse': metrics_dict[year]['test_mse'],
            'test_r2': metrics_dict[year]['test_r2']
        }, model_path)
        
        print(f"Saved best model for year {year} to {model_path}")
        print(f"  Test MSE: {metrics_dict[year]['test_mse']:.6f}, Test R²: {metrics_dict[year]['test_r2']:.4f}")

# Run GPU-accelerated training
results_df, metrics_dict, models_dict = train_model_gpu(df, X_cols, y_col, 1975, 2019)

# Save best model for each year
save_best_model(models_dict, metrics_dict, base_path='models')


# %% [markdown]
# ## 6 Plot Training Metrics for PyTorch
# Plot training loss curves and performance metrics
# 

# %%
def plot_training_curves(metrics_dict, year=None, figsize=(15, 10)):
    """
    Plot training loss curves from metrics_dict
    
    Args:
        metrics_dict: Dictionary returned from train_model_gpu
        year: Specific year to plot, or None to plot all years
        figsize: Figure size tuple
    """
    if year is not None:
        # Plot specific year
        if year not in metrics_dict:
            print(f"Year {year} not found in metrics_dict")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Training Metrics for Year {year}', fontsize=16)
        
        year_metrics = metrics_dict[year]
        
        # Plot CV training histories
        ax1 = axes[0, 0]
        for i, history in enumerate(year_metrics['cv_training_histories']):
            ax1.plot(history, alpha=0.6, label=f'CV Fold {i+1}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Cross-Validation Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot ensemble training histories
        ax2 = axes[0, 1]
        for i, history in enumerate(year_metrics['ensemble_training_histories']):
            ax2.plot(history, alpha=0.6, label=f'Model {i+1}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Training Loss')
        ax2.set_title('Ensemble Models Training Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot average training loss
        ax3 = axes[1, 0]
        avg_cv_history = np.mean([h for h in year_metrics['cv_training_histories']], axis=0)
        avg_ensemble_history = np.mean([h for h in year_metrics['ensemble_training_histories']], axis=0)
        ax3.plot(avg_cv_history, label='Avg CV Training Loss', linewidth=2)
        ax3.plot(avg_ensemble_history, label='Avg Ensemble Training Loss', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Average Loss')
        ax3.set_title('Average Training Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot test metrics
        ax4 = axes[1, 1]
        metrics_names = ['test_mse', 'test_rmse', 'test_mae']
        metrics_values = [year_metrics[m] for m in metrics_names]
        ax4.bar(metrics_names, metrics_values)
        ax4.set_ylabel('Value')
        ax4.set_title('Test Metrics')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
    else:
        # Plot all years - show test metrics over time
        years = sorted([k for k in metrics_dict.keys() if k != 'overall'])
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Metrics Across All Years', fontsize=16)
        
        # Extract metrics
        test_mse = [metrics_dict[y]['test_mse'] for y in years]
        test_r2 = [metrics_dict[y]['test_r2'] for y in years]
        test_mae = [metrics_dict[y]['test_mae'] for y in years]
        test_rmse = [metrics_dict[y]['test_rmse'] for y in years]
        cv_scores = [metrics_dict[y]['best_cv_score'] for y in years]
        training_times = [metrics_dict[y]['training_time'] for y in years]
        
        # Plot test MSE over years
        axes[0, 0].plot(years, test_mse, marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Test MSE')
        axes[0, 0].set_title('Test MSE Over Years')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot test R² over years
        axes[0, 1].plot(years, test_r2, marker='o', linewidth=2, markersize=6, color='green')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Test R²')
        axes[0, 1].set_title('Test R² Over Years')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot CV score vs Test MSE
        axes[1, 0].scatter(cv_scores, test_mse, alpha=0.6, s=50)
        axes[1, 0].set_xlabel('CV Score (MSE)')
        axes[1, 0].set_ylabel('Test MSE')
        axes[1, 0].set_title('CV Score vs Test MSE')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot training time
        axes[1, 1].bar(years, training_times, alpha=0.7)
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Training Time (seconds)')
        axes[1, 1].set_title('Training Time per Year')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()

def plot_predictions_vs_actual(results_df, year=None, sample_size=1000):
    """
    Plot predictions vs actual values
    
    Args:
        results_df: DataFrame returned from train_model_gpu
        year: Specific year to plot, or None to plot all
        sample_size: Number of samples to plot (for large datasets)
    """
    if year is not None:
        year_data = results_df[results_df['datadate'].dt.year == year]
        if len(year_data) == 0:
            print(f"No data for year {year}")
            return
        title = f'Predictions vs Actual - Year {year}'
    else:
        year_data = results_df
        title = 'Predictions vs Actual - All Years'
    
    # Sample if too large
    if len(year_data) > sample_size:
        year_data = year_data.sample(n=sample_size, random_state=42)
        title += f' (sampled {sample_size} points)'
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    axes[0].scatter(year_data['y_actual'], year_data['y_predicted_ann'], alpha=0.5, s=20)
    min_val = min(year_data['y_actual'].min(), year_data['y_predicted_ann'].min())
    max_val = max(year_data['y_actual'].max(), year_data['y_predicted_ann'].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = year_data['y_actual'] - year_data['y_predicted_ann']
    axes[1].scatter(year_data['y_predicted_ann'], residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Residuals (Actual - Predicted)')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Plot training curves and predictions
# Note: Make sure to run train_model_gpu first to get results_df, metrics_dict, models_dict

# Plot training curves for a specific year (e.g., 1990)
if 'metrics_dict' in globals() and metrics_dict:
    # Find a year that exists in metrics_dict
    available_years = [y for y in metrics_dict.keys() if y != 'overall']
    if available_years:
        example_year = available_years[0]
        print(f"Plotting training curves for year {example_year}...")
        plot_training_curves(metrics_dict, year=example_year)
        
        print("\nPlotting metrics across all years...")
        plot_training_curves(metrics_dict)
        
        print("\nPlotting predictions vs actual for all years...")
        plot_predictions_vs_actual(results_df)
        
        print(f"\nPlotting predictions vs actual for year {example_year}...")
        plot_predictions_vs_actual(results_df, year=example_year)
    else:
        print("No training data available. Run train_model_gpu first.")
else:
    print("No metrics_dict available. Run train_model_gpu first.")



