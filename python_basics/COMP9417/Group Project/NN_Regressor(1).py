import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')

#1.data loading and preprocessing
print("=" * 60)
print("1. Data Loading and Preprocessing (Neural Network Regressor)")
print("=" * 60)

df = pd.read_csv("processed_air_quality.csv", parse_dates=['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Time range: {df['Datetime'].min()} to {df['Datetime'].max()}")

target_pollutants = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
available_targets = [t for t in target_pollutants if t in df.columns]
print(f"Available pollutants for prediction: {available_targets}")

# NMHC(GT) Data Statistical Check
if 'NMHC(GT)' in df.columns and 'NMHC_Valid' in df.columns:
    print("\nNMHC(GT) statistics:")
    print(f"  Min: {df['NMHC(GT)'].min():.2f}, Max: {df['NMHC(GT)'].max():.2f}")
    print(f"  Mean: {df['NMHC(GT)'].mean():.2f}, Std: {df['NMHC(GT)'].std():.2f}")
    print(f"  Missing values: {df['NMHC(GT)'].isna().sum()}")
    print(f"  Valid data ratio: {df['NMHC_Valid'].mean():.2%}")

# 2. Feature Engineering

print("\n" + "=" * 60)
print("2. Feature Engineering")
print("=" * 60)

df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Month'] = df['Datetime'].dt.month
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

# lag & roll for each pollution
for target in available_targets:
    for lag in [1, 3, 6, 12, 24]:
        df[f'{target}_lag_{lag}'] = df[target].shift(lag)
    for window in [3, 6, 12, 24]:
        df[f'{target}_MA_{window}'] = df[target].rolling(window=window, min_periods=1).mean()

horizons = [1, 6, 12, 24]
for target in available_targets:
    for h in horizons:
        df[f'{target}_future_{h}'] = df[target].shift(-h)

# Delete NaN caused by shift/rolling
df_clean = df.dropna().copy()
print(f"Data shape after feature engineering: {df_clean.shape}")

for target in available_targets:
    print(f"{target}: {df_clean[target].notna().sum()} samples after cleaning")

# 3. Time Series Split (Train: 2004, Test: 2005)
print("\n" + "=" * 60)
print("3. Time Series Split")
print("=" * 60)

#try Date segmentation(2005), no success(last 20% as test)
train_df = df_clean[df_clean['Datetime'] < '2005-01-01']
test_df = df_clean[df_clean['Datetime'] >= '2005-01-01']

if test_df.shape[0] == 0:
    print("No data >= 2005-01-01 found. Falling back to a time-based ratio split (last 20% as test).")
    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx].copy()
    test_df = df_clean.iloc[split_idx:].copy()

print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# 4. Neural Network Regression for Multi-Pollutant, Multi-Horizon
print("\n" + "=" * 60)
print("4. Multi-Pollutant Multi-Step Prediction using MLPRegressor (Neural Network)")
print("=" * 60)

all_results = {}

for target in available_targets:
    print(f"\n--- Predicting {target} ---")
    pollutant_results = {}
    base_time_feats = ['Hour', 'DayOfWeek', 'Month', 'IsWeekend']

    feat_for_this_target = []

    for col in base_time_feats:
        if col in df_clean.columns:
            feat_for_this_target.append(col)

    # current pollution lag & MA & scaler
    feat_for_this_target.append(target)
    for col in df_clean.columns:
        if col.startswith(f'{target}_lag_') or col.startswith(f'{target}_MA_'):
            feat_for_this_target.append(col)

    feat_for_this_target = list(dict.fromkeys(feat_for_this_target))  # Deduplication insurance
    print(f"  Using {len(feat_for_this_target)} features for {target}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feat_for_this_target])
    X_test  = scaler.transform(test_df[feat_for_this_target])

    for h in horizons:
        print(f"  Horizon: {h} hours ahead")

        y_train = train_df[f'{target}_future_{h}'].values
        y_test  = test_df[f'{target}_future_{h}'].values

        # Naive baseline
        naive_pred = test_df[target].values
        naive_rmse = root_mean_squared_error(y_test, naive_pred)
        naive_mae  = mean_absolute_error(y_test, naive_pred)
        naive_r2   = r2_score(y_test, naive_pred)

        # MLPRegressor 
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=1e-3,
            alpha=1e-3,
            batch_size=64,
            max_iter=400,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.15
        )

        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)

        nn_rmse = root_mean_squared_error(y_test, y_pred)
        nn_mae  = mean_absolute_error(y_test, y_pred)
        nn_r2   = r2_score(y_test, y_pred)

        pollutant_results[f'{h}h'] = {
            'Naive_RMSE': naive_rmse,
            'Naive_MAE':  naive_mae,
            'Naive_R2':   naive_r2,
            'NN_RMSE':    nn_rmse,
            'NN_MAE':     nn_mae,
            'NN_R2':      nn_r2,
            'y_test':     y_test,
            'y_pred':     y_pred,
            'test_times': test_df['Datetime'].values
        }

        print(f"    Naive   - RMSE: {naive_rmse:.4f}, MAE: {naive_mae:.4f}, R²: {naive_r2:.4f}")
        print(f"    MLP NN  - RMSE: {nn_rmse:.4f}, MAE: {nn_mae:.4f}, R²: {nn_r2:.4f}")

    all_results[target] = pollutant_results

# 5. Result Summary and Visualization
print("\n" + "=" * 60)
print("5. Results Summary and Visualization")
print("=" * 60)

summary_rows = []
for target in available_targets:
    for h in horizons:
        if f'{h}h' in all_results[target]:
            res = all_results[target][f'{h}h']
            summary_rows.append({
                'Pollutant': target,
                'Horizon': f'{h}h',
                'Naive_RMSE': res['Naive_RMSE'],
                'NN_RMSE': res['NN_RMSE'],
                'RMSE_Improvement': res['Naive_RMSE'] - res['NN_RMSE'],
                'NN_R2': res['NN_R2']
            })

summary_df = pd.DataFrame(summary_rows)
print("\nNeural Network Regression Performance Summary:")
print(summary_df.round(4))

summary_df.to_csv('NN_Pollutant_Prediction_Results.csv', index=False)
print("\nResults saved to 'NN_Pollutant_Prediction_Results.csv'")

# pollution RMSE plot
if not summary_df.empty:
    pollutants = summary_df['Pollutant'].unique()
    n_pollutants = len(pollutants)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, pollutant in enumerate(pollutants):
        if i >= len(axes):
            break
        sub = summary_df[summary_df['Pollutant'] == pollutant]
        horizons_list = sub['Horizon'].tolist()
        naive_vals = sub['Naive_RMSE'].tolist()
        nn_vals = sub['NN_RMSE'].tolist()

        x = np.arange(len(horizons_list))
        width = 0.35

        axes[i].bar(x - width/2, naive_vals, width, label='Naive', alpha=0.7)
        axes[i].bar(x + width/2, nn_vals, width, label='MLP NN', alpha=0.7)

        axes[i].set_title(f'{pollutant} RMSE Comparison')
        axes[i].set_xlabel('Prediction Horizon')
        axes[i].set_ylabel('RMSE')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(horizons_list)
        axes[i].legend()

    # Hide redundant subgraphs
    for j in range(n_pollutants, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

#Residual plot & Prediction vs Actual Comparison Chart
plots_dir = 'nn_plots'
os.makedirs(plots_dir, exist_ok=True)

for target, res_dict in all_results.items():
    if not res_dict:
        continue

    # Residual plot（4 horizon）
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for i, h in enumerate(horizons):
        key = f'{h}h'
        ax = axes[i]
        if key not in res_dict:
            ax.set_visible(False)
            continue
        y_test = np.array(res_dict[key]['y_test'])
        y_pred = np.array(res_dict[key]['y_pred'])
        residual = y_test - y_pred
        ax.scatter(y_pred, residual, s=8, alpha=0.6)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_title(f'{target} Residuals ({h}h)')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residual (True - Pred)')
    plt.tight_layout()
    fn = os.path.join(plots_dir, f'residuals_{target.replace("/","_")}.png')
    fig.savefig(fn, dpi=150)
    plt.close(fig)

    # Prediction vs Actual Comparison Chart（4 horizon）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, h in enumerate(horizons):
        key = f'{h}h'
        ax = axes[i]
        if key not in res_dict:
            ax.set_visible(False)
            continue
        times = np.array(res_dict[key]['test_times'])
        y_test = np.array(res_dict[key]['y_test'])
        y_pred = np.array(res_dict[key]['y_pred'])
        ax.plot(times, y_test, label='True', alpha=0.8)
        ax.plot(times, y_pred, label='Pred', alpha=0.8)
        ax.set_title(f'{target} True vs Pred ({h}h)')
        ax.set_xlabel('Datetime')
        ax.set_ylabel(target)
        ax.legend()
    plt.tight_layout()
    fn2 = os.path.join(plots_dir, f'comparison_{target.replace("/","_")}.png')
    fig.savefig(fn2, dpi=150)
    plt.close(fig)

    print(f"Saved plots for {target}: {os.path.basename(fn)}, {os.path.basename(fn2)}")

print("\nNeural Network Regressor script completed successfully.")
