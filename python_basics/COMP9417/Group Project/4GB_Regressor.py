from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
plt.style.use("default")

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  
    def root_mean_squared_error(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# data preprocess: ensure timestamp
def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    elif ("Date" in df.columns) and ("Time" in df.columns):
        df["Datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"], errors="coerce"
        )
    else:
        raise ValueError("Datetime or Date/Time columns are missing")
    return df

# data preprocess: build time feature
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_datetime(df)
    df["hour"] = df["Datetime"].dt.hour
    df["weekday"] = df["Datetime"].dt.weekday
    df["month"] = df["Datetime"].dt.month
    df["year"] = df["Datetime"].dt.year
    return df


def make_regression_dataset(df: pd.DataFrame, target_col: str, horizon: int):
    df = df.copy()
    df = add_time_features(df)

    # temperature–humidity interaction (if present)
    if "T" in df.columns and "RH" in df.columns:
        df["T_RH"] = df["T"] * df["RH"]

    # future value as regression target
    df["y"] = df[target_col].shift(-horizon)

    # lag features
    lag_list = [1, 6, 12, 24]
    lag_cols = []
    for lag in lag_list:
        col_name = f"{target_col}_lag_{lag}"
        df[col_name] = df[target_col].shift(lag)
        lag_cols.append(col_name)

    # simple rolling mean feature
    roll_col = f"{target_col}_roll6"
    df[roll_col] = df[target_col].rolling(window=6, min_periods=1).mean()

    # drop rows without valid target/lag data
    df = df.dropna(subset=["y"] + lag_cols)

    # numeric features except the target y
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "y" in feature_cols:
        feature_cols.remove("y")

    X = df[feature_cols].values
    y = df["y"].values
    years = df["year"].values
    times = df["Datetime"].values

    # Chronological split, 2004 for training data and 2005 for testing data mentioned in instruction
    train_mask = years == 2004
    test_mask = years == 2005

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    time_test = times[test_mask]

    return X_train, y_train, X_test, y_test, time_test, df, feature_cols


# gradient boosting evaluation for pollutant horizon pair
def evaluate_gb_for_target(df: pd.DataFrame, target: str, horizon: int):
    (
        X_train,
        y_train,
        X_test,
        y_test,
        time_test,
        df_full,
        feature_cols,
    ) = make_regression_dataset(df, target_col=target, horizon=horizon)

    n_samples = X_train.shape[0]
    n_feats = X_train.shape[1]
    print(f"[GB] Target={target}, horizon={horizon}h, samples={n_samples}, features={n_feats}")

    # skip if there are few saples
    if n_samples < 100:
        print("  -> too few training samples, skip")
        return None

    # construct a future target and evaluate the persistence model for 2005
    df_tmp = add_time_features(df_full).copy()
    df_tmp["y_future"] = df_tmp[target].shift(-horizon)
    df_tmp = df_tmp.dropna(subset=["y_future"])

    years_all = df_tmp["year"].values
    baseline_now = df_tmp[target].values
    baseline_future = df_tmp["y_future"].values

    test_mask = years_all == 2005
    baseline_pred_test = baseline_now[test_mask]
    baseline_true_test = baseline_future[test_mask]

    naive_rmse = root_mean_squared_error(baseline_true_test, baseline_pred_test)
    naive_mae = mean_absolute_error(baseline_true_test, baseline_pred_test)
    try:
        naive_r2 = r2_score(baseline_true_test, baseline_pred_test)
    except Exception:
        naive_r2 = np.nan

    # choose a GB model with 100 trees, 0.1 learning rate and 4 depth
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        max_features="sqrt",
        random_state=42,
        loss="huber",
    )

    try:
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
    except Exception as e:
        print(f"  Model training failed: {e}")
        return None

    # eavaluate GB on test data
    gb_rmse = root_mean_squared_error(y_test, y_pred_test)
    gb_mae = mean_absolute_error(y_test, y_pred_test)
    try:
        gb_r2 = r2_score(y_test, y_pred_test)
    except Exception:
        gb_r2 = np.nan

    print(
        f"  -> GB test RMSE = {gb_rmse:.4f}, R2 = {gb_r2:.4f} "
        f"(naive RMSE = {naive_rmse:.4f})"
    )

    results = {
        "Naive_RMSE": naive_rmse,
        "Naive_MAE": naive_mae,
        "Naive_R2": naive_r2,
        "GB_RMSE": gb_rmse,
        "GB_MAE": gb_mae,
        "GB_R2": gb_r2,
    }

    return {
        "target": target,
        "horizon": horizon,
        "results": results,
        "time_test": time_test,
        "y_test": y_test,
        "y_pred": y_pred_test,
    }


def main():
    # location of processed data
    file_path = Path(__file__).resolve().parent / "processed_air_quality.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find {file_path}")

    df = pd.read_csv(file_path)
    print("Loaded preprocessed data from:", file_path)
    print("Data shape:", df.shape)

    # target pollutants
    regress_targets = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
    available_targets = [t for t in regress_targets if t in df.columns]
    print("Available targets:", available_targets)

    horizons = [1, 6, 12, 24]

    summary_rows = []
    predictions_store = {}

    # Build task list and run in parallel across cores
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

    tasks = [(t, h) for t in available_targets for h in horizons]
    print(f"Running {len(tasks)} tasks in parallel on all available cores...")
    parallel_results = Parallel(n_jobs=-1, backend='loky')(
        delayed(evaluate_gb_for_target)(df, target=t, horizon=h) for t, h in tasks
    )

    # collect prediction results
    for out in parallel_results:
        if out is None:
            continue
        target = out['target']
        h = out['horizon']
        res = out['results']
        summary_rows.append({
            'Pollutant': target,
            'Horizon': f"{h}h",
            'Naive_RMSE': res['Naive_RMSE'],
            'GB_RMSE': res['GB_RMSE'],
            'Improvement': res['Naive_RMSE'] - res['GB_RMSE'],
            'GB_R2': res['GB_R2'],
        })

        pred_dict = predictions_store.setdefault(target, {})
        pred_dict[h] = {'time': out['time_test'], 'y_test': out['y_test'], 'y_pred': out['y_pred']}

    if not summary_rows:
        print("No successful GB runs.")
        return

    # create a dataframe for summary
    summary_df = pd.DataFrame(summary_rows)
    print("\nPerformance Summary (Gradient Boosting):")
    print(summary_df.round(4))

    # save result in csv
    out_csv = Path("GB_Pollutant_Prediction_Results.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv.resolve()}")

    # quick bar plots of RMSE vs naive baseline
    try:
        pollutants = summary_df["Pollutant"].unique()
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for i, pollutant in enumerate(pollutants):
            if i >= len(axes):
                break
            ax = axes[i]
            sub = summary_df[summary_df["Pollutant"] == pollutant]
            x = np.arange(len(sub))
            ax.bar(x - 0.2, sub["Naive_RMSE"], width=0.4, label="Naive")
            ax.bar(x + 0.2, sub["GB_RMSE"], width=0.4, label="GB")
            ax.set_xticks(x)
            ax.set_xticklabels(sub["Horizon"])
            ax.set_title(pollutant)
            ax.set_ylabel("RMSE")
            ax.legend()

        # hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plotting failed: {e}")

    # generate prediction and residual plots
    try:
        print("\nGenerating residual and true-vs-prediction plots...")
        import matplotlib.dates as mdates

        out_dir = Path(__file__).resolve().parent / "figures_gb"
        out_dir.mkdir(parents=True, exist_ok=True)

        for pollutant, horiz_dict in predictions_store.items():
            # Residuals plot (2x2 for horizons 1,6,12,24)
            fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
            axes = axes.flatten()
            for idx, h in enumerate([1, 6, 12, 24]):
                ax = axes[idx]
                if h not in horiz_dict:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(f"{h}h")
                    continue
                rec = horiz_dict[h]
                times = pd.to_datetime(rec['time'])
                y_test = np.array(rec['y_test'], dtype=float)
                y_pred = np.array(rec['y_pred'], dtype=float)
                residuals = y_test - y_pred

                ax.plot(times, residuals, marker='.', linestyle='None', alpha=0.6)
                ax.axhline(0, color='red', linewidth=0.8)
                ax.set_title(f"{h}h")
                ax.set_ylabel('Residual')
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

            plt.suptitle(f"GB - Residuals for {pollutant}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            outname = out_dir / f"residuals_{pollutant.replace('/','_').replace('(','').replace(')','')}.png"
            fig.savefig(str(outname), dpi=150)
            plt.close(fig)
            print(f"Saved residual plot: {outname}")

            # True vs Predicted plot (downsample for plotting)
            fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
            axes = axes.flatten()
            for idx, h in enumerate([1, 6, 12, 24]):
                ax = axes[idx]
                if h not in horiz_dict:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(f"{h}h")
                    continue
                rec = horiz_dict[h]
                times = pd.to_datetime(rec['time'])
                y_test = np.array(rec['y_test'], dtype=float)
                y_pred = np.array(rec['y_pred'], dtype=float)

                # downsample if series is long
                max_points = 1000
                step = max(1, len(times) // max_points)
                times_s = times[::step]
                times_s = times[::step]
                y_test_s = y_test[::step]
                y_pred_s = y_pred[::step]

                ax.plot(times_s, y_test_s, label='True', linewidth=1)
                ax.plot(times_s, y_pred_s, label='Predicted', linewidth=1, alpha=0.8)
                ax.set_title(f"{h}h")
                ax.set_ylabel('Concentration')
                ax.legend()

            plt.suptitle(f"GB - Concentration Comparison for {pollutant}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            outname = out_dir / f"comparison_{pollutant.replace('/','_').replace('(','').replace(')','')}.png"
            fig.savefig(str(outname), dpi=150)
            plt.close(fig)
            print(f"Saved comparison plot: {outname}")

        print("Done generating prediction/residual plots.")
    except Exception as e:
        print(f"Failed to generate prediction/residual plots: {e}")


if __name__ == "__main__":
    main()
