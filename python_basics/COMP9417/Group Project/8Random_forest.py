
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline


# data preprocess: ensure timestamp
def ensure_datetime(df):
    df = df.copy()

    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    elif ("Date" in df.columns) and ("Time" in df.columns):
        df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
    else:
        raise ValueError("Datetime or Date/Time columns are missing")

    return df


# data preprocess: build time feature
def add_time_features(df):
    df = ensure_datetime(df)
    df["hour"] = df["Datetime"].dt.hour
    df["weekday"] = df["Datetime"].dt.weekday
    df["month"] = df["Datetime"].dt.month
    df["year"] = df["Datetime"].dt.year
    return df


# discretise CO concentration values into low-mid-high
def discretise_co(v):
    if pd.isna(v):
        return np.nan
    if v < 1.5:
        return 0
    elif v < 2.5:
        return 1
    else:
        return 2


# build regression dataset and split train-test set
# with time feature, lag feature, rolling mean feature and temperature-humidity interaction
def make_regression_dataset(df, target_col, horizon):
    df = df.copy()
    df = add_time_features(df)

    if "T" in df.columns and "RH" in df.columns:
        df["T_RH"] = df["T"] * df["RH"]

    df["y"] = df[target_col].shift(-horizon)

    lag_list = [1, 6, 12, 24]
    lag_cols = []
    for lag in lag_list:
        col_name = f"{target_col}_lag_{lag}"
        df[col_name] = df[target_col].shift(lag)
        lag_cols.append(col_name)

    roll_col = f"{target_col}_roll6"
    df[roll_col] = df[target_col].rolling(window=6, min_periods=1).mean()

    df = df.dropna(subset=["y"] + lag_cols)

    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols.remove("y")

    X = df[feature_cols]
    y = df["y"]
    years = df["year"].values
    times = df["Datetime"].values

    train_mask = years == 2004
    test_mask = years == 2005

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    time_test = times[test_mask]

    return X_train, y_train, X_test, y_test, time_test, df, feature_cols


# build classification dataset and split train-test set
# with time feature, lag feature, rolling mean feature and temperature-humidity interaction
def make_classification_dataset(df, horizon):
    df = df.copy()
    df = add_time_features(df)

    if "T" in df.columns and "RH" in df.columns:
        df["T_RH"] = df["T"] * df["RH"]

    df["CO_lag_1"] = df["CO(GT)"].shift(1)
    df["CO_lag_6"] = df["CO(GT)"].shift(6)

    df["CO_roll6"] = df["CO(GT)"].rolling(window=6, min_periods=1).mean()

    co_future = df["CO(GT)"].shift(-horizon)

    mask = ~co_future.isna()
    df = df[mask].copy()
    co_future = co_future[mask]

    df["y_class"] = co_future.apply(discretise_co)

    df = df.dropna(subset=["y_class", "CO_lag_1", "CO_lag_6"])

    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols.remove("y_class")

    X = df[feature_cols]
    y = df["y_class"].astype(int)
    years = df["year"].values
    times = df["Datetime"].values

    train_mask = years == 2004
    test_mask = years == 2005

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    time_test = times[test_mask]

    return X_train, y_train, X_test, y_test, time_test, df, feature_cols


# show confusion matrix
def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black" if cm[i, j] > cm.max()/2 else "white")

    plt.tight_layout()
    plt.show()


# show side-by-side bar charts 
# compare accuracy and F1 score of random forest and naive baseline in t+1, t+6, t+12, t+24
def plot_classification_over_time(horizons, acc_rf, acc_base, f1_rf, f1_base):
    x_labels = [f"t+{h}" for h in horizons]
    x = np.arange(len(horizons))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x + width / 2, acc_base, width, label="Naive")
    plt.bar(x - width / 2, acc_rf, width,label="RandomForest")
    plt.xticks(x, x_labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("classification accuracy overtime")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(x + width / 2, f1_base, width, label="Naive")
    plt.bar(x - width / 2, f1_rf, width, label="RandomForest")
    
    plt.xticks(x, x_labels)
    plt.ylim(0, 1.0)
    plt.ylabel("F1-score")
    plt.title("classification F1score overtime")
    plt.legend()
    plt.tight_layout()
    plt.show()



# show RMSE figure for all pollutants and all horizon
def plot_all_targets_rmse_grid(regress_targets, horizons, reg_rmse_rf, reg_rmse_base):
    n_targets = len(regress_targets)
    ncols = 3
    nrows = int(np.ceil(n_targets / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False)
    axes = axes.flatten()

    x = np.arange(len(horizons))
    width = 0.35
    x_labels = [f"t+{h}" for h in horizons]

    for i, target in enumerate(regress_targets):
        ax = axes[i]
        rf_rmse = reg_rmse_rf[target]
        base_rmse = reg_rmse_base[target]

        ax.bar(x - width / 2, base_rmse, width, label="Naive Baseline")
        ax.bar(x + width / 2, rf_rmse, width, label="RandomForest")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("RMSE")
        ax.set_title(f"{target} Prediction Performance")

        if i == 0:
            ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# plot true - predcited comparison
def plot_target_timeseries_grid(target, horizons, time_dict, y_test_dict, y_pred_dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, h in enumerate(horizons):
        time_h = time_dict[target].get(h)
        y_test_h = y_test_dict[target].get(h)
        y_pred_h = y_pred_dict[target].get(h)
        if time_h is None:
            continue

        ax = axes[i]
        ax.plot(time_h, y_test_h, label="True", linewidth=1)
        ax.plot(time_h, y_pred_h, label="Predicted", linewidth=1, alpha=0.7)
        ax.set_title(f"{target} {h}h: True vs Predicted")
        ax.set_ylabel("Concentration")

        if i == 0:
            ax.legend()

    axes[-2].set_xlabel("Time")
    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Concentration Comparison for {target}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



# plot residual figure for all horizon
def plot_target_residuals_grid(target, horizons, time_dict, y_test_dict, y_pred_dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, h in enumerate(horizons):
        time_h = time_dict[target].get(h)
        y_test_h = y_test_dict[target].get(h)
        y_pred_h = y_pred_dict[target].get(h)
        if time_h is None:
            continue

        residuals = y_test_h - y_pred_h

        ax = axes[i]
        ax.scatter(time_h, residuals, s=5)
        ax.axhline(0, color="red", linewidth=1)
        ax.set_title(f"{target} residuals ({h}h)")
        ax.set_ylabel("Residual")

    axes[-2].set_xlabel("Time")
    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Residuals for {target}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# standardization and polynomial feature expansion with degree=2
def build_feature_transformer():
    transformer = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ]
    )
    return transformer


def main():
    # load preprocessed data
    file_path = Path(__file__).resolve().parent / "processed_air_quality.csv"
    df = pd.read_csv(file_path)

    print("Loaded preprocessed data:")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    regress_targets = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
    horizons = [1, 6, 12, 24]

    reg_rmse_rf = {t: [] for t in regress_targets}
    reg_rmse_base = {t: [] for t in regress_targets}

    reg_time = {t: {h: None for h in horizons} for t in regress_targets}
    reg_y_test = {t: {h: None for h in horizons} for t in regress_targets}
    reg_y_pred = {t: {h: None for h in horizons} for t in regress_targets}

    print("\nRandom forest regression")

    # regression loop
    for target in regress_targets:
        print(f"\nTarget: {target}")
        for h in horizons:
            print(f"\nHorizon = {h} hour(s) ahead")

            (
                X_train,
                y_train,
                X_test,
                y_test,
                time_test,
                df_full,
                feature_cols,
            ) = make_regression_dataset(df, target, horizon=h)

            transformer = build_feature_transformer()
            X_train_tf = transformer.fit_transform(X_train)
            X_test_tf = transformer.transform(X_test)

            rf_reg = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            )
            rf_reg.fit(X_train_tf, y_train)

            y_pred_train = rf_reg.predict(X_train_tf)
            y_pred_test = rf_reg.predict(X_test_tf)

            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

            df_tmp = add_time_features(df).copy()
            df_tmp["y_future"] = df_tmp[target].shift(-h)
            df_tmp = df_tmp.dropna(subset=["y_future"])

            years_all = df_tmp["year"].values
            baseline_now = df_tmp[target].values
            baseline_future = df_tmp["y_future"].values

            test_mask = years_all == 2005
            baseline_pred_test = baseline_now[test_mask]
            baseline_true_test = baseline_future[test_mask]

            baseline_rmse = np.sqrt(
                mean_squared_error(baseline_true_test, baseline_pred_test)
            )

            print(f"RandomForest RMSE (train): {rmse_train:.3f}")
            print(f"RandomForest RMSE (test) : {rmse_test:.3f}")
            print(f"Naive baseline RMSE (test): {baseline_rmse:.3f}")

            reg_rmse_rf[target].append(rmse_test)
            reg_rmse_base[target].append(baseline_rmse)

            reg_time[target][h] = time_test
            reg_y_test[target][h] = y_test
            reg_y_pred[target][h] = y_pred_test

    print("\nRandom forest classification")
    
    # classification loop 
    print("Target:CO(GT)")

    class_names = ["low", "mid", "high"]

    acc_rf_list = []
    f1_rf_list = []
    acc_base_list = []
    f1_base_list = []
    cm_for_plot = None

    for h in horizons:
        print(f"\nHorizon = {h} hour(s) ahead")

        (
            X_train,
            y_train,
            X_test,
            y_test,
            time_test,
            df_full,
            feature_cols,
        ) = make_classification_dataset(df, horizon=h)

        transformer = build_feature_transformer()
        X_train_tf = transformer.fit_transform(X_train)
        X_test_tf = transformer.transform(X_test)

        rf_clf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        rf_clf.fit(X_train_tf, y_train)

        y_pred_train = rf_clf.predict(X_train_tf)
        y_pred_test = rf_clf.predict(X_test_tf)

        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        f1_train = f1_score(y_train, y_pred_train, average="macro")
        f1_test = f1_score(y_test, y_pred_test, average="macro")

        print(f"[H={h}] RandomForest ACC (train): {acc_train:.3f}")
        print(f"[H={h}] RandomForest ACC (test) : {acc_test:.3f}")
        print(f"[H={h}] RandomForest F1  (train): {f1_train:.3f}")
        print(f"[H={h}] RandomForest F1  (test) : {f1_test:.3f}")

        acc_rf_list.append(acc_test)
        f1_rf_list.append(f1_test)

        cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2])
        print(f"\n[H={h}] Classification report:")
        print(classification_report(y_test, y_pred_test, target_names=class_names))

        if h == horizons[0]:
            cm_for_plot = cm

        df_tmp = add_time_features(df).copy()
        df_tmp["co_now_cls"] = df_tmp["CO(GT)"].apply(discretise_co)
        df_tmp["co_future"] = df_tmp["CO(GT)"].shift(-h)
        df_tmp["co_future_cls"] = df_tmp["co_future"].apply(discretise_co)
        df_tmp = df_tmp.dropna(subset=["co_now_cls", "co_future_cls"])

        years_all = df_tmp["year"].values
        test_mask = years_all == 2005

        baseline_y_true = df_tmp.loc[test_mask, "co_future_cls"].astype(int)
        baseline_y_pred = df_tmp.loc[test_mask, "co_now_cls"].astype(int)

        baseline_acc = accuracy_score(baseline_y_true, baseline_y_pred)
        baseline_f1 = f1_score(baseline_y_true, baseline_y_pred, average="macro")

        print(f"[H={h}] Naive baseline ACC (test): {baseline_acc:.3f}")
        print(f"[H={h}] Naive baseline F1  (test): {baseline_f1:.3f}")

        acc_base_list.append(baseline_acc)
        f1_base_list.append(baseline_f1)


    # visualization
    target_order = ["C6H6(GT)", "CO(GT)", "NMHC(GT)", "NOx(GT)", "NO2(GT)"]
    plot_all_targets_rmse_grid(target_order, horizons, reg_rmse_rf, reg_rmse_base)

    
    for target in target_order:
        plot_target_timeseries_grid(target, horizons, reg_time, reg_y_test, reg_y_pred)

    
    for target in target_order:
        plot_target_residuals_grid(target, horizons, reg_time, reg_y_test, reg_y_pred)

    
    plot_classification_over_time(
        horizons,
        acc_rf_list,
        acc_base_list,
        f1_rf_list,
        f1_base_list,
    )

    if cm_for_plot is not None:
        plot_confusion_matrix(
            cm_for_plot,
            class_names,
            title="classification confusion matrix",
        )


if __name__ == "__main__":
    main()
