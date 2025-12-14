#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Model choose: Random Forest

# import relevant libraries
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
    plt.tight_layout()
    plt.show()


# show side-by-side bar charts 
# compare accuracy and F1 score of random forest and naive baseline in t+1, t+6, t+12, t+24
def plot_classification_over_time(horizons, acc_rf, acc_base, f1_rf, f1_base):
    x_labels = [f"t+{h}" for h in horizons]
    x = np.arange(len(horizons))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, acc_rf, width, label="RandomForest")
    plt.bar(x + width / 2, acc_base, width, label="Naive")
    plt.xticks(x, x_labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("classification accuracy overtime")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, f1_rf, width, label="RandomForest")
    plt.bar(x + width / 2, f1_base, width, label="Naive")
    plt.xticks(x, x_labels)
    plt.ylim(0, 1.0)
    plt.ylabel("F1-score")
    plt.title("classification F1score overtime")
    plt.legend()
    plt.tight_layout()
    plt.show()


# make a three-in-one chart for each pollutant
# first subplot: compare RMSE between random forest and baseline overtime
# second subplot: time series plot 
def plot_regression_summary_for_target(
    target,
    horizons,
    rmse_rf_list,
    rmse_base_list,
    time_dict,
    y_test_dict,
    y_pred_dict,
):
    title_name = target.replace("(GT)", "").strip()

    # RMSE figure
    x_labels = [f"t+{h}" for h in horizons]
    x = np.arange(len(horizons))
    width = 0.35

    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, rmse_rf_list, width, label="RandomForest")
    plt.bar(x + width / 2, rmse_base_list, width, label="Naive")
    plt.xticks(x, x_labels)
    plt.ylabel("RMSE")
    plt.title(f"{title_name} RMSE vs baseline")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # time-series figures for each horizon
    for h in horizons:
        time_h = time_dict.get(h)
        y_test_h = y_test_dict.get(h)
        y_pred_h = y_pred_dict.get(h)
        if (time_h is None) or (y_test_h is None) or (y_pred_h is None):
            continue

        plt.figure(figsize=(10, 4))
        plt.plot(time_h, y_test_h, label="True")
        plt.plot(time_h, y_pred_h, label="Predicted", alpha=0.7)
        plt.ylabel("Concentration")
        plt.title(f"{title_name} Predicted vs True over time (t+{h})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # residual figure for t+1
    h_res = horizons[0]
    time_h1 = time_dict.get(h_res)
    y_test_h1 = y_test_dict.get(h_res)
    y_pred_h1 = y_pred_dict.get(h_res)
    if (time_h1 is not None) and (y_test_h1 is not None) and (y_pred_h1 is not None):
        residuals = y_test_h1 - y_pred_h1
        plt.figure(figsize=(10, 4))
        plt.plot(time_h1, residuals)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Time")
        plt.ylabel("Residual")
        plt.title(f"{title_name} Residuals over time (t+{h_res})")
        plt.tight_layout()
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
    """
    Main workflow:
      1) Load preprocessed data
      2) Build regression datasets and train RandomForest with transformed features
      3) Build classification datasets and train RandomForest with transformed features
      4) Generate required plots
    """
    
    # load preprocessed data
    df = pd.read_csv("processed_air_quality.csv")

    print("Loaded preprocessed data:")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    regress_targets = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
    horizons = [1, 6, 12, 24]

    reg_rmse_rf = {t: [] for t in regress_targets}
    reg_rmse_base = {t: [] for t in regress_targets}
    # 存各个 horizon 的时间 / 真值 / 预测，用于画 4 张趋势图 + 1 张残差图
    reg_time = {t: {h: None for h in horizons} for t in regress_targets}
    reg_y_test = {t: {h: None for h in horizons} for t in regress_targets}
    reg_y_pred = {t: {h: None for h in horizons} for t in regress_targets}

    print("\n========== Regression (Random Forest) ==========")

    # regression loop
    # outer loop: iterates 5 pollutants
    # inner loop: iterates 4 time windows
    # every loop: build dataset, process feature transformation, train random forest regrassor, calculate RMSE
    for target in regress_targets:
        print(f"\n>>> Target: {target}")
        for h in horizons:
            print(f"\n--- Horizon = {h} hour(s) ahead ---")

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

            # 保存对应 horizon 的时间 / 真值 / 预测，用于画 4 张趋势图和 1 张残差图
            reg_time[target][h] = time_test
            reg_y_test[target][h] = y_test
            reg_y_pred[target][h] = y_pred_test

    print("\n========== Classification (Random Forest) ==========")
    
    # classification loop
    # traversal 4 time windows
    # every loop: build dataset, process feature transformation, train random forest classifier, calculate accuracy and F1 score
    print("Target: discretised CO(GT) -> low / mid / high")

    class_names = ["low", "mid", "high"]

    acc_rf_list = []
    f1_rf_list = []
    acc_base_list = []
    f1_base_list = []
    cm_for_plot = None

    for h in horizons:
        print(f"\n--- Horizon = {h} hour(s) ahead ---")

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

    target_order = ["C6H6(GT)", "CO(GT)", "NMHC(GT)", "NOx(GT)", "NO2(GT)"]
    for target in target_order:
        plot_regression_summary_for_target(
            target=target,
            horizons=horizons,
            rmse_rf_list=reg_rmse_rf[target],
            rmse_base_list=reg_rmse_base[target],
            time_dict=reg_time[target],
            y_test_dict=reg_y_test[target],
            y_pred_dict=reg_y_pred[target],
        )

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
