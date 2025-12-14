from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATA_FILE = Path(__file__).resolve().parent / "AirQualityUCI.csv"
PROCESSED_DATA_FILE = Path(__file__).resolve().parent / "processed_air_quality.csv"
ORIGINAL_COLUMNS = None

# Pollutant columns that should not be negative
POLLUTANT_COLS = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
SENSOR_COLS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]

# Physical constraints
TEMP_MIN, TEMP_MAX = -20, 50  # Temperature range in Celsius
RH_MIN, RH_MAX = 0, 100  # Relative humidity in percentage
AH_MIN = 0  # Absolute humidity should be positive

# Outlier detection method: IQR multiplier
IQR_MULTIPLIER = 1.5


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_raw_data():
    """Load raw Air Quality dataset from CSV."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    df = pd.read_csv(
        DATA_FILE,
        sep=";",
        decimal=",",
        engine="python",
    )

    df = df.rename(columns=lambda c: c.strip())
    df = df.drop(columns=[col for col in df.columns if "Unnamed" in col], errors="ignore")

    if "Date" not in df.columns or "Time" not in df.columns:
        raise ValueError("Original data missing Date or Time columns, cannot create timestamp.")

    return df


# -----------------------------------------------------------------------------
# Missing Value Handling
# -----------------------------------------------------------------------------

def handle_missing_values(df):
    """Replace sentinel values (-200) with NaN and create datetime index."""
    df = df.copy()

    global ORIGINAL_COLUMNS
    if ORIGINAL_COLUMNS is None:
        ORIGINAL_COLUMNS = df.columns.tolist()

    # Create datetime index
    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce",
    )
    df = df.drop(columns=["Date", "Time"])
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()

    # Replace -200 sentinel values with NaN
    numeric_cols = df.columns
    df[numeric_cols] = df[numeric_cols].replace(-200, np.nan)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    return df


# -----------------------------------------------------------------------------
# Outlier Detection
# -----------------------------------------------------------------------------

def detect_negative_values(df, columns):
    """Detect negative values in specified columns."""
    outliers = {}
    for col in columns:
        if col in df.columns:
            negative_mask = df[col] < 0
            negative_count = negative_mask.sum()
            if negative_count > 0:
                outliers[col] = {
                    "count": negative_count,
                    "indices": df[negative_mask].index,
                    "values": df.loc[negative_mask, col].values,
                }
    return outliers


def detect_iqr_outliers(df, columns):
    """Detect outliers using IQR method."""
    outliers = {}
    for col in columns:
        if col not in df.columns:
            continue
        if df[col].notna().sum() < 10:  # Need at least 10 values
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:  # Skip if no variance
            continue

        lower_bound = Q1 - IQR_MULTIPLIER * IQR
        upper_bound = Q3 + IQR_MULTIPLIER * IQR

        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            outliers[col] = {
                "count": outlier_count,
                "percentage": outlier_count / len(df) * 100,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "min_value": df[col].min(),
                "max_value": df[col].max(),
                "indices": df[outlier_mask].index,
            }
    return outliers


def detect_physical_constraints(df):
    """Detect values that violate physical constraints."""
    violations = {}

    # Temperature check
    if "T" in df.columns:
        temp_violations = (df["T"] < TEMP_MIN) | (df["T"] > TEMP_MAX)
        if temp_violations.sum() > 0:
            violations["T"] = {
                "count": temp_violations.sum(),
                "indices": df[temp_violations].index,
                "min": df["T"].min(),
                "max": df["T"].max(),
            }

    # Relative humidity check
    if "RH" in df.columns:
        rh_violations = (df["RH"] < RH_MIN) | (df["RH"] > RH_MAX)
        if rh_violations.sum() > 0:
            violations["RH"] = {
                "count": rh_violations.sum(),
                "indices": df[rh_violations].index,
                "min": df["RH"].min(),
                "max": df["RH"].max(),
            }

    # Absolute humidity check
    if "AH" in df.columns:
        ah_violations = df["AH"] < AH_MIN
        if ah_violations.sum() > 0:
            violations["AH"] = {
                "count": ah_violations.sum(),
                "indices": df[ah_violations].index,
            }

    return violations


# -----------------------------------------------------------------------------
# Outlier Treatment
# -----------------------------------------------------------------------------

def treat_negative_values(df, columns, method="clip"):
    """
    Treat negative values in specified columns.
    
    Parameters:
    -----------
    method : str
        'clip': clip to 0
        'remove': set to NaN
        'median': replace with median
    """
    df = df.copy()
    treated_count = 0

    for col in columns:
        if col not in df.columns:
            continue

        negative_mask = df[col] < 0
        negative_count = negative_mask.sum()

        if negative_count > 0:
            treated_count += negative_count
            if method == "clip":
                df.loc[negative_mask, col] = 0
            elif method == "remove":
                df.loc[negative_mask, col] = np.nan
            elif method == "median":
                median_val = df[col].median()
                if pd.notna(median_val):
                    df.loc[negative_mask, col] = median_val
                else:
                    df.loc[negative_mask, col] = 0

    if treated_count > 0:
        print(f"Treated {treated_count} negative values using method: {method}")

    return df


def treat_iqr_outliers(df, columns, method="clip"):
    """
    Treat IQR outliers in specified columns.
    
    Parameters:
    -----------
    method : str
        'clip': clip to bounds
        'remove': set to NaN
        'median': replace with median
    """
    df = df.copy()
    treated_count = 0

    for col in columns:
        if col not in df.columns:
            continue
        if df[col].notna().sum() < 10:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            continue

        lower_bound = Q1 - IQR_MULTIPLIER * IQR
        upper_bound = Q3 + IQR_MULTIPLIER * IQR

        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            treated_count += outlier_count
            if method == "clip":
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
            elif method == "remove":
                df.loc[outlier_mask, col] = np.nan
            elif method == "median":
                median_val = df[col].median()
                if pd.notna(median_val):
                    df.loc[outlier_mask, col] = median_val

    if treated_count > 0:
        print(f"Treated {treated_count} IQR outliers using method: {method}")

    return df


def treat_physical_constraints(df, method="clip"):
    """Treat values that violate physical constraints."""
    df = df.copy()
    treated_count = 0

    # Temperature
    if "T" in df.columns:
        temp_violations = (df["T"] < TEMP_MIN) | (df["T"] > TEMP_MAX)
        if temp_violations.sum() > 0:
            treated_count += temp_violations.sum()
            if method == "clip":
                df.loc[df["T"] < TEMP_MIN, "T"] = TEMP_MIN
                df.loc[df["T"] > TEMP_MAX, "T"] = TEMP_MAX
            elif method == "remove":
                df.loc[temp_violations, "T"] = np.nan

    # Relative humidity
    if "RH" in df.columns:
        rh_violations = (df["RH"] < RH_MIN) | (df["RH"] > RH_MAX)
        if rh_violations.sum() > 0:
            treated_count += rh_violations.sum()
            if method == "clip":
                df.loc[df["RH"] < RH_MIN, "RH"] = RH_MIN
                df.loc[df["RH"] > RH_MAX, "RH"] = RH_MAX
            elif method == "remove":
                df.loc[rh_violations, "RH"] = np.nan

    # Absolute humidity
    if "AH" in df.columns:
        ah_violations = df["AH"] < AH_MIN
        if ah_violations.sum() > 0:
            treated_count += ah_violations.sum()
            if method == "clip":
                df.loc[ah_violations, "AH"] = AH_MIN
            elif method == "remove":
                df.loc[ah_violations, "AH"] = np.nan

    if treated_count > 0:
        print(f"Treated {treated_count} physical constraint violations using method: {method}")

    return df


# -----------------------------------------------------------------------------
# Sensor drift handling (piecewise, daily-level smoothing)
# -----------------------------------------------------------------------------

def correct_sensor_drift(df, col, verbose=True):
    """
    Handle sensor drift for a given pollutant using a two-step strategy:
    1) Build a daily-level smoothed series (daily median).
    2) Detect the largest mean shift and apply a piecewise offset correction.
    """
    df = df.copy()

    if col not in df.columns:
        return df

    # Build daily median series (step 1: daily-level smoothing)
    daily = df[col].resample("D").median()
    daily = daily.dropna()

    # Need enough data to detect a stable change
    if len(daily) < 30:
        return df

    # Detect the largest day-to-day jump
    diff = daily.diff().abs()
    if diff.dropna().empty:
        return df

    change_point = diff.idxmax()

    # Split into before/after segments around the change point
    before = daily[daily.index < change_point]
    after = daily[daily.index >= change_point]

    # Require minimum days in each segment
    if len(before) < 14 or len(after) < 14:
        return df

    before_median = before.median()
    after_median = after.median()
    offset = after_median - before_median

    # Only correct if the shift is substantial
    apply_correction = False
    if before_median > 0:
        if abs(offset) > 0.5 * before_median:
            apply_correction = True
    else:
        if abs(offset) > 50:
            apply_correction = True

    if not apply_correction:
        return df

    # Apply piecewise offset correction on the original hourly series (step 2)
    mask_after = df.index >= change_point
    df.loc[mask_after, col] = df.loc[mask_after, col] - offset

    if verbose:
        print("\nStep 5b: Handling sensor drift for {}...".format(col))
        print(f"  Detected change point at: {change_point.date()}")
        print(f"  Daily median before: {before_median:.2f}, after: {after_median:.2f}")
        print(f"  Applied offset correction: {offset:.2f} to {col} for timestamps >= {change_point.date()}")

    return df


def correct_nox_sensor_drift(df, verbose=True):
    """Backward-compatible wrapper for NOx(GT) drift correction."""
    return correct_sensor_drift(df, "NOx(GT)", verbose=verbose)


# -----------------------------------------------------------------------------
# Feature Engineering
# -----------------------------------------------------------------------------

def add_time_features(df):
    """Append calendar-based features derived from the datetime index."""
    enriched = df.copy()

    # Original time features
    enriched["hour"] = enriched.index.hour
    enriched["weekday"] = enriched.index.weekday
    enriched["month"] = enriched.index.month
    enriched["is_weekend"] = (enriched.index.weekday >= 5).astype(int)

    # Cyclical encoding for time features (better for neural networks)
    # Hour: 0-23 -> sin/cos encoding
    enriched["hour_sin"] = np.sin(2 * np.pi * enriched["hour"] / 24)
    enriched["hour_cos"] = np.cos(2 * np.pi * enriched["hour"] / 24)

    # Weekday: 0-6 -> sin/cos encoding
    enriched["weekday_sin"] = np.sin(2 * np.pi * enriched["weekday"] / 7)
    enriched["weekday_cos"] = np.cos(2 * np.pi * enriched["weekday"] / 7)

    # Month: 1-12 -> sin/cos encoding
    enriched["month_sin"] = np.sin(2 * np.pi * enriched["month"] / 12)
    enriched["month_cos"] = np.cos(2 * np.pi * enriched["month"] / 12)

    return enriched


def add_rolling_features(df, columns, windows):
    """Add backward-looking rolling means for selected columns."""
    enriched = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for window in windows:
            enriched[f"{col}_rolling_mean_{window}h"] = (
                df[col].rolling(window=window, min_periods=1).mean()
            )
    return enriched


# -----------------------------------------------------------------------------
# Main Preprocessing Pipeline
# -----------------------------------------------------------------------------

def preprocess_air_quality_data(
    outlier_treatment="clip",
    iqr_treatment="clip",
    physical_treatment="clip",
    add_rolling=True,
    rolling_columns=None,
    rolling_windows=None,
    verbose=True,
):
    """
    Complete preprocessing pipeline for Air Quality dataset.
    
    Parameters:
    -----------
    outlier_treatment : str
        Method for treating negative values: 'clip', 'remove', or 'median'
    iqr_treatment : str
        Method for treating IQR outliers: 'clip', 'remove', or 'median'
    physical_treatment : str
        Method for treating physical constraint violations: 'clip' or 'remove'
    add_rolling : bool
        Whether to add rolling features
    rolling_columns : list
        Columns to add rolling features for
    rolling_windows : list
        Rolling window sizes in hours
    verbose : bool
        Whether to print progress information
    """
    if verbose:
        print("=" * 80)
        print("Air Quality Data Preprocessing Pipeline")
        print("=" * 80)

    # Step 1: Load raw data
    if verbose:
        print("\nStep 1: Loading raw data...")
    df = load_raw_data()
    if verbose:
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Step 2: Handle missing values
    if verbose:
        print("\nStep 2: Handling missing values...")
    df = handle_missing_values(df)
    if verbose:
        print(f"  Created datetime index")
        print(f"  Replaced -200 sentinel values with NaN")

    # Step 3: Detect outliers
    if verbose:
        print("\nStep 3: Detecting outliers...")
    negative_outliers = detect_negative_values(df, POLLUTANT_COLS + SENSOR_COLS)
    iqr_outliers = detect_iqr_outliers(df, POLLUTANT_COLS + SENSOR_COLS)
    physical_violations = detect_physical_constraints(df)

    if verbose:
        if negative_outliers:
            total_neg = sum(v["count"] for v in negative_outliers.values())
            print(f"  Found {total_neg} negative values")
        if iqr_outliers:
            total_iqr = sum(v["count"] for v in iqr_outliers.values())
            print(f"  Found {total_iqr} IQR outliers")
        if physical_violations:
            total_phys = sum(v["count"] for v in physical_violations.values())
            print(f"  Found {total_phys} physical constraint violations")

    # Step 4: Treat outliers
    if verbose:
        print("\nStep 4: Treating outliers...")
    df = treat_negative_values(df, POLLUTANT_COLS + SENSOR_COLS, method=outlier_treatment)
    df = treat_iqr_outliers(df, POLLUTANT_COLS + SENSOR_COLS, method=iqr_treatment)
    df = treat_physical_constraints(df, method=physical_treatment)

    # Step 5: Interpolate remaining missing values
    if verbose:
        print("\nStep 5: Interpolating missing values...")
    missing_before = df.isna().sum().sum()
    df = df.interpolate(method="time").ffill().bfill()
    missing_after = df.isna().sum().sum()
    if verbose:
        print(f"  Interpolated {missing_before - missing_after} missing values")

    # Step 5b: Handle sensor drift for pollutants via piecewise daily calibration
    if verbose:
        print("\nStep 5b: Handling sensor drift for pollutants via piecewise daily calibration...")
    for col in ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]:
        df = correct_sensor_drift(df, col, verbose=verbose)

    # Step 6: Remove duplicated timestamps
    if verbose:
        print("\nStep 6: Removing duplicated timestamps...")
    duplicates_before = df.index.duplicated().sum()
    df = df[~df.index.duplicated(keep="first")]
    if verbose:
        print(f"  Removed {duplicates_before} duplicate timestamps")

    # Step 7: Add time features
    if verbose:
        print("\nStep 7: Adding time features...")
    df = add_time_features(df)
    if verbose:
        print(f"  Added time features (hour, weekday, month, is_weekend)")
        print(f"  Added cyclical encoding (sin/cos)")

    # Step 8: Add rolling features
    if add_rolling:
        if verbose:
            print("\nStep 8: Adding rolling features...")
        if rolling_columns is None:
            rolling_columns = ["CO(GT)", "NOx(GT)", "NO2(GT)", "T", "RH", "AH"]
        if rolling_windows is None:
            rolling_windows = [3, 6, 12, 24]
        df = add_rolling_features(df, rolling_columns, rolling_windows)
        if verbose:
            print(f"  Added rolling means for {len(rolling_columns)} columns")

    # Step 9: Select only required columns
    if verbose:
        print("\nStep 9: Selecting required columns...")
    df = select_required_columns(df, verbose=verbose)

    # Step 10: Round numeric columns to match original precision
    if verbose:
        print("\nStep 10: Rounding numeric columns to match original precision...")
    df = round_numeric_columns(df, verbose=verbose)

    if verbose:
        print("\n" + "=" * 80)
        print("Preprocessing complete!")
        print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
        print("=" * 80)

    return df


def select_required_columns(df, verbose=True):
    """
    Select columns compatible with Regressor.py and Classification.py.
    
    Required columns:
    - Datetime: datetime column (from index)
    - All pollutant columns: CO(GT), NMHC(GT), C6H6(GT), NOx(GT), NO2(GT)
    - All sensor columns: PT08.S1(CO), PT08.S2(NMHC), PT08.S3(NOx), PT08.S4(NO2), PT08.S5(O3)
    - Meteorological columns: T, RH, AH
    - NMHC_Valid: if exists in original data
    """
    df = df.copy()

    # Required columns for Regressor.py and Classification.py
    required_cols = []
    
    # Add Datetime column from index (always create if index is DatetimeIndex)
    if isinstance(df.index, pd.DatetimeIndex):
        df["Datetime"] = df.index
        required_cols.append("Datetime")
    elif "Datetime" in df.columns:
        # If Datetime already exists as a column, keep it
        required_cols.append("Datetime")
    else:
        # Fallback: try to create from index name
        if df.index.name == "datetime":
            df["Datetime"] = df.index
            required_cols.append("Datetime")
    
    # Add all pollutant columns
    for col in POLLUTANT_COLS:
        if col in df.columns:
            required_cols.append(col)
    
    # Add all sensor columns
    for col in SENSOR_COLS:
        if col in df.columns:
            required_cols.append(col)
    
    # Add meteorological columns
    meteo_cols = ["T", "RH", "AH"]
    for col in meteo_cols:
        if col in df.columns:
            required_cols.append(col)
    
    # Add NMHC_Valid if it exists (from original preprocessing)
    if "NMHC_Valid" in df.columns:
        required_cols.append("NMHC_Valid")
    
    # Select only required columns
    available_cols = [col for col in required_cols if col in df.columns]
    
    # Ensure Datetime is included
    if "Datetime" not in available_cols and "Datetime" in df.columns:
        available_cols.insert(0, "Datetime")
    
    df_selected = df[available_cols].copy()
    
    # Reset index to regular integer index (Datetime is now a column)
    df_selected = df_selected.reset_index(drop=True)
    
    if verbose:
        removed_cols = set(df.columns) - set(available_cols)
        print(f"  Selected {len(available_cols)} columns for Regressor.py/Classification.py compatibility")
        print(f"  Removed {len(removed_cols)} derived columns (time features, lags, rolling features, etc.)")
        print(f"  Columns: {', '.join(available_cols[:10])}{'...' if len(available_cols) > 10 else ''}")
        if "Datetime" in available_cols:
            print(f"  ✓ Datetime column included")

    return df_selected


def round_numeric_columns(df, verbose=True):
    """
    Round numeric columns to match the precision of the original dataset.
    - Pollutant, sensor, and meteorological columns: 1 decimal place (or integer if appropriate)
    - Datetime column: keep as datetime (not rounded)
    - NMHC_Valid: keep as boolean/integer (not rounded)
    """
    df = df.copy()

    # Exclude non-numeric columns from rounding
    exclude_cols = ["Datetime", "NMHC_Valid"]
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Round pollutant columns to 1 decimal place
    pollutant_cols = [
        col for col in POLLUTANT_COLS if col in numeric_df.columns
    ]
    if pollutant_cols:
        df[pollutant_cols] = df[pollutant_cols].round(1)

    # Round sensor columns (typically integers, but round to 1 decimal for consistency)
    sensor_cols = [
        col for col in SENSOR_COLS if col in numeric_df.columns
    ]
    if sensor_cols:
        df[sensor_cols] = df[sensor_cols].round(1)

    # Round meteorological columns to 1 decimal place
    meteo_cols = ["T", "RH", "AH"]
    meteo_cols = [col for col in meteo_cols if col in numeric_df.columns]
    if meteo_cols:
        df[meteo_cols] = df[meteo_cols].round(1)

    if verbose:
        print(f"  Rounded {len(pollutant_cols)} pollutant columns to 1 decimal place")
        print(f"  Rounded {len(sensor_cols)} sensor columns to 1 decimal place")
        print(f"  Rounded {len(meteo_cols)} meteorological columns to 1 decimal place")
        if "Datetime" in df.columns:
            print(f"  Preserved Datetime column (not rounded)")

    return df


# -----------------------------------------------------------------------------
# Visualisation for NOx preprocessing and sensor drift
# -----------------------------------------------------------------------------

def plot_nox_preprocessing():
    """
    Visualise NOx(GT) before/after preprocessing and sensor drift correction.
    
    Figure 1 (3 subplots):
      1) NOx(GT) original (after datetime + -200 handling)
      2) NOx(GT) after -200 and outlier treatment
      3) NOx(GT) after full preprocessing including sensor drift correction

    Figure 2:
      Difference = original NOx(GT) - NOx(GT) after full preprocessing.
    """
    try:
        # Load and go through the same steps as the main pipeline (up to drift)
        raw_df = load_raw_data()
        df_missing = handle_missing_values(raw_df)

        # Save "original" NOx after -200 handling (视作原数据基准)
        if "NOx(GT)" not in df_missing.columns:
            print("NOx(GT) column not found, skip plotting.")
            return
        nox_original = df_missing["NOx(GT)"].copy()

        # Outlier treatment: negative, IQR, physical constraints
        df_outliers = treat_negative_values(df_missing, POLLUTANT_COLS + SENSOR_COLS, method="clip")
        df_outliers = treat_iqr_outliers(df_outliers, POLLUTANT_COLS + SENSOR_COLS, method="clip")
        df_outliers = treat_physical_constraints(df_outliers, method="clip")
        nox_after_outlier = df_outliers["NOx(GT)"].copy()

        # Interpolate missing values
        df_interp = df_outliers.interpolate(method="time").ffill().bfill()

        # Sensor drift correction for five pollutants
        df_drift = df_interp.copy()
        for col in ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]:
            df_drift = correct_sensor_drift(df_drift, col, verbose=False)
        nox_after_drift = df_drift["NOx(GT)"].copy()

        # Align indices
        common_index = nox_original.index.intersection(nox_after_outlier.index)
        common_index = common_index.intersection(nox_after_drift.index)
        nox_original = nox_original.loc[common_index]
        nox_after_outlier = nox_after_outlier.loc[common_index]
        nox_after_drift = nox_after_drift.loc[common_index]

        # Figure 1: three subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(nox_original.index, nox_original.values)
        axes[0].set_title("NOx(GT) original")
        axes[1].plot(nox_after_outlier.index, nox_after_outlier.values)
        axes[1].set_title("NOx(GT) after -200 and outlier handling")
        axes[2].plot(nox_after_drift.index, nox_after_drift.values)
        axes[2].set_title("NOx(GT) after sensor drift correction")

        for ax in axes:
            ax.set_ylabel("Concentration")
        axes[-1].set_xlabel("Time")
        plt.tight_layout()
        plt.show()

        # Figure 2: difference (original - after full preprocessing)
        diff = nox_original - nox_after_drift
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 4))
        ax2.plot(diff.index, diff.values)
        ax2.set_title("NOx(GT) difference: original - after preprocessing and drift")
        ax2.set_ylabel("Difference")
        ax2.set_xlabel("Time")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Plotting NOx(GT) preprocessing failed: {e}")


def plot_original_pollutants_comparison():
    """
    Plot original time series for five pollutants after datetime and -200 handling.

    One figure with multiple subplots:
      CO(GT), NMHC(GT), C6H6(GT), NOx(GT), NO2(GT).
    """
    try:
        raw_df = load_raw_data()
        df_missing = handle_missing_values(raw_df)

        cols = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
        available_cols = [col for col in cols if col in df_missing.columns]

        if not available_cols:
            print("No pollutant columns found for original comparison plot.")
            return

        fig, axes = plt.subplots(
            len(available_cols),
            1,
            figsize=(12, 2.5 * len(available_cols)),
            sharex=True,
        )
        if len(available_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, available_cols):
            series = df_missing[col]
            ax.plot(series.index, series.values)
            ax.set_title(f"{col} original")
            ax.set_ylabel("Concentration")

        axes[-1].set_xlabel("Time")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Plotting original pollutants comparison failed: {e}")


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Run preprocessing pipeline
    processed_data = preprocess_air_quality_data(
        outlier_treatment="clip",  # Clip negative values to 0
        iqr_treatment="clip",  # Clip outliers to IQR bounds
        physical_treatment="clip",  # Clip to physical bounds
        add_rolling=True,
        verbose=True,
    )

    # Save processed data (Datetime is now a column, so use index=False)
    processed_data.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"\nProcessed data saved to: {PROCESSED_DATA_FILE.resolve()}")

    # Plot NOx(GT) preprocessing and drift effects
    plot_nox_preprocessing()

    # Plot original comparison for five pollutants
    plot_original_pollutants_comparison()
