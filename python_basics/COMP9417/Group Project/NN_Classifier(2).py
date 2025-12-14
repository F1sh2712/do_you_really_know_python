import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

import warnings
warnings.filterwarnings('ignore')
plt.style.use('default')

#Data Loading and Preprocessing 
print("=" * 60)
print("1. Data Loading and Preprocessing (Neural Network Classifier)")
print("=" * 60)

df = pd.read_csv("processed_air_quality.csv", parse_dates=['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Time range: {df['Datetime'].min()} to {df['Datetime'].max()}")


# 2. Feature Engineering  （CO）
print("\n" + "=" * 60)
print("2. Feature Engineering")
print("=" * 60)

target = 'CO(GT)'

# lag & roll (CO ）
lags = [1, 3, 6, 12, 24]
mas  = [3, 6, 12, 24]

for lag in lags:
    df[f'{target}_lag_{lag}'] = df[target].shift(lag)

for w in mas:
    df[f'{target}_MA_{w}'] = df[target].rolling(window=w, min_periods=1).mean()

# Numerical targets for multi-step prediction
horizons = [1, 6, 12, 24]
for h in horizons:
    df[f'CO_future_{h}'] = df[target].shift(-h)

df_clean = df.dropna().copy()
print(f"Data shape after feature engineering: {df_clean.shape}")

# 3. Target Discretization (CO classes)
print("\n" + "=" * 60)
print("3. Target Variable Discretization")
print("=" * 60)

def categorize_co(x):
    if x < 1.5:
        return 'low'
    elif x < 2.5:
        return 'mid'
    else:
        return 'high'

df_clean['CO_class_current'] = df_clean['CO(GT)'].apply(categorize_co)
for h in horizons:
    df_clean[f'CO_class_future_{h}'] = df_clean[f'CO_future_{h}'].apply(categorize_co)

print("\nCurrent CO class distribution:")
print(df_clean['CO_class_current'].value_counts())

print("\nFuture CO class distribution (1-hour ahead):")
print(df_clean['CO_class_future_1'].value_counts())

# 4. Time Series Split (train 2004, test 2005)
print("\n" + "=" * 60)
print("4. Time Series Split")
print("=" * 60)

train_df = df_clean[df_clean['Datetime'] < '2005-01-01']
test_df  = df_clean[df_clean['Datetime'] >= '2005-01-01']

print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# 5. Feature Preparation

print("\n" + "=" * 60)
print("5. Feature Preparation")
print("=" * 60)

feature_columns = (
    ['CO(GT)'] +
    [f'{target}_lag_{lag}' for lag in lags] +
    [f'{target}_MA_{w}' for w in mas]
)

print("Features used for NN:", feature_columns)

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_columns])
X_test  = scaler.transform(test_df[feature_columns])

le = LabelEncoder()
le.fit(['low', 'mid', 'high'])


# 6. Neural Network Classification for Multi-Horizon
print("\n" + "=" * 60)
print("6. Neural Network Classification (MLPClassifier)")
print("=" * 60)

classification_results = {}

for h in horizons:
    print(f"\n--- CO Class Prediction: {h} hours ahead ---")

    y_train_cls = train_df[f'CO_class_future_{h}']
    y_test_cls  = test_df[f'CO_class_future_{h}']

    y_train = le.transform(y_train_cls)
    y_test  = le.transform(y_test_cls)

    # Naive baseline
    y_naive = le.transform(test_df['CO_class_current'])
    naive_acc = accuracy_score(y_test, y_naive)
    print(f"Naive Baseline Accuracy: {naive_acc:.4f}")

    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
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

    mlp_clf.fit(X_train, y_train)
    y_pred = mlp_clf.predict(X_test)
    nn_acc = accuracy_score(y_test, y_pred)

    classification_results[f'{h}h'] = {
        'Naive_Baseline': naive_acc,
        'MLP_NN': nn_acc
    }

    print(f"MLP Neural Network Accuracy: {nn_acc:.4f}")

    if h == 1:
        print("\nDetailed Classification Report (1-hour ahead, MLP NN):")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(7, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix - MLP NN (1-hour ahead)')
        plt.colorbar()
        tick_marks = np.arange(len(le.classes_))
        plt.xticks(tick_marks, le.classes_, rotation=45)
        plt.yticks(tick_marks, le.classes_)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color="white" if cm[i, j] > thresh else "black"
                )

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()


# 7. Accuracy Summary and Visualization
print("\n" + "=" * 60)
print("7. Accuracy Summary and Visualization")
print("=" * 60)

rows = []
for h in horizons:
    res = classification_results[f'{h}h']
    rows.append({
        'Horizon': f'{h}h',
        'Naive_Baseline': res['Naive_Baseline'],
        'MLP_NN': res['MLP_NN'],
        'Improvement': res['MLP_NN'] - res['Naive_Baseline']
    })

summary_df = pd.DataFrame(rows)
print("\nNeural Network Classification Performance Summary:")
print(summary_df.round(4))

summary_df.to_csv('NN_CO_Classification_Results_simple.csv', index=False)
print("\nResults saved to 'NN_CO_Classification_Results_simple.csv'")

x = np.arange(len(horizons))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, summary_df['Naive_Baseline'], width, label='Naive')
plt.bar(x + width/2, summary_df['MLP_NN'], width, label='MLP NN')

plt.xticks(x, summary_df['Horizon'])
plt.xlabel('Prediction Horizon')
plt.ylabel('Accuracy')
plt.title('CO Class Prediction Accuracy Comparison (Naive vs MLP NN, simple features)')
plt.legend()
plt.tight_layout()
plt.show()

print("\nNeural Network Classifier (simple) completed successfully.")
