import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')

# Start data preprocessing
print("=" * 60)
print("1. Data Loading and Preprocessing")
print("=" * 60)

# Load processed data with Datetime column
df = pd.read_csv("AirQuality_Clean_WithNMHC.csv", parse_dates=['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Time range: {df['Datetime'].min()} to {df['Datetime'].max()}")

# Start feature engineering
print("\n" + "=" * 60)
print("2. Feature Engineering")
print("=" * 60)

# Define 4 time features
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Month'] = df['Datetime'].dt.month
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

# Define shift and rolling features only for CO pollutant
target = 'CO(GT)'
for lag in [1, 3, 6, 12, 24]:
    df[f'{target}_lag_{lag}'] = df[target].shift(lag)

for window in [3, 6, 12, 24]:
    df[f'{target}_MA_{window}'] = df[target].rolling(window=window, min_periods=1).mean()

# # For CO, we make multi-step forecasting targets
# at time t for t+1, t+6, t+12 and t+24
horizons = [1, 6, 12, 24]
for horizon in horizons:
    df[f'CO_future_{horizon}'] = df[target].shift(-horizon)

# Delete NaN data
df_clean = df.dropna().copy()
print(f"Data shape after feature engineering: {df_clean.shape}")

# Start target variable discretization
print("\n" + "=" * 60)
print("3. Target Variable Discretization")
print("=" * 60)

def categorize_co(value):
    # From instruction, we define 3 categories
    if value < 1.5:
        return 'low'
    elif value < 2.5:
        return 'mid'
    else:
        return 'high'

# Current and future CO levels for each category
df_clean['CO_class_current'] = df_clean['CO(GT)'].apply(categorize_co)
for horizon in horizons:
    df_clean[f'CO_class_future_{horizon}'] = df_clean[f'CO_future_{horizon}'].apply(categorize_co)

print("\nCurrent CO class distribution:")
print(df_clean['CO_class_current'].value_counts())
print("\nFuture CO class distribution (1-hour):")
print(df_clean['CO_class_future_1'].value_counts())

# Start time series split
print("\n" + "=" * 60)
print("4. Time Series Split")
print("=" * 60)

# We use data from 2004 to train the model, and data from 2005 to test model prediction
train_df = df_clean[df_clean['Datetime'] < '2005-01-01']
test_df = df_clean[df_clean['Datetime'] >= '2005-01-01']

print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# Start feature preparation
print("\n" + "=" * 60)
print("5. Feature Preparation")
print("=" * 60)

# Create base feature
base_features = [col for col in df_clean.columns if col not in 
                ['Datetime', 'NMHC_Valid', 'CO(GT)'] + 
                [f'CO_future_{h}' for h in horizons] +
                [f'CO_class_future_{h}' for h in horizons] +
                ['CO_class_current']]
feature_columns = [col for col in base_features if not col.startswith('CO_future_')]

print(f"Number of features: {len(feature_columns)}")

# Standardize features to fit train data set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df[feature_columns])
X_test_scaled = scaler.transform(test_df[feature_columns])

# Start classification model
print("\n" + "=" * 60)
print("6. Classification Model Development")
print("=" * 60)

# 可能要修改的地方 我们只需要保留梯度提升的类别在这里
classification_models = {     'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),     'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),     'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),     'SVM': SVC(random_state=42) }

classification_results = {}

# Label 3 categories
le = LabelEncoder()
le.fit(['low', 'mid', 'high'])

for horizon in horizons:
    print(f"\n--- Classification Prediction: {horizon} hours ahead ---")
    
    # Targets
    y_train_class = train_df[f'CO_class_future_{horizon}']
    y_test_class = test_df[f'CO_class_future_{horizon}']
    
    # Encode labels to int
    y_train_encoded = le.transform(y_train_class)
    y_test_encoded = le.transform(y_test_class)
    
    # Naive baseline, we assume fureture equals to current
    current_classes = test_df['CO_class_current']
    current_encoded = le.transform(current_classes)
    naive_accuracy = accuracy_score(y_test_encoded, current_encoded)
    
    print(f"Naive Baseline Accuracy: {naive_accuracy:.4f}")
    
    horizon_results = {'Naive Baseline': naive_accuracy}
    
    # 可能需要修改的地方！！！可以不用for循环遍历model类别， 直接使用梯度提升模型
    # 训练和评估每个分类模型
    for name, model in classification_models.items():
        # 训练模型
        model.fit(X_train_scaled, y_train_encoded)
        
        # 预测
        y_pred_class = model.predict(X_test_scaled)
        
        # 计算准确率
        accuracy = accuracy_score(y_test_encoded, y_pred_class)
        horizon_results[name] = accuracy
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        # 为第一个时域和梯度提升模型输出详细报告
        ## 直接使用梯度提升模型就行
        if horizon == 1 and name == 'Gradient Boosting':
            print(f"\nDetailed Classification Report for {name} ({horizon}-hour):")
            print(classification_report(y_test_encoded, y_pred_class, 
                                      target_names=le.classes_))
            
            # Plot confuson matrix
            cm = confusion_matrix(y_test_encoded, y_pred_class)
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {name} ({horizon}-hour)')
            plt.colorbar()
            tick_marks = np.arange(len(le.classes_))
            plt.xticks(tick_marks, le.classes_, rotation=45)
            plt.yticks(tick_marks, le.classes_)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            # Add values for each square in matrix
            thresh = cm.max() / 2.
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.show()
    
    classification_results[f'{horizon}h'] = horizon_results

# Start data visualization
print("\n" + "=" * 60)
print("7. Results Visualization")
print("=" * 60)

# Plot accuracy comparison
accuracy_df = pd.DataFrame(classification_results)
plt.figure(figsize=(12, 6))
accuracy_df.T.plot(kind='bar', figsize=(12, 6))
plt.title('Classification Accuracy Comparison for Different Prediction Horizons')
plt.xlabel('Prediction Horizon')
plt.ylabel('Accuracy')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Start importance analysis
print("\n" + "=" * 60)
print("8. Feature Importance Analysis")
print("=" * 60)

# Analize feature importance using BG classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_train_scaled, le.transform(train_df['CO_class_future_1']))

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': gb_classifier.feature_importances_
}).sort_values('importance', ascending=False)

# Plot top 15 features
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
y_pos = np.arange(len(top_features))
plt.barh(y_pos, top_features['importance'])
plt.yticks(y_pos, top_features['feature'])
plt.xlabel('Importance')
plt.title('Feature Importance for CO Classification (Top 15)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\nTop 10 most important features for CO classification:")
print(feature_importance.head(10))

# Start multi-step prediction analysis
print("\n" + "=" * 60)
print("9. Multi-step Prediction Performance Analysis")
print("=" * 60)

# Performance summary 
performance_summary = []
for horizon in horizons:
    for model_name, accuracy in classification_results[f'{horizon}h'].items():
        performance_summary.append({
            'Model': model_name,
            'Horizon': f'{horizon}h',
            'Accuracy': accuracy
        })

performance_df = pd.DataFrame(performance_summary)

# Plot heatmap for performance
pivot_df = performance_df.pivot(index='Model', columns='Horizon', values='Accuracy')
plt.figure(figsize=(10, 6))
im = plt.imshow(pivot_df.values, cmap='YlGnBu', aspect='auto')
plt.colorbar(im, label='Accuracy')
plt.xticks(range(len(pivot_df.columns)), pivot_df.columns)
plt.yticks(range(len(pivot_df.index)), pivot_df.index)
plt.xlabel('Prediction Horizon')
plt.ylabel('Model')
plt.title('Classification Accuracy Heatmap')

# Add label of accuracy
for i in range(len(pivot_df.index)):
    for j in range(len(pivot_df.columns)):
        plt.text(j, i, f'{pivot_df.iloc[i, j]:.3f}', 
                ha='center', va='center', fontsize=10)

plt.tight_layout()
plt.show()

# Start summary result
print("\n" + "=" * 60)
print("10. Summary of Results")
print("=" * 60)

print("\nPerformance Summary:")
print(performance_df.round(4))

# Compute mean improvement vs naive baselien
baseline_accuracies = performance_df[performance_df['Model'] == 'Naive Baseline']['Accuracy'].values
model_accuracies = {}

for model in classification_models.keys():
    model_data = performance_df[performance_df['Model'] == model]
    avg_accuracy = model_data['Accuracy'].mean()
    avg_improvement = avg_accuracy - baseline_accuracies.mean()
    model_accuracies[model] = {
        'Average Accuracy': avg_accuracy,
        'Average Improvement': avg_improvement
    }

print("\nAverage Performance by Model:")
for model, metrics in model_accuracies.items():
    print(f"{model}:")
    print(f"  Average Accuracy: {metrics['Average Accuracy']:.4f}")
    print(f"  Average Improvement over Baseline: {metrics['Average Improvement']:.4f}")

# Save result
performance_df.to_csv('CO_Classification_Results.csv', index=False)
print(f"\nResults saved to 'CO_Classification_Results.csv'")

# Sart class distribution over time
print("\n" + "=" * 60)
print("11. Class Distribution Over Time")
print("=" * 60)

# Analyze distribution of CO with 4 time periods of a day
time_periods = {
    'Morning (6-12)': (6, 12),
    'Afternoon (12-18)': (12, 18),
    'Evening (18-24)': (18, 24),
    'Night (0-6)': (0, 6)
}

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, (period_name, (start_hour, end_hour)) in enumerate(time_periods.items()):
    if i < len(axes):
        period_data = df_clean[(df_clean['Hour'] >= start_hour) & (df_clean['Hour'] < end_hour)]
        class_dist = period_data['CO_class_current'].value_counts()
        
        axes[i].pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%', startangle=90)
        axes[i].set_title(f'CO Class Distribution - {period_name}')

plt.tight_layout()
plt.show()

print("\nProject completed successfully!")