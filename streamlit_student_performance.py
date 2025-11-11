import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
import joblib

#Loading Dataset
df = pd.read_excel("K:\\GEN-AI FOR DATA SCIENCE\\DAY-4\\DAY4-TASK\\Task____students_performance_dataset.xlsx")

#Preview of dataset
print("\n First ten rows of the dataset: ")
print(df.head(10))

#Info of Dataset
print("\nInformation: ")
print(df.info())

#Statistical Summary
print("\nStatistical Analysis: ")
print(df.describe())

#Count of Missing values
print("Count of missing values: ")
print(df.isnull().sum())

#To check
print(df.head())

#Exclude features for modeling
exclude_cols = ['Student_ID', 'Family_Income', 'Parental_Education', 'School_Type', 'Final_Score']

#Feature Selection
feature_cols = [x for x in df.columns if x not in exclude_cols]
X = df[feature_cols]
y = df['Final_Score']
print(f"Features selected: {len(feature_cols)} columns")
print(f"Target variable: Final Score")

#Display the feature columns
print(feature_cols)

#Important Statistics
print(f"Mean: {y.mean():.2f}")
print(f"Median: {y.median():.2f}")
print(f"Std Dev: {y.std():.2f}")
print(f"Min: {y.min():.2f}")
print(f"Max: {y.max():.2f}")

# Correlation analysis (only numeric columns)
numeric_df = df.select_dtypes(include=['number'])
correlations = numeric_df.corr()['Final_Score'].sort_values(ascending=False)
print("\nTop 10 correlated features with Final Score:")
print(correlations.head(10))

#Visualizing the Data
plt.figure(figsize=(8, 5))
plt.hist(y, bins=30, alpha=0.7, color='brown', edgecolor='black')
plt.title('Exam mark Distribution')
plt.xlabel('Features')
plt.ylabel('Target')
plt.grid(True, alpha=0.3)
plt.show()

# STUDENT PERFORMANCE MODEL
# Exclude Irrelevant Columns
exclude_cols = ['Student_ID', 'Family_Income', 'Parental_Education', 'School_Type', 'Final_Score']

# Feature and Target Selection
X = df.drop(columns=exclude_cols)
y = df['Final_Score']

# Encode Categorical Columns
X = pd.get_dummies(X, drop_first=True)

# Correlation Analysis
correlations = df.select_dtypes(include=['number']).corr()['Final_Score'].sort_values(ascending=False)
print("\nTop 10 correlated features with Final Score:")
print(correlations.head(10))

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap - Student Performance", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection
selector = SelectKBest(score_func=f_regression, k=10)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)
selected_features = X.columns[selector.get_support()]
print("\nSelected Features:")
print(list(selected_features))

# Model Training
model = LinearRegression()
model.fit(X_train_selected, y_train)

# Model Evaluation
def calculate_metrics(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{dataset_name} Set Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae}

y_pred_train = model.predict(X_train_selected)
y_pred_test = model.predict(X_test_selected)
train_metrics = calculate_metrics(y_train, y_pred_train, "Training")
test_metrics = calculate_metrics(y_test, y_pred_test, "Test")

# Overfitting Check
r2_diff = train_metrics['R2'] - test_metrics['R2']
print(f"\nR² Difference: {r2_diff:.4f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Actual vs Predicted
axes[0].scatter(y_test, y_pred_test, alpha=0.6, color='green')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('Actual Final Score')
axes[0].set_ylabel('Predicted Final Score')
axes[0].set_title('Actual vs Predicted')
axes[0].grid(True, alpha=0.3)

# Feature Importance
axes[1].barh(feature_importance['Feature'], feature_importance['Coefficient'])
axes[1].set_title('Feature Importance')
axes[1].set_xlabel('Coefficient Value')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Save Model
joblib.dump(model, 'student_performance_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector, 'feature_selector.pkl')