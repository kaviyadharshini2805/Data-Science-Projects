#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Loading the dataset
df = pd.read_csv('advertising.csv')

print("First five rows: ")
print(df.head(), end="\n") #To print the first five rows of the dataset

print("The last five rows: ")
print(df.tail(), end="\n") #To print the last five rows of the dataset

#To print the information
print("Info of the dataset: ")
print(df.info(), end="\n")

#To print statistical values
print("\nStatistical analysis of the dataset: ")
print(df.describe())

#To find the count of null values
print("\nThe count of null values: ")
print(df.isnull().sum())

#To seperate features and target
X = df[['TV', 'Radio','Newspaper']]
y = df['Sales']

#Feature Selection
selector = SelectKBest(score_func=f_regression, k=1)
X_new = selector.fit_transform(X, y)

#To get the names of selected features
selected_features = X.columns[selector.get_support()]
print("\nSelected Top Features: ", list(selected_features))

#To use only selected features
X = df[selected_features]

#Splitting Training data and Testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Training the model
model = LinearRegression()
model.fit(X_train, y_train)

#Make Predictions
y_pred = model.predict(X_test)

#Evaluation
mae = mean_absolute_error(y_test, y_pred)
print("Mean absolute error: ", mae)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)
rmse = np.sqrt(mse)
print("Root mean squared error: ", rmse)
r2 = r2_score(y_test, y_pred)
print("R-squared score: ", r2)

#Visualization for feature importance
feature_scores = pd.DataFrame({'Feature': X.columns, 'F_Score': selector.scores_[selector.get_support()]}).sort_values(by='F_Score', ascending=False)
plt.figure(figsize=(4,4))
sns.barplot(x='Feature', y='F_Score', hue='Feature', data=feature_scores, palette='Blues_d', width=0.1)
plt.title('Feature Selection - F Score')
plt.xlabel('Feature')
plt.ylabel('F Score')
plt.tight_layout()
plt.savefig("day2_features.png", dpi=150)
plt.show()


# Plot the data and regression line for the selected feature
plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, color='blue', label='Actual Sales')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel(selected_features[0])
plt.ylabel("Sales")
plt.title(f"Linear Regression - {selected_features[0]} vs Sales")
plt.legend()
plt.grid(True)
plt.savefig("day2_regression.png", dpi=150)
plt.show()