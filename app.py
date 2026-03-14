

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Data
df=pd.read_csv("/content/kc_house_data.csv")

# EDA (Exploratory Data Analysis)
df.head()
df.tail()
df.info()
df.shape
df.describe()
     

# Data Cleaning
df.fillna(0, inplace=True)
df.dropna(inplace=True)
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
     

df.columns
     

# Model_Training
x=df[['bedrooms','bathrooms','sqft_living', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']]
y=df['price']

from sklearn.metrics import mean_absolute_error, mean_squared_error
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2,random_state=42)
     

model=LinearRegression()
model.fit(x_train,y_train)
     

print(f"Minimum predicted house price: {y_pred.min():.2f}")
print(f"Maximum predicted house price: {y_pred.max():.2f}")
     

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
     

y_pred = model.predict(x_test)
print("Predicted house prices (first 5):")
print(y_pred[:5])
     

r_squared = model.score(x_test, y_test)
print(f"R-squared score: {r_squared:.2f}")
     

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 7))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2) # Perfect prediction line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")
plt.show()
     