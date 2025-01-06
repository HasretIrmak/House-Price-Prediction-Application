import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # For saving the model

# Load the dataset
file_path = "train.csv"
data = pd.read_csv(file_path)

# Fill missing values with the median
data.fillna(data.median(numeric_only=True), inplace=True)

# Select features and target variable
X = data.select_dtypes(include=[np.number]).drop(columns=['Id', 'SalePrice'])
y = data['SalePrice']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("Random Forest - RMSE:", rmse)
print("Random Forest - R²:", r2)

# Save the trained model
joblib.dump(rf_model, "house_price_model.pkl")
