import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("House Price Prediction Dataset (1).csv")

# Display first few rows
print(df.head())

# Remove unnecessary column
df = df.drop("Id", axis=1)

# Convert categorical columns into numbers
encoder = LabelEncoder()


for column in ["Location", "Condition", "Garage"]:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column].astype(str))

# Features and Target
print(df.dtypes)
print(df.head())
X = df.drop("Price", axis=1)
y = df["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
predictions = model.predict(X_test)

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

print("\nSample Predictions:")
print(results.head(10))

results.to_csv("predictions.csv", index=False)

print("\nPredictions saved as predictions.csv")

# Evaluate Model
print("\nModel Performance")
print("------------------")
print("R2 Score:", r2_score(y_test, predictions))
print("Mean Squared Error:", mean_squared_error(y_test, predictions))

# Save the trained model
joblib.dump(model, "house_price_model.pkl")

print("\nModel saved successfully as house_price_model.pkl")

# Plot Actual vs Predicted Prices
plt.figure(figsize=(8, 6))

plt.scatter(y_test, predictions)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.savefig("actual_vs_predicted.png")
plt.show()

# Display Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Importance:")
print(importance.sort_values(by="Coefficient", ascending=False))