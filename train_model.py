"""
train_model.py
Loads student_scores.csv, trains a Multiple Linear Regression model,
evaluates it, and saves the trained model as model.pkl
"""

import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load data
df = pd.read_csv("student_scores.csv")
print(df.head())
print("\nDataset shape:", df.shape)

# 2. Split features (X) and target (y)
X = df[["hours_studied", "attendance_percent", "assignments_completed"]]
y = df["final_score"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.3f}")

print("\nModel coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.3f}")
print(f"  Intercept: {model.intercept_:.3f}")

# 6. Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as model.pkl")
