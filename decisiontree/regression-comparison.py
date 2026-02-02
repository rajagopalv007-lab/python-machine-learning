import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Set seed for reproducibility
np.random.seed(42)

# ---------------------------------------------------------
# 1. Generate Data (Same definition as your demo)
# ---------------------------------------------------------
n_samples = 100
education_levels = ["Secondary", "Diploma", "Degree", "Masters"]
education_priorities_map = {
    "Secondary": 1,
    "Diploma": 2,
    "Degree": 3,
    "Masters": 4,
}

age = np.random.randint(22, 65, n_samples)
education_labels = np.random.choice(education_levels, n_samples, p=[0.2, 0.3, 0.4, 0.1])
education_numeric = pd.Series(education_labels).map(education_priorities_map)

# The "True" Linear Formula
# Salary = 1200*Age + 5000*Education
salary = (
    (age * 1200) + (education_numeric * 5000) + np.random.normal(0, 5000, n_samples)
)

df = pd.DataFrame({"Age": age, "Education": education_numeric, "Salary": salary})

X = df[["Age", "Education"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 2. Train Decision Tree (The "Step" Model)
# ---------------------------------------------------------
tree_model = DecisionTreeRegressor(random_state=42, max_depth=4)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
tree_score = r2_score(y_test, tree_pred)

# ---------------------------------------------------------
# 3. Train Linear Regression (The "Line" Model)
# ---------------------------------------------------------
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)
linear_score = r2_score(y_test, linear_pred)

# ---------------------------------------------------------
# 4. Compare Results
# ---------------------------------------------------------
print("-" * 30)
print("Model Comparison Results")
print("-" * 30)
print(f"Decision Tree R2 Score:  {tree_score:.4f}")
print(f"Linear Regression R2 Score: {linear_score:.4f}")
print("-" * 30)

if linear_score > tree_score:
    print(">> Linear Regression Won (Expected for linear data)")
else:
    print(">> Decision Tree Won")

# ---------------------------------------------------------
# 5. Visualize the difference (simplified to 1 feature: Age)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(X_test["Age"], y_test, color="black", label="Actual Data", alpha=0.6)

# Sort for clean plotting
sorted_idx = X_test["Age"].argsort()
age_sorted = X_test["Age"].iloc[sorted_idx]
tree_pred_sorted = tree_pred[sorted_idx]
linear_pred_sorted = linear_pred[sorted_idx]

plt.plot(
    age_sorted,
    tree_pred_sorted,
    color="blue",
    linewidth=2,
    label="Decision Tree (Step)",
)
plt.plot(
    age_sorted,
    linear_pred_sorted,
    color="red",
    linewidth=2,
    label="Linear Regression (Line)",
)

plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Visual Comparison: Tree steps vs Linear line")
plt.legend()
plt.savefig("model_comparison.png")
print("Comparison plot saved as 'model_comparison.png'")
