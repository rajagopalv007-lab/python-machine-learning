import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
# Set seed for reproducibility

np.random.seed(42)

# Generate sample data
n_samples = 20
income = np.random.randint(20000, 150000, n_samples)
loan_amount = np.random.randint(5000, 50000, n_samples)

# Logic for default: higher loan-to-income ratio increases default probability
default = (loan_amount / income > 0.2).astype(int)

# Create DataFrame
df = pd.DataFrame({"income": income, "loan_amount": loan_amount, "default": default})

# Print the first few rows of the DataFrame
print(df)

# Draw a scatter plot with loan amount on x-axis and income on y-axis and color the points red or green based on default
"""
plt.figure(figsize=(10, 6))
colors = df["default"].map({0: "green", 1: "red"})

df["default"] = df["default"].map({0: "green", 1: "red"})

plt.scatter(df["loan_amount"], df["income"], c=df["default"], cmap="viridis")
plt.xlabel("Loan Amount")
plt.ylabel("Income")
plt.title("Loan Default Prediction")
plt.show()

sns.boxplot(x="default", y="income", data=df)
plt.show()
"""
# Build a decision tree classifier to predict default, draw the tree


X = df[["income", "loan_amount"]]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


param_grid = {
    "max_depth": [2, 3, 4, 5],
    "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "min_samples_leaf": [1, 2, 3, 4, 5, 6],
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=3)
grid_search.fit(X_train, y_train)
print("Best parameters found:", grid_search.best_params_)


model = DecisionTreeClassifier(random_state=42, **grid_search.best_params_)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy for test data:", accuracy_score(y_test, y_pred))
print("Accuracy for train data:", accuracy_score(y_train, model.predict(X_train)))

plt.figure(figsize=(12, 8))
plot_tree(
    model, filled=True, feature_names=X.columns, class_names=["No Default", "Default"]
)
plt.show()
