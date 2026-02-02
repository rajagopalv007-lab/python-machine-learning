from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Set seed for reproducibility
np.random.seed(42)

# Generate sample data
n_samples = 100
education_levels = ["Secondary", "Diploma", "Degree", "Masters"]
education_priorities_map = {
    "Secondary": 1,
    "Diploma": 2,
    "Degree": 3,
    "Masters": 4,
}

# 1. Generate Age
age = np.random.randint(22, 65, n_samples)

# 2. Assign Education Levels first
education_labels = np.random.choice(education_levels, n_samples, p=[0.2, 0.3, 0.4, 0.1])

# 3. Map Education Levels to numeric priorities for calculation
# We use pandas Series map for convenience
education_numeric = pd.Series(education_labels).map(education_priorities_map)

# 4. Create synthetic Salary based on Age and the ACTUAL Education level
salary = (
    (age * 1200) + (education_numeric * 5000) + np.random.normal(0, 5000, n_samples)
)

# 5. Assemble the data dictionary
data = {"Age": age, "Education": education_labels, "Salary": salary}

df = pd.DataFrame(data)
df["Education"] = df["Education"].map(education_priorities_map)
print(df)

"""# box plot to show the relationship between Age, Education and Salary
plt.figure(figsize=(12, 6))
sns.boxplot(x="Age", y="Salary", data=df)
plt.show()

# box plot to show the relationship between Education and Salary
plt.figure(figsize=(12, 6))
sns.boxplot(x="Education", y="Salary", data=df)
plt.show()
"""

# split the dataframe df into train and test. Salary is the dependent variable. Age and Education are the independent variables.
X = df[["Age", "Education"]]
y = df["Salary"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=X["Education"]
)
# Train the model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# score the model using train and test data

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train Score: {train_score}")
print(f"Test Score: {test_score}")


# Calculate and print the mean absolute error
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")


# ---------------------------------------------------------
# Post Pruning: Find the effective alpha
# ---------------------------------------------------------
# Use an unconstrained model to find the pruning path
unconstrained_model = DecisionTreeRegressor(random_state=42)
path = unconstrained_model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Train models with different alphas
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeRegressor(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# Calculate scores to find the best alpha
test_scores = [clf.score(X_test, y_test) for clf in clfs]
best_index = np.argmax(test_scores)
best_alpha = ccp_alphas[best_index]
best_score = test_scores[best_index]

print(f"\n--- Post Pruning Results ---")
print(f"Best Alpha: {best_alpha}")
print(f"Best Test Score: {best_score}")

# Set the model to the best one found for the final plot
model = clfs[best_index]

# Plot Alpha vs Score
plt.figure(figsize=(10, 6))
train_scores = [clf.score(X_train, y_train) for clf in clfs]
plt.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
plt.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
plt.xlabel("alpha")
plt.ylabel("accuracy")
plt.title("Accuracy vs alpha for training and testing sets")
plt.legend()
plt.savefig("alpha_score_visualization.png")
print("Alpha score visualization saved as 'alpha_score_visualization.png'")

# Plot the DecisionTreeRegressor model.
plt.figure(figsize=(15, 15))
plot_tree(model, filled=True, feature_names=X.columns, rounded=True, max_depth=3)
plt.title("Decision Tree Regressor Visualization")
plt.savefig("decision_tree_regression.png")
print("Decision Tree visualization saved as 'decision_tree_regression.png'")
