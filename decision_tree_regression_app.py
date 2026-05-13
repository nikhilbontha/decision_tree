
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint


st.title("Decision Tree Regression with RandomizedSearchCV")

# Load Dataset
housing = fetch_california_housing()

X = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

y = housing.target

# Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(X.head())

# Parameters on Main Page
st.subheader("Model Parameters")

col1, col2 = st.columns(2)

with col1:
    test_size = st.slider(
        "Test Size",
        0.1, 0.5, 0.2
    )

with col2:
    n_iter = st.slider(
        "Random Search Iterations",
        5, 50, 20
    )

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42
)

# Model
model = DecisionTreeRegressor(random_state=42)

# Hyperparameters
params = {
    'criterion': [
        'squared_error',
        'friedman_mse',
        'absolute_error'
    ],
    'max_depth': randint(2, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=params,
    n_iter=n_iter,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)

with st.spinner("Training Model..."):
    random_search.fit(X_train, y_train)

# Best Model
best_model = random_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Results
st.subheader("Best Parameters")
st.json(random_search.best_params_)

st.subheader("Model Performance")

st.write(f"### Mean Squared Error: {mse:.4f}")
st.write(f"### Root Mean Squared Error: {rmse:.4f}")
st.write(f"### R² Score: {r2:.4f}")

# Visualization
st.subheader("Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(18, 10))

plot_tree(
    best_model,
    feature_names=X.columns,
    filled=True,
    fontsize=8,
    ax=ax
)

st.pyplot(fig)
