
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


st.title("Decision Tree Classifier - Iris Dataset")

# Load Dataset
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(X.head())

# Parameters on Main Page
st.subheader("Model Parameters")

col1, col2 = st.columns(2)

with col1:
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

with col2:
    max_depth = st.slider("Max Depth", 1, 10, 3)

criterion = st.selectbox(
    "Criterion",
    ["gini", "entropy"]
)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42
)

# Model
model = DecisionTreeClassifier(
    max_depth=max_depth,
    criterion=criterion,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
st.subheader("Model Results")

accuracy = accuracy_score(y_test, y_pred)

st.write(f"### Accuracy Score: {accuracy:.4f}")

# Classification Report
st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

st.dataframe(pd.DataFrame(report).transpose())

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

st.write(cm)

# Tree Visualization
st.subheader("Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(14, 8))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    ax=ax
)

st.pyplot(fig)
