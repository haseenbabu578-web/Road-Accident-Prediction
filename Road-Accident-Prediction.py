import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==============================
# Step 2: Upload File
# ==============================

uploaded = files.upload()

# Make sure file name matches exactly
data = pd.read_excel("Road Accident Data.xlsx")

print("Data Loaded Successfully ✅")
print(data.head())


# ==============================
# Step 3: Basic Info
# ==============================

print("\nShape of Dataset:", data.shape)
print("\nMissing Values:\n", data.isnull().sum())


# ==============================
# Step 4: Data Cleaning
# ==============================

# Remove missing values
data = data.dropna()

# Encode categorical columns
le = LabelEncoder()

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

print("\nData After Cleaning:")
print(data.head())


# ==============================
# Step 5: Define Features & Target
# ==============================

# ⚠️ Change column name if needed
target_column = "Accident_Severity"

X = data.drop(target_column, axis=1)
y = data[target_column]


# ==============================
# Step 6: Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# Step 7: Train Model
# ==============================

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nModel Training Complete ✅")


# ==============================
# Step 8: Predictions
# ==============================

y_pred = model.predict(X_test)


# ==============================
# Step 9: Evaluation
# ==============================

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ==============================
# Step 10: Feature Importance
# ==============================

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:\n")
print(feature_importances)


# Plot Feature Importance
plt.figure()
plt.bar(feature_importances['Feature'], feature_importances['Importance'])
plt.xticks(rotation=90)
plt.title("Feature Importance")
plt.show()


# ==============================
# Step 11: Save Model
# ==============================

pickle.dump(model, open("accident_model.pkl", "rb"))

print("\nModel Saved as accident_model.pkl ✅")


