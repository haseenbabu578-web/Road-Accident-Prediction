# 🚗 Road Accident Severity Prediction

## 📌 Project Overview

This project uses **Machine Learning** to predict the **severity of road accidents** using a dataset.
The model is trained using a **Random Forest Classifier** and evaluates the prediction performance using different metrics.

The goal is to analyze accident data and identify the important factors that influence accident severity.

---

## 📂 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Pickle
* Google Colab

---

## 📊 Dataset

The dataset used in this project is:

**Road Accident Data.xlsx**

It contains different features related to road accidents such as:

* Location
* Weather conditions
* Road type
* Vehicle information
* Accident severity

The **target column** used for prediction is:

```
Accident_Severity
```

---

## ⚙️ Project Workflow

### 1️⃣ Import Libraries

Required Python libraries for:

* Data processing
* Visualization
* Machine learning

---

### 2️⃣ Upload Dataset

The dataset is uploaded using **Google Colab file uploader**.


uploaded = files.upload()


Then the dataset is loaded using:


data = pd.read_excel("Road Accident Data.xlsx")


---

### 3️⃣ Data Exploration

Basic dataset information is checked:

* Dataset shape
* Missing values
* First few rows

---

### 4️⃣ Data Cleaning

The dataset is cleaned by:

* Removing missing values
* Encoding categorical columns using **LabelEncoder**


data = data.dropna()


---

### 5️⃣ Feature and Target Selection

The dataset is divided into:

* **Features (X)** → Input variables
* **Target (y)** → Accident Severity


X = data.drop("Accident_Severity", axis=1)
y = data["Accident_Severity"]


---

### 6️⃣ Train-Test Split

The dataset is split into:

* **80% Training Data**
* **20% Testing Data**


train_test_split(X, y, test_size=0.2, random_state=42)


---

### 7️⃣ Model Training

A **Random Forest Classifier** is used to train the model.

```
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

---

### 8️⃣ Prediction

The trained model predicts accident severity on the test data.

```
y_pred = model.predict(X_test)
```

---

### 9️⃣ Model Evaluation

Model performance is evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

---

### 🔟 Feature Importance

The model identifies the **most important features affecting accident severity**.

A bar chart is generated using **Matplotlib**.

---

### 1️⃣1️⃣ Save Model

The trained model is saved using **Pickle**.

```python
pickle.dump(model, open("accident_model.pkl", "wb"))
```

Saved file:

```
accident_model.pkl
```

---

## 📈 Output

The program produces:

* Model Accuracy
* Confusion Matrix
* Classification Report
* Feature Importance Graph
* Saved Machine Learning Model

---

## ▶️ How to Run the Project

1. Open **Google Colab**
2. Upload the dataset
3. Run all code cells
4. The model will train and display results
5. The trained model will be saved as **accident_model.pkl**

---

## 🎯 Future Improvements

* Use larger datasets
* Apply advanced models (XGBoost, Neural Networks)
* Build a **web application using Flask or Streamlit**
* Add real-time accident prediction

---

## 👨‍💻 Author

**Haseen Babu**

---

