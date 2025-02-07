# Heart Disease Risk Prediction Dataset
_Synthetic Heart Disease Risk Prediction Dataset: A Comprehensive Collection_

---

## **Overview**

This repository contains a synthetic dataset of 70,000 samples designed for predicting the risk of heart disease. Each row in the dataset represents a patient, with binary (Yes/No) indicators for symptoms, lifestyle factors, and medical history, along with a computed risk label indicating whether the patient is at high or low risk of developing heart disease.

The dataset is part of the **EarlyMed** initiative, a student-driven project developed by students from **Vellore Institute of Technology (VIT)**. EarlyMed aims to leverage data science and machine learning for early detection and prevention of chronic diseases.

---

## **Dataset Description**

### **Features**
The dataset consists of **16 columns** (features) and **70,000 rows** (samples). Below is a detailed breakdown of the features:

| Column Name              | Description                                                                 | Data Type   | Example Values       |
|--------------------------|-----------------------------------------------------------------------------|-------------|----------------------|
| `chest_pain`             | Presence of chest pain (Yes/No)                                             | Binary      | 0 (No), 1 (Yes)      |
| `shortness_of_breath`    | Difficulty breathing (Yes/No)                                               | Binary      | 0 (No), 1 (Yes)      |
| `fatigue`                | Persistent tiredness without an obvious cause (Yes/No)                      | Binary      | 0 (No), 1 (Yes)      |
| `palpitations`           | Irregular or rapid heartbeat (Yes/No)                                       | Binary      | 0 (No), 1 (Yes)      |
| `dizziness`              | Episodes of lightheadedness or fainting (Yes/No)                            | Binary      | 0 (No), 1 (Yes)      |
| `swelling`               | Swelling in legs/ankles due to fluid retention (Yes/No)                     | Binary      | 0 (No), 1 (Yes)      |
| `radiating_pain`         | Pain radiating to arms, jaw, neck, or back (Yes/No)                         | Binary      | 0 (No), 1 (Yes)      |
| `cold_sweats`            | Cold sweats and nausea (Yes/No)                                             | Binary      | 0 (No), 1 (Yes)      |
| `age`                    | Patient's age in years                                                      | Continuous  | 25, 45, 65           |
| `hypertension`           | History of high blood pressure (Yes/No)                                     | Binary      | 0 (No), 1 (Yes)      |
| `cholesterol_high`       | Elevated cholesterol levels (Yes/No)                                        | Binary      | 0 (No), 1 (Yes)      |
| `diabetes`               | Diagnosis of diabetes (Yes/No)                                              | Binary      | 0 (No), 1 (Yes)      |
| `smoker`                 | Smoking history (Yes/No)                                                    | Binary      | 0 (No), 1 (Yes)      |
| `obesity`                | Obesity status (Yes/No)                                                     | Binary      | 0 (No), 1 (Yes)      |
| `family_history`         | Family history of heart disease (Yes/No)                                    | Binary      | 0 (No), 1 (Yes)      |
| `risk_label`             | Risk of heart disease (Binary label: 0 = Low risk, 1 = High risk)           | Binary      | 0, 1                 |

---

## **Provenance**

### **Sources**
The dataset is inspired by a combination of clinical guidelines, research studies, and publicly available datasets on heart disease. Key sources include:
- *Harrison's Principles of Internal Medicine* by J. Larry Jameson et al.
- *Mayo Clinic Cardiology* by Joseph G. Murphy et al.
- Framingham Heart Study
- American Heart Association (AHA) Guidelines
- Centers for Disease Control and Prevention (CDC)
- World Health Organization (WHO)

### **Collection Methodology**
This dataset was synthetically generated using Python libraries such as `numpy` and `pandas`. The process involved:
1. **Feature Selection**: Symptoms and risk factors were chosen based on their clinical relevance to heart disease.
2. **Data Generation**: Binary features were randomly generated using realistic probabilities, while continuous variables like `age` were sampled from a realistic range (e.g., 20–80 years).
3. **Risk Label Assignment**: The output label (`risk_label`) was computed based on a combination of risk factors and symptoms. Patients with multiple high-risk factors (e.g., smoking, hypertension, diabetes) were more likely to be labeled as high risk (`1`).
4. **Balancing Classes**: The dataset was intentionally balanced to ensure an equal distribution of high-risk (`1`) and low-risk (`0`) cases.
5. **Validation**: Synthetic data patterns were cross-referenced with clinical guidelines and real-world datasets to ensure realism and consistency.

### **Ethical Considerations**
Since the dataset is synthetically generated, it does not contain any personally identifiable information (PII) or real patient data, ensuring compliance with ethical and privacy standards. The synthetic nature of the dataset avoids potential biases present in real-world data while maintaining realistic correlations between features.

---

## **How to Use This Dataset**

### **Prerequisites**
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

### **Steps**

#### **1. Load the Dataset**
Use Python's `pandas` library to load the dataset:
```python
import pandas as pd
df = pd.read_csv('heart_disease_risk_dataset_earlymed.csv')
```

## 2. Exploratory Data Analysis (EDA)
Analyze feature distributions, correlations, and class balance:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize class distribution
sns.countplot(x='risk_label', data=df)
plt.title('Class Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
```
---
## 3. Preprocessing

Normalize continuous variables (e.g., age) and split the dataset into training and testing sets:
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = df.drop('risk_label', axis=1)
y = df['risk_label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize continuous variables
scaler = StandardScaler()
X_train[['age']] = scaler.fit_transform(X_train[['age']])
X_test[['age']] = scaler.transform(X_test[['age']])
```
## 4. Train a Machine Learning Model
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```
## Applications
This dataset can be used for:
1. Training machine learning models for heart disease risk prediction.
2. Identifying key risk factors contributing to heart disease.
3. Developing decision support systems for early detection of cardiovascular risks.
4. Educational purposes, such as teaching predictive modeling in healthcare.

To contribute in the kaggle: https://www.kaggle.com/datasets/mahatiratusher/heart-disease-risk-prediction-dataset/data

## Limitations
- **Synthetic Nature :** The dataset is synthetically generated and may not fully replicate real-world complexities.
- **Simplified Output :** The output label is binary (0 or 1). In real-world scenarios, risk stratification might involve multiple levels (e.g., low, moderate, high).
- **Missing Data :** No missing values are included in this dataset.

## Contributors
This dataset was created as part of the EarlyMed initiative by students from Vellore Institute of Technology (VIT) :

- Mahatir Ahmed Tusher
- Saket Choudary Kongara
- Vangapalli Sivamani

## License
This project is licensed under the MIT License . You are free to use, modify, and distribute the dataset for any purpose, provided you include the original license and copyright notice.
For more details, see the LICENSE file.

## Contact
For questions, feedback, or collaboration opportunities, please reach out to the contributors:
1. Mahatir Ahmed Tusher: mahatirahmedtusher123@gmail.com
2. Saket Choudary Kongara: saketchoudarykongara@gmail.com
3. Vangapalli Sivamani: vangapallisivamani@gmail.com

### Acknowledgments
We would like to thank the authors of the books, research papers, and existing datasets that inspired this work. Special thanks to the open-source community for tools like numpy, pandas, and scikit-learn, which made data generation and analysis possible.

Thank you for using this dataset! We hope it contributes to meaningful advancements in healthcare analytics and predictive modeling.
