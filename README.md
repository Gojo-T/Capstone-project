# _Hi, "I am Tushar" 👋


## Capstone-project

## 📖 Project Overview
Forensic dentistry is a branch of forensic medicine that helps identify individuals using dental measurements. This project utilizes **machine learning** techniques to predict **gender** based on dental metrics.

## 🎯 Objectives
- **Analyze** dental data and its relationship with gender.
- **Implement machine learning models** for gender classification.
- **Evaluate and compare model performance**.

## 🚀 Tech Stack
- **Programming Language**: Python
- **Tools**: Jupyter Notebook, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost
- **Project Difficulty level** : Rookie/ Basic

## 📂 Dataset Description
**File:** `Dentistry Dataset.csv`

| Feature Name                           | Description |
|----------------------------------------|-------------|
| **Age**                                | The age of the individual |
| **Gender (Target Variable)**           | Male (1) / Female (0) |
| **Inter-canine distance intraoral**    | Measurement between upper canine teeth |
| **Right & Left Canine Width Casts**    | Width of the right and left canines |
| **Canine Index**                       | Canine index measurement |

## 🛠 Methodology
### 1️⃣ **Plot gender distribution**
```python

plt.figure(figsize=(6, 6))
gender_counts.plot(kind='bar', color=['skyblue', 'pink'], title="Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()
```
![Image](https://github.com/user-attachments/assets/a1e70dc3-b9af-4427-a3ee-6130ba028a7f)


### 2️⃣ **Scatter plot: Inter-canine distances (intraoral vs. cast)**
```python
plt.figure(figsize=(8, 8))
plt.scatter(data['inter canine distance intraoral'], data['intercanine distance casts'], alpha=0.5, c='green')
plt.title("Inter-Canine Distance Comparison (Intraoral vs. Cast)")
plt.xlabel("Intraoral Inter-Canine Distance")
plt.ylabel("Cast Inter-Canine Distance")
plt.show()
```
![Image](https://github.com/user-attachments/assets/10980cdd-1678-42a0-9821-d6a083331eaa)


### 3️⃣ **Drop any non-numerical columns that won't contribute to the correlation matrix**
```python
numerical_data = data.drop(['Sl No', 'Sample ID', 'Gender'], axis=1)

# Compute the correlation matrix
correlation_matrix = numerical_data.corr()

# Plot a heat map of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title("Heat Map of Feature Correlations")
plt.show()
```
![Image](https://github.com/user-attachments/assets/2067cc43-2c16-41ae-a48c-14a82d49f98e)


### 3️⃣ **Model Building**
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV

# Evaluate model with Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Calculate ROC-AUC Score
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) 
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='blue')
plt.plot([0, 1], [0, 1], 'r--', label="Random Classifier")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Perform Hyperparameter Tuning (Grid Search)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

# Output the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy Score from Grid Search:", grid_search.best_score_)
```
![Image](https://github.com/user-attachments/assets/a9317daa-899e-4b90-9247-7bef657f48ba)

## 📊 Results & Analysis
| Model                  | Accuracy |
|------------------------|-------------|
| Random Forest        | 0.895454545454545         |


## 📂 Installation & Usage
### 🔹 **Clone the Repository**
```bash
git clone https://github.com/Gojo-T?tab=repositories
```

### 🔹 **Install Dependencies**
```bash
!pip install numpy pandas matplotlib seaborn scikit-learn xgboost

```

### 🔹 **Run the Jupyter Notebook**
```bash
jupyter notebook
```

## 📂 References
- **Scikit-learn Documentation**: [https://scikit-learn.org](https://scikit-learn.org)
---

📌 **Author:** _**Tushar Govind Khairnar**_  

📌 **GitHub Repository:** _https://github.com/Gojo-T?tab=repositories

