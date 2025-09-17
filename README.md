# MLOps Assignment 1 – Model Training, Tracking & Registration

📌 Problem Statement
The goal of this assignment is to:
1. Use GitHub for version control and reproducibility.  
2. Train and compare multiple ML models on a dataset.  
3. Use MLflow for experiment tracking and logging (parameters, metrics, artifacts).  
4. Monitor model performance and register the best model in MLflow Model Registry.  

---

📊 Dataset Description
- Dataset: Iris dataset (`sklearn.datasets.load_iris`)  
- Features: 4 numerical features (sepal length, sepal width, petal length, petal width)  
- Target: 3 classes of Iris flowers (`setosa`, `versicolor`, `virginica`)  
- Train/Test Split: 80% train, 20% test  

---

🤖 Model Selection & Comparison
We trained and compared 3 machine learning models:
1. Logistic Regression  
2. Random Forest Classifier  
3. Support Vector Machine (SVM, linear kernel)  

 📈 Metrics Used
- Accuracy  
- Precision  
- Recall  
- F1-Score  

| Model                | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 1.00     | 1.00      | 1.00   | 1.00     |
| Random Forest         | 1.00     | 1.00      | 1.00   | 1.00     |
| SVM                   | 1.00     | 1.00      | 1.00   | 1.00     |





📂 MLflow Tracking & Logging
For each model, MLflow tracked:
- Parameters (hyperparameters, kernel type, n_estimators, etc.)  
- Metrics (accuracy, precision, recall, F1)  
- Artifacts:
  - Saved models (`.joblib`)  
  - Confusion matrix plots  




- Best model selected based on **F1-score**.  
- Registered in MLflow Model Registry.  



**Model Info:**
- Name: `mlops_assignment_1_best_model`  
- Version: `1 and 2`  
- Artifact Path: `mlruns/.../artifacts/model`  

---

⚡ How to Run the Project

 1️⃣ Clone Repository
```bash
git clone https://github.com/SobanSageer/mlops-assignment-1.git
cd mlops-assignment-1
python src/train_models.py
mlflow ui
