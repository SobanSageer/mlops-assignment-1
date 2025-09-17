import mlflow
import mlflow.sklearn
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Define models
models = {
    "logistic_regression": LogisticRegression(max_iter=200),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "svm": SVC(kernel="linear", probability=True)
}

# Experiment setup
mlflow.set_experiment("mlops_assignment_1")

best_model_name = None
best_f1 = 0.0
best_model = None

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        # Log params & metrics
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        # Save model
        model_path = f"models/{name}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=iris.target_names,
                    yticklabels=iris.target_names)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plot_path = f"results/{name}_confusion_matrix.png"
        os.makedirs("results", exist_ok=True)
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        # Track best model by F1
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model
            best_run_id = run.info.run_id

# ✅ Register best model in MLflow Model Registry
if best_model is not None:
    print(f"\n🏆 Best model: {best_model_name} with F1 = {best_f1:.4f}")
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="mlops_assignment_1_best_model"
    )
