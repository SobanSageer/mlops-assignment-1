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

# Define models with params
models = {
    "logistic_regression": LogisticRegression(max_iter=200),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "svm": SVC(kernel="linear", probability=True)
}

# Make sure models folder exists
os.makedirs("models", exist_ok=True)

# Start MLflow experiment
mlflow.set_experiment("mlops_assignment_1")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        # Log params (example: hyperparameters)
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Save model artifact
        model_path = f"models/{name}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plot_path = f"results/{name}_confusion_matrix.png"
        os.makedirs("results", exist_ok=True)
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        print(f"✅ Finished training and logging {name}")
