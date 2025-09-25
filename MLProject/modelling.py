import mlflow
import warnings
import numpy as np
import pandas as pd
import argparse
from dotenv import load_dotenv
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, log_loss
)

# load env
load_dotenv()

warnings.filterwarnings("ignore")
np.random.seed(42)

def lda_dim_reduction(X, y, n_comp=3):
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    return lda.fit_transform(X, y)


def data_splitting(X, y, rand_state=42, test_size=0.2):
    return train_test_split(X, y, random_state=rand_state, test_size=test_size)


def compute_metrics(X_train, y_train, y_pred_train,
                    X_test, y_test, y_pred_test,
                    model):
    y_prob_train = model.predict_proba(X_train)
    y_prob_test = model.predict_proba(X_test)

    return {
        "val_accuracy": accuracy_score(y_test, y_pred_test),
        "val_f1": f1_score(y_test, y_pred_test, average="weighted"),
        "val_recall": recall_score(y_test, y_pred_test, average="weighted"),
        "val_precision": precision_score(y_test, y_pred_test, average="weighted"),
        "val_roc_auc": roc_auc_score(y_test, y_prob_test, multi_class="ovr"),
        "val_log_loss": log_loss(y_test, y_prob_test),

        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "train_f1": f1_score(y_train, y_pred_train, average="weighted"),
        "train_recall": recall_score(y_train, y_pred_train, average="weighted"),
        "train_precision": precision_score(y_train, y_pred_train, average="weighted"),
        "train_roc_auc": roc_auc_score(y_train, y_prob_train, multi_class="ovr"),
        "train_log_loss": log_loss(y_train, y_prob_train),
    }


def main(args):
    # load dataset
    df = pd.read_csv(args.dataset)
    X, y = df.drop(columns=["price_range"]), df["price_range"]

    # preprocessing
    X_lda = lda_dim_reduction(X, y, n_comp=args.ncomps)
    X_train, X_test, y_train, y_test = data_splitting(X_lda, y)

    # Random search ranges (contoh: C dan gamma SVM)
    C_range = np.logspace(-2, 2, 5)         # 0.01 → 100
    gamma_range = np.logspace(-3, 1, 5)     # 0.001 → 10
    kernels = ["rbf", "linear", "poly", "sigmoid"]

    best_accuracy = 0
    best_params = {}
    best_metrics = None
    best_model = None

    for C in C_range:
        for gamma in gamma_range:
            for kernel in kernels:
                run_name = f"random_C{C:.3f}_gamma{gamma:.3f}_kernel{kernel}"
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_params({"C": C, "gamma": gamma, "kernel": kernel})

                    # Train
                    model = SVC(C=C, gamma=gamma, kernel=kernel,
                                probability=True, random_state=42)
                    model.fit(X_train, y_train)

                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)

                    metrics = compute_metrics(
                        X_train, y_train, y_pred_train,
                        X_test, y_test, y_pred_test, model
                    )

                    mlflow.log_metrics(metrics)

                    # Cek best
                    if metrics["val_accuracy"] > best_accuracy:
                        best_accuracy = metrics["val_accuracy"]
                        best_params = {"C": C, "gamma": gamma, "kernel": kernel}
                        best_metrics = metrics
                        best_model = model

    # Save best run
    if best_model is not None:
        with mlflow.start_run(run_name="Best_Model"):
            mlflow.log_params(best_params)
            mlflow.log_metrics({"best_" + k: v for k, v in best_metrics.items()})
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model"
            )

        print("Best Params:", best_params)
        print("Best Metrics:", best_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cleaned_data.csv")
    parser.add_argument("--ncomps", type=int, default=3)
    args = parser.parse_args()
    main(args)
