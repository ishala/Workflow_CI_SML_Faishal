import mlflow
import dagshub
import os
import warnings
import numpy as np
import pandas as pd
import argparse
import optuna
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

# --- DAGsHub + MLflow setup sekali ---
# auth
dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"), host="https://dagshub.com")

# init mlflow ke repo dagshub
dagshub.init(
    repo_owner=os.getenv("DAGSHUB_USER"),
    repo_name="SML_Membangun_Model",
    mlflow=True
)

mlflow.set_experiment("Mobile Price Range Prediction")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI_PROD", "https://dagshub.com/ishala/SML_Membangun_Model.mlflow"))
mlflow.set_experiment("Mobile Price Range Prediction")

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


def objective(trial, X_train, X_test, y_train, y_test):
    C = trial.suggest_loguniform("C", 1e-2, 1e2)
    gamma = trial.suggest_loguniform("gamma", 1e-3, 1e1)
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"])

    model = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=42)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = compute_metrics(X_train, y_train, y_pred_train,
                              X_test, y_test, y_pred_test, model)

    # nested run khusus untuk trial ini
    with mlflow.start_run(nested=True):
        mlflow.log_params({"C": C, "gamma": gamma, "kernel": kernel})
        mlflow.log_metrics(metrics)

    return metrics["val_accuracy"]


def main(args):
    # load dataset
    df = pd.read_csv(args.dataset)
    X, y = df.drop(columns=["price_range"]), df["price_range"]

    X_lda = lda_dim_reduction(X, y, n_comp=args.ncomps)
    X_train, X_test, y_train, y_test = data_splitting(X_lda, y)

    # di sini TIDAK ada start_run() manual
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, X_test, y_train, y_test),
        n_trials=args.ntrials
    )

    best_trial = study.best_trial
    # ini aman, karena parent run dari MLProject masih aktif
    mlflow.log_params(best_trial.params)

    best_model = SVC(**best_trial.params, probability=True, random_state=42)
    best_model.fit(X_train, y_train)
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    best_metrics = compute_metrics(X_train, y_train, y_pred_train,
                                   X_test, y_test, best_model)

    mlflow.log_metrics({"best_" + k: v for k, v in best_metrics.items()})

    print("Best Trial:", best_trial.params)
    print("Best Metrics:", best_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cleaned_data.csv")
    parser.add_argument("--ncomps", type=int, default=3)
    parser.add_argument("--ntrials", type=int, default=30)
    args = parser.parse_args()
    main(args)