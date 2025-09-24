import mlflow
import pandas as pd
import numpy as np
import argparse
import warnings
import optuna
import os
from dotenv import load_dotenv
# import dagshub

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss
)

# load .env
load_dotenv()

def lda_dim_reduction(X: pd.DataFrame, y: pd.Series, n_comp: int = 3) -> np.ndarray:
    """LDA dimensionality reduction"""
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    return lda.fit_transform(X, y)


def data_splitting(X: np.ndarray, y: pd.Series,
                   rand_state: int = 42, test_size: float = 0.2):
    """Split data into train and test"""
    return train_test_split(X, y, random_state=rand_state, test_size=test_size)


def compute_metrics(X_train, y_train, y_pred_train,
                    X_test, y_test, y_pred_test,
                    model) -> dict:
    """Compute training & validation metrics"""
    y_prob_train = model.predict_proba(X_train)
    y_prob_test = model.predict_proba(X_test)

    metrics = {
        # test/validation
        "val_accuracy": accuracy_score(y_test, y_pred_test),
        "val_f1": f1_score(y_test, y_pred_test, average="weighted"),
        "val_recall": recall_score(y_test, y_pred_test, average="weighted"),
        "val_precision": precision_score(y_test, y_pred_test, average="weighted"),
        "val_roc_auc": roc_auc_score(y_test, y_prob_test, multi_class="ovr"),
        "val_log_loss": log_loss(y_test, y_prob_test),

        # training
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "train_f1": f1_score(y_train, y_pred_train, average="weighted"),
        "train_recall": recall_score(y_train, y_pred_train, average="weighted"),
        "train_precision": precision_score(y_train, y_pred_train, average="weighted"),
        "train_roc_auc": roc_auc_score(y_train, y_prob_train, multi_class="ovr"),
        "train_log_loss": log_loss(y_train, y_prob_train),
    }
    return metrics


def objective(trial, X_train, X_test, y_train, y_test):
    """Objective function for Optuna"""
    C = trial.suggest_loguniform("C", 1e-2, 1e2)
    gamma = trial.suggest_loguniform("gamma", 1e-3, 1e1)
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"])

    model = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=42)
    model.fit(X_train, y_train)

    # predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # metrics
    metrics = compute_metrics(X_train, y_train, y_pred_train,
                              X_test, y_test, y_pred_test, model)

    # logging ke MLflow (per trial)
    with mlflow.start_run(nested=True):
        mlflow.log_params({"C": C, "gamma": gamma, "kernel": kernel})
        mlflow.log_metrics(metrics)

    return metrics["val_accuracy"]

def main(args):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI_DEV"))
    mlflow.set_experiment("Mobile Price Range Prediction")
    # dagshub.init(
    #     repo_owner="ishala",
    #     repo_name="Membangun_Model_SML_Faishal",
    #     mlflow=True
    # )

    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # load dataset
    df = pd.read_csv(args.dataset)
    X, y = df.drop(columns=["price_range"]), df["price_range"]

    # reduce & split
    X_lda = lda_dim_reduction(X, y, n_comp=args.ncomps)
    X_train, X_test, y_train, y_test = data_splitting(X_lda, y)

    with mlflow.start_run(run_name="Optuna_Tuning") as parent_run:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, X_train, X_test, y_train, y_test),
            n_trials=args.n_trials
        )

        # log best trial
        best_trial = study.best_trial
        mlflow.log_params(best_trial.params)

        # retrain model with best params to recompute metrics
        best_model = SVC(**best_trial.params, probability=True, random_state=42)
        best_model.fit(X_train, y_train)
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        best_metrics = compute_metrics(X_train, y_train, y_pred_train,
                                       X_test, y_test, y_pred_test, best_model)

        # log best metrics
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