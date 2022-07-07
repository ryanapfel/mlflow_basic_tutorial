import mlflow
from dotenv import load_dotenv
import os
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature

import click


def pre_process_one_hot(df):
    cat = df.select_dtypes(include="O").keys()
    return pd.get_dummies(df, columns=cat)


def pre_process_drop_categoricals(df):
    cat = df.select_dtypes(include="O").keys()
    return df.drop(cat, axis=1)


def get_split_data_predict_mean(df, random_state=42):
    df["mean_grade"] = (df["G1"] + df["G2"] + df["G3"]) / 3

    Y = df["mean_grade"]
    X = df.drop(["G1", "G2", "G3", "mean_grade"], axis=1)

    return train_test_split(X, Y, test_size=0.33, random_state=random_state)


def get_split_data_predict_percentile(df, random_state=42):
    df["mean_grade"] = (df["G1"] + df["G2"] + df["G3"]) / 3
    Y = df["mean_grade"]
    X = df.drop(["G1", "G2", "G3", "mean_grade"], axis=1)

    return train_test_split(X, Y, test_size=0.33, random_state=random_state)


@click.group()
def cli():
    pass


@cli.command()
def lowerquality():
    load_dotenv()
    # create an experiment with the models that you want to save
    mlflow.set_tracking_uri(os.environ.get("TRACKING_URI"))
    mlflow.set_experiment("test-sklearn-autolog")

    mlflow.sklearn.autolog()
    # train a model
    model = LinearRegression()

    df = pd.read_csv("data/student_data.csv")
    # dropping categoricals here makes this lesser quality
    df = pre_process_drop_categoricals(df)
    X_train, X_test, Y_train, Y_test = get_split_data_predict_mean(df)

    with mlflow.start_run() as run:
        model.fit(X_train, Y_train)
        metrics = mlflow.sklearn.eval_and_log_metrics(
            model, X_test, Y_test, prefix="val_"
        )
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            registered_model_name="sk-learn-random-forest-reg-model",
            input_example=X_test.iloc[:1],
            signature=signature,
        )

        print(metrics)


@cli.command()
def higherquality():
    load_dotenv()
    # create an experiment with the models that you want to save
    mlflow.set_tracking_uri(os.environ.get("TRACKING_URI"))
    mlflow.set_experiment("test-sklearn-autolog")

    mlflow.sklearn.autolog()
    # train a model

    df = pd.read_csv("data/student_data.csv")
    # one hotting to show improved quality
    df = pre_process_one_hot(df)
    X_train, X_test, Y_train, Y_test = get_split_data_predict_mean(df)

    model = LinearRegression()

    with mlflow.start_run() as run:
        model.fit(X_train, Y_train)
        metrics = mlflow.sklearn.eval_and_log_metrics(
            model, X_test, Y_test, prefix="val_"
        )
        signature = infer_signature(X_train, model.predict(X_train))

        print(metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            registered_model_name="sk-learn-random-forest-reg-model",
            input_example=X_test.iloc[:1],
            signature=signature,
        )


if __name__ == "__main__":
    cli()
