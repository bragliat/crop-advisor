"""Caricamento del dataset e divisione in train, validation e test."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CARTELLA_PROGETTO = Path(__file__).resolve().parent.parent
PERCORSO_DATI = CARTELLA_PROGETTO / "data" / "Crop_recommendation.csv"
SEED = 42


def carica_dati():
    """Legge il CSV e restituisce features (X) e target (y)."""
    df = pd.read_csv(PERCORSO_DATI)
    X = df.drop(columns="label")
    y = df["label"]
    return X, y


def dividi(X, y):
    """Divide in 70% train, 15% validation e 15% test, stratificati per coltura."""
    X_train, X_resto, y_train, y_resto = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_resto, y_resto, test_size=0.5, random_state=SEED, stratify=y_resto
    )
    return X_train, X_val, X_test, y_train, y_val, y_test