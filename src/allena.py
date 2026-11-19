"""Confronta albero e Random Forest sul validation, valuta il vincitore sul test e lo salva."""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from dati import CARTELLA_PROGETTO, SEED, carica_dati, dividi

CARTELLA_MODELLI = CARTELLA_PROGETTO / "models"
CARTELLA_REPORT = CARTELLA_PROGETTO / "reports"


def salva_grafico_importanza(modello, colonne):
    """Salva un grafico a barre con l'importanza di ogni colonna."""
    importanze = pd.Series(modello.feature_importances_, index=colonne).sort_values()

    fig, ax = plt.subplots(figsize=(7, 4))
    importanze.plot.barh(ax=ax, color="#1f4e79")
    ax.set_title("Quanto conta ogni colonna per il modello")
    ax.set_xlabel("Importanza")
    fig.tight_layout()

    CARTELLA_REPORT.mkdir(exist_ok=True)
    fig.savefig(CARTELLA_REPORT / "importanza_colonne.png", dpi=120)
    plt.close(fig)


def main():
    X, y = carica_dati()
    X_train, X_val, X_test, y_train, y_val, y_test = dividi(X, y)

    candidati = {
        "Albero (profondità 12)": DecisionTreeClassifier(max_depth=12, random_state=SEED),
        "Random Forest (100 alberi)": RandomForestClassifier(n_estimators=100, random_state=SEED),
    }

    # 1. Confronto sul validation
    print("Accuratezza sul validation:")
    risultati = {}
    for nome, modello in candidati.items():
        modello.fit(X_train, y_train)
        risultati[nome] = accuracy_score(y_val, modello.predict(X_val))
        print(f"  {nome:<28} {risultati[nome]:.1%}")

    # 2. Il vincitore, e solo lui, fa il test
    vincitore = max(risultati, key=risultati.get)
    modello = candidati[vincitore]
    acc_test = accuracy_score(y_test, modello.predict(X_test))
    print(f"\nVincitore: {vincitore}")
    print(f"Accuratezza sul test: {acc_test:.1%}")

    # 3. Salvo grafico e modello
    salva_grafico_importanza(modello, X.columns)
    CARTELLA_MODELLI.mkdir(exist_ok=True)
    joblib.dump(modello, CARTELLA_MODELLI / "modello.joblib")
    print(f"\nModello salvato in {CARTELLA_MODELLI / 'modello.joblib'}")
    print(f"Grafico salvato in {CARTELLA_REPORT / 'importanza_colonne.png'}")


if __name__ == "__main__":
    main()