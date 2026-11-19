"""Allena una rete neurale (PyTorch) con early stopping e la valuta sul validation."""
import copy

import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn

from dati import SEED, carica_dati, dividi

EPOCHE_MAX = 300
PAZIENZA = 30          # epoche senza miglioramenti prima di fermarsi
LEARNING_RATE = 0.01


def crea_rete(n_input, n_classi):
    """Rete con due strati nascosti: n_input → 64 → 32 → n_classi."""
    return nn.Sequential(
        nn.Linear(n_input, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, n_classi),
    )


def allena_rete(X_train, y_train, X_val, y_val):
    """Allena la rete e restituisce l'accuratezza migliore sul validation e l'epoca in cui l'ha raggiunta."""
    torch.manual_seed(SEED)

    scaler = StandardScaler().fit(X_train)
    encoder = LabelEncoder().fit(y_train)

    X_train_t = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    X_val_t = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    y_train_t = torch.tensor(encoder.transform(y_train))
    y_val_num = encoder.transform(y_val)

    rete = crea_rete(X_train.shape[1], len(encoder.classes_))
    funzione_errore = nn.CrossEntropyLoss()
    ottimizzatore = torch.optim.Adam(rete.parameters(), lr=LEARNING_RATE)

    migliore_acc = 0
    migliore_epoca = 0
    migliori_pesi = None

    for epoca in range(1, EPOCHE_MAX + 1):
        punteggi = rete(X_train_t)
        errore = funzione_errore(punteggi, y_train_t)
        ottimizzatore.zero_grad()
        errore.backward()
        ottimizzatore.step()

        with torch.no_grad():
            previsioni_val = rete(X_val_t).argmax(dim=1)
        acc_val = accuracy_score(y_val_num, previsioni_val.numpy())

        # Early stopping: ricordo i pesi dell'epoca migliore...
        if acc_val > migliore_acc:
            migliore_acc = acc_val
            migliore_epoca = epoca
            migliori_pesi = copy.deepcopy(rete.state_dict())
        # ...e mi fermo se da troppe epoche non migliora
        elif epoca - migliore_epoca >= PAZIENZA:
            break

    rete.load_state_dict(migliori_pesi)
    return migliore_acc, migliore_epoca, epoca


def main():
    X, y = carica_dati()
    X_train, X_val, X_test, y_train, y_val, y_test = dividi(X, y)

    acc_val, epoca_migliore, epoca_finale = allena_rete(X_train, y_train, X_val, y_val)
    print(f"Rete neurale: validation {acc_val:.1%} all'epoca {epoca_migliore}")
    print(f"Allenamento fermato all'epoca {epoca_finale} (early stopping)")


if __name__ == "__main__":
    main()