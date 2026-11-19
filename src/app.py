"""App Streamlit: la Random Forest sceglie la coltura, Gemini spiega il perché."""
import os

import joblib
import pandas as pd
import streamlit as st
from google import genai

from dati import CARTELLA_PROGETTO

PERCORSO_MODELLO = CARTELLA_PROGETTO / "models" / "modello.joblib"
MODELLO_GEMINI = "gemini-3.5-flash"

# Per ogni colonna: etichetta da mostrare, minimo, massimo, valore iniziale
SLIDER = {
    "N": ("Azoto (N)", 0, 140, 90),
    "P": ("Fosforo (P)", 5, 145, 42),
    "K": ("Potassio (K)", 5, 205, 43),
    "temperature": ("Temperatura (°C)", 8.0, 44.0, 21.0),
    "humidity": ("Umidità (%)", 14.0, 100.0, 82.0),
    "ph": ("pH del terreno", 3.5, 10.0, 6.5),
    "rainfall": ("Pioggia (mm)", 20.0, 300.0, 203.0),
}


@st.cache_resource
def carica_modello():
    """Carica il modello salvato da allena.py."""
    return joblib.load(PERCORSO_MODELLO)


def crea_prompt(valori, classifica):
    """Costruisce il prompt per Gemini con i dati del terreno e le previsioni del modello."""
    righe_terreno = []
    for colonna, valore in valori.items():
        etichetta = SLIDER[colonna][0]
        righe_terreno.append(f"- {etichetta}: {valore}")

    righe_colture = []
    for coltura, prob in classifica.items():
        righe_colture.append(f"- {coltura}: {prob:.0%}")

    testo_terreno = "\n".join(righe_terreno)
    testo_colture = "\n".join(righe_colture)

    return f"""Sei un agronomo che spiega a un agricoltore italiano, in modo semplice e concreto.

Dati del terreno e del clima:
{testo_terreno}

Un modello di machine learning ha stimato queste colture come le più adatte (con la probabilità):
{testo_colture}

Compito:
1. Spiega in 3-4 frasi perché la prima coltura è adatta, collegandoti ai valori sopra.
2. Se la probabilità della prima coltura è bassa, dillo chiaramente.
3. Dai 2 consigli pratici.

Regole: rispondi in italiano, massimo 120 parole, non inventare dati che non sono sopra."""


def chiedi_a_gemini(prompt):
    """Manda il prompt a Gemini e restituisce il testo della risposta."""
    client = genai.Client(http_options={"retry_options": {"attempts": 1}})
    risposta = client.interactions.create(model=MODELLO_GEMINI, input=prompt)
    return risposta.output_text


# ---------------- L'interfaccia ----------------

st.title("🌱 Crop Advisor")
st.caption("La Random Forest sceglie la coltura, Gemini spiega il perché")

if not PERCORSO_MODELLO.exists():
    st.error("Modello non trovato: lancia prima `python src/allena.py`.")
    st.stop()

modello = carica_modello()

valori = {}
colonna_sx, colonna_dx = st.columns(2)
for i, (colonna, (etichetta, minimo, massimo, iniziale)) in enumerate(SLIDER.items()):
    if i % 2 == 0:
        with colonna_sx:
            valori[colonna] = st.slider(etichetta, minimo, massimo, iniziale)
    else:
        with colonna_dx:
            valori[colonna] = st.slider(etichetta, minimo, massimo, iniziale)

terreno = pd.DataFrame([valori])
probabilita = modello.predict_proba(terreno)[0]
classifica = pd.Series(probabilita, index=modello.classes_).sort_values(ascending=False).head(3)

st.subheader(f"Coltura consigliata: {classifica.index[0]} ({classifica.iloc[0]:.0%})")
if classifica.iloc[0] < 0.5:
    st.warning("Terreno insolito: il modello è poco sicuro di questa previsione.")
st.bar_chart(classifica)

if st.button("Spiegami perché"):
    if not os.getenv("GEMINI_API_KEY"):
        st.error("Manca la chiave: imposta GEMINI_API_KEY nel terminale prima di lanciare l'app.")
    else:
        with st.spinner("Gemini sta scrivendo..."):
            st.write(chiedi_a_gemini(crea_prompt(valori, classifica)))