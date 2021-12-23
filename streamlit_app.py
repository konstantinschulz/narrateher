import streamlit as st

from enums import Reaction
from eval import predict_score, map_score_to_reaction
from train import train_sklearn

train_sklearn()

st.title('Lass dich von Femi-Bot Maggie bewerten!')
st.subheader('Beispiele')
sample_text: str = "Das Männerbild der meisten Feministinnen ist unverrückbar. Die wollen gar keinen einsichtigen, reflektierten Mann, die wollen einen Punch Bag zum nimmermüden Draufhauen."
st.write(sample_text + " (Christian Ulmen)")
sample_text = ""
st.subheader(f"Femi-Score: {predict_score(sample_text)}% {Reaction.bad.value}")
st.subheader("Probier es selbst:")
input_text: str = st.text_input('Eingabe', '', placeholder="Feminismus ist für alle da.")
current_score: int = predict_score(input_text)
st.subheader('Dein Score: ' + (f"{current_score}% {map_score_to_reaction(current_score)}" if input_text else ""))
