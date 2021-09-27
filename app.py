import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib 
import pandas as pd 
import fr_core_news_sm
nlp = fr_core_news_sm.load()
from spacy.lang.fr.stop_words import STOP_WORDS
import streamlit as st
import numpy as np

def main():
    
    optionSide = st.sidebar.selectbox(
        'Summary',
        ['Chargé','Copié/Collé'])


    if optionSide == 'Chargé':
     file = st.file_uploader("file", type = ["txt"])
     if st.button("Process"):
        if file is not None:
            text = str(file.read(), "utf-8")
    if (file != '') and (file != None):
        st.write('Texte initial : ')
        st.write(file)
        text_lemma = lemmatize_text(file)
        st.write('Resultat : ')
        texte_preprocess = text_preprocess(text_lemma)
        prediction= predict(texte_preprocess)
        st.success(prediction)

def lemmatize_text(texte):
    texte = "".join(ch for ch in texte if ch.isalnum() or ch==" ")
    texte = texte.replace(" +"," ").lower().strip()
    title= nlp(texte)
    title=" ".join([token.lemma_ for token in title if token.text not in STOP_WORDS])
    return title

def text_preprocess(lemma_texte):
    lemma_texte= lemma_texte.replace(" \r"," ").strip()
    lemma_texte= " ".join([token.lemma_ for token in nlp(lemma_texte)])
    tokenizer= tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(lemma_texte)
    tokenize_text= tokenizer.texts_to_sequences([lemma_texte])
    paddle_text= tf.keras.preprocessing.sequence.pad_sequences(tokenize_text, maxlen=248, padding="post")
    return paddle_text

def predict(texte_final):
    classifier = tf.keras.models.load_model("my_model.h5")
    prediction = np.argmax(classifier.predict(texte_final))
    probabilite = np.max(classifier.predict(texte_final) * 100).astype("float")
    if probabilite <65 :
        return prediction, 'Probabilité de prédiction faible'
    elif probabilite < 90:
        return prediction, 'Probabilité de prédiction élevée'
    else :
        return prediction, "Probabilité de prédiction très élevée"




if __name__ == "__main__":
       main()
