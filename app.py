import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib 
import pandas as pd 
import fr_core_news_sm
nlp = fr_core_news_sm.load()
from spacy.lang.fr.stop_words import STOP_WORDS
import streamlit as st
import numpy as np
import io
import json

def main():
    
    optionSide = st.sidebar.selectbox(
        'Options',
        ['Chargé','Copié/Collé'])


    if optionSide == 'Chargé':
     file = st.file_uploader("file", type = ["txt"])
     if st.button("Process"):
        if file is not None:
            text = str(file.read(), "utf-8")
            st.write('Texte initial : ')
            st.write(text)
            text_lemma = lemmatize_text(text)
            st.write('Resultat : ')
            texte_preprocess = text_preprocess(text_lemma)
            prediction, probabilite= predict(texte_preprocess)
            st.success("Le nombre d'étoile prédit pour cet avis est de : {}  Avec une probabilité de : {} %".format(prediction, probabilite))       

    elif optionSide== "Copié/Collé":
        file = st.text_area("Entrez le commentaire", height=250)
        file_lemma = lemmatize_text(file)
        texte_preprocess = text_preprocess(file_lemma)
        prediction, probabilite= predict(texte_preprocess)
        st.success("Le nombre d'étoile prédit pour cet avis est de : {}  Avec une probabilité de : {}".format(prediction, probabilite))       
    elif (file != '') and (file != None):
        st.write('Texte initial : ')
        st.write(file)
        text_lemma = lemmatize_text(file)
        st.write('Resultat : ')
        texte_preprocess = text_preprocess(text_lemma)
        prediction, probabilite= predict(texte_preprocess)
        st.success(prediction, probabilite)


def lemmatize_text(texte):
    texte = "".join(ch for ch in texte if ch.isalnum() or ch==" ")
    texte = texte.replace(" +"," ").lower().strip()
    title= nlp(texte)
    title=" ".join([token.lemma_ for token in title if token.text not in STOP_WORDS])
    return title

def text_preprocess(lemma_texte):
    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer =  tf.keras.preprocessing.text.tokenizer_from_json(data)
    tokenizer.fit_on_texts([lemma_texte])
    tokenize_text= tokenizer.texts_to_sequences([lemma_texte])
    paddle_text= tf.keras.preprocessing.sequence.pad_sequences(tokenize_text, maxlen=248, padding="post")
    return paddle_text

def predict(texte_final):
    classifier = tf.keras.models.load_model("my_model.h5")
    prediction = np.argmax(classifier.predict(texte_final))
    probabilite = np.max(classifier.predict(texte_final)) * 100
    return prediction, probabilite


if __name__ == "__main__":
       main()
