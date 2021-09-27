import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib 
import pandas as pd 
import fr_core_news_sm
nlp = fr_core_news_sm.load()
from spacy.lang.fr.stop_words import STOP_WORDS

def main():
    methode = input("Veuillez sélectionner la méthode de prédiction : \n input \n upload \n     ")
    methode = methode.lower().replace(" ","")
    if methode=="input":
        texte = input("Veuillez entrer un commentaire : \n")
        lemma_texte = lemmatize_text(texte)
        texte_final= text_preprocess(lemma_texte)
        prediction= predict(texte_final)
        result= resultat(prediction)
    elif methode=="upload":
        print("Cette méthode n'est pas encore utilisable")
    else:
        print("Veuillez choisir une méthode utilisable")

def lemmatize_text(texte):
    texte = "".join(ch for ch in texte if ch.isalnum() or ch==" ")
    texte = texte.replace(" +"," ").lower().strip()
    title= nlp(title)
    title=" ".join([token.lemma_ for token in title if token.text not in STOP_WORDS])
    return title

def text_preprocess(lemma_texte):
    lemma_texte= lemma_texte.replace(" \r"," ").strip()
    lemma_texte= " ".join([token.lemma_ for token in nlp(lemma_texte)])
    tokenizer= tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(lemma_texte)
    tokenize_text= tokenizer.text_to_sequences([lemma_texte])
    paddle_text= tf.keras.preprocessing.sequence.pad_sequences(title, maxlen=163, padding="post")
    return paddle_text

def predict(texte_final):
    classifier = joblib.load(r"")
