from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import numpy as np

def activation_function(prediction):
    if prediction > 0.5:  # Modifier le seuil selon vos besoins
        return "determinant"
    else:
        return "nom"

def preprocess_input(user_input, tokenizer, max_length):
    if len(user_input.split()) == 1:  # Vérifier si l'entrée est un seul mot
        if user_input.isdigit() or user_input.isalpha():
            sequence = tokenizer.texts_to_sequences([user_input])
            padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
            return padded_sequence
        else:
            print("L'entrée doit être un seul mot.")
            return None
    else:
        print("L'entrée doit être un seul mot.")
        return None

def main():
    max_length = 4  # Modifier la longueur de l'entrée pour correspondre à celle attendue par le modèle

    # Charger le modèle entraîné
    model = load_model("nomDeterminantAI.h5")

    # Charger le Tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    while True:
        # Demander à l'utilisateur de saisir une entrée
        user_input = input("Saisissez votre entrée (ou 'q' pour quitter) : ")
    
        # Vérifier si l'utilisateur souhaite quitter
        if user_input.lower() == 'q':
            print("Au revoir !")
            break
      
        # Prétraiter l'entrée utilisateur
        preprocessed_input = preprocess_input(user_input, tokenizer, max_length)
        if preprocessed_input is not None and preprocessed_input.size > 0:
            # Faire une prédiction avec le modèle
            prediction = model.predict(preprocessed_input)

            # Afficher la prédiction
            print("Prédiction :", prediction)
            print("Type de l'entrée :", activation_function(prediction))

if __name__ == '__main__':
    main()
