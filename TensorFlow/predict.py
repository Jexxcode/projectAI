import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def activation_function(prediction):
    # Extraire l'indice de la classe prédite (0 pour lettre, 1 pour nombre)
    predicted_class_index = np.argmax(prediction)
    
    # Utiliser l'indice pour déterminer le type de l'entrée
    if predicted_class_index == 0:
        return "lettre"
    else:
        return "nombre"

def preprocess_input(user_input, tokenizer, max_length):
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    return padded_sequence

def main():
    max_length = 20

    # Charger le modèle entraîné
    model = load_model("mon_modele.h5")

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

        # Faire une prédiction avec le modèle
        prediction = model.predict(preprocessed_input)

        # Afficher la prédiction
        print("Prédiction :", prediction)
        print("Type de l'entrée :", activation_function(prediction))


if __name__ == '__main__':
    main()
