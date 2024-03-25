import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
import pickle

# Créer le modèle
def create_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Couche de sortie avec activation sigmoïde pour une classification binaire
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Concaténer les séquences de lettres et de nombres
def concat_sequences(sequences_letters, sequences_numbers):
    return sequences_letters + sequences_numbers

# Convertir les caractères en codes ASCII
def char_to_ascii(characters):
    return [ord(char) for char in characters]

# Préparer les étiquettes
def prepare_labels(total_samples):
    labels = np.concatenate((np.ones(total_samples // 2), np.zeros(total_samples // 2)))
    return labels

# Prétraitement de l'entrée
def preprocess_input(input_str):
    if input_str.isdigit() or input_str.isalpha():
        return [input_str]
    else:
        print("Input doit être une seule lettre ou un seul chiffre.")
        return None

# Processus d'entrée
def input_process(input_str):
    input_str = preprocess_input(input_str)
    if input_str:
        input_sequences = tokenizer.texts_to_sequences(input_str)
        padded_sequences = pad_sequences(input_sequences, maxlen=max_length, padding='post')
        return padded_sequences
    else:
        return None

def main():
    # Charger les données d'entraînement
    with open('C:/Users/Jeffrey/Documents/GitHub/projectAI/TensorFlow/ChiffreLettre/lettre.txt', 'r') as f:
        words_letters = f.readlines()

    with open('C:/Users/Jeffrey/Documents/GitHub/projectAI/TensorFlow/ChiffreLettre/chiffre.txt', 'r') as f:
        words_numbers = f.readlines()
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(words_letters + words_numbers)
    sequences_letters = tokenizer.texts_to_sequences(words_letters)
    sequences_numbers = tokenizer.texts_to_sequences(words_numbers)
    
    # Concaténer les séquences pour former les caractéristiques
    features_train = concat_sequences(sequences_letters, sequences_numbers)

    # Générer les étiquettes
    labels = prepare_labels(len(features_train))

    # Remplir les séquences pour s'assurer qu'elles ont toutes la même longueur
    max_length = 20
    features_train = pad_sequences(features_train, maxlen=max_length, padding='post')
    
    # Convertir les listes en tableaux numpy
    features_train = np.array(features_train)
    labels = np.array(labels)

    # Mélanger les données
    indices = np.random.permutation(len(features_train))
    features_train = features_train[indices]
    labels = labels[indices]

    # Créer le modèle
    model = create_model(input_shape=features_train.shape[1])

    # Entraîner le modèle
    model.fit(features_train, labels, epochs=30, batch_size=70)

    # Sauvegarder le modèle entraîné
    model.save("C:/Users/Jeffrey/Documents/GitHub/projectAI/TensorFlow/ChiffreLettre/ChiffreLettreAI.h5")

    # Sauvegarder le Tokenizer
    with open('C:/Users/Jeffrey/Documents/GitHub/projectAI/TensorFlow/ChiffreLettre/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

if __name__ == '__main__':
    main()
