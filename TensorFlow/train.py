import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential

import pickle

# Créer le modèle

def create_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

 
def concatener_sequences(sequences_lettres, sequences_nombres ):
    return sequences_lettres + sequences_nombres

def mots_to_sequences(mots_lettres, mots_nombres):
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(mots_lettres + mots_nombres)
  sequences_lettres = tokenizer.texts_to_sequences(mots_lettres)
  sequences_nombres = tokenizer.texts_to_sequences(mots_nombres)
  return sequences_lettres, sequences_nombres, tokenizer


def pad_sequences_to_max_length(sequences, max_length):
    return pad_sequences(sequences, maxlen=max_length, padding='post')

def my_concatenate(arr1, arr2):
    return arr1 + arr2

def charToAscii(listeChar):
    resultat = [0] * len(listeChar)
    i = 0
    for str in listeChar:
       resultat[i] = [ord(char) for char in str]
       i = i + 1
    return resultat


def preparingLabels():
    i = 0
    resultat = [0] * 20000
    while(i < 20000):
        if (i < 10000):
            resultat[i] = 1
        else:
            resultat[i] = 0 
        i += 1

    return resultat

def main():
    # Charger des données d'entraînement
    with open('lettre.txt', 'r') as f:
        mots_lettres = f.readlines()

    with open('chiffre.txt', 'r') as f:
        mots_nombres = f.readlines()
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(mots_lettres + mots_nombres)
    sequences_lettres = tokenizer.texts_to_sequences(mots_lettres)
    sequences_nombres = tokenizer.texts_to_sequences(mots_nombres)

    # Concatenate the sequences to form the features
    features_train = sequences_lettres + sequences_nombres

    # Generate labels
    labels = [1] * len(sequences_lettres) + [0] * len(sequences_nombres)

    # Pad sequences to ensure they all have the same length
    max_length = 20
    features_train = pad_sequences(features_train, maxlen=max_length, padding='post')

    # Convert lists to numpy arrays
    features_train = np.array(features_train)
    labels = np.array(labels)

    # Shuffle the data
    indices = np.random.permutation(len(features_train))
    features_train = features_train[indices]
    labels = labels[indices]

    # Create the model
    model = create_model(input_shape=features_train.shape[1])

    # Train the model
    model.fit(features_train, labels, epochs=30, batch_size=70)

    # Save the trained model
    model.save("mon_modele.h5")

    # Save the Tokenizer
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

if __name__ == '__main__':
    main()