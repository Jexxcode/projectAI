import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Créer le modèle
def create_model(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 128),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(2, activation='softmax')  # Utiliser softmax pour la classification multiclasse
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
 
def concatener_sequences(sequences_lettres, sequences_nombres, max_length):
    features_train = []
    for seq_lettre, seq_nombre in zip(sequences_lettres, sequences_nombres):
        concatenated_seq = list(seq_lettre[:max_length]) + list(seq_nombre[:max_length])  # Convertir en liste pour concaténer
        padded_seq = concatenated_seq + [0] * max(0, max_length - len(concatenated_seq))
        features_train.append(padded_seq)
    return np.array(features_train)


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

def main():
    max_length = 20

    # Charger des données d'entraînement
    with open('C:\\Users\\Jeffrey\\Documents\\GitHub\\projectAI\\TensorFlow\\mots.txt', 'r') as f:
        mots_lettres = f.readlines()

    with open('C:\\Users\\Jeffrey\\Documents\\GitHub\\projectAI\\TensorFlow\\nombres.txt', 'r') as f:
        mots_nombres = f.readlines()
    
    # Convertir les mots en séquences d'entiers
    sequences_lettres, sequences_nombres, tokenizer = mots_to_sequences(mots_lettres, mots_nombres)

    # Padder les séquences pour qu'elles aient toutes la même longueur
    padded_sequences_lettres = pad_sequences_to_max_length(sequences_lettres, max_length)
    padded_sequences_nombres = pad_sequences_to_max_length(sequences_nombres, max_length)

    # Concaténer les séquences pour former les caractéristiques
    features_train = concatener_sequences(padded_sequences_lettres, padded_sequences_nombres,max_length)

    # Créer les étiquettes
    labels_lettres = np.ones((len(padded_sequences_lettres),), dtype=int)
    labels_nombres = np.zeros((len(padded_sequences_nombres),), dtype=int)
    labels_train = np.concatenate((labels_lettres, labels_nombres))
    labels_train = to_categorical(labels_train, num_classes=2)  # Encodage one-hot pour la classification binaire

    # Mélanger (shuffle) les caractéristiques et les étiquettes de manière synchronisée
    indices = np.random.permutation(len(features_train))
    features_train = features_train[indices]
    labels_train = labels_train[indices]

    # Créer le modèle à l'aide de la fonction create_model
    model = create_model(20000)  # le nombre de mots uniques dans le tokenizer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entraîner le modèle
    model.fit(features_train, labels_train, epochs=30, batch_size=100)

    # Enregistrer le modèle entraîné
    model.save("mon_modele.h5")

    # Sauvegarder le Tokenizer
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

if __name__ == '__main__':
  main()


