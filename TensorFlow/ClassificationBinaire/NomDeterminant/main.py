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

def main():
    # Charger les données d'entraînement
    with open('C:/Users/Jeffrey/Documents/GitHub/projectAI/TensorFlow/ClassificationBinaire/NomDeterminant/dictionnaireFR.txt', 'r') as f:
        mots_nom = f.readlines()

    with open('C:/Users/Jeffrey/Documents/GitHub/projectAI/TensorFlow/ClassificationBinaire/NomDeterminant/determinant.txt', 'r') as f:
        mots_determinant = f.readlines()
    
    # Concaténer les mots pour former les caractéristiques
    features_train = mots_nom + mots_determinant

    # Générer les étiquettes
    labels = np.concatenate((np.zeros(len(mots_nom)), np.ones(len(mots_determinant))))

    # Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(features_train)
    word_sequences = tokenizer.texts_to_sequences(features_train)

    # Remplir les séquences pour s'assurer qu'elles ont toutes la même longueur
    max_length = max(len(seq) for seq in word_sequences)
    features_train = pad_sequences(word_sequences, maxlen=max_length, padding='post')
    
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
    model.save("C:/Users/Jeffrey/Documents/GitHub/projectAI/TensorFlow/ClassificationBinaire/NomDeterminant/nomDeterminantAI.h5")

    # Sauvegarder le Tokenizer
    with open('C:/Users/Jeffrey/Documents/GitHub/projectAI/TensorFlow/ClassificationBinaire/NomDeterminant/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

if __name__ == '__main__':
    main()
