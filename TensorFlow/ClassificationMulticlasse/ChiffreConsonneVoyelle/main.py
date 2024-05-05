import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
import pickle

# Créer le modèle pour une classification multiclasse avec trois classes de sortie
def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')  # Utilisation de softmax pour la classification multiclasse avec trois classes de sortie
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Préparer les étiquettes en one-hot encoding pour la classification multiclasse
def prepare_labels(total_samples, num_classes):
    labels = []
    samples_per_class = total_samples // num_classes

    for i in range(num_classes):
        labels.extend([i] * samples_per_class)

    return tf.keras.utils.to_categorical(labels, num_classes=num_classes)


# Reste du code inchangé

def main():
    # Charger les données d'entraînement TensorFlow/ClassificationMulticlasse/ChiffreConsonneVoyelle
    with open('TensorFlow/ClassificationMulticlasse/ChiffreConsonneVoyelle/consonne.txt', 'r') as f:
        consonnes = f.readlines()

    with open('TensorFlow/ClassificationMulticlasse/ChiffreConsonneVoyelle/voyelle.txt', 'r') as f:
        voyelles = f.readlines()

    with open('TensorFlow/ClassificationMulticlasse/ChiffreConsonneVoyelle/chiffre.txt', 'r') as f:
        chiffres = f.readlines()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(consonnes + voyelles + chiffres)
    sequences_consonnes = tokenizer.texts_to_sequences(consonnes)
    sequences_voyelles = tokenizer.texts_to_sequences(voyelles)
    sequences_chiffres = tokenizer.texts_to_sequences(chiffres)

    # Concaténer les séquences pour former les caractéristiques
    features_train = sequences_consonnes + sequences_voyelles + sequences_chiffres


    # Générer les étiquettes
    num_classes = 3  # Trois classes : consonne, voyelle, chiffre
    labels = prepare_labels(len(features_train), num_classes)

    # Remplir les séquences pour s'assurer qu'elles ont toutes la même longueur
    max_length = 1
    features_train = pad_sequences(features_train, maxlen=max_length, padding='post')

    # Convertir les listes en tableaux numpy
    features_train = np.array(features_train)

    # Mélanger les données
    indices = np.random.permutation(len(features_train))
    features_train = features_train[indices]
    labels = labels[indices]

    # Créer le modèle
    model = create_model(input_shape=features_train.shape[1], num_classes=num_classes)

    # Entraîner le modèle
    model.fit(features_train, labels, epochs=30, batch_size=70)

    # Sauvegarder le modèle entraîné
    model.save("TensorFlow/ClassificationMulticlasse/ChiffreConsonneVoyelle/modele_multiclasse.h5")

    # Sauvegarder le Tokenizer
    with open('TensorFlow/ClassificationMulticlasse/ChiffreConsonneVoyelle/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

if __name__ == '__main__':
    main()
