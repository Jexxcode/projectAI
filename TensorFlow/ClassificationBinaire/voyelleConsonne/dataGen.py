from random import random
from math import floor

# Fonction pour générer une consonne aléatoire
def generer_consonne():
    consonnes = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z']
    return consonnes[floor(random() * len(consonnes))]

# Fonction pour générer une voyelle aléatoire
def generer_voyelle():
    voyelles = ['a', 'e', 'i', 'o', 'u', 'y']
    return voyelles[floor(random() * len(voyelles))]

# Générer et écrire 10 000 consonnes dans le fichier consonne.txt
with open("C:/Users/Jeffrey/Documents/GitHub/projectAI/TensorFlow/ClassificationBinaire/voyelleConsonne/consonne.txt", "w") as file_consonne:
    for _ in range(10000):
        consonne = generer_consonne()
        file_consonne.write(consonne + "\n")

# Générer et écrire 10 000 voyelles dans le fichier voyelle.txt
with open("C:/Users/Jeffrey/Documents/GitHub/projectAI/TensorFlow/ClassificationBinaire/voyelleConsonne/voyelle.txt", "w") as file_voyelle:
    for _ in range(10000):
        voyelle = generer_voyelle()
        file_voyelle.write(voyelle + "\n")
