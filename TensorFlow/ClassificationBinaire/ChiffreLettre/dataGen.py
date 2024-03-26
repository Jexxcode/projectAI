from random import random
from math import floor
from random import randint

with open("TensorFlow/ClassificationBinaire/ChiffreLettre/chiffre.txt", "w") as file:
    # Générer et écrire 10 000 chiffres aléatoires entre 0 et 9 dans le fichier
    for _ in range(10000):
        s = str(randint(0, 9))  # Générer un chiffre aléatoire entre 0 et 9
        file.write(s + "\n")
    
file = open("TensorFlow/ClassificationBinaire/ChiffreLettre/lettre.txt", "w")
for i in range(10000):
    nbLettres = 1
    s = ""
    for i in range(nbLettres):
        ordChar = floor(random() * 26)
        baseCarac : int = ord('a')
        if random() < 0.5:
            baseCarac : int = ord('A')
        s += chr(ordChar + baseCarac)
    file.write(s + "\n")
