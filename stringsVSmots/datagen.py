from random import random
from math import floor

lignes = open("mots.txt").readlines()
mots = [mot.rstrip("\n") for mot in lignes]

file = open("strings.txt", "w")
for i in range(len(mots)):
    nbLettres = len(mots[i])
    s = ""
    for i in range(nbLettres):
        ordChar = floor(random() * 26)
        baseCarac : int = ord('a')
        s += chr(ordChar + baseCarac)
    file.write(s + "\n")
