from random import random
from math import floor

file = open("nombres.txt", "w")
for i in range(10000):
    s = str(floor(random() * 1000000000))
    file.write(s[0:floor(random() * 15 + 1)] + "\n")
    
file = open("mots.txt", "w")
for i in range(10000):
    nbLettres = floor(random() * 16) + 1
    s = ""
    for i in range(nbLettres):
        ordChar = floor(random() * 26)
        baseCarac : int = ord('a')
        if random() < 0.5:
            baseCarac : int = ord('A')
        s += chr(ordChar + baseCarac)
    file.write(s + "\n")
