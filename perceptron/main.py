from random import random
from random import shuffle
from math import floor, ceil

def FonctionActivation(value : float) -> bool:
    if(value < 0.5):
        return False
    else:
        return True

# ------------------------------------------------------------------------------------------------
class Perceptron:
    p1 : float = 0
    p2 : float = 0
    def __init__(self):
        self.p1 = random()
        self.p2 = random()
        pass

    def f(self, i1 : float, i2 : float) -> float:
        return self.p1 * i1 + self.p2 * i2

# ------------------------------------------------------------------------------------------------
def CreerReseauNeurones(longueurChaineMax : int):
    couches = []
    nbPerceptronsDansCoucheSuivante = floor(longueurChaineMax / 2)
    while nbPerceptronsDansCoucheSuivante > 0:
        couches.append([Perceptron()] * nbPerceptronsDansCoucheSuivante)
        nbPerceptronsDansCoucheSuivante = floor(nbPerceptronsDansCoucheSuivante / 2)
    return couches

# ------------------------------------------------------------------------------------------------
def AppliquerResultatsSurCouche(couche, valeursDeLaCouchePrecedente):
    assert(len(valeursDeLaCouchePrecedente) == 2 * len(couche))
    valeursDeLaCoucheCourante = []
    for iPerceptron in range(len(couche)):
        valeursDeLaCoucheCourante.append( couche[iPerceptron].f(valeursDeLaCouchePrecedente[iPerceptron * 2], valeursDeLaCouchePrecedente[iPerceptron * 2 + 1]) )
    assert(len(valeursDeLaCoucheCourante) == len(couche))
    return valeursDeLaCoucheCourante

# ------------------------------------------------------------------------------------------------
def Inference(reseau, tableauEntree):
    valeurs = tableauEntree
    for iCouche in range(len(reseau)):
        nombrePerceptronDansLaCouche = len(reseau[iCouche])
        valeurs = AppliquerResultatsSurCouche(reseau[iCouche], valeurs)
    assert(len(valeurs) == 1)
    return valeurs[0]

# ------------------------------------------------------------------------------------------------
def ObtenirEntree(longueurChaineMax):
    entree : str = input("Entrer une chaine de caractères: ")
    return String2Tableau(longueurChaineMax, entree)

def String2Tableau(longueurChaineMax : int, chaine : str):
    tableauEntree = [0.0] * longueurChaineMax
    for i in range(min(longueurChaineMax, len(chaine))):
        tableauEntree[i] = ord(chaine[i]) / float(128.0)
    return tableauEntree

def Charger() :
    mots = open("mots.txt").readlines()
    mots = [mot.rstrip("\n") for mot in mots]
    nombres = open("nombres.txt").readlines()
    nombres = [nombre.rstrip("\n") for nombre in nombres]
    entrees = list(zip(mots, [True] * len(mots)))
    entrees.extend(zip(nombres,[False]* len(nombres)))
    shuffle(entrees)
    return entrees

def InferenceEnsemble(longueurChaineMax : int, reseau : list, entrees : list):
    resultats = []
    for entree in entrees:
        sortie = Inference(reseau, String2Tableau(longueurChaineMax, entree[0]))
        resultats.append(FonctionActivation(sortie))
    return resultats

def Entrainer(longueurChaineMax : int, reseau : list, entrees : list):
    sortiesAttendues = zip(*entrees)[1]
    sorties = InferenceEnsemble(longueurChaineMax, reseau, entrees)
    for i in range(len(sorties)):
        print(str(FonctionActivation(sortiesAttendues[i])) + " [" + entrees[i][0] + "]")

# ------------------------------------------------------------------------------------------------
def main():
    entrees = Charger()
    longueurChaineMax = 16
    tableauEntree = ObtenirEntree(longueurChaineMax)
    ensembleEntrainement = entrees[0:floor(len(entrees) * 0.80)]
    ensembleValidation = entrees[ceil(len(entrees) * 0.80):]
    reseau = CreerReseauNeurones(longueurChaineMax)
    Entrainer(longueurChaineMax, reseau, ensembleEntrainement)
    sortie = Inference(reseau, tableauEntree)

# ------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
