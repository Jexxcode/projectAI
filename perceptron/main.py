from random import random
from random import shuffle
from math import floor, ceil, log2

def FonctionActivation(value : float) -> bool:
    if(value < 0.5):
        return False
    else:
        return True

# ------------------------------------------------------------------------------------------------
class Perceptron:
    p1 : float = 0
    p2 : float = 0
    lastI1 : float = 0
    lastI2 : float = 0
    def __init__(self):
        self.p1 = random()
        self.p2 = random()
        pass

    def f(self, i1 : float, i2 : float) -> float:
        self.lastI1 = i1
        self.lastI2 = i2
        return self.p1 * i1 + self.p2 * i2

    # Add a method to update weights
    def update_weights(self, learning_rate, delta):
        # Update p1 and p2 based on delta and learning_rate
        self.p1 -= learning_rate * delta * self.lastI1
        self.p2 -= learning_rate * delta * self.lastI2

# ------------------------------------------------------------------------------------------------
def CreerReseauNeurones(longueurChaineMax : int):
    nombredeCouches = int(log2(longueurChaineMax))
    couches = []
    nbPerceptronsDansCoucheSuivante = floor(longueurChaineMax / 2)
    for iCouche in range(nombredeCouches):
        couche = []
        for i in range(nbPerceptronsDansCoucheSuivante):
            couche.append(Perceptron())
        couches.append(couche)
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

# ------------------------------------------------------------------------------------------------
def String2Tableau(longueurChaineMax : int, chaine : str):
    tableauEntree = [0.0] * longueurChaineMax
    for i in range(min(longueurChaineMax, len(chaine))):
        tableauEntree[i] = ord(chaine[i]) / float(128.0)
    return tableauEntree

# ------------------------------------------------------------------------------------------------
def Charger() :
    mots = open("perceptron/mots.txt").readlines()
    mots = [mot.rstrip("\n") for mot in mots]
    nombres = open("perceptron/nombres.txt").readlines()
    nombres = [nombre.rstrip("\n") for nombre in nombres]
    entrees = list(zip(mots, [True] * len(mots)))
    entrees.extend(zip(nombres,[False]* len(nombres)))
    shuffle(entrees)
    return entrees

# ------------------------------------------------------------------------------------------------
def InferenceEnsemble(longueurChaineMax : int, reseau : list, entrees : list):
    resultats = []
    for entree in entrees:
        sortie = Inference(reseau, String2Tableau(longueurChaineMax, entree[0]))
        resultats.append(FonctionActivation(sortie))
    return resultats

# ------------------------------------------------------------------------------------------------
def Bool2Int(b):
    if b:
        return 1
    return 0

# Add a simple loss function for demonstration
def mean_squared_error(predicted, target):
    return (predicted - target) ** 2

# Add derivative of the activation function (assuming binary step for simplicity)
def derivative_activation_function(output):
    # Derivative of binary step is not well-defined at 0, so this is a placeholder
    return 1 if output != 0.5 else 0

def AfficherReseau(reseau):
    for i, couche in enumerate(reseau):
        print(f"Couche {i}:")
        for j, perceptron in enumerate(couche):
            print(f"   Perceptron {j}: p1 = {perceptron.p1}, p2 = {perceptron.p2}")

def Entrainer(longueurChaineMax, reseau, entrees):
    learning_rate = 0.01
    for epoch in range(2000):  # Number of epochs
        total_error = 0
        for entree in entrees:
            sortie = Inference(reseau, String2Tableau(longueurChaineMax, entree[0]))
            expected_output = Bool2Int(entree[1])
            error = mean_squared_error(sortie, expected_output)
            total_error += error
            
            # Backpropagation step
            # Calculate delta for output layer
            derivative_loss = 2 * (sortie - expected_output)  # dError/dOutput
            derivative_output = derivative_activation_function(sortie)  # dOutput/dNetInput
            delta = derivative_loss * derivative_output
            
            for couche in reseau[::-1]:
                for perceptron in couche:
                    perceptron.update_weights(learning_rate, delta / (len(reseau) * len(couche)))
        print("-----------------------------")
        # AfficherReseau(reseau)
        print(f"Epoch {epoch}, Total Error: {total_error}")

# ------------------------------------------------------------------------------------------------
def main():
    entrees = Charger()
    longueurChaineMax = 16
    ensembleEntrainement = entrees[0:floor(len(entrees) * 0.80)]
    ensembleValidation = entrees[ceil(len(entrees) * 0.80):]
    reseau = CreerReseauNeurones(longueurChaineMax)
    Entrainer(longueurChaineMax, reseau, ensembleEntrainement)

    while True:
        tableauEntree = ObtenirEntree(longueurChaineMax)
        sortie = Inference(reseau, tableauEntree)
        if FonctionActivation(sortie):
            print("Est un mot")
        else:
            print("Est un nombre")

# ------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
