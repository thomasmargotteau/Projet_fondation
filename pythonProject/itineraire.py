# Dijkstra avec des petits points ça devrait marcher
from enum import Enum

class Case(Enum):
  ACCESSIBLE = 0
  OBSTACLE = 1
  DEPART = 2
  ARRIVEE = 3

def trouverDepartEtArrivee(grille):
    caseDepart=None
    caseArrivee=None
    for i in range(0,len(grille)):
        for j in range(0,len(grille[0])):
            if (grille[i][j]==Case.DEPART.value) :
                caseDepart=(j,i)
            elif (grille[i][j]==Case.ARRIVEE.value) :
                caseArrivee=(j,i)
            elif (grille[i][j]!=Case.ACCESSIBLE.value and grille[i][j]!=Case.OBSTACLE.value) :
                print("ERREUR : La case (",j,",",i,") n'a aucun type répertorié !")
            
    if (not caseDepart):
        print("ERREUR : Pas de case départ !")
    if (not caseArrivee):
        print("ERREUR : Pas de case arrivée !")
    return (caseDepart, caseArrivee)

# Fonction qui renvoie toutes les cases valides de la grille
def casesValides(grille):
    cases = []
    for i in range(len(grille)):
        for j in range(len(grille[0])):
            if grille[i][j] != Case.OBSTACLE.value:
                cases.append((i, j))
    return cases

# Fonction qui renvoie toutes les cases voisines d'une case donnée
def obtenirVoisins(grille, case):
    tabVoisins = []
    ligneAct, colonneAct = case
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        ligne = ligneAct + dy
        col = colonneAct + dx
        if 0 <= ligne < len(grille) and 0 <= col < len(grille[0]) and grille[ligne][col] != Case.OBSTACLE.value:
            tabVoisins.append((ligne, ligneAct))
    return tabVoisins

def dijkstra(grille):

    # Trouver le départ et l'arrivée
    caseDepart, caseArrivee = trouverDepartEtArrivee(grille)

    # Initialisation du dictionnaire de distances
    tabDistances = {case: float("inf") for case in casesValides(grille)}
    tabDistances[caseDepart]=0

    # Initilisation des prédécesseurs
    tabParents = {case: None for case in casesValides(grille)}

    # Initialisation de la fifo
    fifo = [(0, caseDepart)]

    while fifo :
       
        # On selectionne le sommet le plus proche et on le retire de la fifo
        distance, caseAct = fifo[0]
        print(distance, caseAct)
        fifo.pop(0)

        # Si la distance est plus petite
        if (distance<=tabDistances[caseAct]) :
            # On cherche tous les voisins
            for caseVoisine in obtenirVoisins(grille,caseAct) :
                nouvelleDistance = distance + 1
                # Si la distance vers la case voisine est meilleure, la mettre à jour et la marquer comme explorée
                if nouvelleDistance < tabDistances[caseVoisine]:
                    tabDistances[caseVoisine] = nouvelleDistance
                    tabParents[caseVoisine] = caseAct
                    fifo.append((nouvelleDistance, caseVoisine))


    # On retrouve le chemin le plus court
    PCC = []
    caseAct = caseArrivee
    while caseAct != caseDepart:
        print(caseAct)
        PCC.append(caseAct)
        caseAct = tabParents[caseAct]

    PCC.reverse()

    return PCC


# Test

grille = [
  [2, 0, 0, 0, 0],
  [0, 1, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 1, 0],
  [0, 0, 0, 0, 3]
]

# Exécuter l'algorithme de Dijkstra et afficher le chemin le plus court
PCC = dijkstra(grille)

if PCC:
  print("Chemin le plus court:", PCC)
else:
  print("Aucun chemin trouvé :'(")
