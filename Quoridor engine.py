# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:34:15 2025

@author: W0rk-in-Progress
Still not finished - guess what, it's a WIP!

Remerciements à S. Capdevielle : pédagogue de prodige qui mèle savamment la joie de la découverte au plaisir d'apprendre !
Merci de m'avoir poussé jusqu'au bout du chemin.
"""

import numpy as np
from copy import deepcopy
from time import time
import subprocess

def git_push(commit_message="Auto commit"):
    """
    courtesy of ChatGPT cuz I'm new to this Git biz.
    """
    try:
        # Add all changes
        subprocess.run(["git", "add", "."], check=True)
        
        # Commit with a message
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        
        # Push to GitHub
        subprocess.run(["git", "push", "origin", "main"], check=True)

        print("Changes pushed successfully!")
    except subprocess.CalledProcessError as e:
        print("Error during Git operation:", e)


# On se propose de programmer un algorithme jouant de façon optimisée au jeu de
# Quoridor.


#################
# Règles du jeu #
#################

"""
Le Quoridor se joue à 2 (ou 4) joueurs, sur un plateau de 9x9 cases séparées
par des gouttières. Chaque joueur commence sur au bord du plateau, au milieu
de sa rangée en face d'un adversaire, et dispose de 10 (ou 5) cloisons pour la
partie.
L'objectif est d'atteindre une case quelconque de la rangée opposée à celle de
départ. Chacun son tour, les joueurs ont le choix de déplacer leur pion d'une
case verticalement ou horizontalement, ou bien de placer un mur dans les
gouttières entre les cases.
Les murs font 2 cases de long, et empêchent le passage de tous les pions. Une
extrémité peut s'appuyer contre une façade, mais ils ne peuvent, autrement, ni
se croiser, ni se superposer. Il est aussi interdit de créer un labyrinthe qui
empêche rigoureusement à un des joueurs d'atteindre sa rangée cible : un chemin
doit toujouors être possible.
Enfin, si le pion adverse empêche un déplacement, il est alors possible de le
'sauter', en plaçant son pion sur la case de l'autre côté. Si cette case n'est
pas disponible, on peut alors déplacer son pion sur une des deux cases adja-
centes restantes.

*   *   *   *   *       *   *   *   *   *
                                         
*   *   *   *   *       *   *   *   *   *
      J   A                       A   J  
*   *   *   *   *       *   *   *   *   *
                                         
*   *   *   *   *       *   *   *   *   *



*   *   *   *   *       *   *   *   *   *
            |                       |    
*   *   *   |   *       *   *   *   |   *
      J   A |                     A |    
*   *   *   *   *       *   *   *   *   *
                                  J      
*   *   *   *   *       *   *   *   *   *
exemples de déplacements possibles
"""


positions = 'positions'
nb_murs = 'nb_murs'
murs = 'murs'

gauche = (0, -1)
droite = (0, +1)
haut   = (-1, 0)
bas    = (+1, 0)

Profondeur_défaut = 2

def générer_plateau():
    """
    Renvoie un dictionnaire représentant le plateau au début de la partie,
    sous la forme d'un dictionnaire. Il contient :
     - la position des joueurs,
     - leur nombre de murs restant,
    (Ces informations sont données d'abord pour le "joueur" dont c'est le tour,
     puis pour son "adversaire".)
     - un array des murs posés et leur orientation.
    """
    plateau = {}
    plateau[positions] = np.array([[8, 4], [0, 4]])
    plateau[nb_murs] = [10, 10]
    plateau[murs] = np.zeros((10, 10), dtype=str)
    return plateau


def retourne_(plateau):
    """
    Retourne le plateau pour remplacer le joueur par l'adversaire et vice-versa,
    puis le renvoie
    """
    # nouveau plateau
    nP = {}
    
    nP[positions] = 8 - plateau[positions][::-1]
    nP[nb_murs] = plateau[nb_murs][::-1]
    nP[murs] = plateau[murs][::-1, ::-1]
    
    return nP


dic_deplacement = {gauche : ('V', np.array([[0, 1],[0, 0]])),
                   droite : ('V', np.array([[0, 1],[1, 1]])),
                   haut : ('H', np.array([[0, 0],[0, 1]])),
                   bas : ('H', np.array([[1, 1],[0, 1]]))}

def EstLegal_(deplacement, pos, Murs):
    MurOriente, RegarderMurs = dic_deplacement[deplacement]
    # Coordonnées, en l'axe 0, des obstacles potentiels
    x_obstacles, y_obstacles = pos[:,np.newaxis] + RegarderMurs
    if MurOriente in Murs[x_obstacles, y_obstacles]:
        return False # Présence d'obstacle
    
    # Le pion ne doit pas sortir du plateau
    pos_n = pos + deplacement
    return 0<= pos_n[0] <=8 and 0<= pos_n[1] <=8



def liste_coups(plateau, EstAdversaire):
    """
    Renvoie la liste des coups jouables par le joueur ou adversaire d'un
    plateau donné
    """
    pos_j, pos_a = plateau[positions][::Signe_(EstAdversaire)]
    Murs = plateau[murs]
    L = []
    
    # Déplacements
    
    
    for deplacement in dic_deplacement:
        if EstLegal_(deplacement, pos_j, Murs):
            # Tenter le déplacement
            if (pos_j + deplacement != pos_a).any():
                # Les pions ne se superposent pas
                # déplacement normal
                L.append(np.array(deplacement))
            else:
                # Sauter l'adversaire
                if EstLegal_(deplacement, pos_a, Murs):
                    # Sauter en ligne droite
                    L.append(np.array(deplacement) +deplacement)
                else:
                    # Sauter en biais
                    dep_1 = deplacement[::-1]
                    dep_2 = tuple(-np.array(dep_1))
                    if EstLegal_(dep_1, pos_a, Murs):
                        L.append(np.array(deplacement) +dep_1)
                    if EstLegal_(dep_2, pos_a, Murs):
                        L.append(np.array(deplacement) +dep_2)
    
    
    # À court de murs
    if not(plateau[nb_murs][int(EstAdversaire)]):
        return L
    
    # Murs
    for i in range(1, 9):
        for j in range(1, 9):
            # On sélectionne uniquement les emplacements VOISINS à d'autres murs
            # ou à un des joueurs
            if not Murs[i, j] and ('H' in Murs[i-1:i+2, j-1:j+2] or
                                   'V' in Murs[i-1:i+2, j-1:j+2] or
                                   (i-1<=pos_j[0]<=i and j-1<=pos_j[1]<=j) or
                                   (i-1<=pos_a[0]<=i and j-1<=pos_a[1]<=j)
                                   ):
                # Pas de chevauchement avec un voisin tangent.
                if 'H' not in Murs[i, j-1:j+2]:
                    L.append(('H', (i, j)))
                if 'V' not in Murs[i-1:i+2, j]:
                    L.append(('V', (i, j)))
    
    return L


def jouer_(coup, plateau, EstAdversaire):
    """
    Joue un coup sur une copie du plateau, puis la renvoie.
    """
    nP = deepcopy(plateau)
    if type(coup)==tuple:
        # Poser un mur
        nP[murs][coup[1]] = coup[0]
        nP[nb_murs][int(EstAdversaire)] -= 1
        return nP
    # Déplacer le pion du "joueur"
    nP[positions][int(EstAdversaire)] += coup
    return nP



def Afficher_(plateau):
    pos_j, pos_a = plateau[positions]
    Murs = plateau[murs]
    ecran = """
*   *   *   *   *   *   *   *   *   *
                                     
*   *   *   *   *   *   *   *   *   *
                                     
*   *   *   *   *   *   *   *   *   *
                                     
*   *   *   *   *   *   *   *   *   *
                                     
*   *   *   *   *   *   *   *   *   *
                                     
*   *   *   *   *   *   *   *   *   *
                                     
*   *   *   *   *   *   *   *   *   *
                                     
*   *   *   *   *   *   *   *   *   *
                                     
*   *   *   *   *   *   *   *   *   *
                                     
*   *   *   *   *   *   *   *   *   *
"""
    def indexe_joueur(pos):
        X, Y = pos
        i = 41 + 2*38*X
        i += Y*4
        return i
    def indexe_mur(i, j):
        i = 1 + 2*38*i
        i += j*4
        return i
    
    def insere(ecran, i, s):
        return ecran[:i] + s + ecran[i+len(s):]
    
    # joueurs
    ecran = insere(ecran, indexe_joueur(pos_j), 'J')
    ecran = insere(ecran, indexe_joueur(pos_a), 'A')
    
    # Murs
    for i in range(1, 9):
        for j in range(1, 9):
            if Murs[i, j] == 'H':
                ecran = insere(ecran, indexe_mur(i, j) -3, '-------')
            elif Murs[i, j] == 'V':
                ecran = insere(ecran, indexe_mur(i, j)-38, '|')
                ecran = insere(ecran, indexe_mur(i, j)+ 0, '|')
                ecran = insere(ecran, indexe_mur(i, j)+38, '|')
    
    ecran += f'Joueur : {plateau[nb_murs][0]} murs    Adversaire : {plateau[nb_murs][1]} murs'
    
    print(ecran)




##############
# Algorithme #
##############

# valeur relative de la posession d'un mur par rapport à la distance à parcourir
ValeurMur = 2


def hacher_labyrinthe(Murs):
    """
    Transforme un array labyrinthe en entier hachable,
    pour en faire une clé de dictionnaire valide.
    """
    dic = {'':0, 'H':1, 'V':2}
    S = 0
    for i in range(1, 9):
        for j in range(1, 9):
            S += dic[Murs[i, j]] * 3**( (i-1)*8 + j-1 )
            # À chaque emplacement de mur, hormis les bord du plateau,
            # correspond un puissance de 3.
    return S


def hacher_jeu(plateau):
    """
    Transforme un dictionnaire plateau en tuple hachable,
    pour en faire une clé de dictionnaire valide.
    """
    return (tuple(plateau[positions][0]),
            tuple(plateau[positions][1]),
            tuple(plateau[nb_murs]),
            hacher_labyrinthe(plateau[murs])
            )

def Signe_(EstAdversaire):
    if EstAdversaire:
        return -1
    return +1

def minimax_score(plateau, heuristique, EstAdversaire = False,
                  profondeur = Profondeur_défaut - 1, memo_score = {}):
    """
    Renvoie le score actuel du plateau, soit celui obtenu après avoir joué le
    meilleur coup.
    L'information de ce coup n'est utile que pour le premier
    appel d'une fonction minimax ; elle peut donc être écrasée pour tous les
    appels de moindre profondeur.
    Cette structure en double score permet d'utiliser un dictionnaire de mémoi-
    sation simplifié, mettant en commun les scores pour les deux joueurs.
    """
    global iteration
    global memoisation
    iteration +=1
    
    # L'ennemi vient de jouer un coup : détection de défaite
    Pos_ennemi = plateau[positions][int(not(EstAdversaire))]
    if Altitude(Pos_ennemi, EstAdversaire=not(EstAdversaire)) == 0:
        return -float('inf')
    
    # Memoisation :
    # La profondeur est prise en compte : on ne garde que le résultat avec la
    # plus grande profondeur de calcul restante, et il est réutilisé pour
    # chaque calcul de profondeur moindre ou égale. On intègre donc la valeur
    # maximale de profondeur calculée parmis les images du dictionnaire.
    # De plus, ce dernier est calibré pour le premier joueur : l'adversaire
    # devra prendre l'opposé des scores.
    Cle_jeu = hacher_jeu(plateau)
    if Cle_jeu in memo_score and profondeur <= memo_score[Cle_jeu][1]:
        memoisation += 1
        return memo_score[Cle_jeu][0] *Signe_(EstAdversaire)
    
    # Calcul d'heuristique
    if profondeur <= 0:
        Score = heuristique(plateau)
        memo_score[Cle_jeu] = Score, 0
        return Score *Signe_(EstAdversaire)
    
    
    # Détermination par récurrence:
    Successeurs = liste_coups(plateau, EstAdversaire)
    
    Coup_max = Successeurs[0]
    # Plateau "adversaire"
    plateau_a = jouer_(Coup_max, plateau, EstAdversaire)
    Score_max = -minimax_score(plateau_a, heuristique, not(EstAdversaire),
                               profondeur-1, memo_score)
    for coup in Successeurs[1:]:
        # Trouver le meilleur coup
        plateau_a = jouer_(coup, plateau, EstAdversaire)
        score = -minimax_score(plateau_a, heuristique, not(EstAdversaire),
                               profondeur-1, memo_score)
        
        if score == float('inf'):
            # Une stratégie gagnante a été trouvée ; on arrête le calcul.
            memo_score[Cle_jeu] = score*Signe_(EstAdversaire), profondeur
            return score
        
        if score > Score_max:
            Coup_max, Score_max = coup, score
    
    memo_score[Cle_jeu] = Score_max*Signe_(EstAdversaire), profondeur
    return Score_max


def minimax_coup(plateau, heuristique, EstAdversaire = False,
                 profondeur = Profondeur_défaut, memo_score={}):
    """
    Renvoie le meilleur coup à jouer, et le score actuel du plateau
    (soit celui obtenu après avoir joué ce coup).
    """
    global iteration
    iteration +=1
    
    # L'ennemi ne vient pas de jouer, mais on vérifie par précaution que le
    # jeu est toujours en cours.
    Pos_joueur, Pos_adversaire = plateau[positions]
    if Altitude(Pos_joueur, None, False)==0 or Altitude(Pos_adversaire, None, True)==0:
        return None, 0
    
    # Memoisation : elle ne sera quasiment jamais utilisée pour des appels de
    # profondeur maximale.
    
    # Pas d'heuristique nécessaire
    
    # Détermination par récurrence :
    Successeurs = liste_coups(plateau, EstAdversaire)
    
    Coup_max = Successeurs[0]
    # Plateau "adversaire"
    plateau_a = jouer_(Coup_max, plateau, EstAdversaire)
    Score_max = -minimax_score(plateau_a, heuristique, not(EstAdversaire),
                               profondeur-1, memo_score)
    # print(EstAdversaire, Coup_max, Score_max)
    for coup in Successeurs[1:]:
        # Trouver le meilleur coup
        plateau_a = jouer_(coup, plateau, EstAdversaire)
        score = -minimax_score(plateau_a, heuristique, not(EstAdversaire),
                               profondeur-1, memo_score)
        # print(EstAdversaire, coup, score)
        if score == float('inf'):
            # Une stratégie gagnante a été trouvée ; on arrête le calcul.
            return coup, score
        
        if score > Score_max:
            Coup_max, Score_max = coup, score
    
    return Coup_max, Score_max


def Heuristique(f_Distance):
    """
    définit une fonction heuristique, en fonction de la méthode de calcul pour
    la distance à l'objectif : Altitude, Astar, ou BoiteDeCereales.
    """
    def H(plateau):
        Pos_joueur, Pos_adversaire = plateau[positions]
        Murs = plateau[murs]
        
        # Distances relatives à l'objective
        distances_rel = f_Distance(tuple(Pos_joueur), Murs, False)\
                      - f_Distance(tuple(Pos_adversaire), Murs, True)
        
        # Détection d'un labyrinthe illégal. Son accomplissement est
        # techniquement possible, mais infiniment déconseillé.
        if distances_rel in (float('inf'), -float('inf'), float('nan')):
            return float('nan')
        
        Nb_murs = plateau[nb_murs]
        # "Valeur" des murs
        Vmurs = 2* (Nb_murs[0] - Nb_murs[1])
        
        return Vmurs - distances_rel
    
    return H


# Heuristique de base, indiquant à l'alogrithme la bonne direction.
def Altitude(Pos, Murs = None, EstAdversaire=False):
    if EstAdversaire:
        return 8 - Pos[0]
    return Pos[0]


# Cependant, 

def Dist_Astar(Pos, Murs, EstAdversaire=False):
    distance = {(a, b): float('inf') for a in range(9) for b in range(9)}
    distance[Pos] = 0
    non_visites = {Pos}
    
    while non_visites:
        # on recherche un sommet non visité à distance minimale
        
        # Distance + Altitude :
        Hdistance_min = float('inf')
        
        for chemin in non_visites:
            if distance[chemin] + Altitude(chemin, None, EstAdversaire) <= Hdistance_min:
                chemin_min = chemin
                Hdistance_min = distance[chemin] + Altitude(chemin, None, EstAdversaire)
        
        non_visites.remove(chemin_min)
        # Détermination des successeurs
        for dx, dy in (haut, bas, gauche, droite):
            successeur = (chemin_min[0] +dx, chemin_min[1] +dy)
            
            if not EstLegal_((dx, dy), np.array(chemin_min), Murs):
                # Déplacement impossible : le chemin n'existe pas
                continue
            if distance[successeur] == float('inf'):
                # Ce chemin n'a encore jamais été observé.
                non_visites.add(successeur)
                distance[successeur] = distance[chemin_min] + 1
            # Toutes les distances entre les cases étant identiques, on n'a
            # pas besoin de détecter pour la formation d'un raccourci.
            
            # On regarde si la rangée opposée a été atteinte
            if Altitude(successeur, None, EstAdversaire) == 0:
                return distance[successeur]
    
    # Fin de la boucle, sans solution : le labyrinthe est fermé, donc illégal.
    return float('inf')



def BoiteDeCereales(Pos, Murs, EstAdversaire=False):
    """
    Renvoie la distance à l'objectif, en procédant comme pour résoudre le
    labyrinthe d'une boite de céréales : en commançant par la fin. Contraire-
    ment à Djikstra, cet algorithme ne part pas d'une position en particulier,
    mais calcule la distance de toutes les positions accessible selon un
    labyrinthe donné, et les enregistre afin de grandement réduire les calculs.
    """
    # mémoisation : comme je rechigne à ajouter un paramètre facultatif à la
    # fonction heuristique, qui ne sera utilisé que par cette fonction-ci, je
    # tente d'utiliser plus simplement une variable globale.
    global memo_BDC
    global memoisation_lab
    try:
        memo_BDC
    except:
        memo_BDC = {}
    
    Cle_labyrinthe = (hacher_labyrinthe(Murs), EstAdversaire)
    if Cle_labyrinthe in memo_BDC:
        memoisation_lab += 1
        return memo_BDC[Cle_labyrinthe][Pos]
    
    memo_BDC[Cle_labyrinthe] = update_BoiteDeCereales(*Cle_labyrinthe)
    return memo_BDC[Cle_labyrinthe][Pos]



def update_BoiteDeCereales(Murs, EstAdversaire=False):
    """
    Génère un nouveau dictionnaire contenant toutes les distances à l'objectif
    pour un labyrinthe donné. Cet algorithme est inspiré de Djikstra.
    """
    distance = {(a, b): float('inf') for a in range(9) for b in range(9)}
    predec = {}
    
    # Au lieu de partir d'une seule position, on part de toute la rangée à atteindre.
    non_visites = {(8*EstAdversaire, Y) for Y in range(9)}
    for e in non_visites:
        distance[e] = 0
    
    while non_visites:
        # on recherche un sommet non visité à distance minimale
        
        # Distance sans heuristique de nécessaire, puisque l'objectif est de
        # parcourir l'entièreté du plateau.
        distance_min = float('inf')
        
        for chemin in non_visites:
            if distance[chemin] <= distance_min:
                chemin_min = chemin
                distance_min = distance[chemin]
        
        non_visites.remove(chemin_min)
        # Détermination des successeurs
        for dx, dy in (haut, bas, gauche, droite):
            successeur = (chemin_min[0] +dx, chemin_min[1] +dy)
            
            if not EstLegal_((dx, dy), np.array(chemin_min), Murs):
                # Déplacement impossible : le chemin n'existe pas
                continue
            if distance[successeur] == float('inf'):
                # Ce chemin n'a encore jamais été observé.
                non_visites.add(successeur)
                distance[successeur] = distance[chemin_min] + 1
                predec[successeur] = chemin_min
            # Toutes les distances entre les cases étant identiques, on n'a
            # pas besoin de détecter pour la formation d'un raccourci.
    
    # Fin de la boucle sans condition d'arrêt : tous les chemins non visités
    # sont inaccessibles, donc reçoivent une distance infinie.
    return distance



#################
# Partie de Jeu #
#################


memo = {}

Plateau = générer_plateau()
H = Heuristique(Dist_Astar)
EstAdversaire = True
I = ''

print("""
Appuyer sur entrée pour passer au tour suivant.
Taper '!stop', puis appuyer sur entrée pour finir la partie.
""")

while I!='!stop':
    Afficher_(Plateau)
    
    I = input('')
    if I == '!stop':
        break
    print('\nEn cours...')
    
    # Plateau = retourne_(Plateau)
    EstAdversaire = not(EstAdversaire)
    
    iteration = 0
    memoisation = 0
    memoisation_lab = 0
    Coup, Score = minimax_coup(Plateau, H, EstAdversaire,
                               profondeur = 2, memo_score = memo)
    print(iteration, "jeux calculés")
    print(f'memo utilisé {memoisation} fois')
    print(f'memo_labyrinthe utilisé {memoisation_lab} fois')
    if Coup is None:
        break
    print("Coup joué :", Coup, Score)
    
    Plateau = jouer_(Coup, Plateau, EstAdversaire)

print('Fin du jeu!')
