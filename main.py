"""
Le code (nom des variables, fonctions, ...) à été écrit anglais par habitude. Si vous ne comprenez
pas l'anglais, concentrez vous sur les commentaires et docstrings ou demandez à une Gen AI de
traduire ce script en français.

X -> LARGEUR
Y -> PROFONDEUR
Z -> HAUTEUR
"""

import math
from functools import lru_cache  # Memoization afin d'améliorer les performances en sauvegardant
                                 # les résultats des fonctions "déterministiques"


DEBUG_MODE = True


@lru_cache(maxsize=None)
def prime_factors(n: int) -> list[int]:
    """Décomposition en facteurs premiers"""

    result = []

    while n > 1:
        for i in range(2, n+1):
            if (n/i) % 1 == 0:
                result.append(i)
                n //= i
                break

    return result



def calc_square_pyramid(
    pyramid_top: float,
    pyramid_height: float,

    glasses: int,
    edge_thickness: float,
    target_glass_volume: float,  # cm3 -> mL
    max_height: float,
) -> tuple[int, int, int] | None:

    """Calcule les dimensions de la zone occupé par les verres tel que les conditions
    target_glass_volume et glasses soit satisfait.

    pyramid_top / pyramide_height donne les dimensions intérieurs. Les bords du verres sont
    considérés comme à l'extérieur de ces dimensions, ce qui signifie que la largeur du verre sera
    par exemple pyramid_top + 2 * edge_width.

    Args:
        pyramid_top (float): 

    Returns:
        list[tuple[int, int, int]]: Dimensions X Y Z occupés par les verres de chaque arrangement
            de verre dans l'étagère optimale
        None: Les paramètres ne permettent pas de satisfaire les \
            conditions imposées
    """

    # Alias
    b = pyramid_top
    H = pyramid_height
    
    n = glasses
    e = edge_thickness

    v = target_glass_volume
    m = max_height

    # ========================================
    # ETAPE 1: DETERMINER LA TAILLE D'UNE PILE
    # ========================================

    # -----------------------------------------------------------------------------------
    # Déterminer la hauteur intérieur du verre tel que le volume vaut target_glass_volume
    # -----------------------------------------------------------------------------------

    """
    Un frustum (pyramide tronqué) peut être vu comme une grande pyramide auquel on retire une plus
    petite pyramide.
    La formule du volume d'une pyramide à base carée est (1/3) * aire de la base * hauteur.
    Avec a et b réels supérieurs ou égaux à 0 et représentant la taille des bases de la "petite"
    et grande pyramides respectivement, H la hauteur de la pyramide, et v le volume du frustum, la
    formule du volume du frustum est:

            v = (1/3)b²H - (1/3)a²H
        <=> v = (H/3)(b² - a²)

    Avec cette formule, nous pouvons déterminer a, la taille de la base de la petite pyramide:

            v         = (H/3)(b² - a²)
        <=> v / (H/3) = b² - a²
        <=> 3v/H - b² = -a²
        <=> a²        = (-3v / H) + b²
        <=> a         = sqrt(b² - 3v / H)  avec a >= 0

    Si b² - 3v / H est négatif, alors la racine sera imaginaire. Si la largeur de base est
    imaginaire, alors il est impossible de satisfaire les contraintes donc on ne peut pas émettre
    une solution.
    """

    # Calculer la base intérieur de la pyramide (aire puis largeur)
    a2 = (b ** 2) - (3 * v / H)
    if a2 < 0:  # Vérifier que la formule ne va pas produire un nombre imaginaire
        return None
    a = math.sqrt(a2)

    # S'assurer que la formule est correcte en retrouvant le volume à partir de la base
    if DEBUG_MODE:
        sub_pyramid_volume = (H / 3) * (b * b + a2)
        full_pyramid_volume = (1/3) * b * b * H
        assert math.isclose(full_pyramid_volume - sub_pyramid_volume, v), \
            f"Formule de la taille de la base intérieur incorrecte: {full_pyramid_volume - sub_pyramid_volume=}"


    # ------------------------------------------------------------------------------------------
    # Déterminer l'épaisseur horizontale du verre puis l'espace entre différent verres de la pile
    # -------------------------------------------------------------------------------------------
    """
    Nous devons déterminer l'espace entre deux verres. Nous divisons ce calcul en plusieur étapes:
    - Calcul de la hauteur du frustum
    - Calcul du coefficient directeur de la paroi latérale
    - Calcul de l'épaisseur horizontale du verre
    - Calcul de la base extérieur à une hauteur définie
    - Calcul de la hauteur de contact entre deux verres empilés

    Avec a réel la base inférieur (petite base), b réel la base supérieur (grande base) et H la
    hauteur de la pyramide entière,

    Pour obtenir la hauteur du frustum relatif à la pyramide complète, nous utiliserons
    l'expression 1 - a/b. (et non a/b seul car a/b vaut 1 lorsque la grande base et la petite sont
    égales, ce qui arrive lorsqu'on à un frustum de hauteur 0). Nous pouvons ensuite multiplier par
    H pour obtenir la hauteur h du frustum:

            h = d_y = (1 - a/b) * H

    Pour déterminer le coefficient directeur, on a aussi besoin de la largeur de la bordure. Celle
    ci se calcule simplement par (b-a) / 2 (on ajoute /2 car on a besoin de la demi-base). On peut
    donc calculer notre coefficient directeur:
    
            c = d_y / d_x
        <=> c = ((1 - a/b) * H) / ((b - a) / 2)
        <=> c = (((b-a) / b) * H * 2) / (b - a)
        <=> c = ((b-a) / b) * (2H / (b - a))
        <=> c = ((b - a)2H) / ((b - a)b)
        <=> c = 2H / b

    Avec e réel l'épaisseur du verre demandé par les consignes, nous déterminons l'épaisseur
    horizontale du verre qui est nécéssaire pour les prochains calculs.
    
            cos theta = côté adjacent / hypoténuse = e / e_h
        <=> e_h = e / cos(theta)
        <=> e_h = e / arctan(d_y / d_x)
        <=> e_h = e / arctan(c)
    
    Avec e réel l'épaisseur du verre et e_h réel l'épaisseur horizontale du verre, on calcule la
    largeur extérieur de la base inférieur.
    À la base intérieur, nous ajoutons l'épaisseur horizontale du verre des deux côtés, avant de
    soustraire la perte d'épaisseur lié à l'angle de la paroi calculable avec 2e_h / c, des deux
    côtés.

            a_e = a + 2(e_h - e_h / c)

    À partir de la base extérieure a_e, nous pouvons déterminer la distance verticale minimale
    nécessaire entre deux verres empilés.

    On cherche la hauteur y_c (mesurée depuis la base du frustum) telle que la largeur intérieure
    disponible dans le premier verre soit exactement égale à la base extérieure du second verre.

    La largeur intérieure du frustum varie linéairement avec la hauteur. On définit donc la
    fonction:

        f(y) = a + (b - a) * (y / h)

    où y appartient à [0, h].

    Le point de contact y_c est défini par :

            f(y_c) = a_e
        <=> a + (b - a) * (y_c / h) = a_e
        <=> (b - a) * (y_c / h) = a_e - a
        <=> (b - a) * y_c = h(a_e - a)
        <=> y_c = h * (a_e - a) / (b - a)

    La distance verticale d entre les bases intérieures de deux verres empilés est alors obtenue
    en ajoutant l'épaisseur horizontale du verre :

            d = y_c + e_h
    """

    h = (1 - a / b) * H
    c = 2 * H / b
    e_h = e / math.atan(c)
    a_e = a + 2 * (e_h - e_h / c)
    y_c = h * (a_e - a) / (b - a)
    d = y_c + e_h


    return 0, 0, 0




if __name__ == "__main__":

    print((-2) ** 0.5)