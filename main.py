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


@lru_cache(maxsize=1100)
def prime_factors(n: int) -> tuple[int]:
    """Décomposition en facteurs premiers"""

    result = []

    while n > 1:
        for i in range(2, n+1):
            if (n/i) % 1 == 0:
                result.append(i)
                n //= i
                break

    return tuple(result)

@lru_cache(maxsize=1100)
def pairs(elements: tuple[int]) -> set[tuple[int, int]]:
    """
    Algorithmes qui donne toutes les paires que l'on peut créer à partir de n tel que
    x * y = produits de p
    """
    cmb = []

    # Calculer les produits de p:
    p = 1
    for elem in elements:
        p *= elem

    # Créer toutes les paires possibles
    for i in range(2 ** len(elements)): # Cette approche est acceptable. Pour 512 verres il y aura
                                        # 9 facteurs premiers -> 512 itérations de cet algorithme:
        x = 1
        for j, k in enumerate(elements):
            if (i >> k) & 1 == 1:
                x *= elements[j]
        y = p // x

        # Optimization pour éviter d'avoir à la fin des combinaisons "dupliquées" ex. (5,2) et (2,5)
        if x >= y:
            cmb.append((x, y))

    # Renvoyer les combinaisons en supprimant les éléments dupliqués
    return set(cmb)




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
        None: Les paramètres ne permettent pas de satisfaire les conditions imposées
    """

    # Alias
    b = pyramid_top
    H = pyramid_height

    n = glasses
    e = edge_thickness

    v = target_glass_volume
    m = max_height

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

    # En théorie, a ne devrait pas être supérieur à b. Nous nous assurons néanmoins que cela est vrai
    assert a < b

    # S'assurer que la formule est correcte en retrouvant le volume à partir de la base
    if DEBUG_MODE:
        sub_pyramid_volume = (H / 3) * (b * b + a2)
        full_pyramid_volume = (1/3) * b * b * H
        assert math.isclose(full_pyramid_volume - sub_pyramid_volume, v), \
            f"Formule de la taille de la base intérieur incorrecte: {full_pyramid_volume - sub_pyramid_volume=}"


    # -------------------------------------------------------------------------------------------
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

    Le point de contact SELON LA PENTE y_c est défini par :

            f(y_c) = a_e
        <=> a + (b - a) * (y_c / h) = a_e
        <=> (b - a) * (y_c / h) = a_e - a
        <=> (b - a) * y_c = h(a_e - a)
        <=> y_c = h * (a_e - a) / (b - a)

    La distance verticale d entre les bases intérieures de deux verres empilés est alors obtenue en
    ajoutant l'épaisseur du verre e (et non e_h car la base n'est pas anglée). À noter que si les
    angles sont trop horizontaux, alors les verres s'empileront les bords des précédents car ils ne
    peuvent pas être magiquement suspendu en l'air:

            d = { y_c + e   si y_c + e < h + e
                { h + e     sinon
    """

    h = (1 - a / b) * H
    c = 2 * H / b
    e_h = e / math.atan(c)
    a_e = a + 2 * (e_h - e_h / c)
    y_c = h * (a_e - a) / (b - a)

    d = y_c + e
    if d > h + e:
        d = h + e

    # ------------------------------------------------
    # Déterminer le nombre de verre max. dans une pile
    # ------------------------------------------------

    """
    Nous devons déterminer le nombre de verres maximaux dans une pile. Nous créerons f(x) pour
    calculer la hauteur d'une pile. Il faut ajouter à la hauteur laissé par l espace entre les
    verres empilés (égal à kd) la hauteur intérieur du dernier verre, puis l'épaisseur du verre
    pour prendre en compte le dernier verre correctement.

            f(x) = h + xd + e

    Nous ajouterons également une vérification pour s'assurer que h soit inférieur à m la hauteur
    maximale.

    Nous pouvons désormais résoudre la hauteur maximale d'une pile. À noter que d > 0:
    
            f(x) <= m
        <=> (h + e) + xd <= m
        <=> (h + e) / d + x <= m / d
        <=> x <= (m - h - e) / d

    Enfin, on ajuste cette formule car nous avons un nombre entier de verres maximaux et obtenons
    p_m entier naturel la taille maximale d'une pile de verres:
    
            p_m = ⌊(m - h - e) / d⌋
    """

    if h > m or not d > 0:
        return None
    p_m = int((m - h - e) / d)


    # -----------------------------------------------------
    # Déterminer les caractéristiques des piles nécéssaires
    # -----------------------------------------------------
    """
    Pour déterminer les dimensions du placard, nous aurons besoin de déterminer la taille d'un
    verre. Sa hauteur extérieur peut facilement être déterminée en ajoutant l'épaisseur du verre à
    sa base -> (h + e). Nous avons cependant besoin de calculer la largeur extérieur de la base
    la plus grande -> base extérieur. Nous devons simplement ajouter l'épaisseur horizontalle du*
    verre aux deux bords:
    
            b_e = b + 2e_h
    
    Nous déterminons d'abord le nombre de piles qui doivent être créées. Pour ca, nous arrondissons
    vers le haut la division du nombre de verres au nombre de verres que l'on peut mettre par pile.

            p = ⌈n / p_m⌉

    Ensuite, nous déterminons toutes les combinaisons possibles de grilles de verre. Pour chaque
    combinaison de largeur et profondeur x y entiers naturels:

    TODO
    """

    b_e = b + 2 * e_h
    p = math.ceil(n / p_m)

    # Trouver les facteurs premiers
    glass_prime_factors = prime_factors(p)
    # Déterminer toutes les combinaisons possibles de ces variables
    cmb = pairs(glass_prime_factors)


    return 0, 0, 0




if __name__ == "__main__":

    print((-2) ** 0.5)