"""
Le code (nom des variables, fonctions, ...) à été écrit anglais par habitude. Si vous ne comprenez
pas l'anglais, concentrez vous sur les commentaires et docstrings ou demandez à une Gen AI de
traduire ce script en français.

X -> LARGEUR
Y -> PROFONDEUR
Z -> HAUTEUR
"""

import math
from dataclasses import dataclass
from functools import lru_cache  # Memoization afin d'améliorer les performances en sauvegardant
                                 # les résultats des fonctions "déterministiques"
from typing import Any


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
            if (i >> j) & 1:
                x *= elements[j]
        y = p // x
        # Optimization pour éviter d'avoir à la fin des combinaisons "dupliquées" ex. (5,2) et (2,5)
        if x >= y:
            cmb.append((x, y))

    # Renvoyer les combinaisons en supprimant les éléments dupliqués
    return set(cmb)

@dataclass(frozen=True, slots=True)
class Resultat:
    dim: tuple[float, float, float]
    volume: float
    grid: tuple[int, int]
    max_glass_per_pile: int
    variables: dict[str, Any]


def calc_square_pyramid(
    pyramid_top: float,
    pyramid_height: float,

    glasses: int,
    edge_thickness: float,
    target_glass_volume: float,  # cm3 -> mL
    max_height: float,
) -> list[Resultat] | None:

    """Calcule les dimensions de la zone occupé par les verres tel que les conditions
    target_glass_volume et glasses soit satisfait.

    pyramid_top / pyramide_height donne les dimensions intérieurs. Les bords du verres sont
    considérés comme à l'extérieur de ces dimensions, ce qui signifie que la largeur du verre sera
    par exemple pyramid_top + 2 * (...).

    Args:
        pyramid_top (float): TODO

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

    Avec cette formule, nous pouvons déterminer a (réel **supérieur ou égal à 0**), la taille de la
    base de la petite pyramide:

            v         = (H/3)(b² - a²)
        <=> v / (H/3) = b² - a²
        <=> 3v/H - b² = -a²
        <=> a²        = (-3v / H) + b²
        <=> a         = sqrt(b² - 3v / H)

    Si b² - 3v / H est négatif, alors la racine sera imaginaire. Si la largeur de base est
    imaginaire, alors il est impossible de satisfaire les contraintes donc on ne peut pas émettre
    une solution.
    """

    # Calculer la base intérieur de la pyramide (aire puis largeur)
    a2 = (b ** 2) - (3 * v / H)
    if a2 < 0:  # Vérifier que la formule ne va pas produire un nombre imaginaire
        return None
    a = math.sqrt(a2)

    # En théorie, a ne devrait pas être supérieur à b. Nous vérifions quand même.
    assert a < b

    # S'assurer que la formule est correcte en retrouvant le volume à partir de la base
    if DEBUG_MODE:
        sub_pyramid_volume = (H / 3) * (b * b + a2)
        full_pyramid_volume = (1/3) * b * b * H
        assert math.isclose(full_pyramid_volume - sub_pyramid_volume, v), \
            f"Formule de la taille de la base intérieur incorrecte: {full_pyramid_volume - sub_pyramid_volume=}, {a, b=}"


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
    theta = e / e_h  # Utilisé pour plus tard
    a_e = a + 2 * (e_h - e_h / c)
    if a_e <= a:
        return None
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
    la plus grande -> base extérieur. Nous devons simplement ajouter l'épaisseur horizontalle du
    verre aux deux bords:
    
            b_e = b + 2e_h
    
    Nous déterminons d'abord le nombre de piles qui doivent être créées. Pour ca, nous arrondissons
    vers le haut la division du nombre de verres au nombre de verres que l'on peut mettre par pile.

            n_p = ⌈n / p_m⌉

    Ici, nous pouvons avoir des piles pleines et une pile pratiquement vide. Nous déterminons le
    nombre de verres de la pile la plus haute en prenant le plancher de la division du nombre de
    verres demandés et le nombre de piles à créer. Ainsi, nous réduisons la hauteur du placard.

            p = ⌈n / n_p⌉

    Ensuite, nous déterminons à l'aide d'un algorithme basé sur la décomposition en facteurs
    premiers toutes les combinaisons possibles de grilles de verre. (dans le code, nous utilisons
    cette algorithme avec `pairs(prime_factors(p))`. Pour chaque combinaison de largeur et
    profondeur x y entiers naturels:

    Nous déterminons la hauteur et profondeur de l'armore:
            s_x = b_e * x
            s_y = b_e * y
            s_y = f(p)
    """

    f = lambda x: h + x*d + e
    b_e = b + 2 * e_h
    n_p = math.ceil(n / p_m)
    p = math.ceil(n / n_p)

    # Trouver les facteurs premiers
    glass_prime_factors = prime_factors(n_p)
    # Déterminer toutes les combinaisons possibles de ces variables
    cmb = pairs(glass_prime_factors)


    resultats = []
    for x, y in cmb:
        dim = (b_e * x, b_e * y, f(p))
        resultats.append(Resultat(
            dim=dim,
            volume=dim[0]*dim[1]*dim[2],
            grid=(x, y),
            max_glass_per_pile=p,
            variables={
                # Variables de taille
                'b': b,
                'b_e': b_e,
                'a': a,
                'a_e': a_e,
                'e': e,
                'e_h': e_h,
                # Caractéristiques des piles
                'n_p': n_p,
                'p': p,
                'H': H,
                'h': h,
                # Autres informations
                'theta_pi': theta,  # Angle de la paroi du verre
                'theta': math.degrees(theta)  # 90° -> a = b
            }
        ))
        

    return resultats

def line_by_line_repr(l):
    return '[\n\t' + '\n\t'.join([str(elem) for elem in l]) + ']'


def fetch_result(results: list[Resultat] | None) -> Resultat | None:

    if results is None:
        return None

    retained = results[-1]  # Avec l'implémentation actuelle, les premiers résultats sont des
                            # grilles très longues et peu profondes et les derniers des
                            # grilles plus équilibrées.

    if (True
        # Par ex. (et car 90° -> cube): `and retained.variables['theta'] > 60`
    ):  # Ajouter des conditions ici.
        return retained
    else:
        return None


if __name__ == "__main__":

    DEBUG_MODE=False

    print("Essai de la tentative 1:")

    """
            v            = (H/3)(b² - a²)
        <=> 3v           = h(b² - a²)
        <=> 3v / (b²-a²) = h
    
    Dans cette tentative, b = 7cm et a = 4cm et v = 20cL = 200mL = 200cm²
    """


    print(line_by_line_repr(calc_square_pyramid(
        pyramid_top=7,
        pyramid_height=(3*200) / (7**2 - 4**2),
        glasses=1000,
        edge_thickness=0.2,  # 2mm = 0.2cm
        target_glass_volume=200,
        max_height=40
    )))
    
    print(line_by_line_repr(calc_square_pyramid(
        pyramid_top=10.8,
        pyramid_height=(3*202.1) / (10.8**2 - 3.6**2),
        glasses=1000,
        edge_thickness=0.2,  # 2mm = 0.2cm
        target_glass_volume=200,
        max_height=40
    )))


    precision = 50
    resultats = []
    for i in range(1_000, 30_000, precision):
        print(f"New iteration... {i / 1_000}")
        for j in range(1_000, 30_000, precision):
            result: list[Resultat] | None = calc_square_pyramid(
                pyramid_top=i/1000,
                pyramid_height=j/1000,
                glasses=1000,
                edge_thickness=0.2,
                target_glass_volume=200,
                max_height=40
            )
            if result is None:
                continue
            taken_result: Resultat = result[-1]
            if (
                True
                # and taken_result.variables['theta'] > 60
            ):
                resultats.append(taken_result)

    resultats.sort(key=lambda x: x.volume)

    print(line_by_line_repr(resultats[:10]))