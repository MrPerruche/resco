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
from typing import Any, Callable  # Gardons quelque chose de propre


DEBUG_LEVEL = 0


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
    variables: dict[str, tuple[Any, str]]

    def __str__(self) -> str:
        longest_var = max([len(k) for k in self.variables])
        longest_val = max([len(str(v[0])) for v in self.variables.values()])
        return f"""\
Dimensions X Y Z   : {' x '.join([f'{dim} cm' for dim in self.dim])}
Volume             : {self.volume} cm3
Grille proposé     : {self.grid[0]} x {self.grid[1]} ({self.grid[0] * self.grid[1]} piles)
Verres / piles max : {self.max_glass_per_pile}
Variables... {''.join([f'\n  - {k + ' ' * (longest_var - len(k) + 1)}: {v[0]!r}{' '*(longest_val - len(str(v[0])) + 1)} ({v[1]})' for k, v in self.variables.items()])}\
"""


def calc_square_pyramid(
    pyramid_top: float,
    pyramid_height: float,

    glasses: int,
    edge_thickness: float,
    target_glass_volume: float,  # cm3 -> mL
    max_height: float,

    parameters: 'Parameters'
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

    # Vérification précoce pour éviter des divisions par zéro
    if H == 0:
        return None

    # Calculer la base intérieur de la pyramide (aire puis largeur)
    a2 = (b ** 2) - (3 * v / H)
    if a2 < 0:  # Vérifier que la formule ne va pas produire un nombre imaginaire
        return None
    a = math.sqrt(a2)

    # En théorie, a ne devrait pas être supérieur à b. Nous vérifions quand même.
    assert a < b

    # S'assurer que la formule est correcte en retrouvant le volume à partir de la base
    if DEBUG_LEVEL >= 1:
        # Vérification du volume
        calc_v = (H / 3) * (b**2 - a**2)
        assert math.isclose(calc_v, v, rel_tol=1e-9), \
            f"Volume incohérent: {calc_v=} != {v=}"
        if DEBUG_LEVEL >= 2:
            print(f"Calcul du volume OK: {b, a, H, calc_v = }")

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
        <=> e_h = e * sqrt(1 + c**2)

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


    # Vérification de la hauteur
    if DEBUG_LEVEL >= 1:
        # h = (1 - a/b) * H  <=>  a/b = (H - h)/H
        assert math.isclose(a/b, (H - h)/H, rel_tol=1e-9), \
            "Erreur dans la hauteur du frustum"
        if DEBUG_LEVEL >= 2:
            print(f"Calcul de la hauteur OK: {a/b = } = {(H - h)/H = }")

    c = 2 * H / b
    e_h = e * math.sqrt(1 + c**2)
    theta = math.atan(c)  # Utilisé pour plus tard
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

    À noter que d > 0 and h <= m.

    Nous pouvons désormais résoudre la hauteur maximale d'une pile. À noter que d > 0.
    Nous avons un paramètre qui détermine si l'on doit rajouter un espace supplémentaire. Dans le
    cas ou celui ci n'est pas activé, alors nous devons résoudre f(x) <= m:
    
            f(x) <= m
        <=> (h + e) + xd <= m
        <=> (h + e) / d + x <= m / d
        <=> x <= (m - h - e) / d

    Si l’on souhaite conserver un espace supplémentaire afin de pouvoir retirer le verre supérieur,
    il ne suffit pas de réserver la hauteur d’un verre complet. En effet, lorsque les verres sont
    empilés, chaque nouveau verre n’ajoute qu’une hauteur d, inférieure à sa hauteur extérieure
    réelle h+e, en raison de l’imbrication. La hauteur "perdue" à chaque empilement vaut donc
    (h+e)−d.

            f(x) + ((h + e) - d) <= m
        <=> (h + e) + xd + ((h + e) - d) <= m
        <=> 2(h + e) + (x - 1)d <= m
        <=> x - 1 <= (m - 2h - 2e) / d
        <=> x <= (m - 2h - 2e) / d + 1

    Enfin, on ajuste cette formule car nous avons un nombre entier de verres maximaux et obtenons
    p_m entier naturel la taille maximale d'une pile de verres:
    
            Ne pas laisser d'espace -> p_m = ⌊(m - h - e) / d⌋
            Laisser un espace       -> p_m = ⌊(m - 2h - 2e) / d + 1⌋
    """

    if h > m or not d > 0:
        return None
    p_m = int((m - 2*h - 2*e) / d + 1) if parameters.extra_space else int((m - h - e) / d)


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
    cette algorithme avec `pairs(prime_factors(p))`). Pour chaque combinaison de largeur et
    profondeur x y entiers naturels:

    Nous déterminons la hauteur et profondeur de l'armore:
            s_x = b_e * x
            s_y = b_e * y
            
            Ne pas laisser un espace -> s_z = f(p)
            Laisser un espace        -> s_z = f(p) + ((h + e) - d)
    """

    f = lambda x: h + x*d + e
    b_e = b + 2 * e_h
    if p_m == 0:  # Eviter la division par 0
        return None
    n_p = math.ceil(n / p_m)
    p = math.ceil(n / n_p)

    # Trouver les facteurs premiers
    glass_prime_factors = prime_factors(n_p)
    # Déterminer toutes les combinaisons possibles de ces variables
    cmb = pairs(glass_prime_factors)


    resultats = []
    for x, y in cmb:
        dim_z = f(p) + ((h+e) - d) if parameters.extra_space else f(p)
        dim = (b_e * x, b_e * y, dim_z)
        resultats.append(Resultat(
            dim=dim,
            volume=dim[0]*dim[1]*dim[2],
            grid=(x, y),
            max_glass_per_pile=p,
            variables={
                # Variables de taille
                'b': (b, 'Base haute intérieur (cm)'),
                'b_e': (b_e, 'Base haute extérieur (cm)'),
                'a': (a, 'Base basse intérieure (cm)'),
                'a_e': (a_e, 'Base basse extérieure (cm)'),
                'h': (h, 'Hauteur intérieur (cm)'),
                # Caractéristiques peut utiles
                'H': (H, 'Hauteur de la pyramide non-tronquée (cm)'),
                'e': (e, 'Epaisseur du verre (cm)'),
                'e_h': (e_h, 'Epaisseur du verre horizontale (cm)'),
                # Caractéristiques des piles
                'n_p': (n_p, 'Nombre de piles'),
                'p': (p, 'Nombre de verres par pile'),
                'n_p*p': (n_p*p, 'Nombre total de verres qui peuvent être placés'),
                # Autres informations
                'theta_pi': (theta, 'Angle de la paroi du verre (r)'),  # Angle de la paroi du verre
                'theta': (math.degrees(theta), 'Angle de la paroi du verre (°) (90° = vertical)')  # 90° -> a = b
            }
        ))
        

    return resultats



# -----------------------------------------------
# Entrée utilisateur
# -----------------------------------------------

@dataclass(frozen=True, slots=True)
class Parameters:

    min_angle: float | None
    min_base_ratio: float | None
    extra_space: bool

    @staticmethod
    def input() -> 'Parameters':
        """Créée un nouvelle instance de la classe à partir d'entrées utilisateur"""

        min_angle_in = input("Entrer l'angle minimal (en °) (vide -> aucune contrainte)\n> ").strip()
        min_angle = float(min_angle_in) if min_angle_in else None

        min_base_ratio_in = input("Entrez un ratio minimal haut : bas (x > 1) (vide -> aucune contrainte)\n> ").strip()
        min_base_ratio = float(min_base_ratio_in) if min_base_ratio_in else None
        assert min_base_ratio is None or min_base_ratio > 1

        extra_space_in = input("Laisser un espace supplémentaire pour pouvoir lever les verres tout en haut de la pile ? (Y/n) (par défault: n)\n> ").strip().lower()
        assert len(extra_space_in) < 2, "Veuillez entrer Y ou n"
        extra_space = extra_space_in == 'y'

        return Parameters(
            min_angle=min_angle,
            min_base_ratio=min_base_ratio,
            extra_space=extra_space
        )


# -----------------------------------------------
# Fonctions liés à l'interprétation des résultats
# -----------------------------------------------

def line_by_line_repr(l):
    return '[\n\t' + '\n\t'.join([str(elem) for elem in l]) + ']'


def fetch_result(results: list[Resultat] | None, parameters: Parameters) -> Resultat | None:
    """Interpréte les résultats avec des paramètres donnés"""

    if results is None:
        return None

    retained = results[0]  # Avec l'implémentation actuelle, les derniers résultats sont des
                           # grilles très longues et peu profondes et les premiers des grilles
                           # plus équilibrées.
    var = retained.variables

    if parameters.min_angle is not None and var['theta'][0] < parameters.min_angle:
        return None

    if parameters.min_base_ratio is not None and var['b'][0] / var['a'][0] > parameters.min_base_ratio:
        return None

    return retained


# ----------------------------------
# TESTS
# ----------------------------------

def basic_test():
    # Entrées utilisateur
    precision = int(input("Ecart entre les essais (1/1_000 cm) (Recommandé: 50-15)\n> "))
    parameters = Parameters.input()

    # Algorithme
    resultats = []
    for i in range(0, 30_000, precision):
        print(f"Nouvelle étape de recherche... {i / 1_000 = }")
        for j in range(0, 30_000, precision):

            # Effectuer les calculs
            result: Resultat | None = fetch_result(calc_square_pyramid(
                pyramid_top=i/1000,
                pyramid_height=j/1000,
                glasses=1000,
                edge_thickness=0.2,
                target_glass_volume=200,
                max_height=40,
                parameters=parameters
            ), parameters)

            if result is None:
                continue

            resultats.append(result)

    # On trie et affiche les 10 meilleurs résultats
    resultats.sort(key=lambda x: x.volume)
    # print('\n'.join([f'#{i}: {elem}' for i, elem in enumerate(resultats[:10])]))



def bin_like_test():
    precision = int(input("Nombre d'essais par variable par étape de recherche (itérations -> x²) (recommandé: 100 - 1_000)\n> "))
    assert precision >= 10, "Précision insuffisante"
    search_depth = int(input("Nombres d'étapes de recherches (profondeur) (recommandé: 5 - 10)\n> "))
    assert search_depth >= 1, "Nombre d'étapes de recherches (profondeur) insuffisant."
    parameters = Parameters.input()

    # Préparer les variables
    pyramid_top_range = (0, 50)
    pyramid_height_range = (0, 50)
    old_range = 50
    scan_width = 3

    # Afficher des informations à l'utilisateur...
    # Estimation de la précision
    final_range = old_range * ((scan_width / (precision - 1)) ** search_depth)
    final_step = final_range / (precision - 1)
    estimated_precision = math.log10(final_step)
    # Affichage
    print(f"Début... {precision**2*search_depth:,} tests seront effectués. Précision estimé: 10^{estimated_precision} cm")
    if estimated_precision < -14:
        print("Précision extrême. Les flottants à double précision risque de ne pas être assez précis.")

    # Algorithme de recherche
    pyramid_top_range = (0, 50)
    pyramid_height_range = (0, 50)
    best_result = None
    for i in range(search_depth):
        print(f"Nouvelle étape de recherche... Profondeur {i+1} / {search_depth}")
        # Calculer le nouveau meilleur résultat...
        best_result = run_tests(
            precision,
            pyramid_top_range,
            pyramid_height_range,
            parameters
        )
        b = best_result.variables['b'][0]
        H = best_result.variables['H'][0]
        # Redéfinir le rayon de recherche
        # 1.5 pour ajouter de la marge. En théorie, 0.5 pourrais fonctionner.
        old_range = pyramid_top_range[1] - pyramid_top_range[0]
        step = old_range / (precision - 1)

        new_range = scan_width * step  # Multiplier pour tester les voisins (évite de louper la solution)

        pyramid_top_range = (
            b - new_range / 2,
            b + new_range / 2
        )

        pyramid_height_range = (
            H - new_range / 2,
            H + new_range / 2
        )

    # Désormais best_result est le meilleur résultat. On peut afficher
    print(f'\n=== RESULTAT TROUVE ===\n\n{best_result}')


def run_tests(
    precision: int,
    pyramid_top_range: tuple[float, float],
    pyramid_height_range: tuple[float, float],
    parameters: Parameters
) -> Resultat:
    # Calculer les pas
    pyramid_top_step = (pyramid_top_range[1] - pyramid_top_range[0]) / (precision - 1)
    pyramid_height_step = (pyramid_height_range[1] - pyramid_height_range[0]) / (precision - 1)

    # Tester toutes les combinaisons
    resultats = []
    for i in range(precision):
        for j in range(precision):
            result = fetch_result(calc_square_pyramid(
                pyramid_top=pyramid_top_range[0] + pyramid_top_step * i,
                pyramid_height=pyramid_height_range[0] + pyramid_height_step * j,
                glasses=1000,
                edge_thickness=0.2,
                target_glass_volume=200,
                max_height=40,
                parameters=parameters
            ), parameters)
            # None est renvoyé si les entrées n'aboutissent pas à une solution. On passe
            if result is not None:
                resultats.append(result)

    # Retourner le meilleur resultat
    resultats.sort(key=lambda x: x.volume)
    return resultats[1]

# ----------------------

# Ansi escape codes
FM_CLEAR = '\x1b[0m'
FM_INVERSE = '\x1b[7m'

def main():

    DEBUG_MODE=False

    """
    ##
    ANCIENS ESSAIS MANUELS DE PRECENDENTS RESULTATS
    ##
    
            v            = (H/3)(b² - a²)
        <=> 3v           = H(b² - a²)
        <=> 3v / (b²-a²) = H
    
    Dans cette tentative, b = 7cm et a = 4cm et v = 20cL = 200mL = 200cm²
    ```
    # Premier essai -> ~81k cm3?
    print("Premier essai:")
    print(line_by_line_repr(calc_square_pyramid(
        pyramid_top=7,
        pyramid_height=(3*200) / (7**2 - 4**2),
        glasses=1000,
        edge_thickness=0.2,  # 2mm = 0.2cm
        target_glass_volume=200,
        max_height=40
    )))

    # Essai d'Edgar -> ~32k cm3
    print("Essai d'Edgar:")
    print(line_by_line_repr(calc_square_pyramid(
        pyramid_top=10.8,
        pyramid_height=(3*202.1) / (10.8**2 - 3.6**2),
        glasses=1000,
        edge_thickness=0.2,  # 2mm = 0.2cm
        target_glass_volume=200,
        max_height=40
    )))

    # Essai de Matei
    print("Essai de Matei:")
    b, a, v = 7.06, 1, 200
    H = (3*v) / (b**2 - a**2)
    print(line_by_line_repr(calc_square_pyramid(
        pyramid_top=b,
        pyramid_height=H,
        glasses=1000,
        edge_thickness=0.2,  # 2mm = 0.2cm
        target_glass_volume=200,
        max_height=40,
    )))
    ```
    """

    test_type = input(f"""\
Script python écrit dans le cadre du projet ResCo.
License MIT. Voir le fichier LICENSE.

Options:
{FM_INVERSE}BASIQUE{FM_CLEAR} Essaie aveuglement des millions de possibilités.
{FM_INVERSE}BINAIRE{FM_CLEAR} Recherche des milliers ou millions de possibiliés en divisant petit à petit le rayon de recherche

> """)
    print('\n')  # 2 new lines

    match test_type:
        case 'BASIQUE':
            basic_test()
        case 'BINAIRE':
            bin_like_test()
        case _:
            print(f'{FM_INVERSE}INVALIDE. REESSAYEZ.{FM_CLEAR}\n\n')
            main()


if __name__ == "__main__":
    main()
