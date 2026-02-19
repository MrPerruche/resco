# Installation

Pour utiliser ce script, vous devez avoir:
- Python 3.13+ (recommendé: Python [3.13.12](https://www.python.org/downloads/release/python-31312/))
- NumPy (recommendé: dernière version)

**Si vous ne souhaitez pas installer ce script Python, mettre en place l'environnement virtuel etc., vous pouvez éxécuter ce script sur un interpréteur en ligne comme [Online Python](www.online-python.com). À noter que pratiquement tous les interpréteurs en ligne mettera fin au script s'il met trop longtemps à s'exécuter. N'entrez pas des valeurs excessives.**

Vous devez d'abord vérifier quelle version de python vous avez actuellement si vous l'avez déjà installé, pour savoir si vous avez une version suffisament à jour.
- Dans le fichier où se trouve le fichier .py, ouvrez le terminal (windows: clic droit -> ouvrir dans le terminal)
- Entrez `py --version` et vérifiez que vous avez Python 3.13 ou supérieur. Il est possible que deux versions différentes de Python sont installés, dans ce cas la deuxiè_me est accessible avec `python` au lieu de `py` -> Essayez aussi `python --version` si `py --version` affiche une version inférieur à la 3.13.

Une fois que vous avez la bonne version, dans le terminal, exécutez le script en entrant `py main.py` ou `python main.py` (selon la commande qui vous a affiché la bonne version de Python.)

Alternativement, vous pouvez double cliquer sur le fichier `main.py` pour éxécuter le script, mais cela pourrais utiliser la mauvaise version de Python si vous avez plusieurs versions d'installé.

Si le script est exécuté dans une version qui n'est pas prise en charge, un avertissement s'affichera dans le terminal. Vous pourrez appuyer sur Entrer pour ignorer l'avertissement, mais il est fort probable que vous rencontrez des erreurs (notamment des erreurs de syntaxe liés aux match case et aux f-string imbriquées).
