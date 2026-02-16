# Installation

Pour utiliser ce script, vous devez avoir:
- Python 3.13+ (recommendé: Python [3.13.12](https://www.python.org/downloads/release/python-31312/))
- NumPy (recommendé: dernière version)

**Si vous ne souhaitez pas installer ce script Python, mettre en place l'environnement virtuel etc., vous pouvez éxécuter ce script sur un interpréteur en ligne qui supporte NumPy comme [Online Python](www.online-python.com). À noter que pratiquement tous les interpréteurs en ligne mettera fin au script s'il met trop longtemps à s'exécuter. N'entrez pas des valeurs excessives.**

Vous devez d'abord vérifier quelle version de python vous avez actuellement si vous l'avez déjà installé, pour savoir si vous avezla bonne version:
`py --version`. Une deuxième version de python peut être présente et accessible avec `python` au lieu de `py` -> Essayez aussi `python --version` si `py --version` affiche une version inférieur à la 3.13.

Dans le fichier que vous souhaitez modifier, ouvrez le terminal (windows: clic droit -> ouvrir dans le terminal)

Dans le terminal, créez un environnement virtuel: `python -m venv .venv`

Entrez dans l'environnement virtuel: `.\.venv\Scripts\Activate.ps1` ou `.venv\Scripts\activate` si le précédent ne fonctionne pas

Installez numpy: `py -m pip install numpy`

Exécutez le script: `py main.py`
