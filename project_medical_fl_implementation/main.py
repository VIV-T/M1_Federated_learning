
import fluke
print(dir(fluke))  # Liste les modules disponibles


# 3. Importe la fonction centralized de fluke
import custom_dataset
from fluke.run import centralized

# 4. Charge tes fichiers YAML et lance l'entraînement
centralized("config/exp-iid.yml", "config/fedavg-logreg.yml")