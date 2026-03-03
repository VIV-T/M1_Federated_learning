import subprocess
import os

# Redirige stdout vers un fichier temporaire pour éviter l'affichage de la bannière
with open(os.devnull, 'w', encoding='utf-8') as devnull:
    result = subprocess.run(
        ["fluke", "federation", "config/exp-fedavg-logreg-iid.yml", "config/fedavg-logreg.yml"],
        stdout=devnull,  # Ignore la sortie standard (bannière)
        stderr=subprocess.PIPE,  # Capture les erreurs
        text=True,
        encoding='utf-8'
    )

# Affiche les erreurs (si elles existent)
if result.stderr:
    print("Erreur :", result.stderr)
else:
    print("La commande s'est exécutée avec succès (sortie ignorée pour éviter les erreurs d'encodage).")
