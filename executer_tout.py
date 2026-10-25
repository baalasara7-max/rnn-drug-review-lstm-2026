"""Executer toutes les etapes dans l'ordre depuis le dossier du projet.

Les TSV originaux doivent etre dans data/. Les checkpoints complets deja presents
peuvent etre reutilises par l'etape 10 selon ses regles de reprise.
"""
from pathlib import Path
import json
import subprocess
import sys
import time


def main():
    dossier = Path(__file__).resolve().parent
    journaux = dossier / "execution"
    journaux.mkdir(exist_ok=True)
    commandes = [
        ["telecharger_donnees.py"],
        ["etape_01_explorer.py", "data/drugsComTrain_raw.tsv"],
        ["etape_02_categories.py", "data/drugsComTrain_raw.tsv"],
        ["etape_03_nettoyer.py"],
        ["etape_04_separer.py"],
        ["etape_05_vocabulaire.py"],
        ["etape_06_embedding.py"],
        ["etape_07_modele_vanilla.py"],
        ["-m", "unittest", "discover", "-s", "tests", "-v"],
        ["etape_08_entrainer.py"],
        ["etape_09_variantes.py"],
        ["etape_10_experiences.py"],
        ["etape_11_predire.py", "This medicine helped me and I feel better.", "--architecture", "cifg"],
    ]
    bilan = {"python": sys.version, "etapes": []}
    for numero, arguments in enumerate(commandes, 1):
        nom = arguments[0] if arguments[0] != "-m" else "tests"
        chemin = journaux / f"{numero:02d}_{Path(nom).stem}.log"
        print(f"[{numero}/{len(commandes)}] {' '.join(arguments)}", flush=True)
        debut = time.perf_counter()
        with chemin.open("w", encoding="utf-8") as sortie:
            processus = subprocess.Popen([sys.executable, "-u", *arguments], cwd=dossier,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, encoding="utf-8", errors="replace",
                                          env={**__import__('os').environ, "PYTHONIOENCODING": "utf-8"})
            for ligne in processus.stdout:
                sortie.write(ligne)
                sortie.flush()
                # Les avis complets restent disponibles dans le journal de l'etape 1.
                if "explorer" not in nom:
                    print(ligne, end="", flush=True)
            code = processus.wait()
        bilan["etapes"].append({"commande": arguments, "code_retour": code,
                                "secondes": time.perf_counter() - debut,
                                "journal": chemin.name})
        (journaux / "bilan_execution.json").write_text(json.dumps(bilan, indent=2), encoding="utf-8")
        if code:
            raise SystemExit(f"Echec : {nom}. Voir {chemin}")
    print("Toutes les etapes se sont terminees sans erreur.", flush=True)


if __name__ == "__main__":
    main()
