"""Recuperer les deux TSV depuis le miroir cite, sans ecraser les fichiers existants."""
from pathlib import Path
import urllib.request

if __name__ == "__main__":
    data = Path(__file__).resolve().parent / "data"
    data.mkdir(exist_ok=True)
    base = "https://raw.githubusercontent.com/semnan-university-ai/Drug-Review-Dataset/master/"
    for nom in ("drugsComTrain_raw.tsv", "drugsComTest_raw.tsv"):
        cible = data / nom
        if cible.exists():
            print("Deja present :", cible)
            continue
        temporaire = cible.with_suffix(".tsv.part")
        print("Telechargement :", nom, flush=True)
        with urllib.request.urlopen(base + nom, timeout=60) as reponse, temporaire.open("wb") as fichier:
            longueur = reponse.headers.get("Content-Length")
            total = 0
            while bloc := reponse.read(1024 * 1024):
                fichier.write(bloc)
                total += len(bloc)
            if longueur and total != int(longueur):
                raise RuntimeError("Telechargement incomplet : relancer le script.")
        temporaire.replace(cible)
        print("Termine :", cible)
