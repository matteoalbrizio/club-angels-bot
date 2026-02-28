import time
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin

BASE = "https://www.clubdeglinvestitori.it"
LIST_URL = "https://www.clubdeglinvestitori.it/it/people/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MatteoBot/1.0; +https://example.com)"
}

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fetch(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def extract_people_links(list_soup: BeautifulSoup):
    """
    La pagina /it/people/ contiene una lista di link con i nomi.
    Prendiamo SOLO quelli che puntano a /it/people/<slug>/.
    """
    people = []
    for a in list_soup.select("a[href]"):
        href = a.get("href", "")
        text = clean_text(a.get_text(" "))
        if not text:
            continue
        # link relativi o assoluti
        full = urljoin(BASE, href)

        if "/it/people/" in full and full.rstrip("/").count("/") >= 5:
            # evita la pagina indice /it/people/
            if full.rstrip("/") == LIST_URL.rstrip("/"):
                continue
            # evita roba strana tipo anchor
            if "#" in full:
                continue
            people.append((text, full))

    # dedup per URL
    uniq = {}
    for name, url in people:
        uniq[url] = name
    return [(name, url) for url, name in uniq.items()]

def extract_bio(profile_soup: BeautifulSoup) -> str:
    """
    Strategia semplice:
    - prendiamo il contenuto testuale principale della pagina
    - togliamo pezzi inutili (menu/footer)
    """
    # spesso il corpo principale sta in article / main / entry-content
    candidates = []
    for sel in ["article", "main", ".entry-content", ".wp-block-post-content"]:
        el = profile_soup.select_one(sel)
        if el:
            candidates.append(el)

    container = candidates[0] if candidates else profile_soup.body

    # rimuovi elementi non utili
    for junk in container.select("nav, footer, header, form, script, style"):
        junk.decompose()

    text = clean_text(container.get_text(" "))

    # spesso nel testo c’è il nome ripetuto e “Torna ai soci” ecc.
    # qui facciamo una pulizia “good enough”
    text = text.replace("Torna ai soci", " ")
    text = text.replace("Rimani Aggiornato", " ")

    return clean_text(text)

def main():
    list_soup = fetch(LIST_URL)
    people = extract_people_links(list_soup)
    print(f"Trovati {len(people)} profili")

    rows = []
    for name, url in tqdm(people):
        try:
            soup = fetch(url)
            bio = extract_bio(soup)
            rows.append({"name": name, "url": url, "bio": bio})
            time.sleep(0.3)  # gentilezza verso il server
        except Exception as e:
            rows.append({"name": name, "url": url, "bio": "", "error": str(e)})
            continue

    df = pd.DataFrame(rows)
    df.to_csv("club_people.csv", index=False)
    df.to_json("club_people.json", orient="records", force_ascii=False, indent=2)
    print("Salvati: club_people.csv e club_people.json")

if __name__ == "__main__":
    main()