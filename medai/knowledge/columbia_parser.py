"""
Columbia University Disease-Symptom Knowledge Base parser.

Scrapes and parses the KB from:
https://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html

Run directly: python knowledge/columbia_parser.py
"""

import json
import re
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from difflib import SequenceMatcher

import requests
from bs4 import BeautifulSoup


KB_URL = "https://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "knowledge_base" / "columbia_kb.json"


def _parse_umls_entry(entry: str) -> List[Dict]:
    """
    Parse a UMLS-coded entry, handling compound codes joined by ^.

    e.g. "UMLS:C0038990_sweat^UMLS:C0700590_sweating increased"
    → [{"umls_code": "C0038990", "name": "sweat"},
       {"umls_code": "C0700590", "name": "sweating increased"}]
    """
    entry = entry.strip()
    parts = entry.split("^")
    results = []
    pattern = re.compile(r"UMLS:([A-Z0-9]+)_(.+)", re.IGNORECASE)
    for part in parts:
        part = part.strip()
        m = pattern.match(part)
        if m:
            results.append({
                "umls_code": m.group(1).strip(),
                "name": m.group(2).strip(),
            })
    return results


def parse_columbia_kb(url: str = KB_URL) -> dict:
    """Scrape and parse the Columbia Disease-Symptom KB HTML table."""
    print(f"Fetching KB from {url} ...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("No table found on the Columbia KB page.")

    rows = table.find_all("tr")

    diseases: List[Dict] = []
    disease_index: Dict[str, int] = {}  # umls_code → index in diseases list

    current_disease: Optional[Dict] = None

    for row in rows:
        cells = row.find_all("td")
        if not cells:
            continue  # header row or empty

        # Rows can have 3 columns (disease, count, symptom) or
        # just 1-2 cols when disease cell is empty (continuation).
        cell_texts = [c.get_text(separator=" ", strip=True) for c in cells]

        # Detect if disease column is populated
        disease_col = cell_texts[0].strip() if len(cell_texts) > 0 else ""
        count_col = cell_texts[1].strip() if len(cell_texts) > 1 else ""
        symptom_col = cell_texts[2].strip() if len(cell_texts) > 2 else ""

        # If disease column is non-empty and looks like a UMLS entry,
        # start a new disease (or retrieve existing one).
        if disease_col and "UMLS:" in disease_col:
            disease_entries = _parse_umls_entry(disease_col)
            if not disease_entries:
                continue
            d = disease_entries[0]  # disease cell is always a single entry
            try:
                occ = int(re.sub(r"[^\d]", "", count_col))
            except ValueError:
                occ = 0

            key = d["umls_code"]
            if key in disease_index:
                current_disease = diseases[disease_index[key]]
            else:
                current_disease = {
                    "umls_code": key,
                    "name": d["name"],
                    "occurrence_count": occ,
                    "symptoms": [],
                }
                disease_index[key] = len(diseases)
                diseases.append(current_disease)

        if current_disease is None:
            continue  # no disease context yet

        # Parse symptom column (may be compound)
        if symptom_col and "UMLS:" in symptom_col:
            symptom_entries = _parse_umls_entry(symptom_col)
            rank = len(current_disease["symptoms"]) + 1
            for s in symptom_entries:
                current_disease["symptoms"].append({
                    "umls_code": s["umls_code"],
                    "name": s["name"],
                    "rank": rank,
                })

    # Collect unique symptoms
    unique_symptoms: set = set()
    for d in diseases:
        for s in d["symptoms"]:
            unique_symptoms.add(s["name"].lower())

    kb = {
        "diseases": diseases,
        "metadata": {
            "total_diseases": len(diseases),
            "total_unique_symptoms": len(unique_symptoms),
            "source": "Columbia University DBMI Disease-Symptom KB",
            "url": url,
            "parsed_at": datetime.utcnow().isoformat() + "Z",
        },
    }

    return kb


def save_kb(kb: dict, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
    print(f"Saved KB to {path}")


def load_kb(path: Path = OUTPUT_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fuzzy_match(query: str, candidates: List[str], threshold: float = 0.6) -> List[str]:
    """Return candidates that fuzzy-match the query above threshold."""
    query_lower = query.lower()
    matches = []
    for c in candidates:
        ratio = SequenceMatcher(None, query_lower, c.lower()).ratio()
        if ratio >= threshold:
            matches.append((ratio, c))
    matches.sort(reverse=True)
    return [m[1] for m in matches]


def get_diseases_for_symptom(
    symptom_name: str, kb_path: Path = OUTPUT_PATH
) -> List[dict]:
    """
    Fuzzy-match symptom_name against the KB and return a ranked list
    of diseases whose symptom lists include it.
    """
    kb = load_kb(kb_path)
    symptom_lower = symptom_name.lower()

    results = []
    for disease in kb["diseases"]:
        for sym in disease["symptoms"]:
            sym_name = sym["name"].lower()
            ratio = SequenceMatcher(None, symptom_lower, sym_name).ratio()
            if ratio >= 0.5 or symptom_lower in sym_name or sym_name in symptom_lower:
                results.append({
                    "disease_name": disease["name"],
                    "umls_code": disease["umls_code"],
                    "occurrence_count": disease["occurrence_count"],
                    "symptom_rank": sym["rank"],
                    "symptom_matched": sym["name"],
                    "match_score": ratio,
                })
                break  # each disease counted once per call

    results.sort(key=lambda x: (x["match_score"], -x["symptom_rank"]), reverse=True)
    return results


def get_symptoms_for_disease(
    disease_name: str, kb_path: Path = OUTPUT_PATH
) -> List[dict]:
    """
    Fuzzy-match disease_name and return its ranked symptom list.
    """
    kb = load_kb(kb_path)
    disease_lower = disease_name.lower()

    best_match = None
    best_ratio = 0.0
    for disease in kb["diseases"]:
        ratio = SequenceMatcher(None, disease_lower, disease["name"].lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = disease

    if best_match and best_ratio >= 0.4:
        return sorted(best_match["symptoms"], key=lambda s: s["rank"])
    return []


if __name__ == "__main__":
    kb = parse_columbia_kb()
    n_diseases = kb["metadata"]["total_diseases"]
    n_symptoms = kb["metadata"]["total_unique_symptoms"]
    print(f"Parsed {n_diseases} diseases, {n_symptoms} unique symptoms")
    save_kb(kb)
