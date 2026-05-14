"""
Valutazione dell'accuratezza dell'annotazione automatica (LLM)
rispetto all'annotazione manuale (gold standard).

Campi valutati:
  - characters         (lista, ordine irrilevante)
  - scene_setting      (valore singolo)
  - faces_position     (lista) – due varianti: fine-grained e aggregata
  - social_class       (lista) – due varianti: fine-grained e aggregata
  - toy                (lista, ordine irrilevante)
  - emotional_intensity (valore numerico)

Per i campi lista: accuratezza normale + accuratezza "partial credit"
(solo elementi effettivamente presenti nel LLM, indipendentemente
dal numero totale atteso dall'annotatore).

Per immagini con 2 annotatori: si usa la media dei due.
"""

import json
import re
import pandas as pd
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────
# 1. CARICAMENTO FILE
# ─────────────────────────────────────────────

CSV_PATH  = "/Users/deniseatzori/Library/Mobile Documents/com~apple~CloudDocs/ENC-PSL/Memoire/Github_tesi/Results/MLLM_annotation/human_annotations.csv"
JSON_PATH = "/Users/deniseatzori/Library/Mobile Documents/com~apple~CloudDocs/ENC-PSL/Memoire/Github_tesi/Results/MLLM_annotation/llm_annotations.json"

df = pd.read_csv(CSV_PATH)
with open(JSON_PATH) as f:
    llm_data = json.load(f)

# Dizionario di lookup: (subfolder, filename) -> entry LLM
llm_lookup = {}
for entry in llm_data:
    key = (entry["subfolder"].strip(), entry["file_name"].strip())
    llm_lookup[key] = entry.get("llm_output", {})

print(f"Righe CSV: {len(df)}")
print(f"Entrate JSON: {len(llm_data)}")

# ─────────────────────────────────────────────
# 2. NORMALIZZAZIONE VALORI
# ─────────────────────────────────────────────

# Artefatti noti nel JSON: stringhe spurie da "None" mal-parsato
SPURIOUS = {"n", "o", "e", "none"}  # lower-case


def normalize_str(s: str) -> str:
    """Pulizia generica: strip, case canonico."""
    return s.strip()


def normalize_llm_list(raw_list: list) -> list:
    """
    Rimuove artefatti spuri ('N','e','n','o','none','None')
    e normalizza varianti note (es. 'Middle' -> 'Middle Class',
    'Ambiguous' -> 'Both / Ambiguous').
    """
    result = []
    for item in raw_list:
        s = str(item).strip()
        if s.lower() in SPURIOUS:
            continue
        # Alias noti
        aliases = {
            "Middle":     "Middle Class",
            "Ambiguous":  "Both / Ambiguous",
            "toy":        "other",   # 'toy' generico -> 'other'
        }
        s = aliases.get(s, s)
        result.append(s)
    return result


def parse_csv_list(cell_value) -> list:
    """
    Trasforma una cella CSV semicolon-separated in lista.
    NaN / None -> lista vuota.
    """
    if pd.isna(cell_value):
        return []
    return [normalize_str(x) for x in str(cell_value).split(";")]


def aggregate_three_quarter(labels: list) -> list:
    """Unifica three_quarter_look_up/down -> three_quarter."""
    return [
        "three_quarter" if re.match(r"three_quarter_look", lbl) else lbl
        for lbl in labels
    ]


def aggregate_class(labels: list) -> list:
    """Unifica Middle Class + Upper Class -> 'Middle/Upper Class'."""
    return [
        "Middle/Upper Class" if lbl in ("Middle Class", "Upper Class") else lbl
        for lbl in labels
    ]


# ─────────────────────────────────────────────
# 3. METRICHE DI CONFRONTO PER CAMPI LISTA
# ─────────────────────────────────────────────

def list_accuracy(human: list, llm: list) -> dict:
    """
    Confronta due liste ignorando l'ordine.

    Ritorna un dizionario con:
      - 'full'   : percentuale di elementi umani presenti nell'LLM
                   (penalizza se LLM ha meno elementi)
      - 'partial': percentuale di elementi LLM corretti rispetto
                   a quelli LLM presenti (utile quando LLM < umano)

    Entrambe le metriche usano multiset (contatori):
      corrispondenze = min(count_umano[x], count_llm[x]) per ogni x.
    """
    from collections import Counter

    # Escludi None / valori vuoti dal conteggio
    def clean(lst):
        return [x for x in lst if x and x.lower() not in ("none", "")]

    h = clean(human)
    l = clean(llm)

    if not h and not l:
        return {"full": 1.0, "partial": 1.0}
    if not h:
        # Umano non annota nulla, LLM aggiunge qualcosa: non penalizziamo
        return {"full": 1.0, "partial": 1.0}
    if not l:
        return {"full": 0.0, "partial": float("nan")}

    ch = Counter(h)
    cl = Counter(l)

    matches = sum(min(ch[k], cl[k]) for k in ch)

    full    = matches / len(h)          # rispetto a quanti elementi attesi
    partial = matches / len(l)          # rispetto a quanti ne ha messo il LLM

    return {"full": full, "partial": partial}


def scalar_accuracy(human_val, llm_val, numeric=False) -> dict:
    """
    Confronto per valori scalari.
    Se numeric=True (emotional_intensity), ritorna anche acc con tolleranza ±1.
    """
    if pd.isna(human_val) or llm_val is None:
        return {"exact": float("nan"), "tol1": float("nan")} if numeric else {"exact": float("nan")}

    if numeric:
        try:
            h = round(float(human_val))
            l = int(llm_val)
        except (ValueError, TypeError):
            return {"exact": float("nan"), "tol1": float("nan")}
        return {
            "exact": 1.0 if h == l else 0.0,
            "tol1":  1.0 if abs(h - l) <= 1 else 0.0,
        }
    else:
        # Alias per scene_setting
        aliases = {"Ambiguous": "Both / Ambiguous"}
        h = str(human_val).strip()
        l = aliases.get(str(llm_val).strip(), str(llm_val).strip())
        return {"exact": 1.0 if h == l else 0.0}


# ─────────────────────────────────────────────
# 4. CICLO PRINCIPALE
# ─────────────────────────────────────────────

# Raccoglie risultati per immagine (media su annotatori se >1)
# Struttura: results[key] = {campo: valore_o_dict, ...}
results = defaultdict(lambda: defaultdict(list))

missing_in_llm = []

for _, row in df.iterrows():
    subfolder = row["subfolder"].strip()
    filename  = row["filename"].strip()
    key = (subfolder, filename)

    if key not in llm_lookup:
        missing_in_llm.append(key)
        continue

    llm = llm_lookup[key]

    # ── characters ──────────────────────────────
    h_char = parse_csv_list(row["characters"])
    l_char = normalize_llm_list(llm.get("characters") or [])
    results[key]["characters"].append(list_accuracy(h_char, l_char))

    # ── scene_setting ────────────────────────────
    h_ss = row["scene_setting"]
    l_ss = llm.get("scene_setting")
    results[key]["scene_setting"].append(scalar_accuracy(h_ss, l_ss))

    # ── faces_position (fine-grained) ───────────
    h_fp = parse_csv_list(row["faces_position"])
    l_fp = normalize_llm_list(llm.get("faces_position") or [])
    results[key]["faces_position_fine"].append(list_accuracy(h_fp, l_fp))

    # ── faces_position (aggregata: three_quarter*) ──
    h_fp_agg = aggregate_three_quarter(h_fp)
    l_fp_agg = aggregate_three_quarter(l_fp)
    results[key]["faces_position_agg"].append(list_accuracy(h_fp_agg, l_fp_agg))

    # ── social_class (fine-grained) ─────────────
    h_sc = parse_csv_list(row["social_class"])
    l_sc = normalize_llm_list(llm.get("social_class") or [])
    results[key]["social_class_fine"].append(list_accuracy(h_sc, l_sc))

    # ── social_class (aggregata: Middle+Upper) ──
    h_sc_agg = aggregate_class(h_sc)
    l_sc_agg = aggregate_class(l_sc)
    results[key]["social_class_agg"].append(list_accuracy(h_sc_agg, l_sc_agg))

    # ── toy ─────────────────────────────────────
    h_toy = parse_csv_list(row["toy"])
    l_toy = normalize_llm_list(llm.get("toy") or [])
    results[key]["toy"].append(list_accuracy(h_toy, l_toy))

    # ── emotional_intensity ──────────────────────
    h_ei = row["emotional_intensity"]
    l_ei = llm.get("emotional_intensity")
    results[key]["emotional_intensity"].append(scalar_accuracy(h_ei, l_ei, numeric=True))

if missing_in_llm:
    print(f"\nATTENZIONE: {len(missing_in_llm)} immagini non trovate nel JSON:")
    for k in missing_in_llm:
        print(f"  {k[0]} / {k[1]}")

# ─────────────────────────────────────────────
# 5. AGGREGAZIONE: media su annotatori, poi media su immagini
# ─────────────────────────────────────────────

def avg_dicts(list_of_dicts: list) -> dict:
    """Media di una lista di dict (per list_accuracy o scalar con sub-keys)."""
    keys = list_of_dicts[0].keys()
    out = {}
    for k in keys:
        vals = [d[k] for d in list_of_dicts if not np.isnan(d[k])]
        out[k] = np.mean(vals) if vals else float("nan")
    return out


# Per ogni immagine: media sui suoi annotatori
img_scores = {}
for key, fields in results.items():
    img_scores[key] = {}
    for field, vals in fields.items():
        img_scores[key][field] = avg_dicts(vals)

# Media finale su tutte le immagini
list_fields  = ["characters", "faces_position_fine", "faces_position_agg",
                "social_class_fine", "social_class_agg", "toy"]
scalar_fields = ["scene_setting", "emotional_intensity"]

final = {}
for field in list_fields:
    full_vals    = [img_scores[k][field]["full"]    for k in img_scores if field in img_scores[k] and not np.isnan(img_scores[k][field]["full"])]
    partial_vals = [img_scores[k][field]["partial"] for k in img_scores if field in img_scores[k] and not np.isnan(img_scores[k][field]["partial"])]
    final[field] = {
        "full":    np.mean(full_vals)    if full_vals    else float("nan"),
        "partial": np.mean(partial_vals) if partial_vals else float("nan"),
        "n_images": len(full_vals),
    }

for field in scalar_fields:
    sub_keys = list(next(
        img_scores[k][field] for k in img_scores if field in img_scores[k]
    ).keys())
    final[field] = {"n_images": 0}
    for sk in sub_keys:
        vals = [img_scores[k][field][sk] for k in img_scores
                if field in img_scores[k] and not np.isnan(img_scores[k][field][sk])]
        final[field][sk] = np.mean(vals) if vals else float("nan")
        final[field]["n_images"] = len(vals)

# ─────────────────────────────────────────────
# 6. STAMPA RISULTATI
# ─────────────────────────────────────────────

DIVIDER = "─" * 70

def pct(v):
    return f"{v*100:.1f}%" if not np.isnan(v) else "N/A"


print("\n")
print(DIVIDER)
print("  VALUTAZIONE ACCURATEZZA: LLM vs ANNOTAZIONE UMANA (gold standard)")
print(DIVIDER)
print(f"  Immagini valutate: {len(img_scores)}")
print(f"  Immagini mancanti nel JSON: {len(missing_in_llm)}")
print()

# ── Campi scalari ────────────────────────────
print("CAMPI SCALARI")
print(DIVIDER)
print(f"  {'Campo':<35} {'Esatta':>8} {'Tol. ±1':>10}   {'N':>6}")
print(DIVIDER)
for field in scalar_fields:
    r = final[field]
    if "tol1" in r:
        print(f"  {field:<35} {pct(r['exact']):>8} {pct(r['tol1']):>10}   {r['n_images']:>6}")
    else:
        print(f"  {field:<35} {pct(r['exact']):>8} {'—':>10}   {r['n_images']:>6}")

print()

# ── Campi lista ──────────────────────────────
print("CAMPI LISTA (corrispondenza multiset, ordine irrilevante)")
print()
print("  'full'    = % elementi umani trovati nell'LLM")
print("              (penalizza omissioni dell'LLM)")
print("  'partial' = % elementi LLM che sono corretti")
print("              (utile quando LLM inserisce meno elementi)")
print()
print(DIVIDER)
print(f"  {'Campo':<35} {'full':>8} {'partial':>10}   {'N':>6}")
print(DIVIDER)

list_labels = {
    "characters":          "characters",
    "faces_position_fine": "faces_position  (fine-grained)",
    "faces_position_agg":  "faces_position  (three_quarter aggregato)",
    "social_class_fine":   "social_class    (fine-grained)",
    "social_class_agg":    "social_class    (Middle+Upper aggregati)",
    "toy":                 "toy",
}

for field, label in list_labels.items():
    r = final[field]
    print(f"  {label:<35} {pct(r['full']):>8} {pct(r['partial']):>10}   {r['n_images']:>6}")

print(DIVIDER)
print()


# ─────────────────────────────────────────────
# 7. EXPORT CSV CON DETTAGLIO PER IMMAGINE
# ─────────────────────────────────────────────

rows = []
for (subfolder, filename), fields in img_scores.items():
    row = {"subfolder": subfolder, "filename": filename}
    for field in list_fields:
        if field in fields:
            row[f"{field}_full"]    = fields[field]["full"]
            row[f"{field}_partial"] = fields[field]["partial"]
    for field in scalar_fields:
        if field in fields:
            for sk, sv in fields[field].items():
                row[f"{field}_{sk}"] = sv
    rows.append(row)

out_df = pd.DataFrame(rows)
out_path = "/mnt/user-data/outputs/accuracy_per_image.csv"
out_df.to_csv(out_path, index=False)
print(f"  → Dettaglio per immagine salvato in: {out_path}")
print()
