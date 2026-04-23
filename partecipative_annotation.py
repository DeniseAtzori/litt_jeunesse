import streamlit as st
import pandas as pd
import os
from PIL import Image

DATA_DIR = "Img_annotation"
CSV_PATH = "annotations.csv"

# -----------------------------
# COSTANTI
# -----------------------------
CHARACTER_OPTIONS = [
    'Man', 'Woman', 'Girl', 'Boy', 'Animal',
    'Animate Object', 'Mythical / Imaginary Being',
    'Other', 'None'
]

FACES_OPTIONS = [
    'front', 'three_quarter', 'profile',
    'three_quarter_look_up', 'three_quarter_look_down',
    'behind', 'None'
]

SOCIAL_OPTIONS = [
    'Upper Class', 'Middle Class', 'Working Class', 'Poor',
    'Ambigous', 'None'
]

TOY_OPTIONS = [
    'doll', 'train', 'toy soldier',
    'rocking horse', 'hoop', 'puppet', 'other'
]

# -----------------------------
# Utility
# -----------------------------
def load_images():
    rows = []
    for subfolder in os.listdir(DATA_DIR):
        sub_path = os.path.join(DATA_DIR, subfolder)
        if os.path.isdir(sub_path):
            for file in os.listdir(sub_path):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    rows.append({
                        "subfolder": subfolder,
                        "filename": file
                    })
    return pd.DataFrame(rows)


def load_or_create_csv():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        df = load_images()
        for col in [
            "annotator", "characters", "scene_setting",
            "faces_position", "social_class", "toy",
            "emotional_intensity"
        ]:
            df[col] = ""
        df["done"] = False
        df.to_csv(CSV_PATH, index=False)

    return df.fillna("")


def save_csv(df):
    df.to_csv(CSV_PATH, index=False)


def safe_split(value):
    if pd.isna(value) or value == "":
        return []
    return str(value).split(";")

# -----------------------------
# INIT
# -----------------------------
df = load_or_create_csv()

if "mode" not in st.session_state:
    st.session_state.mode = "Da annotare"

if "filtered_pos" not in st.session_state:
    st.session_state.filtered_pos = 0

if "annotator_name" not in st.session_state:
    st.session_state.annotator_name = ""

if "annotated_count" not in st.session_state:
    st.session_state.annotated_count = 0

# -----------------------------
# MODALITÀ + FILTRO
# -----------------------------
mode = st.radio(
    "Modalità",
    ["Da annotare", "Già annotate"],
    index=0 if st.session_state.mode == "Da annotare" else 1
)

if mode != st.session_state.mode:
    st.session_state.mode = mode
    st.session_state.filtered_pos = 0

done_value = True if mode == "Già annotate" else False
filtered = df[df["done"] == done_value].reset_index()

if len(filtered) == 0:
    st.warning("Nessuna immagine in questa modalità")
    st.stop()

st.session_state.filtered_pos = min(
    st.session_state.filtered_pos,
    len(filtered) - 1
)

current_idx = filtered.loc[st.session_state.filtered_pos, "index"]
row = df.loc[current_idx]

# -----------------------------
# INIT STATE PER IMMAGINE
# -----------------------------
def init_state():
    if st.session_state.get("loaded_idx") != current_idx:

        chars = safe_split(row["characters"])
        faces = safe_split(row["faces_position"])
        social = safe_split(row["social_class"])
        toys = safe_split(row["toy"])

        if st.session_state.annotator_name == "":
            st.session_state.annotator_name = row["annotator"]

        st.session_state.n_chars = max(1, len(chars))

        st.session_state.characters = chars if chars else ["Man"]
        st.session_state.faces = faces if faces else ["front"]
        st.session_state.social = social if social else ["Middle Class"]

        st.session_state.scene = row["scene_setting"] if row["scene_setting"] else "Indoors"
        st.session_state.toys = toys
        st.session_state.n_toys = len(toys)

        st.session_state.emotion = int(row["emotional_intensity"]) if row["emotional_intensity"] else 3

        st.session_state.loaded_idx = current_idx

init_state()

# -----------------------------
# HEADER
# -----------------------------
col_title, col_counter = st.columns([4,1])
with col_counter:
    if mode == "Già annotate":
        unique_imgs = df[df["done"] == True][["subfolder","filename"]].drop_duplicates()
        st.markdown(f"Annotate totali: 📊 {len(unique_imgs)}")
    else:
        st.markdown(f"Annotate in questa sessione: ✅ {st.session_state.annotated_count}")

# -----------------------------
# IMMAGINE
# -----------------------------
img_path = os.path.join(DATA_DIR, row["subfolder"], row["filename"])
image = Image.open(img_path)
st.image(image, use_container_width=True)

# -----------------------------
# INFO FILE
# -----------------------------
st.markdown(f"**Libro:** {row['subfolder']}  \n**Immagine:** {row['filename']}")

# -----------------------------
# NAVIGAZIONE
# -----------------------------
col1, col2, col3 = st.columns([1,2,1])

with col1:
    if st.button("⬅️ Indietro"):
        st.session_state.filtered_pos = max(0, st.session_state.filtered_pos - 1)
        st.rerun()

with col3:
    if st.button("Avanti ➡️"):
        st.session_state.filtered_pos = min(
            len(filtered) - 1,
            st.session_state.filtered_pos + 1
        )
        st.rerun()

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:

    st.header("Annotazione ")

    st.session_state.annotator_name = st.text_input(
        "Annotatore (inserisci il tuo primo nome in maiuscolo)",
        value=st.session_state.annotator_name
    )

    new_n = st.number_input(
        "Numero di personaggi presenti sulla scena",
        min_value=1,
        max_value=10,
        value=st.session_state.n_chars
    )

    if new_n != st.session_state.n_chars:
        st.session_state.n_chars = new_n

        def resize(lst, default):
            if len(lst) < new_n:
                lst += [default] * (new_n - len(lst))
            else:
                lst = lst[:new_n]
            return lst

        st.session_state.characters = resize(st.session_state.characters, "Man")
        st.session_state.faces = resize(st.session_state.faces, "front")
        st.session_state.social = resize(st.session_state.social, "Middle Class")

    st.markdown("### Personaggi")
    st.markdown("Chi sono i personaggi? Se sono presenti più personaggi dello stesso tipo, inseriscili ogni volta.")
    for i in range(st.session_state.n_chars):
        st.session_state.characters[i] = st.selectbox(
            f"Personaggio {i+1}",
            CHARACTER_OPTIONS,
            index=CHARACTER_OPTIONS.index(st.session_state.characters[i]),
            key=f"char_{i}"
        )

    st.markdown("### Ambientazione")
    st.session_state.scene = st.radio(
        "Dov'è ambientata la scena?",
        ['Indoors', 'Outdoors', 'Both / Ambiguous'],
        index=['Indoors','Outdoors','Both / Ambiguous'].index(st.session_state.scene)
    )

    st.markdown("### Volti")
    st.markdown("In che posizione sono i volti dei personaggi? Inseriscili nello stesso ordine che hai usato per i personaggi.")
    for i in range(st.session_state.n_chars):
        st.session_state.faces[i] = st.selectbox(
            f"Volto {i+1}",
            FACES_OPTIONS,
            index=FACES_OPTIONS.index(st.session_state.faces[i]),
            key=f"face_{i}"
        )

    st.markdown("### Classe sociale")
    st.markdown("A che classe sociale appartengono? Inseriscile nello stesso ordine che hai usato per i personaggi.")
    for i in range(st.session_state.n_chars):
        st.session_state.social[i] = st.selectbox(
            f"Classe {i+1}",
            SOCIAL_OPTIONS,
            index=SOCIAL_OPTIONS.index(st.session_state.social[i]),
            key=f"social_{i}"
        )

    st.session_state.emotion = st.slider(
        "Intensità emotiva della scena: 1 molto calma, 5 molto intensa.",
        1, 5,
        value=st.session_state.emotion
    )

    if st.button("💾 Salva"):

        if st.session_state.annotator_name.strip() == "":
            st.error("Inserisci annotatore")
        else:
            annotator = st.session_state.annotator_name.lower()

            # 🔥 MULTI-ANNOTATOR LOGIC
            mask = (
                (df["subfolder"] == row["subfolder"]) &
                (df["filename"] == row["filename"]) &
                (df["annotator"] == annotator)
            )

            if mask.any():
                idx = df[mask].index[0]
            else:
                new_row = row.copy()
                new_row["annotator"] = annotator
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                idx = df.index[-1]

            df.at[idx, "characters"] = ";".join(st.session_state.characters)
            df.at[idx, "scene_setting"] = st.session_state.scene
            df.at[idx, "faces_position"] = ";".join(st.session_state.faces)
            df.at[idx, "social_class"] = ";".join(st.session_state.social)
            df.at[idx, "emotional_intensity"] = st.session_state.emotion
            df.at[idx, "done"] = True

            save_csv(df)

            st.session_state.annotated_count += 1

            filtered = df[df["done"] == False].reset_index()

            if len(filtered) > 0:
                st.session_state.filtered_pos = min(
                    st.session_state.filtered_pos,
                    len(filtered) - 1
                )
            else:
                st.session_state.filtered_pos = 0

            st.success("Salvato!")
            st.rerun()