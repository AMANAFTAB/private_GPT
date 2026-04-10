import streamlit as st
import os
import json
import uuid
import base64
import shutil
import string
import random
from datetime import datetime
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from PIL import Image
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from openai import AzureOpenAI
import pyrebase
from streamlit_autorefresh import st_autorefresh

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AM Intelligence",
    page_icon="🔒",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════
# AZURE Authentication
# All models share the same endpoint/key since they are deployed
# under the same Azure AI Foundry resource.
# ══════════════════════════════════════════════════════════════════════
AZURE_ENDPOINT       = st.secrets["MODELS_ENDPOINT"]
AZURE_API_KEY        = st.secrets["LLM_API_KEY"]
AZURE_API_VERSION    = "2025-04-01-preview"
LLM_DEPLOYMENT       = "Llama-4-Maverick-17B-128E-Instruct-FP8"
EMBEDDING_DEPLOYMENT = "text-embedding-ada-002"

# ══════════════════════════════════════════════════════════════════════
# FIREBASE Authentication
# ══════════════════════════════════════════════════════════════════════
FIREBASE_CONFIG = {
    "apiKey":            st.secrets["firebase"]["apiKey"],
    "authDomain":        st.secrets["firebase"]["authDomain"],
    "databaseURL":       st.secrets["firebase"]["databaseURL"],
    "projectId":         st.secrets["firebase"]["projectId"],
    "storageBucket":     st.secrets["firebase"]["storageBucket"],
    "messagingSenderId": st.secrets["firebase"]["messagingSenderId"],
    "appId":             st.secrets["firebase"]["appId"],
}

# ══════════════════════════════════════════════════════════════════════
# AUTHORIZED USERS
# ══════════════════════════════════════════════════════════════════════
AUTHORIZED_USERS = {
    "authuser": "password321",
}

# ══════════════════════════════════════════════════════════════════════
# LOCAL STORAGE PATHS
# ══════════════════════════════════════════════════════════════════════
APP_DATA_DIR = "./app_data"
SESSIONS_DIR = os.path.join(APP_DATA_DIR, "sessions")
ROOMS_DIR    = os.path.join(APP_DATA_DIR, "rooms")
os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(ROOMS_DIR,    exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# SESSION STATE BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "authenticated":      False,
    "username":           "",
    "mode":               "private",
    "current_session_id": None,
    "current_room_code":  None,
    "pdf_stores":         {},
    "image_files":        {},
    "selected_files":     [],
    "chat_history":       [],
    "room_thinking":      False,
    "room_selected_files": [],   # tracks file selection specifically for rooms
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ══════════════════════════════════════════════════════════════════════
# FIREBASE INIT --> uses the [firebase] config parameters mentioned in the .secrets/secrets.toml file
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource
def init_firebase():
    firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
    return firebase.database()

db = init_firebase()

# ══════════════════════════════════════════════════════════════════════
# AZURE CLIENTS --> calling the embedding model
# ══════════════════════════════════════════════════════════════════════
class AzureOpenAIEmbeddings(Embeddings):
    """
    Wraps the Azure OpenAI embeddings API to be compatible with FAISS.
    embed_documents()  called when indexing uploaded file chunks.
    embed_query()      called when converting a user question to a vector.
    """
    def __init__(self, client, deployment: str):
        self.client     = client
        self.deployment = deployment

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            input=texts, model=self.deployment
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            input=[text], model=self.deployment
        )
        return response.data[0].embedding


@st.cache_resource
def load_azure_client():
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )

@st.cache_resource
def load_embeddings(_client):
    return AzureOpenAIEmbeddings(client=_client, deployment=EMBEDDING_DEPLOYMENT)

azure_client = load_azure_client()
embeddings   = load_embeddings(azure_client)

# ══════════════════════════════════════════════════════════════════════
# LLM HELPERS
#
# Temperature guide:
# 0.0 ─── 0.2 ─── 0.5 ─── 0.7 ─── 1.0+
#  |        |       |       |        |
# Robotic Precise Balance Natural  Chaotic
#
# 0.18 → document Q&A  (precise, factual)
# 0.85 → open chat     (natural, conversational)
# ══════════════════════════════════════════════════════════════════════
def ask_llm(prompt: str, has_context: bool = False,
            images: list = None) -> str:
    system_prompt = (
        "You are a helpful assistant. "
        "When document context is provided, answer based on it and always "
        "mention which document your answer comes from. "
        "When no context is provided, answer normally using your own knowledge."
    )

    if images:
        content = [{"type": "text", "text": prompt}]
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": content},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ]

    response = azure_client.chat.completions.create(
        model=LLM_DEPLOYMENT,
        messages=messages,
        max_tokens=1000,
        temperature=0.4 if has_context else 0.85,
    )
    return response.choices[0].message.content


def build_response(user_question: str) -> str:
    """
    Build context from selected files and return LLM answer.
    In room mode, uses room_selected_files.
    In private mode, uses selected_files.
    """
    # Use room selection if in room mode, else private selection
    if st.session_state["mode"] == "room":
        selected = st.session_state["room_selected_files"]
    else:
        selected = st.session_state["selected_files"]

    pdf_stores  = st.session_state["pdf_stores"]
    image_files = st.session_state["image_files"]

    sel_text = [f for f in selected if f in pdf_stores]
    sel_imgs = [f for f in selected if f in image_files]
    imgs_b64 = [image_files[f] for f in sel_imgs]

    if not sel_text and not sel_imgs:
        return ask_llm(user_question, has_context=False)

    context = ""
    if sel_text:
        all_matches = []
        for fname in sel_text:
            matches = pdf_stores[fname].similarity_search(user_question, k=2)
            for m in matches:
                m.metadata["source"] = fname
            all_matches.extend(matches)
        context = "\n\n".join(
            f"[From: {m.metadata.get('source', 'unknown')}]\n{m.page_content}"
            for m in all_matches
        )

    if context and sel_imgs:
        prompt = (
            f"Use the document excerpts AND the provided images to answer.\n\n"
            f"Document Context:\n{context}\n\n"
            f"Question: {user_question}\n\nAnswer:"
        )
    elif context:
        prompt = (
            f"Use the following document excerpts to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_question}\n\nAnswer:"
        )
    else:
        prompt = (
            f"Use the provided images to answer the question.\n\n"
            f"Question: {user_question}\n\nAnswer:"
        )

    return ask_llm(prompt, has_context=True, images=imgs_b64 or None)


# ══════════════════════════════════════════════════════════════════════
# FILE PROCESSING
# ══════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(f) -> str:
    reader = PdfReader(f)
    return "".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_docx(f) -> str:
    doc = DocxDocument(f)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def image_to_base64(f) -> str:
    img = Image.open(f)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def chunk_and_embed(text: str, filename: str, index_dir: str):
    chunks = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    ).split_text(text)
    if not chunks:
        return None
    store     = FAISS.from_texts(chunks, embeddings)
    save_path = os.path.join(index_dir, filename)
    os.makedirs(save_path, exist_ok=True)
    store.save_local(save_path)
    return store

def load_faiss_index(filename: str, index_dir: str):
    path = os.path.join(index_dir, filename)
    if os.path.exists(path):
        return FAISS.load_local(
            path, embeddings, allow_dangerous_deserialization=True
        )
    return None

def process_uploaded_file(uploaded_file, index_dir: str,
                           mode: str = "private",
                           session_id: str = None,
                           room_code: str = None):
    fname = uploaded_file.name
    ext   = fname.rsplit(".", 1)[-1].lower()

    already_loaded = (
        list(st.session_state["pdf_stores"].keys()) +
        list(st.session_state["image_files"].keys())
    )
    if fname in already_loaded:
        st.info(f"'{fname}' is already loaded!")
        return

    if ext == "pdf":
        text  = extract_text_from_pdf(uploaded_file)
        store = chunk_and_embed(text, fname, index_dir)
        if store:
            st.session_state["pdf_stores"][fname] = store
            if mode == "private" and session_id:
                _save_file_meta(session_id, fname, "pdf")
            elif mode == "room" and room_code:
                _add_file_to_room_db(room_code, fname, "pdf")

    elif ext == "docx":
        text  = extract_text_from_docx(uploaded_file)
        store = chunk_and_embed(text, fname, index_dir)
        if store:
            st.session_state["pdf_stores"][fname] = store
            if mode == "private" and session_id:
                _save_file_meta(session_id, fname, "docx")
            elif mode == "room" and room_code:
                _add_file_to_room_db(room_code, fname, "docx")

    elif ext in ["jpg", "jpeg", "png", "webp"]:
        img_b64 = image_to_base64(uploaded_file)
        st.session_state["image_files"][fname] = img_b64
        img_dir = (
            os.path.join(SESSIONS_DIR, session_id, "images")
            if mode == "private"
            else os.path.join(ROOMS_DIR, room_code, "images")
        )
        os.makedirs(img_dir, exist_ok=True)
        with open(os.path.join(img_dir, fname), "wb") as f:
            f.write(base64.b64decode(img_b64))
        if mode == "private" and session_id:
            _save_file_meta(session_id, fname, "image")
        elif mode == "room" and room_code:
            _add_file_to_room_db(room_code, fname, "image")

    # Update both selection lists
    all_files = (
        list(st.session_state["pdf_stores"].keys()) +
        list(st.session_state["image_files"].keys())
    )
    st.session_state["selected_files"]      = all_files
    st.session_state["room_selected_files"] = all_files


# ══════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════

def get_all_sessions() -> list:
    sessions = []
    for sid in os.listdir(SESSIONS_DIR):
        mp = os.path.join(SESSIONS_DIR, sid, "metadata.json")
        if os.path.exists(mp):
            with open(mp) as f:
                sessions.append(json.load(f))
    return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)

def create_new_session(name: str = None) -> str:
    sid         = str(uuid.uuid4())[:8]
    session_dir = os.path.join(SESSIONS_DIR, sid)
    os.makedirs(os.path.join(session_dir, "indexes"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "images"),  exist_ok=True)
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta = {
        "id":         sid,
        "name":       name or f"New Chat · {now}",
        "created_at": now,
        "updated_at": now,
        "files":      [],
    }
    with open(os.path.join(session_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(session_dir, "chat_history.json"), "w") as f:
        json.dump([], f)
    return sid

def load_session(sid: str):
    session_dir = os.path.join(SESSIONS_DIR, sid)
    hist_path   = os.path.join(session_dir, "chat_history.json")
    st.session_state["chat_history"] = (
        json.load(open(hist_path)) if os.path.exists(hist_path) else []
    )
    st.session_state["pdf_stores"]  = {}
    st.session_state["image_files"] = {}
    meta_path = os.path.join(session_dir, "metadata.json")
    if os.path.exists(meta_path):
        meta      = json.load(open(meta_path))
        index_dir = os.path.join(session_dir, "indexes")
        for fi in meta.get("files", []):
            fname, ftype = fi["name"], fi["type"]
            if ftype in ["pdf", "docx"]:
                store = load_faiss_index(fname, index_dir)
                if store:
                    st.session_state["pdf_stores"][fname] = store
            elif ftype == "image":
                img_path = os.path.join(session_dir, "images", fname)
                if os.path.exists(img_path):
                    with open(img_path, "rb") as f:
                        st.session_state["image_files"][fname] = (
                            base64.b64encode(f.read()).decode()
                        )
    st.session_state["selected_files"] = (
        list(st.session_state["pdf_stores"].keys()) +
        list(st.session_state["image_files"].keys())
    )
    st.session_state["current_session_id"] = sid

def save_chat_history(sid: str):
    session_dir = os.path.join(SESSIONS_DIR, sid)
    with open(os.path.join(session_dir, "chat_history.json"), "w") as f:
        json.dump(st.session_state["chat_history"], f, indent=2)
    meta_path = os.path.join(session_dir, "metadata.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        meta["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        if meta["name"].startswith("New Chat") and st.session_state["chat_history"]:
            first        = st.session_state["chat_history"][0]["content"]
            meta["name"] = first[:45] + ("…" if len(first) > 45 else "")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

def _save_file_meta(sid: str, filename: str, filetype: str):
    meta_path = os.path.join(SESSIONS_DIR, sid, "metadata.json")
    meta      = json.load(open(meta_path))
    if not any(fi["name"] == filename for fi in meta["files"]):
        meta["files"].append({"name": filename, "type": filetype})
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

def rename_session(sid: str, new_name: str):
    meta_path    = os.path.join(SESSIONS_DIR, sid, "metadata.json")
    meta         = json.load(open(meta_path))
    meta["name"] = new_name
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

def delete_session(sid: str):
    path = os.path.join(SESSIONS_DIR, sid)
    if os.path.exists(path):
        shutil.rmtree(path)


# ══════════════════════════════════════════════════════════════════════
# ROOM MANAGEMENT
# ══════════════════════════════════════════════════════════════════════

def _generate_room_code() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

def _safe_firebase_key(filename: str) -> str:
    """Firebase keys cannot contain dots, slashes or spaces."""
    return filename.replace(".", "_").replace("/", "_").replace(" ", "_")

def create_room(room_name: str) -> str:
    code     = _generate_room_code()
    room_dir = os.path.join(ROOMS_DIR, code)
    os.makedirs(os.path.join(room_dir, "indexes"), exist_ok=True)
    os.makedirs(os.path.join(room_dir, "images"),  exist_ok=True)
    db.child("rooms").child(code).set({
        "name":       room_name,
        "created_by": st.session_state["username"],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "active":     True,
        "files":      {},
    })
    return code

def room_exists(code: str) -> bool:
    try:
        val = db.child("rooms").child(code).get().val()
        return val is not None and val.get("active", False)
    except:
        return False

def get_room_messages(code: str) -> list:
    try:
        msgs = db.child("rooms").child(code).child("messages").get()
        if msgs.val():
            return sorted(
                msgs.val().items(),
                key=lambda x: x[1].get("timestamp", "")
            )
    except Exception as e:
        st.warning(f"{e}")

def send_room_message(code: str, user: str, content: str,
                      is_bot: bool = False, is_system: bool = False):
    """
    Send a message to the room.
    is_bot    = True for LLM responses
    is_system = True for notifications (file selection changes etc.)
    """
    msg_id = str(uuid.uuid4())[:8]
    db.child("rooms").child(code).child("messages").child(msg_id).set({
        "user":      user,
        "content":   content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_bot":    is_bot,
        "is_system": is_system,
    })

def get_room_files(code: str) -> list:
    """Return real filenames from the room's Firebase file registry."""
    try:
        files = db.child("rooms").child(code).child("files").get()
        if files.val():
            return [v.get("filename", k) for k, v in files.val().items()]
        return []
    except:
        return []

def _add_file_to_room_db(code: str, filename: str, filetype: str):
    safe_key = _safe_firebase_key(filename)
    db.child("rooms").child(code).child("files").child(safe_key).set({
        "type":        filetype,
        "filename":    filename,
        "uploaded_by": st.session_state["username"],
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })

def get_all_rooms() -> dict:
    try:
        rooms = db.child("rooms").get()
        if rooms.val():
            return {
                c: d for c, d in rooms.val().items()
                if d.get("active")
            }
        return {}
    except:
        return {}

def delete_room(code: str):
    db.child("rooms").child(code).update({"active": False})
    room_dir = os.path.join(ROOMS_DIR, code)
    if os.path.exists(room_dir):
        shutil.rmtree(room_dir)

def load_room_files_into_state(code: str):
    """
    Called on every auto-refresh so files uploaded by the authorized
    user become immediately available to all room members.
    Also ensures room_selected_files defaults to all available files.
    """
    room_files = get_room_files(code)
    index_dir  = os.path.join(ROOMS_DIR, code, "indexes")
    img_dir    = os.path.join(ROOMS_DIR, code, "images")

    newly_loaded = []
    for fname in room_files:
        if (fname in st.session_state["pdf_stores"] or
                fname in st.session_state["image_files"]):
            continue
        ext = fname.rsplit(".", 1)[-1].lower()
        if ext in ["pdf", "docx"]:
            store = load_faiss_index(fname, index_dir)
            if store:
                st.session_state["pdf_stores"][fname] = store
                newly_loaded.append(fname)
        elif ext in ["jpg", "jpeg", "png", "webp"]:
            img_path = os.path.join(img_dir, fname)
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    st.session_state["image_files"][fname] = (
                        base64.b64encode(f.read()).decode()
                    )
                newly_loaded.append(fname)

    all_files = (
        list(st.session_state["pdf_stores"].keys()) +
        list(st.session_state["image_files"].keys())
    )
    st.session_state["selected_files"] = all_files

    # Auto-select all files for room querying by default
    if not st.session_state["room_selected_files"] and all_files:
        st.session_state["room_selected_files"] = all_files
    else:
        # Add any newly loaded files to selection automatically
        for f in newly_loaded:
            if f not in st.session_state["room_selected_files"]:
                st.session_state["room_selected_files"].append(f)


# ══════════════════════════════════════════════════════════════════════
# ██████████████████████████  S I D E B A R  ██████████████████████████
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🔒 AM Intelligence")
    st.divider()

    # ── UNAUTHENTICATED ──────────────────────────────────────────────
    if not st.session_state["authenticated"]:

        if not st.session_state.get("current_room_code"):
            st.caption("💡 Join a group room with a code.")
            guest_code = st.text_input("Room Code", max_chars=6).upper().strip()
            guest_name = st.text_input("Your display name")
            if st.button("Join Room as Guest", use_container_width=True):
                if not guest_name:
                    st.error("Please enter a display name.")
                elif not room_exists(guest_code):
                    st.error("Room not found.")
                else:
                    st.session_state["mode"]               = "room"
                    st.session_state["current_room_code"]  = guest_code
                    st.session_state["username"]           = guest_name
                    st.session_state["pdf_stores"]         = {}
                    st.session_state["image_files"]        = {}
                    st.session_state["selected_files"]     = []
                    st.session_state["room_selected_files"]= []
                    st.rerun()
        else:
            st.info(f"👤 **{st.session_state['username']}** (Guest)")
            st.caption(f"Room: **{st.session_state['current_room_code']}**")
            if st.button("Leave Room", use_container_width=True):
                st.session_state["mode"]               = "private"
                st.session_state["current_room_code"]  = None
                st.session_state["username"]           = ""
                st.session_state["pdf_stores"]         = {}
                st.session_state["image_files"]        = {}
                st.session_state["selected_files"]     = []
                st.session_state["room_selected_files"]= []
                st.rerun()

        st.divider()
        with st.expander("🔐 Login to upload files"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login", use_container_width=True):
                if (username in AUTHORIZED_USERS and
                        AUTHORIZED_USERS[username] == password):
                    st.session_state["authenticated"] = True
                    st.session_state["username"]      = username
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

    # ── AUTHENTICATED ────────────────────────────────────────────────
    else:
        st.success(f"👤 **{st.session_state['username']}**")

        mode_choice = st.radio(
            "App mode",
            ["💬 Private Chat", "👥 Group Room"],
            label_visibility="collapsed",
        )
        st.session_state["mode"] = (
            "private" if mode_choice == "💬 Private Chat" else "room"
        )

        if st.button("Logout", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

        st.divider()

        # ── PRIVATE CHAT SIDEBAR ─────────────────────────────────────
        if st.session_state["mode"] == "private":
            st.subheader("💬 Sessions")

            if st.button("➕ New Chat", use_container_width=True):
                sid = create_new_session()
                load_session(sid)
                st.rerun()

            for s in get_all_sessions():
                is_active = s["id"] == st.session_state.get("current_session_id")
                col1, col2 = st.columns([5, 1])
                with col1:
                    label = ("▶ " if is_active else "") + s["name"][:32]
                    if st.button(label, key=f"s_{s['id']}", use_container_width=True):
                        load_session(s["id"])
                        st.rerun()
                with col2:
                    if st.button("🗑", key=f"d_{s['id']}"):
                        delete_session(s["id"])
                        if s["id"] == st.session_state.get("current_session_id"):
                            st.session_state["current_session_id"] = None
                            st.session_state["chat_history"]       = []
                            st.session_state["pdf_stores"]         = {}
                            st.session_state["image_files"]        = {}
                        st.rerun()

            if st.session_state["current_session_id"]:
                st.divider()

                with st.expander("✏️ Rename session"):
                    new_name = st.text_input("New name", key="rename_inp")
                    if st.button("Save") and new_name:
                        rename_session(
                            st.session_state["current_session_id"], new_name
                        )
                        st.rerun()

                st.subheader("📁 Upload Files")
                uploaded = st.file_uploader(
                    "PDFs, DOCs or Images",
                    type=["pdf", "docx", "jpg", "jpeg", "png", "webp"],
                    key="priv_uploader",
                )
                if uploaded:
                    sid       = st.session_state["current_session_id"]
                    index_dir = os.path.join(SESSIONS_DIR, sid, "indexes")
                    with st.spinner(f"Processing {uploaded.name}…"):
                        process_uploaded_file(
                            uploaded, index_dir,
                            mode="private", session_id=sid,
                        )
                    st.success(f"✅ {uploaded.name} ready!")

                all_files = (
                    list(st.session_state["pdf_stores"].keys()) +
                    list(st.session_state["image_files"].keys())
                )
                if all_files:
                    st.subheader("🔍 Search in")
                    srch = st.radio(
                        "Search scope",
                        ["All files", "Select files"],
                        label_visibility="collapsed",
                    )
                    if srch == "All files":
                        st.session_state["selected_files"] = all_files
                    else:
                        st.session_state["selected_files"] = st.multiselect(
                            "Choose files",
                            options=all_files,
                            default=st.session_state["selected_files"] or all_files,
                        )

        # ── GROUP ROOM SIDEBAR ───────────────────────────────────────
        else:
            st.subheader("👥 Group Rooms")

            with st.expander("➕ Create New Room"):
                r_name = st.text_input("Room name", key="cr_name")
                if st.button("Create", use_container_width=True) and r_name:
                    code = create_room(r_name)
                    st.session_state["current_room_code"]  = code
                    st.session_state["pdf_stores"]         = {}
                    st.session_state["image_files"]        = {}
                    st.session_state["selected_files"]     = []
                    st.session_state["room_selected_files"]= []
                    st.success(f"✅ Room created!  Code: **{code}**")
                    st.rerun()

            with st.expander("🚪 Join a Room"):
                j_code = (
                    st.text_input("Room code", max_chars=6, key="j_code")
                    .upper().strip()
                )
                if st.button("Join", use_container_width=True, key="j_btn"):
                    if room_exists(j_code):
                        st.session_state["current_room_code"]  = j_code
                        st.session_state["pdf_stores"]         = {}
                        st.session_state["image_files"]        = {}
                        st.session_state["selected_files"]     = []
                        st.session_state["room_selected_files"]= []
                        st.rerun()
                    else:
                        st.error("Room not found.")

            all_rooms = get_all_rooms()
            my_rooms  = {
                c: d for c, d in all_rooms.items()
                if d.get("created_by") == st.session_state["username"]
            }
            if my_rooms:
                st.divider()
                st.caption("Your rooms:")
                for code, data in my_rooms.items():
                    is_active = code == st.session_state.get("current_room_code")
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        lbl = ("▶ " if is_active else "") + f"{data['name']} ({code})"
                        if st.button(lbl, key=f"r_{code}", use_container_width=True):
                            st.session_state["current_room_code"]  = code
                            st.session_state["pdf_stores"]         = {}
                            st.session_state["image_files"]        = {}
                            st.session_state["selected_files"]     = []
                            st.session_state["room_selected_files"]= []
                            st.rerun()
                    with col2:
                        if st.button("🗑", key=f"dr_{code}"):
                            delete_room(code)
                            if st.session_state.get("current_room_code") == code:
                                st.session_state["current_room_code"] = None
                            st.rerun()

            # ── File upload + file selector for active room ──────────
            if st.session_state.get("current_room_code"):
                st.divider()
                st.subheader("📁 Upload to Room")
                room_file = st.file_uploader(
                    "PDF, DOCX, or Image",
                    type=["pdf", "docx", "jpg", "jpeg", "png", "webp"],
                    key="room_uploader",
                )
                if room_file:
                    code      = st.session_state["current_room_code"]
                    index_dir = os.path.join(ROOMS_DIR, code, "indexes")
                    with st.spinner(f"Processing {room_file.name}…"):
                        process_uploaded_file(
                            room_file, index_dir,
                            mode="room", room_code=code,
                        )
                    st.success(f"✅ {room_file.name} added to room!")

                # ── File selector (auth user only, visible in room) ──
                all_room_files = (
                    list(st.session_state["pdf_stores"].keys()) +
                    list(st.session_state["image_files"].keys())
                )
                if all_room_files:
                    st.subheader("🔍 Query scope")
                    room_srch = st.radio(
                        "Room search scope",
                        ["All files", "Select files"],
                        label_visibility="collapsed",
                        key="room_srch_radio",
                    )

                    prev_selection = list(st.session_state["room_selected_files"])

                    if room_srch == "All files":
                        st.session_state["room_selected_files"] = all_room_files
                    else:
                        st.session_state["room_selected_files"] = st.multiselect(
                            "Choose files",
                            options=all_room_files,
                            default=st.session_state["room_selected_files"] or all_room_files,
                            key="room_file_multiselect",
                        )

                    # Send notification if selection changed
                    new_selection = st.session_state["room_selected_files"]
                    if set(new_selection) != set(prev_selection) and new_selection:
                        code = st.session_state["current_room_code"]
                        notif = (
                            f"📌 **{st.session_state['username']}** changed the active "
                            f"documents to: **{', '.join(new_selection)}**"
                        )
                        send_room_message(code, "System", notif, is_system=True)


# ══════════════════════════════════════════════════════════════════════
# ████████████████████████  M A I N   A R E A  ████████████████████████
# ══════════════════════════════════════════════════════════════════════

mode = st.session_state["mode"]

# ── PRIVATE CHAT ─────────────────────────────────────────────────────
if mode == "private":

    st.header("🔒 AM Intelligence")

    if not st.session_state["authenticated"]:
        st.caption("💡 Log in from the sidebar to upload files and save sessions.")
        st.divider()

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_q = st.chat_input("Ask anything…")
        if user_q:
            with st.chat_message("user"):
                st.write(user_q)
            with st.spinner("Thinking…"):
                response = ask_llm(user_q, has_context=False)
            with st.chat_message("assistant"):
                st.write(response)
            st.session_state["chat_history"].append({"role": "user",      "content": user_q})
            st.session_state["chat_history"].append({"role": "assistant", "content": response})

    elif not st.session_state["current_session_id"]:
        st.info("Click **➕ New Chat** in the sidebar to begin a saved session.")
        st.divider()

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_q = st.chat_input("Ask anything…")
        if user_q:
            with st.chat_message("user"):
                st.write(user_q)
            with st.spinner("Thinking…"):
                response = ask_llm(user_q, has_context=False)
            with st.chat_message("assistant"):
                st.write(response)
            st.session_state["chat_history"].append({"role": "user",      "content": user_q})
            st.session_state["chat_history"].append({"role": "assistant", "content": response})

    else:
        sid       = st.session_state["current_session_id"]
        meta_path = os.path.join(SESSIONS_DIR, sid, "metadata.json")
        meta      = json.load(open(meta_path))

        st.header(f"💬 {meta['name']}")
        st.caption(f"🕐 Created: {meta['created_at']}  ·  Last updated: {meta['updated_at']}")

        if st.session_state["selected_files"]:
            st.caption(f"📂 Searching in: {', '.join(st.session_state['selected_files'])}")

        st.divider()

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_q = st.chat_input("Ask anything…")
        if user_q:
            with st.chat_message("user"):
                st.write(user_q)
            with st.spinner("Thinking…"):
                response = build_response(user_q)
            with st.chat_message("assistant"):
                st.write(response)
            st.session_state["chat_history"].append({"role": "user",      "content": user_q})
            st.session_state["chat_history"].append({"role": "assistant", "content": response})
            save_chat_history(sid)

# ── GROUP ROOM ────────────────────────────────────────────────────────
elif mode == "room":
    code = st.session_state.get("current_room_code")

    if not code:
        st.title("👥 Group Rooms")
        if st.session_state["authenticated"]:
            st.info("Create or join a room from the sidebar.")
        else:
            st.info("Enter a room code in the sidebar to join.")

    else:
        # Pause auto-refresh while LLM is thinking so call isn't killed
        if not st.session_state.get("room_thinking", False):
            st_autorefresh(interval=15000, key="room_refresh")

        # Validate room
        try:
            room_data = db.child("rooms").child(code).get().val()
            if not room_data or not room_data.get("active"):
                st.error("This room no longer exists.")
                st.session_state["current_room_code"] = None
                st.stop()
        except Exception as e:
            st.error(f"Could not connect to room: {e}")
            st.stop()

        # Sync new files from disk into session_state
        load_room_files_into_state(code)

        # Header
        st.header(f"👥 {room_data['name']}")
        st.caption(
            f"Room Code: **{code}**  ·  "
            f"Created by: {room_data['created_by']}  ·  "
            f"{room_data['created_at']}"
        )

        room_files = get_room_files(code)
        if room_files:
            st.caption(f"📂 Files in room: {', '.join(room_files)}")

        # Show currently active query scope
        active = st.session_state.get("room_selected_files", [])
        if active and set(active) != set(room_files):
            st.caption(f"🔍 Currently querying: {', '.join(active)}")

        st.divider()

        # Render all messages from Firebase
        for _, msg in get_room_messages(code):
            if msg.get("is_system"):
                # System notifications shown as info banners
                st.info(msg.get("content", ""))
            else:
                role = "assistant" if msg.get("is_bot") else "user"
                with st.chat_message(role):
                    if not msg.get("is_bot"):
                        st.caption(
                            f"**{msg.get('user', 'unknown')}**  ·  "
                            f"{msg.get('timestamp', '')}"
                        )
                    st.write(msg.get("content", ""))

        # Chat input — available to ALL users in the room
        user_q = st.chat_input("Ask the room…")
        if user_q:
            # Pause auto-refresh so LLM call completes uninterrupted
            st.session_state["room_thinking"] = True

            send_room_message(
                code, st.session_state["username"], user_q, is_bot=False
            )

            with st.spinner("Thinking…"):
                try:
                    response = build_response(user_q)
                except Exception as e:
                    response = f"⚠️ Error getting response: {str(e)}"

            send_room_message(code, "AM Intelligence", response, is_bot=True)

            st.session_state["room_thinking"] = False
            st.rerun()
