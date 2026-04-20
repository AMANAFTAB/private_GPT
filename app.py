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
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from openai import AzureOpenAI, BadRequestError
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
# All models share the same endpoint/key since they are deployed under the same Azure AI Foundry resource.
# ══════════════════════════════════════════════════════════════════════
AZURE_ENDPOINT       = st.secrets["MODELS_ENDPOINT"]
AZURE_API_KEY        = st.secrets["LLM_API_KEY"]
AZURE_API_VERSION    = "2025-04-01-preview"
LLM_DEPLOYMENT       = "Llama-4-Maverick-17B-128E-Instruct-FP8"
EMBEDDING_DEPLOYMENT = "embed-v-4-0"
EMBEDDING_BATCH_SIZE = 96
MAX_UPLOAD_MB        = 20
MAX_UPLOAD_BYTES     = MAX_UPLOAD_MB * 1024 * 1024
PAGE_QUERY_RE        = re.compile(
    r"\bpage(?:\s+number|\s+no\.?)?\s+(\d+)\b", re.IGNORECASE
)
SECTION_TITLE_RE     = re.compile(
    r'\bsection\s+[\"“”\']([^\"“”\']+)[\"“”\']', re.IGNORECASE
)
SECTION_FOLLOWUP_RE  = re.compile(r"\b(that|this|the)\s+section\b", re.IGNORECASE)

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
    "private_upload_sig": None,
    "room_upload_sig":    None,
    "room_unavailable_files": [],
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ══════════════════════════════════════════════════════════════════════
# FIREBASE INIT --> uses the [firebase] config parameters mentioned in the .secrets/secrets.toml file
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource
def init_firebase():
    return pyrebase.initialize_app(FIREBASE_CONFIG)

firebase_app = init_firebase()
db = firebase_app.database()

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
        if not texts:
            return []

        all_embeddings = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start:start + EMBEDDING_BATCH_SIZE]
            response = self.client.embeddings.create(
                input=batch, model=self.deployment
            )
            all_embeddings.extend(item.embedding for item in response.data)
        return all_embeddings

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
            images: list = None, history: list = None) -> str:
    """
    FIX – Azure AI Foundry gateway does not support the 'system' role.
    Conversation rules are injected into the first user turn instead.
    The optional `history` list ({"role", "content"} dicts) gives the LLM
    conversational memory across turns.
    """
    conversation_rules = (
        "Conversation rules:\n"
        "- Be a helpful assistant.\n"
        "- If document context is provided, answer from it and mention the "
        "document it came from.\n"
        "- If no document context is provided, answer normally using your "
        "general knowledge."
    )

    messages = []
    injected = False

    # Replay history, injecting the conversation rules into the first user turn.
    for turn in (history or []):
        if turn["role"] == "user" and not injected:
            messages.append({
                "role":    "user",
                "content": (
                    f"{conversation_rules}\n\n"
                    f"User message:\n{turn['content']}"
                ),
            })
            injected = True
        else:
            messages.append({"role": turn["role"], "content": turn["content"]})

    # Current user turn
    current_text = (
        prompt if injected else f"{conversation_rules}\n\nUser message:\n{prompt}"
    )
    if images:
        content = [{"type": "text", "text": current_text}]
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": current_text})

    try:
        response = azure_client.chat.completions.create(
            model=LLM_DEPLOYMENT,
            messages=messages,
            max_tokens=1000,
            temperature=0.4 if has_context else 0.85,
        )
    except BadRequestError as exc:
        err = getattr(exc, "body", {}) or {}
        if err.get("error", {}).get("code") != "content_filter":
            raise

        # False positives can happen when provider filters react to the
        # injected conversation rules or full replayed history. Retry once
        # with only the current turn.
        retry_messages = []
        if images:
            retry_content = [{"type": "text", "text": prompt}]
            for img_b64 in images:
                retry_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                })
            retry_messages.append({"role": "user", "content": retry_content})
        else:
            retry_messages.append({"role": "user", "content": prompt})

        try:
            response = azure_client.chat.completions.create(
                model=LLM_DEPLOYMENT,
                messages=retry_messages,
                max_tokens=1000,
                temperature=0.4 if has_context else 0.85,
            )
        except BadRequestError as retry_exc:
            retry_err = getattr(retry_exc, "body", {}) or {}
            if retry_err.get("error", {}).get("code") == "content_filter":
                return (
                    "The model provider blocked this prompt under its content "
                    "filter. This can be a false positive. Please try "
                    "rephrasing the question slightly and try again."
                )
            raise
    return response.choices[0].message.content


def _format_context(matches: list[Document]) -> str:
    blocks = []
    for match in matches:
        source = match.metadata.get("source", "unknown")
        page_number = match.metadata.get("page_number")
        heading = f"[From: {source}"
        if page_number is not None:
            heading += f", page {page_number}"
        heading += "]"
        blocks.append(f"{heading}\n{match.page_content}")
    return "\n\n".join(blocks)


def _extract_requested_page(user_question: str):
    match = PAGE_QUERY_RE.search(user_question)
    return int(match.group(1)) if match else None


def _normalize_lookup_text(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _extract_explicit_section_title(text: str):
    match = SECTION_TITLE_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


def _resolve_section_title(user_question: str, history: list = None):
    explicit = _extract_explicit_section_title(user_question)
    if explicit:
        return explicit

    if not SECTION_FOLLOWUP_RE.search(user_question):
        return None

    for turn in reversed(history or []):
        if turn.get("role") != "user":
            continue
        prior_explicit = _extract_explicit_section_title(turn.get("content", ""))
        if prior_explicit:
            return prior_explicit
    return None


def _store_has_page_metadata(store) -> bool:
    doc_map = getattr(getattr(store, "docstore", None), "_dict", {})
    return any("page_number" in doc.metadata for doc in doc_map.values())


def _get_sorted_store_docs(store, filename: str) -> list[Document]:
    doc_map = getattr(getattr(store, "docstore", None), "_dict", {})
    docs = []
    for doc in doc_map.values():
        if "source" not in doc.metadata:
            doc.metadata["source"] = filename
        docs.append(doc)
    return sorted(
        docs,
        key=lambda doc: (
            doc.metadata.get("page_number", 10**9),
            doc.metadata.get("chunk_index", 10**9),
        )
    )


def _get_page_chunks(store, filename: str, page_number: int) -> list[Document]:
    matches = []
    for doc in _get_sorted_store_docs(store, filename):
        if doc.metadata.get("page_number") != page_number:
            continue
        matches.append(doc)
    return sorted(matches, key=lambda doc: doc.metadata.get("chunk_index", 0))


def _find_section_context(store, filename: str, section_title: str) -> list[Document]:
    docs = _get_sorted_store_docs(store, filename)
    needle = _normalize_lookup_text(section_title)
    anchor_index = None
    for index, doc in enumerate(docs):
        if needle in _normalize_lookup_text(doc.page_content):
            anchor_index = index
            break

    if anchor_index is None:
        return []

    anchor = docs[anchor_index]
    anchor_page = anchor.metadata.get("page_number")
    anchor_chunk = anchor.metadata.get("chunk_index", 0)

    if anchor_page is None:
        return docs[anchor_index:anchor_index + 6]

    section_docs = []
    for doc in docs:
        page_number = doc.metadata.get("page_number")
        if page_number is None or page_number < anchor_page or page_number > anchor_page + 2:
            continue
        if page_number == anchor_page and doc.metadata.get("chunk_index", 0) < anchor_chunk:
            continue
        section_docs.append(doc)
    return section_docs


def build_response(user_question: str, history: list = None) -> str:
    """
    Build context from selected files and return LLM answer.
    In room mode, uses room_selected_files.
    In private mode, uses selected_files.
    FIX: passes history so the LLM has conversational memory.
    """
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
        return ask_llm(user_question, has_context=False, history=history)

    requested_page = _extract_requested_page(user_question)
    section_title = _resolve_section_title(user_question, history=history)
    context = ""
    direct_page_lookup = False
    direct_section_lookup = False

    if sel_text:
        if requested_page is not None:
            page_matches = []
            page_aware_files = []
            for fname in sel_text:
                store = pdf_stores[fname]
                if not _store_has_page_metadata(store):
                    continue
                page_aware_files.append(fname)
                page_matches.extend(_get_page_chunks(store, fname, requested_page))

            if page_matches:
                context = _format_context(page_matches)
                direct_page_lookup = True
            elif page_aware_files and not sel_imgs:
                filenames = ", ".join(page_aware_files)
                return (
                    f"I could not find indexed text for page {requested_page} in "
                    f"the selected document(s): {filenames}. Please check the "
                    f"page number and try again."
                )
            elif not page_aware_files and not sel_imgs:
                return (
                    "Page-specific lookup is not available for the selected "
                    "document(s) yet. Please re-upload the PDF so it can be "
                    "indexed with page numbers."
                )

        if section_title and not context:
            section_matches = []
            for fname in sel_text:
                section_matches.extend(
                    _find_section_context(pdf_stores[fname], fname, section_title)
                )
            if section_matches:
                context = _format_context(section_matches)
                direct_section_lookup = True

        if not context:
            all_matches = []
            retrieval_query = user_question
            if section_title:
                retrieval_query = f"{section_title}\n{user_question}"
            for fname in sel_text:
                matches = pdf_stores[fname].similarity_search(
                    retrieval_query,
                    k=4 if section_title else 2
                )
                for match in matches:
                    match.metadata["source"] = fname
                all_matches.extend(matches)
            context = _format_context(all_matches)

    if context and sel_imgs:
        if direct_page_lookup:
            prompt = (
                f"Use the exact excerpts from page {requested_page} AND the "
                f"provided images to answer.\n\n"
                f"Document Context:\n{context}\n\n"
                f"Question: {user_question}\n\nAnswer:"
            )
        elif direct_section_lookup:
            prompt = (
                f"Use the exact excerpts from the section titled "
                f"\"{section_title}\" AND the provided images to answer.\n\n"
                f"Document Context:\n{context}\n\n"
                f"Question: {user_question}\n\nAnswer:"
            )
        else:
            prompt = (
                f"Use the document excerpts AND the provided images to answer.\n\n"
                f"Document Context:\n{context}\n\n"
                f"Question: {user_question}\n\nAnswer:"
            )
    elif context:
        if direct_page_lookup:
            prompt = (
                f"Use the following exact excerpts from page {requested_page} to "
                f"answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {user_question}\n\nAnswer:"
            )
        elif direct_section_lookup:
            prompt = (
                f"Use the following exact excerpts from the section titled "
                f"\"{section_title}\" to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {user_question}\n\nAnswer:"
            )
        else:
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

    return ask_llm(prompt, has_context=True, images=imgs_b64 or None, history=history)


# ══════════════════════════════════════════════════════════════════════
# FILE PROCESSING
# ══════════════════════════════════════════════════════════════════════

def extract_pages_from_pdf(f) -> list[Document]:
    f.seek(0)
    reader = PdfReader(f)
    page_docs = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            page_docs.append(Document(
                page_content=text,
                metadata={"page_number": page_number, "filetype": "pdf"},
            ))
    return page_docs

def extract_text_from_docx(f) -> str:
    f.seek(0)
    doc = DocxDocument(f)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def image_to_base64(f) -> str:
    f.seek(0)
    img = Image.open(f)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _chunk_documents(documents: list[Document], filename: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = []
    for document in documents:
        doc_chunks = splitter.split_text(document.page_content)
        for chunk_index, chunk in enumerate(doc_chunks):
            metadata = dict(document.metadata)
            metadata["source"] = filename
            metadata["chunk_index"] = chunk_index
            chunks.append(Document(page_content=chunk, metadata=metadata))
    return chunks

def chunk_and_embed(documents: list[Document], filename: str, index_dir: str):
    chunks = _chunk_documents(documents, filename)
    if not chunks:
        return None
    store     = FAISS.from_documents(chunks, embeddings)
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
    file_size = getattr(uploaded_file, "size", 0)

    if file_size and file_size > MAX_UPLOAD_BYTES:
        size_mb = file_size / (1024 * 1024)
        raise ValueError(
            f"File is {size_mb:.1f} MB. Please upload a file under "
            f"{MAX_UPLOAD_MB} MB."
        )

    already_loaded = (
        list(st.session_state["pdf_stores"].keys()) +
        list(st.session_state["image_files"].keys())
    )
    if fname in already_loaded:
        existing_store = st.session_state["pdf_stores"].get(fname)
        if not (ext == "pdf" and existing_store and not _store_has_page_metadata(existing_store)):
            st.info(f"'{fname}' is already loaded!")
            return

    if ext == "pdf":
        documents = extract_pages_from_pdf(uploaded_file)
        if not documents:
            raise ValueError(
                "No readable text could be extracted from this PDF. "
                "If it is a scanned document, OCR would be needed first."
            )
        store = chunk_and_embed(documents, fname, index_dir)
        if store:
            st.session_state["pdf_stores"][fname] = store
            if mode == "private" and session_id:
                _save_file_meta(session_id, fname, "pdf")
            elif mode == "room" and room_code:
                _add_file_to_room_db(room_code, fname, "pdf")
                if not _sync_room_file_to_cloud(room_code, fname, "pdf"):
                    st.warning(
                        f"{fname} was indexed locally, but shared room sync to "
                        f"Firebase Storage did not complete."
                    )

    elif ext == "docx":
        text = extract_text_from_docx(uploaded_file)
        if not text.strip():
            raise ValueError("No readable text could be extracted from this DOCX.")
        documents = [Document(
            page_content=text,
            metadata={"filetype": "docx"},
        )]
        store = chunk_and_embed(documents, fname, index_dir)
        if store:
            st.session_state["pdf_stores"][fname] = store
            if mode == "private" and session_id:
                _save_file_meta(session_id, fname, "docx")
            elif mode == "room" and room_code:
                _add_file_to_room_db(room_code, fname, "docx")
                if not _sync_room_file_to_cloud(room_code, fname, "docx"):
                    st.warning(
                        f"{fname} was indexed locally, but shared room sync to "
                        f"Firebase Storage did not complete."
                    )

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
            if not _sync_room_file_to_cloud(room_code, fname, "image"):
                st.warning(
                    f"{fname} was saved locally, but shared room sync to "
                    f"Firebase Storage did not complete."
                )

    # FIX: Only touch room_selected_files when actually in room mode,
    # so a private upload never clobbers the room's file selection.
    all_files = (
        list(st.session_state["pdf_stores"].keys()) +
        list(st.session_state["image_files"].keys())
    )
    st.session_state["selected_files"] = all_files
    if mode == "room":
        st.session_state["room_selected_files"] = all_files


def make_upload_signature(uploaded_file, scope_id: str) -> str:
    return f"{scope_id}:{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"


def _room_file_ref(code: str, filename: str):
    safe_key = _safe_firebase_key(filename)
    return db.child("rooms").child(code).child("files").child(safe_key)


def _room_index_dir(code: str, filename: str) -> str:
    return os.path.join(ROOMS_DIR, code, "indexes", filename)


def _room_image_path(code: str, filename: str) -> str:
    return os.path.join(ROOMS_DIR, code, "images", filename)


def _room_index_storage_base(code: str, filename: str) -> str:
    return f"room_assets/{code}/indexes/{_safe_firebase_key(filename)}"


def _room_image_storage_path(code: str, filename: str) -> str:
    return f"room_assets/{code}/images/{_safe_firebase_key(filename)}"


def _update_room_file_db(code: str, filename: str, updates: dict):
    _room_file_ref(code, filename).update(updates)


def _firebase_storage():
    return firebase_app.storage()


def _upload_to_storage(local_path: str, storage_path: str):
    _firebase_storage().child(storage_path).put(local_path)


def _download_from_storage(storage_path: str, local_path: str) -> bool:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    _firebase_storage().child(storage_path).download(storage_path, local_path)
    return os.path.exists(local_path)


def _room_index_files_exist(code: str, filename: str) -> bool:
    index_dir = _room_index_dir(code, filename)
    return (
        os.path.exists(os.path.join(index_dir, "index.faiss")) and
        os.path.exists(os.path.join(index_dir, "index.pkl"))
    )


def _sync_room_file_to_cloud(code: str, filename: str, filetype: str) -> bool:
    try:
        if filetype in ["pdf", "docx"]:
            if not _room_index_files_exist(code, filename):
                return False
            storage_base = _room_index_storage_base(code, filename)
            index_dir = _room_index_dir(code, filename)
            _upload_to_storage(
                os.path.join(index_dir, "index.faiss"),
                f"{storage_base}/index.faiss",
            )
            _upload_to_storage(
                os.path.join(index_dir, "index.pkl"),
                f"{storage_base}/index.pkl",
            )
            _update_room_file_db(code, filename, {
                "cloud_synced": True,
                "storage_kind": "faiss_index",
                "storage_base": storage_base,
            })
            return True

        if filetype == "image":
            image_path = _room_image_path(code, filename)
            if not os.path.exists(image_path):
                return False
            storage_path = _room_image_storage_path(code, filename)
            _upload_to_storage(image_path, storage_path)
            _update_room_file_db(code, filename, {
                "cloud_synced": True,
                "storage_kind": "image",
                "storage_path": storage_path,
            })
            return True
    except Exception:
        return False

    return False


def _download_room_file_from_cloud(code: str, filename: str, meta: dict) -> bool:
    try:
        filetype = meta.get("type", "")
        if filetype in ["pdf", "docx"]:
            storage_base = meta.get("storage_base") or _room_index_storage_base(code, filename)
            index_dir = _room_index_dir(code, filename)
            os.makedirs(index_dir, exist_ok=True)
            faiss_ok = _download_from_storage(
                f"{storage_base}/index.faiss",
                os.path.join(index_dir, "index.faiss"),
            )
            pkl_ok = _download_from_storage(
                f"{storage_base}/index.pkl",
                os.path.join(index_dir, "index.pkl"),
            )
            return faiss_ok and pkl_ok

        if filetype == "image":
            storage_path = meta.get("storage_path") or _room_image_storage_path(code, filename)
            return _download_from_storage(storage_path, _room_image_path(code, filename))
    except Exception:
        return False

    return False


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
    return []

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
    registry = get_room_file_registry(code)
    return list(registry.keys())


def get_room_file_registry(code: str) -> dict:
    try:
        files = db.child("rooms").child(code).child("files").get()
        if files.val():
            return {
                v.get("filename", k): v
                for k, v in files.val().items()
                if isinstance(v, dict)
            }
        return {}
    except:
        return {}

def _room_history_for_llm(code: str) -> list:
    """
    FIX (room memory): Read Firebase room messages, strip system notifications,
    and return a list of {"role", "content"} dicts for the LLM history param.
    """
    try:
        raw = get_room_messages(code)
    except Exception:
        return []
    history = []
    for _, msg in raw:
        if msg.get("is_system"):
            continue
        role = "assistant" if msg.get("is_bot") else "user"
        content = msg.get("content", "")
        if content:
            history.append({"role": role, "content": content})
    return history


def _add_file_to_room_db(code: str, filename: str, filetype: str):
    safe_key = _safe_firebase_key(filename)
    db.child("rooms").child(code).child("files").child(safe_key).set({
        "type":        filetype,
        "filename":    filename,
        "uploaded_by": st.session_state["username"],
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "cloud_synced": False,
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
    room_files = get_room_file_registry(code)

    newly_loaded = []
    unavailable = []
    for fname, meta in room_files.items():
        filetype = meta.get("type", "")

        # Migration path: if this device already has a local copy/index for an
        # older room file, backfill it to Firebase Storage so other devices can
        # download it too.
        if not meta.get("cloud_synced"):
            _sync_room_file_to_cloud(code, fname, filetype)

        if (fname in st.session_state["pdf_stores"] or
                fname in st.session_state["image_files"]):
            continue

        if filetype in ["pdf", "docx"]:
            if not _room_index_files_exist(code, fname):
                if meta.get("cloud_synced") or meta.get("storage_base"):
                    _download_room_file_from_cloud(code, fname, meta)
            store = load_faiss_index(fname, os.path.join(ROOMS_DIR, code, "indexes"))
            if store:
                st.session_state["pdf_stores"][fname] = store
                newly_loaded.append(fname)
            else:
                unavailable.append(fname)
        elif filetype == "image":
            img_path = _room_image_path(code, fname)
            if not os.path.exists(img_path):
                if meta.get("cloud_synced") or meta.get("storage_path"):
                    _download_room_file_from_cloud(code, fname, meta)
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    st.session_state["image_files"][fname] = (
                        base64.b64encode(f.read()).decode()
                    )
                newly_loaded.append(fname)
            else:
                unavailable.append(fname)

    all_files = (
        list(st.session_state["pdf_stores"].keys()) +
        list(st.session_state["image_files"].keys())
    )
    st.session_state["selected_files"] = all_files
    st.session_state["room_unavailable_files"] = unavailable

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
                    st.session_state["room_unavailable_files"] = []
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
                st.session_state["room_unavailable_files"] = []
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
                st.caption(
                    f"Reliable indexing works best with files up to "
                    f"{MAX_UPLOAD_MB} MB."
                )
                uploaded = st.file_uploader(
                    "PDFs, DOCs or Images",
                    type=["pdf", "docx", "jpg", "jpeg", "png", "webp"],
                    key=f"priv_uploader_{st.session_state['current_session_id']}",
                )
                if uploaded:
                    sid       = st.session_state["current_session_id"]
                    index_dir = os.path.join(SESSIONS_DIR, sid, "indexes")
                    current_sig = make_upload_signature(uploaded, sid)
                    if st.session_state["private_upload_sig"] != current_sig:
                        with st.spinner(f"Processing {uploaded.name}…"):
                            try:
                                process_uploaded_file(
                                    uploaded, index_dir,
                                    mode="private", session_id=sid,
                                )
                            except Exception as e:
                                st.error(f"Could not process {uploaded.name}: {e}")
                            else:
                                st.session_state["private_upload_sig"] = current_sig
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
                    st.session_state["room_unavailable_files"] = []
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
                        st.session_state["room_unavailable_files"] = []
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
                            st.session_state["room_unavailable_files"] = []
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
                st.caption(
                    f"Reliable indexing works best with files up to "
                    f"{MAX_UPLOAD_MB} MB."
                )
                room_file = st.file_uploader(
                    "PDF, DOCX, or Image",
                    type=["pdf", "docx", "jpg", "jpeg", "png", "webp"],
                    key=f"room_uploader_{st.session_state['current_room_code']}",
                )
                if room_file:
                    code      = st.session_state["current_room_code"]
                    index_dir = os.path.join(ROOMS_DIR, code, "indexes")
                    current_sig = make_upload_signature(room_file, code)
                    if st.session_state["room_upload_sig"] != current_sig:
                        with st.spinner(f"Processing {room_file.name}…"):
                            try:
                                process_uploaded_file(
                                    room_file, index_dir,
                                    mode="room", room_code=code,
                                )
                            except Exception as e:
                                st.error(f"Could not process {room_file.name}: {e}")
                            else:
                                st.session_state["room_upload_sig"] = current_sig
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
                response = ask_llm(user_q, has_context=False,
                                   history=st.session_state["chat_history"])
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
                response = ask_llm(user_q, has_context=False,
                                   history=st.session_state["chat_history"])
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
                response = build_response(user_q, history=st.session_state["chat_history"])
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
        # ── GHOST-RESPONSE FIX ─────────────────────────────────────────
        # Read the chat input FIRST and immediately set room_thinking=True
        # so that st_autorefresh is never armed in the same render as an
        # active LLM call. Previously the autorefresh timer could fire
        # mid-call, silently killing it and producing no bot response.
        user_q = st.chat_input("Ask the room…")
        if user_q:
            st.session_state["room_thinking"] = True

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
        unavailable = st.session_state.get("room_unavailable_files", [])
        if unavailable:
            st.warning(
                "These room files are registered in Firebase, but this device "
                f"could not load them yet: {', '.join(unavailable)}"
            )

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

        # Chat input already read above — send message and get LLM response
        if user_q:
            send_room_message(
                code, st.session_state["username"], user_q, is_bot=False
            )

            with st.spinner("Thinking…"):
                try:
                    # FIX: pass room history so LLM has conversational memory
                    room_hist = _room_history_for_llm(code)
                    response  = build_response(user_q, history=room_hist)
                except Exception as e:
                    response = f"⚠️ Error getting response: {str(e)}"

            send_room_message(code, "AM Intelligence", response, is_bot=True)

            st.session_state["room_thinking"] = False
            st.rerun()
