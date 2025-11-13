import os
import re
import logging
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from datetime import datetime, timedelta
import google.generativeai as genai
from langchain_community.embeddings import OllamaEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ================================================
# ENVIRONMENT VARIABLES
# ================================================
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "brookstone_verify_token_2024")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
BROCHURE_URL = os.getenv("BROCHURE_URL", "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/BROOKSTONE.pdf")

# WorkVEU CRM
WORKVEU_WEBHOOK_URL = os.getenv("WORKVEU_WEBHOOK_URL")
WORKVEU_API_KEY = os.getenv("WORKVEU_API_KEY")

MEDIA_STATE_FILE = os.path.join(os.path.dirname(__file__), "media_state.json")
MEDIA_ID = None
MEDIA_EXPIRY_DAYS = 29


def load_media_state():
    try:
        if os.path.exists(MEDIA_STATE_FILE):
            with open(MEDIA_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"❌ Error loading media state: {e}")
    return {}


def save_media_state(state: dict):
    try:
        with open(MEDIA_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception as e:
        logging.error(f"❌ Error saving media state: {e}")


def _find_brochure_file():
    candidates = [
        os.path.join(os.path.dirname(__file__), "static", "brochure", "BROOKSTONE.pdf"),
        os.path.join(os.path.dirname(__file__), "static", "BROCHURE.pdf"),
        os.path.join(os.path.dirname(__file__), "BROOKSTONE.pdf"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def upload_brochure_media():
    global MEDIA_ID
    file_path = _find_brochure_file()
    if not file_path:
        logging.error("❌ Brochure file not found in static folders.")
        return None

    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/media"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    data = {"messaging_product": "whatsapp"}
    try:
        with open(file_path, "rb") as fh:
            files = {"file": (os.path.basename(file_path), fh, "application/pdf")}
            resp = requests.post(url, headers=headers, data=data, files=files, timeout=60)
        if resp.status_code == 200:
            j = resp.json()
            new_media_id = j.get("id")
            if new_media_id:
                MEDIA_ID = new_media_id
                state = {"media_id": MEDIA_ID, "uploaded_at": datetime.utcnow().isoformat()}
                save_media_state(state)
                logging.info(f"✅ Uploaded brochure media, id={MEDIA_ID}")
                return MEDIA_ID
            else:
                logging.error(f"❌ Upload succeeded but no media id returned: {resp.text}")
        else:
            logging.error(f"❌ Failed to upload media: {resp.status_code} - {resp.text}")
    except Exception as e:
        logging.error(f"❌ Exception uploading media: {e}")
    return None


def ensure_media_up_to_date():
    global MEDIA_ID
    state = load_media_state()
    media_id = state.get("media_id")
    uploaded_at = state.get("uploaded_at")
    need_upload = True
    if media_id and uploaded_at:
        try:
            uploaded_dt = datetime.fromisoformat(uploaded_at)
            if datetime.utcnow() - uploaded_dt < timedelta(days=MEDIA_EXPIRY_DAYS):
                MEDIA_ID = media_id
                need_upload = False
                logging.info(f"ℹ️ Using existing media_id (uploaded {uploaded_at})")
        except Exception:
            need_upload = True

    if need_upload:
        logging.info("ℹ️ Uploading brochure media to WhatsApp Cloud")
        upload_brochure_media()


try:
    ensure_media_up_to_date()
    logging.info(f"✅ Media management initialized.")
except Exception as e:
    logging.error(f"❌ Error initializing media: {e}")


# ================================================
# GEMINI SETUP
# ================================================
if not GEMINI_API_KEY or not PINECONE_API_KEY:
    logging.error("❌ Missing GEMINI_API_KEY or PINECONE_API_KEY")

gemini_model = None
gemini_chat = None

try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    gemini_chat = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=0
    )
    logging.info("✅ Gemini API configured successfully")
except Exception as e:
    logging.error(f"❌ Error initializing Gemini: {e}")


# ================================================
# OLLAMA + PINECONE RETRIEVAL SETUP (latest SDK)
# ================================================
INDEX_NAME = "brookstone-faq"

try:
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    logging.info("✅ Ollama nomic-text-embedding configured successfully")
except Exception as e:
    logging.error(f"❌ Error initializing Ollama embeddings: {e}")
    ollama_embeddings = None

# Create new Pinecone client (latest SDK)
pc = None
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    logging.info("✅ Pinecone client created successfully")
except Exception as e:
    logging.error(f"❌ Failed to create Pinecone client: {e}")
    pc = None


def load_vectorstore():
    """Connect to existing Pinecone index using new client."""
    if not ollama_embeddings:
        logging.error("❌ Ollama embeddings not available for Pinecone retrieval")
        return None
    if not pc:
        logging.error("❌ Pinecone client not initialized")
        return None

    try:
        # Ensure index exists
        indexes = [idx["name"] for idx in pc.list_indexes()]
        if INDEX_NAME not in indexes:
            logging.error(f"❌ Index '{INDEX_NAME}' not found in Pinecone. Available: {indexes}")
            return None

        # Connect to the existing index
        index = pc.Index(INDEX_NAME)
        vectorstore = PineconeVectorStore(index=index, embedding=ollama_embeddings)
        logging.info(f"✅ Connected to Pinecone index '{INDEX_NAME}' with Ollama embeddings")
        return vectorstore
    except Exception as e:
        logging.error(f"❌ Error connecting to Pinecone index: {e}")
        return None


try:
    vectorstore = load_vectorstore()
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        logging.info("✅ Retriever initialized successfully using Ollama embeddings + Pinecone")
    else:
        retriever = None
        logging.error("❌ Failed to load retriever from Pinecone")
except Exception as e:
    logging.error(f"❌ Error initializing retriever: {e}")
    retriever = None


# ================================================
# TRANSLATION FUNCTIONS
# ================================================
def translate_gujarati_to_english(text):
    try:
        if not gemini_model:
            return text
        translation_prompt = f"Translate Gujarati to English: {text}"
        response = gemini_model.generate_content(translation_prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text


def translate_english_to_gujarati(text):
    try:
        if not gemini_model:
            return text
        translation_prompt = f"Translate English to Gujarati (short and conversational): {text}"
        response = gemini_model.generate_content(translation_prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text


# ================================================
# HEALTH CHECK ROUTE
# ================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "ollama_configured": bool(ollama_embeddings),
        "gemini_configured": bool(GEMINI_API_KEY and gemini_model and gemini_chat),
        "pinecone_configured": bool(PINECONE_API_KEY),
        "index_name": INDEX_NAME,
        "retriever_ready": bool(retriever)
    }), 200


# ================================================
# RUN APP
# ================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Starting Brookstone WhatsApp Bot on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
