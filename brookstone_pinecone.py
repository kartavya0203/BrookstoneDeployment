import os
import re
import logging
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.embeddings import OpenAIEmbeddings
import json
from datetime import datetime, timedelta
import google.generativeai as genai

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Keep for Pinecone embeddings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
BROCHURE_URL = os.getenv("BROCHURE_URL", "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/BROOKSTONE.pdf")

# >>> Added for WorkVEU CRM Integration <<<
WORKVEU_WEBHOOK_URL = os.getenv("WORKVEU_WEBHOOK_URL")
WORKVEU_API_KEY = os.getenv("WORKVEU_API_KEY")

# Media state file - stores last uploaded media_id and timestamp
MEDIA_STATE_FILE = os.path.join(os.path.dirname(__file__), "media_state.json")
MEDIA_ID = None
MEDIA_EXPIRY_DAYS = 29  # refresh media every 29 days


def load_media_state():
    """Load media state (media_id and uploaded_at) from file."""
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
    # common locations
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
    """Upload brochure PDF to WhatsApp Cloud and store media id."""
    global MEDIA_ID
    file_path = _find_brochure_file()
    if not file_path:
        logging.error("❌ Brochure file not found in static folders. Skipping media upload.")
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
    """Ensure we have a media_id and it's not expired (older than MEDIA_EXPIRY_DAYS)."""
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
        logging.info("ℹ️ Uploading brochure media to WhatsApp Cloud (initial/refresh)")
        upload_brochure_media()


# Initialize media state at startup (without scheduler)
try:
    ensure_media_up_to_date()
    logging.info(f"� Media management initialized. Use refresh_media.py for 29-day renewals.")
except Exception as e:
    logging.error(f"❌ Error initializing media: {e}")

if not GEMINI_API_KEY or not PINECONE_API_KEY or not OPENAI_API_KEY:
    logging.error("❌ Missing API keys! Need GEMINI_API_KEY, PINECONE_API_KEY, and OPENAI_API_KEY")

# Initialize Gemini for chat and translations, OpenAI for Pinecone embeddings
gemini_model = None
gemini_chat = None
openai_embeddings = None

if not GEMINI_API_KEY:
    logging.error("❌ Missing Gemini API key! Chat and translation features will not work.")
else:
    try:
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Initialize Gemini chat for LangChain
        gemini_chat = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0
        )
        
        logging.info("✅ Gemini API configured for chat and translations")
    except Exception as e:
        logging.error(f"❌ Error initializing Gemini: {e}")
        gemini_model = None
        gemini_chat = None

# Initialize OpenAI embeddings for Pinecone (to work with existing data)
if not OPENAI_API_KEY:
    logging.error("❌ Missing OpenAI API key! Pinecone search will not work.")
else:
    try:
        openai_embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=OPENAI_API_KEY
        )
        logging.info("✅ OpenAI embeddings configured for Pinecone search")
    except Exception as e:
        logging.error(f"❌ Error initializing OpenAI embeddings: {e}")
        openai_embeddings = None

# ================================================
# PINECONE SETUP
# ================================================
INDEX_NAME = "brookstone-faq-json"

def load_vectorstore():
    if not openai_embeddings:
        logging.error("❌ OpenAI embeddings not available for Pinecone")
        return None
    return PineconeVectorStore(index_name=INDEX_NAME, embedding=openai_embeddings)

try:
    vectorstore = load_vectorstore()
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        logging.info("✅ Pinecone vectorstore with OpenAI embeddings loaded successfully")
    else:
        retriever = None
        logging.error("❌ Failed to load vectorstore")
except Exception as e:
    logging.error(f"❌ Error loading Pinecone: {e}")
    retriever = None

# ================================================
# LLM SETUP
# ================================================
# LLM is now initialized above with gemini_chat

# ================================================
# TRANSLATION FUNCTIONS
# ================================================
def translate_gujarati_to_english(text):
    """Translate Gujarati text to English using Gemini"""
    try:
        if not gemini_model:
            logging.error("❌ Gemini model not available for translation")
            return text
            
        translation_prompt = f"""
Translate the following Gujarati text to English. Provide only the English translation, nothing else.

Gujarati text: {text}

English translation:
        """
        
        logging.info(f"🔄 Translating Gujarati to English: {text[:50]}...")
        response = gemini_model.generate_content(translation_prompt)
        translated_text = response.text.strip()
        logging.info(f"✅ Translation result: {translated_text[:50]}...")
        return translated_text
        
    except Exception as e:
        logging.error(f"❌ Error translating Gujarati to English with Gemini: {e}")
        return text  # Return original text if translation fails

def translate_english_to_gujarati(text):
    """Translate English text to Gujarati using Gemini"""
    try:
        if not gemini_model:
            logging.error("❌ Gemini model not available for translation")
            return text
            
        translation_prompt = f"""
Translate the following English text to Gujarati. Keep the same tone, style, and LENGTH - make it brief and concise like the original. Provide only the Gujarati translation, nothing else.

English text: {text}

Gujarati translation (keep it brief and concise):
        """
        
        logging.info(f"🔄 Translating English to Gujarati: {text[:50]}...")
        response = gemini_model.generate_content(translation_prompt)
        translated_text = response.text.strip()
        logging.info(f"✅ Translation result: {translated_text[:50]}...")
        return translated_text
        
    except Exception as e:
        logging.error(f"❌ Error translating English to Gujarati with Gemini: {e}")
        return text  # Return original text if translation fails

# ================================================
# AREA INFORMATION DATABASE
# ================================================
AREA_INFO = {
    "3bhk": {
        "super_buildup": "2650 sqft",
        "display_name": "3BHK"
    },
    "4bhk": {
        "super_buildup": "3850 sqft", 
        "display_name": "4BHK"
    },
    "3bhk_tower_duplex": {
        "super_buildup": "5300 sqft + 700 sqft carpet terrace",
        "display_name": "3BHK Tower Duplex"
    },
    "4bhk_tower_duplex": {
        "super_buildup": "7700 sqft + 1000 sqft carpet terrace",
        "display_name": "4BHK Tower Duplex" 
    },
    "3bhk_tower_simplex": {
        "super_buildup": "2650 sqft + 700 sqft carpet terrace",
        "display_name": "3BHK Tower Simplex"
    },
    "4bhk_tower_simplex": {
        "super_buildup": "3850 sqft + 1000 sqft carpet terrace", 
        "display_name": "4BHK Tower Simplex"
    }
}

def get_area_information(query):
    """Get area information from hardcoded database"""
    query_lower = query.lower()
    results = []
    
    # Check for specific unit types
    if "tower duplex" in query_lower:
        if "3bhk" in query_lower or "3 bhk" in query_lower:
            results.append(f"{AREA_INFO['3bhk_tower_duplex']['display_name']}: {AREA_INFO['3bhk_tower_duplex']['super_buildup']}")
        elif "4bhk" in query_lower or "4 bhk" in query_lower:
            results.append(f"{AREA_INFO['4bhk_tower_duplex']['display_name']}: {AREA_INFO['4bhk_tower_duplex']['super_buildup']}")
        else:
            # If tower duplex mentioned but no specific BHK, show both tower duplex units
            results.append(f"{AREA_INFO['3bhk_tower_duplex']['display_name']}: {AREA_INFO['3bhk_tower_duplex']['super_buildup']}")
            results.append(f"{AREA_INFO['4bhk_tower_duplex']['display_name']}: {AREA_INFO['4bhk_tower_duplex']['super_buildup']}")
    elif "tower simplex" in query_lower:
        if "3bhk" in query_lower or "3 bhk" in query_lower:
            results.append(f"{AREA_INFO['3bhk_tower_simplex']['display_name']}: {AREA_INFO['3bhk_tower_simplex']['super_buildup']}")
        elif "4bhk" in query_lower or "4 bhk" in query_lower:
            results.append(f"{AREA_INFO['4bhk_tower_simplex']['display_name']}: {AREA_INFO['4bhk_tower_simplex']['super_buildup']}")
        else:
            # If tower simplex mentioned but no specific BHK, show both tower simplex units
            results.append(f"{AREA_INFO['3bhk_tower_simplex']['display_name']}: {AREA_INFO['3bhk_tower_simplex']['super_buildup']}")
            results.append(f"{AREA_INFO['4bhk_tower_simplex']['display_name']}: {AREA_INFO['4bhk_tower_simplex']['super_buildup']}")
    else:
        # Regular units - only add specific matches
        if "3bhk" in query_lower or "3 bhk" in query_lower:
            results.append(f"{AREA_INFO['3bhk']['display_name']}: {AREA_INFO['3bhk']['super_buildup']}")
        if "4bhk" in query_lower or "4 bhk" in query_lower:
            results.append(f"{AREA_INFO['4bhk']['display_name']}: {AREA_INFO['4bhk']['super_buildup']}")
        
        # Only return all regular units if query is very general (like "what are the sizes" or "area information")
        if not results and any(general_term in query_lower for general_term in ["what are", "all", "available", "sizes", "options"]) and not any(specific in query_lower for specific in ["5bhk", "penthouse", "villa", "studio"]):
            results = [
                f"{AREA_INFO['3bhk']['display_name']}: {AREA_INFO['3bhk']['super_buildup']}",
                f"{AREA_INFO['4bhk']['display_name']}: {AREA_INFO['4bhk']['super_buildup']}"
            ]
    
    return results

# ================================================
# CONVERSATION STATE & CONTEXT ANALYSIS WITH GEMINI
# ================================================
CONV_STATE = {}

def ensure_conversation_state(from_phone):
    """Ensure conversation state has all required fields"""
    if from_phone not in CONV_STATE:
        CONV_STATE[from_phone] = {
            "chat_history": [], 
            "language": "english",  # Default to English, will be dynamically detected
            "waiting_for": None,
            "last_context_topics": [],
            "user_interests": [],
            "last_follow_up": None,  # Store the last follow-up question asked
            "follow_up_context": None,  # Store context for the follow-up
            "is_first_message": True,  # Track if this is the first interaction
            "conversation_summary": "",  # Gemini-generated conversation summary
            "user_preferences": {}  # AI-inferred user preferences
        }
    else:
        # Ensure all required fields exist
        if "waiting_for" not in CONV_STATE[from_phone]:
            CONV_STATE[from_phone]["waiting_for"] = None
        if "last_context_topics" not in CONV_STATE[from_phone]:
            CONV_STATE[from_phone]["last_context_topics"] = []
        if "user_interests" not in CONV_STATE[from_phone]:
            CONV_STATE[from_phone]["user_interests"] = []
        if "last_follow_up" not in CONV_STATE[from_phone]:
            CONV_STATE[from_phone]["last_follow_up"] = None
        if "follow_up_context" not in CONV_STATE[from_phone]:
            CONV_STATE[from_phone]["follow_up_context"] = None
        if "is_first_message" not in CONV_STATE[from_phone]:
            CONV_STATE[from_phone]["is_first_message"] = True
        if "conversation_summary" not in CONV_STATE[from_phone]:
            CONV_STATE[from_phone]["conversation_summary"] = ""
        if "user_preferences" not in CONV_STATE[from_phone]:
            CONV_STATE[from_phone]["user_preferences"] = {}

def update_conversation_memory_with_gemini(state, user_message, bot_response):
    """Use Gemini to analyze and update conversation memory"""
    try:
        if not gemini_model:
            return
        
        # Only update memory every few messages to avoid too many API calls
        if len(state["chat_history"]) % 4 != 0:
            return
            
        recent_history = state["chat_history"][-8:]  # Last 4 exchanges
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        
        memory_prompt = f"""
Analyze this real estate conversation and extract:
1. User's key interests and preferences
2. Important conversation points to remember
3. User's likely budget range or property requirements
4. Any specific questions or concerns raised

Conversation History:
{history_text}

Current Summary: {state.get('conversation_summary', 'New conversation')}

Provide a concise updated summary (max 100 words) and key user preferences in JSON format:
{{
    "summary": "brief conversation summary",
    "preferences": {{
        "budget_range": "inferred budget or 'unknown'",
        "preferred_bhk": "3BHK/4BHK/both/unknown",
        "key_interests": ["list", "of", "interests"],
        "concerns": ["any", "concerns", "raised"],
        "visit_intent": "high/medium/low/unknown"
    }}
}}
        """
        
        response = gemini_model.generate_content(memory_prompt)
        
        # Try to parse JSON response
        try:
            import json
            memory_data = json.loads(response.text.strip())
            state["conversation_summary"] = memory_data.get("summary", "")
            state["user_preferences"].update(memory_data.get("preferences", {}))
            logging.info(f"🧠 Updated conversation memory for user")
        except:
            # If JSON parsing fails, just store as text summary
            state["conversation_summary"] = response.text.strip()[:200]
            
    except Exception as e:
        logging.error(f"❌ Error updating conversation memory: {e}")

def analyze_user_interests(message_text, state):
    """Analyze user message to understand their interests (keyword-based fallback)"""
    message_lower = message_text.lower()
    interests = []
    
    # Interest categories - these help understand user intent
    interest_keywords = {
        "pricing": ["price", "cost", "budget", "expensive", "cheap", "affordable", "rate"],
        "size": ["size", "area", "bhk", "bedroom", "space", "sqft", "square"],
        "amenities": ["amenities", "facilities", "gym", "pool", "parking", "security"],
        "location": ["location", "address", "nearby", "connectivity", "metro", "airport"],
        "availability": ["available", "ready", "possession", "when", "booking"],
        "visit": ["visit", "see", "tour", "show", "check", "viewing"]
    }
    
    for category, keywords in interest_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            interests.append(category)
    
    # Update user interests (keep last 5 to avoid memory bloat)
    state["user_interests"].extend(interests)
    state["user_interests"] = list(set(state["user_interests"][-5:]))
    
    return interests

def detect_location_request_with_gemini(message_text):
    """Use Gemini to intelligently detect if user is asking for location/address"""
    try:
        if not gemini_model:
            return False
        
        # First check if message contains property inquiry keywords - if yes, don't treat as location request
        property_inquiry_keywords = [
            "bhk", "flat", "apartment", "house", "home", "property", "unit", 
            "bedroom", "price", "cost", "rate", "booking", "buy", "purchase",
            "interested", "looking", "want", "need", "show", "see", "visit",
            "ફ્લેટ", "ઘર", "પ્રોપર્ટી", "બેડરૂમ", "કિંમત", "દર", "ખરીદી", "જોઈએ", "શોધી"
        ]
        
        message_lower = message_text.lower()
        if any(keyword in message_lower for keyword in property_inquiry_keywords):
            logging.info(f"🏠 Property inquiry detected, not treating as location request: '{message_text[:30]}...'")
            return False
        
        location_detection_prompt = f"""
Analyze this message and determine if the user is SPECIFICALLY asking for location, address, or directions to a property/site.

User Message: "{message_text}"

ONLY consider these as location requests:
- Direct requests for address, location, directions, map
- "Where is it located?" / "કયાં સ્થિત છે?"
- "Can you share the location?" / "લોકેશન શેર કરો"
- "What's the address?" / "સરનામું શું છે?"
- "How to reach there?" / "ત્યાં કેવી રીતે પહોંચવું?"
- "Send location" / "લોકેશન મોકલો"

DO NOT consider these as location requests:
- General property inquiries about flats/apartments
- Questions about prices, features, amenities
- Interest in visiting or seeing properties
- Questions about availability or types of units

Respond with only "YES" if it's a SPECIFIC location/address request, or "NO" if it's any other type of inquiry.
        """
        
        response = gemini_model.generate_content(location_detection_prompt)
        result = response.text.strip().upper()
        
        logging.info(f"🗺️ Location request detection: '{message_text[:30]}...' → {result}")
        return result == "YES"
        
    except Exception as e:
        logging.error(f"❌ Error in location detection: {e}")
        return False

def analyze_user_interests_with_gemini(message_text, state):
    """Enhanced user interest analysis using Gemini AI"""
    try:
        if not gemini_model:
            return analyze_user_interests(message_text, state)  # Fallback to keyword-based
        
        interest_prompt = f"""
Analyze this real estate inquiry and identify the user's interests/intent:

User Message: "{message_text}"
Previous Interests: {state.get('user_interests', [])}

Categorize interests from: pricing, size, amenities, location, availability, visit, brochure, general_info

Return only a comma-separated list of relevant categories. Example: "pricing, size, visit"
        """
        
        response = gemini_model.generate_content(interest_prompt)
        interests = [interest.strip() for interest in response.text.strip().split(",") if interest.strip()]
        
        # Update user interests (keep last 5 to avoid memory bloat)
        state["user_interests"].extend(interests)
        state["user_interests"] = list(set(state["user_interests"][-5:]))
        
        return interests
        
    except Exception as e:
        logging.error(f"❌ Error in Gemini interest analysis: {e}")
        return analyze_user_interests(message_text, state)  # Fallback

def analyze_user_interests(message_text, state):
    """Analyze user message to understand their interests"""
    message_lower = message_text.lower()
    interests = []
    
    # Interest categories - these help understand user intent
    interest_keywords = {
        "pricing": ["price", "cost", "budget", "expensive", "cheap", "affordable", "rate"],
        "size": ["size", "area", "bhk", "bedroom", "space", "sqft", "square"],
        "amenities": ["amenities", "facilities", "gym", "pool", "parking", "security"],
        "location": ["location", "address", "nearby", "connectivity", "metro", "airport"],
        "availability": ["available", "ready", "possession", "when", "booking"],
        "visit": ["visit", "see", "tour", "show", "check", "viewing"]
    }
    
    for category, keywords in interest_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            interests.append(category)
    
    # Update user interests (keep last 5 to avoid memory bloat)
    state["user_interests"].extend(interests)
    state["user_interests"] = list(set(state["user_interests"][-5:]))
    
    return interests

# ================================================
# WHATSAPP FUNCTIONS
# ================================================
def send_whatsapp_text(to_phone, message):
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "text",
        "text": {"body": message}
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            logging.info(f"✅ Message sent to {to_phone}")
        else:
            logging.error(f"❌ Failed to send message: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"❌ Error sending message: {e}")

def send_whatsapp_location(to_phone):
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "location",
        "location": {
            "latitude": "23.0433468",
            "longitude": "72.4594457",
            "name": "Brookstone",
            "address": "Brookstone, Vaikunth Bungalows, Beside DPS Bopal Rd, next to A. Shridhar Oxygen Park, Bopal, Shilaj, Ahmedabad, Gujarat 380058"
        }
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            logging.info(f"✅ Location sent to {to_phone}")
        else:
            logging.error(f"❌ Failed to send location: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"❌ Error sending location: {e}")

def send_whatsapp_document(to_phone, caption="Here is your Brookstone Brochure 📄"):
    # If we have a valid MEDIA_ID, send the document by media id, otherwise fallback to link
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}

    if MEDIA_ID:
        payload = {
            "messaging_product": "whatsapp",
            "to": to_phone,
            "type": "document",
            "document": {"id": MEDIA_ID, "caption": caption, "filename": "Brookstone_Brochure.pdf"}
        }
    else:
        # fallback to sending by link if media id is not available
        payload = {
            "messaging_product": "whatsapp",
            "to": to_phone,
            "type": "document",
            "document": {"link": BROCHURE_URL, "caption": caption, "filename": "Brookstone_Brochure.pdf"}
        }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            logging.info(f"✅ Document sent to {to_phone}")
        else:
            logging.error(f"❌ Failed to send document: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"❌ Error sending document: {e}")

def mark_message_as_read(message_id):
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "status": "read", "message_id": message_id}
    try:
        requests.post(url, headers=headers, json=payload, timeout=10)
    except Exception as e:
        logging.error(f"Error marking message as read: {e}")

# ================================================
# WORKVEU CRM INTEGRATION
# ================================================
def push_to_workveu(name, wa_id, message_text, direction="inbound"):
    """Push chat messages to WorkVEU CRM for admin monitoring"""
    if not WORKVEU_WEBHOOK_URL or not WORKVEU_API_KEY:
        logging.warning("⚠️ WorkVEU integration skipped: Missing WORKVEU_WEBHOOK_URL or WORKVEU_API_KEY in .env file.")
        return

    payload = {
        "api_key": WORKVEU_API_KEY,
        "contacts": [
            {
                "profile": {"name": name or "Unknown User"},
                "wa_id": wa_id,
                "remarks": f"[{direction.upper()}] {message_text}"
            }
        ]
    }

    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(WORKVEU_WEBHOOK_URL, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            logging.info(f"✅ WorkVEU message synced ({direction}) for {wa_id}")
        else:
            logging.error(f"❌ WorkVEU sync failed: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"❌ Error pushing to WorkVEU: {e}")

# ================================================
# MESSAGE PROCESSING
# ================================================
def process_incoming_message(from_phone, message_text, message_id):
    ensure_conversation_state(from_phone)
    state = CONV_STATE[from_phone]
    
    # Dynamic language detection - respond in the same language as current message
    current_msg_has_gujarati = any("\u0A80" <= c <= "\u0AFF" for c in message_text)
    
    # Set language based on current message (dynamic response)
    if current_msg_has_gujarati:
        state["language"] = "gujarati"
    else:
        # Check if it's mostly English text (not just numbers/symbols)
        english_chars = sum(1 for c in message_text if c.isalpha() and ord(c) < 128)
        total_chars = len([c for c in message_text if c.isalpha()])
        
        if total_chars > 0 and (english_chars / total_chars) > 0.7:
            state["language"] = "english"
        else:
            # If unclear, check recent conversation history for context
            recent_gujarati = False
            for msg in state["chat_history"][-3:]:  # Check last 3 messages
                if any("\u0A80" <= c <= "\u0AFF" for c in msg.get("content", "")):
                    recent_gujarati = True
                    break
            state["language"] = "gujarati" if recent_gujarati else "english"
    
    logging.info(f"🌐 Language detected: {state['language']} for message: {message_text[:30]}...")
    
    state["chat_history"].append({"role": "user", "content": message_text})

    # >>> Added for WorkVEU CRM Integration <<<
    push_to_workveu(name=None, wa_id=from_phone, message_text=message_text, direction="inbound")

    # Check if this is the first message and send welcome
    if state.get("is_first_message", True):
        state["is_first_message"] = False
        welcome_text = "Hello! Welcome to Brookstone. How could I assist you today? 🏠✨"
        if state["language"] == "gujarati":
            welcome_text = translate_english_to_gujarati(welcome_text)
        send_whatsapp_text(from_phone, welcome_text)
        
        # >>> Added for WorkVEU CRM Integration <<<
        push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=welcome_text, direction="outbound")
        return

    # 🗺️ PRIORITY: Check for location requests first (using Gemini AI detection)
    if detect_location_request_with_gemini(message_text):
        logging.info(f"🗺️ Location request detected from {from_phone}, sending location pin only")
        send_whatsapp_location(from_phone)
        return

    # 📄 PRIORITY: Check for direct brochure requests (Enhanced Gujarati Support)
    brochure_keywords = ["brochure", "pdf", "document", "file", "download", "send", "details", "ब्रोशर", "બ્રોશર"]
    gujarati_action_words = ["મોકલો", "આપો", "મોકલાવો", "મોકલ", "આપ", "જોઈએ", "પાઠવો", "મેળવવા", "લેવા"]
    
    # Check for Gujarati brochure requests specifically
    if "બ્રોશર" in message_text:
        # If user mentions "બ્રોશર" in any context, send the brochure immediately
        logging.info(f"📄 Gujarati brochure request detected from {from_phone} - 'બ્રોશર' found")
        send_whatsapp_document(from_phone)
        brochure_sent_text = "📄 Here's your Brookstone brochure with complete details! ✨ Any questions after reviewing it? 🏠😊"
        if state["language"] == "gujarati":
            brochure_sent_text = translate_english_to_gujarati(brochure_sent_text)
        send_whatsapp_text(from_phone, brochure_sent_text)
        
        # >>> Added for WorkVEU CRM Integration <<<
        push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=f"📄 Brochure sent + {brochure_sent_text}", direction="outbound")
        return
    
    # Check for English brochure requests
    if any(keyword in message_text.lower() for keyword in brochure_keywords):
        if any(word in message_text.lower() for word in ["send", "share", "give", "want", "need", "show"] + gujarati_action_words):
            logging.info(f"📄 Direct brochure request detected from {from_phone}")
            send_whatsapp_document(from_phone)
            brochure_sent_text = "📄 Here's your Brookstone brochure with complete details! ✨ Any questions after reviewing it? 🏠😊"
            if state["language"] == "gujarati":
                brochure_sent_text = translate_english_to_gujarati(brochure_sent_text)
            send_whatsapp_text(from_phone, brochure_sent_text)
            
            # >>> Added for WorkVEU CRM Integration <<<
            push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=f"📄 Brochure sent + {brochure_sent_text}", direction="outbound")
            return

    # Analyze user interests for better follow-up questions (using Gemini)
    current_interests = analyze_user_interests_with_gemini(message_text, state)
    
    logging.info(f"📱 Processing message from {from_phone}: {message_text} [Language: {state['language']}] [Interests: {current_interests}]")

    # Check for follow-up responses
    message_lower = message_text.lower().strip()
    
    # Handle brochure confirmation
    if state.get("waiting_for") == "brochure_confirmation":
        if any(word in message_lower for word in ["yes", "yeah", "yep", "sure", "please", "send", "brochure", "pdf", "ok", "okay", "हाँ", "હા", "જી", "આપો"]):
            state["waiting_for"] = None
            state["last_follow_up"] = None  # Clear previous follow-up
            send_whatsapp_document(from_phone)
            brochure_text = "📄 Here's your Brookstone brochure! It has all the details you need. Any questions after going through it? ✨😊"
            if state["language"] == "gujarati":
                brochure_text = translate_english_to_gujarati(brochure_text)
            send_whatsapp_text(from_phone, brochure_text)
            
            # >>> Added for WorkVEU CRM Integration <<<
            push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=f"📄 Brochure sent + {brochure_text}", direction="outbound")
            return
        elif any(word in message_lower for word in ["no", "not now", "later", "નહીં", "ના"]):
            state["waiting_for"] = None
            state["last_follow_up"] = None  # Clear previous follow-up
            later_text = "Sure! Let me know if you'd like the brochure later or have any other questions about Brookstone. 🏠😊"
            if state["language"] == "gujarati":
                later_text = translate_english_to_gujarati(later_text)
            send_whatsapp_text(from_phone, later_text)
            
            # >>> Added for WorkVEU CRM Integration <<<
            push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=later_text, direction="outbound")
            return
    
    # Handle ambiguous responses (when user says "sure" but it's unclear what they want)
    if state.get("waiting_for") == "clarification_needed":
        # Check if user wants layout/details
        if any(word in message_lower for word in ["layout", "details", "size", "area", "plan", "floor", "design", "લેઆઉટ", "વિગત"]):
            state["waiting_for"] = "brochure_confirmation"
            state["last_follow_up"] = None  # Clear previous follow-up
            clarify_text = "Great! Would you like me to send you our detailed brochure with all floor plans and specifications? 📄✨"
            if state["language"] == "gujarati":
                clarify_text = translate_english_to_gujarati(clarify_text)
            send_whatsapp_text(from_phone, clarify_text)
            
            # >>> Added for WorkVEU CRM Integration <<<
            push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=clarify_text, direction="outbound")
            return
        # Check if user wants site visit
        elif any(word in message_lower for word in ["visit", "site", "see", "tour", "book", "appointment", "schedule", "મુલાકાત", "સાઇટ"]):
            state["waiting_for"] = None
            state["last_follow_up"] = None  # Clear previous follow-up
            visit_text = "Perfect! Please contact *Mr. Nilesh at 7600612701* to book your site visit. He'll help you schedule a convenient time. 📞✨"
            if state["language"] == "gujarati":
                visit_text = translate_english_to_gujarati(visit_text)
            send_whatsapp_text(from_phone, visit_text)
            
            # >>> Added for WorkVEU CRM Integration <<<
            push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=visit_text, direction="outbound")
            return
        else:
            # If still unclear, ask again
            state["waiting_for"] = None
            unclear_text = "I want to help you properly! Are you interested in seeing the layout details and brochure, or would you like to schedule a site visit? 🏠✨"
            if state["language"] == "gujarati":
                unclear_text = translate_english_to_gujarati(unclear_text)
            send_whatsapp_text(from_phone, unclear_text)
            
            # >>> Added for WorkVEU CRM Integration <<<
            push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=unclear_text, direction="outbound")
            return

    if not retriever:
        error_text = "Please contact our agents at 8238477697 or 9974812701 for more info."
        if state["language"] == "gujarati":
            error_text = translate_english_to_gujarati(error_text)
        send_whatsapp_text(from_phone, error_text)
        
        # >>> Added for WorkVEU CRM Integration <<<
        push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=error_text, direction="outbound")
        return

    try:
        # Translate Gujarati query to English for Pinecone search
        search_query = message_text
        if state["language"] == "gujarati":
            search_query = translate_gujarati_to_english(message_text)
            logging.info(f"🔄 Translated query: {search_query}")

        # Area terminology mapping for search and response
        area_response_mapping = {}
        original_query = search_query
        
        # Check if user is asking about area-related queries
        search_query_lower = search_query.lower()
        
        # Check if this is an area-related query that we have hardcoded information for
        area_keywords = ["area", "sqft", "square feet", "size", "carpet", "super build", "buildup", "built-up", "sbu"]
        is_area_query = any(keyword in search_query_lower for keyword in area_keywords)
        
        # Check for area terminology mapping first
        area_response_mapping = {}
        if any(term in search_query_lower for term in ["carpet area", "carpet"]):
            area_response_mapping["user_term"] = "carpet area"
            area_response_mapping["response_term"] = "Super Build-up area"
            logging.info("🏠 User asked about carpet area - will respond with Super Build-up area")
        elif any(term in search_query_lower for term in ["super build-up", "super buildup", "build-up", "buildup", "built-up", "sbu", "super build up", "build up"]):
            # Extract the original term user used
            original_term = "Super Build-up area"
            if "sbu" in search_query_lower:
                original_term = "SBU"
            elif "build-up" in search_query_lower or "buildup" in search_query_lower or "built-up" in search_query_lower:
                original_term = "Build-up area"
            elif "super build" in search_query_lower:
                original_term = "Super Build-up area"
            
            area_response_mapping["user_term"] = original_term.lower()
            area_response_mapping["response_term"] = original_term
            logging.info(f"🏠 User asked about {original_term}")
        
        # Try to get hardcoded information for area queries FIRST
        hardcoded_area_info = []
        use_hardcoded_only = False
        
        if is_area_query and area_response_mapping:
            # For area-related queries, check hardcoded information first
            hardcoded_area_info = get_area_information(search_query)
            if hardcoded_area_info:
                logging.info(f"🏠 Found hardcoded area information: {hardcoded_area_info}")
                use_hardcoded_only = True
                # Skip Pinecone search for area queries when hardcoded info is available
                docs = []
                context = ""
            else:
                logging.info(f"🏠 No hardcoded area info found, will search Pinecone")
        
        # If not using hardcoded info only, proceed with Pinecone search
        if not use_hardcoded_only:
            # Apply search term mapping for Pinecone search
            if area_response_mapping:
                if area_response_mapping["user_term"] == "carpet area":
                    # Search for carpet area in Pinecone
                    logging.info("🏠 Searching Pinecone for carpet area but will respond with Super Build-up area")
                else:
                    # Replace their term with "carpet area" for Pinecone search
                    search_query = re.sub(r'\b(super build-?up|build-?up|built-?up|sbu)(\s+area)?\b', 'carpet area', search_query, flags=re.IGNORECASE)
                    logging.info(f"🏠 Modified search query for Pinecone: {search_query}")
            
            docs = retriever.invoke(search_query)
            logging.info(f"📚 Retrieved {len(docs)} relevant documents")
        else:
            docs = []

        context = "\n\n".join(
            [(d.page_content or "") + ("\n" + "\n".join(f"{k}: {v}" for k, v in (d.metadata or {}).items())) for d in docs]
        )

        # Store current context topics for future reference
        state["last_context_topics"] = [d.metadata.get("topic", "") for d in docs if d.metadata.get("topic")]

        # Enhanced system prompt with Gemini-generated user context and memory
        user_context = f"User's previous interests: {', '.join(state['user_interests'])}" if state['user_interests'] else "New conversation"
        
        # Include Gemini-generated conversation summary and preferences
        conversation_context = ""
        if state.get("conversation_summary"):
            conversation_context = f"\nCONVERSATION SUMMARY: {state['conversation_summary']}"
        
        user_preferences = state.get("user_preferences", {})
        preferences_context = ""
        if user_preferences:
            pref_items = []
            if user_preferences.get("budget_range", "unknown") != "unknown":
                pref_items.append(f"Budget: {user_preferences['budget_range']}")
            if user_preferences.get("preferred_bhk", "unknown") != "unknown":
                pref_items.append(f"Preferred: {user_preferences['preferred_bhk']}")
            if user_preferences.get("key_interests"):
                pref_items.append(f"Interests: {', '.join(user_preferences['key_interests'])}")
            if user_preferences.get("visit_intent", "unknown") != "unknown":
                pref_items.append(f"Visit Intent: {user_preferences['visit_intent']}")
            
            if pref_items:
                preferences_context = f"\nUSER PREFERENCES: {' | '.join(pref_items)}"
        
        # Include memory of last follow-up question
        follow_up_memory = ""
        if state.get("last_follow_up"):
            follow_up_memory = f"\nRECENT FOLLOW-UP: I recently asked '{state['last_follow_up']}' and user is now responding to that question."
        
        # Area terminology instruction based on user query
        area_terminology_instruction = ""
        hardcoded_area_context = ""
        
        # Include hardcoded area information if available and this is an area query
        if hardcoded_area_info and use_hardcoded_only:
            hardcoded_area_context = f"\nHARDCODED AREA INFORMATION (USE THIS): {' | '.join(hardcoded_area_info)}"
            area_terminology_instruction += f"\nIMPORTANT: Use ONLY the hardcoded area information provided above for area-related questions. This is the most accurate and up-to-date information. Do NOT search or use any other sources for area information."
        elif hardcoded_area_info and not use_hardcoded_only:
            # Hardcoded info available but also using Pinecone context
            hardcoded_area_context = f"\nHARDCODED AREA INFORMATION: {' | '.join(hardcoded_area_info)}"
            area_terminology_instruction += f"\nIMPORTANT: Prefer the hardcoded area information when available, but you can also use context information if needed."
        
        if area_response_mapping:
            if area_response_mapping["user_term"] == "carpet area":
                area_terminology_instruction += f"\nIMPORTANT AREA TERMINOLOGY: User asked about '{area_response_mapping['user_term']}' but you must respond using '{area_response_mapping['response_term']}' instead. Say something like 'The {area_response_mapping['response_term']} is...' or 'Our {area_response_mapping['response_term']} for...' - NEVER mention 'carpet area' in your response."
            else:
                area_terminology_instruction += f"\nIMPORTANT AREA TERMINOLOGY: User asked about '{area_response_mapping['response_term']}'. Use their exact term '{area_response_mapping['response_term']}' in your response - do NOT mention 'carpet area'."
        
        # Determine language for system prompt
        language_instruction = ""
        if state["language"] == "gujarati":
            language_instruction = "IMPORTANT: User is asking in Gujarati. Respond in ENGLISH first (keep it VERY SHORT), then it will be translated to Gujarati automatically. The Gujarati translation should also be brief and concise."
        
        system_prompt = f"""
You are a friendly real estate assistant for Brookstone project. Be conversational, natural, and convincing.

{language_instruction}

{area_terminology_instruction}

{hardcoded_area_context}

CORE INSTRUCTIONS:
- Be EXTREMELY CONCISE - Maximum 1-2 sentences for initial response
- Answer using context below when available
- Use 2-3 relevant emojis only
- Keep responses WhatsApp-friendly and brief
- Do NOT invent details
- Do NOT give long explanations unless specifically asked
- ALWAYS try to convince user in a friendly way
- Use the conversation memory and user preferences provided
- Be NATURAL and CONTEXTUAL - don't repeat the same phrases in every response
- Only mention flat types (3&4BHK) when user specifically asks about them

MEMORY CONTEXT: {follow_up_memory}{conversation_context}{preferences_context}

SMART FLAT MENTIONS:
- ONLY mention "Brookstone offers luxurious 3&4BHK flats" when user specifically asks about:
  * Flat types/configurations (3BHK, 4BHK)
  * "What do you have?" / "What's available?"
  * Property types or unit options
  * First-time inquiries about the project
- Whether user asks about 3BHK or whether user asks about 4BHK, mention both types "We have luxurious 3&4BHK flats available at Brookstone!"
- Do NOT force this phrase into every response - be natural and contextual
- For queries about amenities, location, pricing, etc. - just answer directly without mentioning flat types

RESPONSE LENGTH RULES:
- For flat availability questions: "Yes! We have luxury 3&4BHK flats available 🏠 Interested in details? ✨"
- For general questions: Keep to 1 short sentence + 1 follow-up question
- NO detailed explanations unless specifically asked for details
- NO long paragraphs or multiple sentences

BROCHURE STRATEGY:
- ACTIVELY offer brochure when user shows interest in details, layout, floor plans, specifications, amenities
- Use phrases like "Would you like me to send you our detailed brochure?" 
- The brochure contains complete information about Brookstone's luxury offerings
- Make brochure sound valuable and comprehensive
- Offer brochure for queries about layouts, floor plans, unit details, specifications

SPECIAL HANDLING:

1. TIMINGS: "Our site office is open from *10:30 AM to 7:00 PM* every day. �"

2. SITE VISIT BOOKING: "Perfect! Please contact *Mr. Nilesh at 7600612701* to book your site visit. 📞✨"

3. GENERAL QUERIES: "You can contact our agents at 8238477697 or 9974812701 for any queries. 📱😊"

4. PRICING: Check context first. If no pricing info: "For latest pricing details, please contact our agents at 8238477697 or 9974812701. 💰📞"

5. BROCHURE OFFERING: When user asks about details/layout/plans/amenities/specifications: "Would you like me to send you our detailed brochure with all floor plans and specifications? 📄✨"

IMPORTANT: Do NOT handle location/address requests here - they are processed separately and will send location pin automatically.

CONVINCING STRATEGY:
- Use positive, enthusiastic language
- Highlight luxury and quality aspects
- Create urgency subtly ("perfect time to visit", "great opportunity")
- Use emojis that convey excitement: 🏠✨🌟💎🎉😊🔥💫

FOLLOW-UP STRATEGY:
- If user is responding to a previous follow-up question, acknowledge it and provide relevant information
- After answering, ask ONE simple follow-up question to continue conversation
- Make follow-ups natural and contextual
- This helps maintain conversation flow

RESPONSE PATTERN:
1. If responding to previous follow-up, acknowledge and answer appropriately
2. Give brief answer from context (always mention 3&4BHK when relevant)
3. Ask ONE clear follow-up question to keep conversation flowing

USER CONTEXT: {user_context}

CONVERSATION FLOW:
- If user is answering my previous question, provide relevant info based on their response
- Then naturally continue with another relevant question
- Keep the conversation engaging and helpful
- Always sound excited about Brookstone!

Example Responses (be contextual, not repetitive):
- When user asks about flat types: "Yes! We have luxury 3&4BHK flats 🏠 Which interests you more? ✨"
- When user asks about amenities: "Amazing amenities available! 💎 Want the brochure? �"
- When user asks about location: "Great location with excellent connectivity! 🗺️ Want to visit? 📞"
- When user asks about pricing: "Please contact 8238477697 for pricing 📞 Interested in floor plans? 📄"

Remember: Keep responses EXTREMELY brief - maximum 1-2 sentences!

---
Available Knowledge Context:
{context}

User Question: {search_query}

IMPORTANT: Be natural and contextual. Don't force "Brookstone offers luxurious 3&4BHK flats" into every response. Only mention flat types when the user specifically asks about configurations, availability, or what types of units you have.

KEEP RESPONSES VERY SHORT - Maximum 1-2 sentences + 1 simple follow-up question.
Assistant:
        """.strip()

        # Use Gemini for generating response
        if not gemini_chat:
            error_text = "AI service unavailable. Please contact our agents at 8238477697 or 9974812701."
            if state["language"] == "gujarati":
                error_text = translate_english_to_gujarati(error_text)
            send_whatsapp_text(from_phone, error_text)
            
            # >>> Added for WorkVEU CRM Integration <<<
            push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=error_text, direction="outbound")
            return

        response = gemini_chat.invoke(system_prompt).content.strip()
        logging.info(f"🧠 LLM Response: {response}")

        # Apply area terminology replacement in the response if needed
        if area_response_mapping:
            if area_response_mapping["user_term"] == "carpet area":
                # User asked about carpet area, replace any mention of "carpet area" with "Super Build-up area"
                response = re.sub(r'\bcarpet\s+area\b', area_response_mapping["response_term"], response, flags=re.IGNORECASE)
                response = re.sub(r'\bcarpet\b(?!\s+area)', area_response_mapping["response_term"], response, flags=re.IGNORECASE)
                logging.info(f"🏠 Replaced carpet area mentions with {area_response_mapping['response_term']}")
            else:
                # User asked about super build-up/build-up/SBU, make sure we don't mention carpet area
                response = re.sub(r'\bcarpet\s+area\b', area_response_mapping["response_term"], response, flags=re.IGNORECASE)
                response = re.sub(r'\bcarpet\b(?!\s+area)', area_response_mapping["response_term"], response, flags=re.IGNORECASE)
                logging.info(f"🏠 Ensured response uses {area_response_mapping['response_term']} instead of carpet area")

        # Translate response to Gujarati if user language is Gujarati
        final_response = response
        if state["language"] == "gujarati":
            final_response = translate_english_to_gujarati(response)
            logging.info(f"🔄 Translated response: {final_response}")

        # --- Send primary text response ---
        send_whatsapp_text(from_phone, final_response)

        # >>> Added for WorkVEU CRM Integration <<<
        push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=final_response, direction="outbound")

        # Store the follow-up question asked by the bot for memory
        # Extract follow-up question from response (look for question marks)
        sentences = final_response.split('.')
        follow_up_question = None
        for sentence in sentences:
            if '?' in sentence:
                follow_up_question = sentence.strip()
                break
        
        if follow_up_question:
            state["last_follow_up"] = follow_up_question
            state["follow_up_context"] = context[:500]  # Store some context for reference
            logging.info(f"🧠 Stored follow-up: {follow_up_question}")

        # --- Set conversation states based on bot's response ---
        response_lower = response.lower()  # Use original English response for state detection
        
        # Check if bot is asking multiple choice question (layout or site visit)
        if ("layout" in response_lower or "details" in response_lower) and ("site visit" in response_lower or "schedule" in response_lower):
            state["waiting_for"] = "clarification_needed"
            logging.info(f"🎯 Set state to clarification_needed for {from_phone}")
        
        # Check if bot mentioned site visit booking contact
        elif "mr. nilesh" in response_lower or "7600612701" in response_lower:
            # Bot already provided site visit booking info, no state change needed
            logging.info(f"📞 Site visit booking info provided to {from_phone}")
        
        # Check if bot is asking for brochure
        elif "would you like" in response_lower and ("brochure" in response_lower or "send you" in response_lower):
            state["waiting_for"] = "brochure_confirmation"
            logging.info(f"🎯 Set state to brochure_confirmation for {from_phone}")
        
        # Check if bot mentioned agent contact numbers
        elif "8238477697" in response_lower or "9974812701" in response_lower:
            # Bot already provided agent contact info, no state change needed
            logging.info(f"📞 Agent contact info provided to {from_phone}")

        # Legacy intent detection for immediate actions (without confirmation) 
        # Note: Location requests are now handled separately by Gemini detection
        if re.search(r"\bhere.*brochure\b|\bsending.*brochure\b", response_lower) and state.get("waiting_for") != "brochure_confirmation":
            logging.info(f"📄 Legacy brochure trigger for {from_phone}")
            send_whatsapp_document(from_phone)

        state["chat_history"].append({"role": "assistant", "content": final_response})
        
        # Update conversation memory using Gemini
        update_conversation_memory_with_gemini(state, message_text, final_response)

    except Exception as e:
        logging.error(f"❌ Error in RAG processing: {e}")
        error_text = "Sorry, I'm facing a technical issue. Please contact 8238477697 / 9974812701."
        if state["language"] == "gujarati":
            error_text = translate_english_to_gujarati(error_text)
        send_whatsapp_text(from_phone, error_text)
        
        # >>> Added for WorkVEU CRM Integration <<<
        push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=error_text, direction="outbound")

# ================================================
# WEBHOOK ROUTES
# ================================================
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logging.info("✅ WEBHOOK VERIFIED")
        return challenge, 200
    else:
        return "Forbidden", 403

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    logging.info("Incoming webhook data")

    try:
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])

                for message in messages:
                    from_phone = message.get("from")
                    message_id = message.get("id")
                    msg_type = message.get("type")

                    text = ""
                    if msg_type == "text":
                        text = message.get("text", {}).get("body", "")
                    elif msg_type == "button":
                        text = message.get("button", {}).get("text", "")
                    elif msg_type == "interactive":
                        interactive = message.get("interactive", {})
                        if "button_reply" in interactive:
                            text = interactive["button_reply"].get("title", "")
                        elif "list_reply" in interactive:
                            text = interactive["list_reply"].get("title", "")

                    if not text:
                        continue

                    mark_message_as_read(message_id)
                    process_incoming_message(from_phone, text, message_id)

    except Exception as e:
        logging.exception("❌ Error processing webhook")

    return jsonify({"status": "ok"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "whatsapp_configured": bool(WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID),
        "gemini_configured": bool(GEMINI_API_KEY and gemini_model and gemini_chat),
        "pinecone_configured": bool(PINECONE_API_KEY and openai_embeddings),
        "workveu_configured": bool(WORKVEU_WEBHOOK_URL and WORKVEU_API_KEY),
        "hybrid_mode": "Gemini for chat, OpenAI for search"
    }), 200

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Brookstone WhatsApp RAG Bot is running!",
        "brochure_url": BROCHURE_URL,
        "endpoints": {"webhook": "/webhook", "health": "/health"}
    }), 200

# ================================================
# RUN APP
# ================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Starting Brookstone WhatsApp Bot on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
