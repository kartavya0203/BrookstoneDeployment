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
# CONVERSATION STATE & CONTEXT ANALYSIS WITH GEMINI
# ================================================
CONV_STATE = {}

def ensure_conversation_state(from_phone):
    """Ensure conversation state has all required fields"""
    if from_phone not in CONV_STATE:
        CONV_STATE[from_phone] = {
            "chat_history": [], 
            "language": "english", 
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
# MESSAGE PROCESSING
# ================================================
def process_incoming_message(from_phone, message_text, message_id):
    ensure_conversation_state(from_phone)
    state = CONV_STATE[from_phone]
    guj = any("\u0A80" <= c <= "\u0AFF" for c in message_text)
    state["language"] = "gujarati" if guj else "english"
    state["chat_history"].append({"role": "user", "content": message_text})

    # Check if this is the first message and send welcome
    if state.get("is_first_message", True):
        state["is_first_message"] = False
        welcome_text = "Hello! Welcome to Brookstone. How could I assist you today? 🏠✨"
        if state["language"] == "gujarati":
            welcome_text = translate_english_to_gujarati(welcome_text)
        send_whatsapp_text(from_phone, welcome_text)
        return

    # Analyze user interests for better follow-up questions (using Gemini)
    current_interests = analyze_user_interests_with_gemini(message_text, state)
    
    logging.info(f"📱 Processing message from {from_phone}: {message_text} [Language: {state['language']}] [Interests: {current_interests}]")

    # Check for follow-up responses
    message_lower = message_text.lower().strip()
    
    # Handle location confirmation
    if state.get("waiting_for") == "location_confirmation":
        if any(word in message_lower for word in ["yes", "yeah", "yep", "sure", "please", "ok", "okay", "send", "हाँ", "હા"]):
            state["waiting_for"] = None
            state["last_follow_up"] = None  # Clear previous follow-up
            send_whatsapp_location(from_phone)
            confirmation_text = "📍 Here's our location! We're open from 10:30 AM to 7:00 PM. Looking forward to see you! 🏠✨"
            if state["language"] == "gujarati":
                confirmation_text = translate_english_to_gujarati(confirmation_text)
            send_whatsapp_text(from_phone, confirmation_text)
            return
        elif any(word in message_lower for word in ["no", "nope", "not now", "later", "નહીં", "ના"]):
            state["waiting_for"] = None
            state["last_follow_up"] = None  # Clear previous follow-up
            decline_text = "No problem! Feel free to ask if you need anything else. You can contact our agents at 8238477697 or 9974812701 anytime! ��😊"
            if state["language"] == "gujarati":
                decline_text = translate_english_to_gujarati(decline_text)
            send_whatsapp_text(from_phone, decline_text)
            return
    
    # Handle brochure confirmation
    if state.get("waiting_for") == "brochure_confirmation":
        if any(word in message_lower for word in ["yes", "yeah", "yep", "sure", "please", "send", "brochure", "pdf", "हाँ", "હા"]):
            state["waiting_for"] = None
            state["last_follow_up"] = None  # Clear previous follow-up
            send_whatsapp_document(from_phone)
            brochure_text = "📄 Here's your Brookstone brochure! It has all the details about our luxury 3&4BHK flats. Any questions after going through it? ✨😊"
            if state["language"] == "gujarati":
                brochure_text = translate_english_to_gujarati(brochure_text)
            send_whatsapp_text(from_phone, brochure_text)
            return
        elif any(word in message_lower for word in ["no", "not now", "later", "નહીં", "ના"]):
            state["waiting_for"] = None
            state["last_follow_up"] = None  # Clear previous follow-up
            later_text = "Sure! Let me know if you'd like the brochure later or have any other questions about Brookstone. 🏠�"
            if state["language"] == "gujarati":
                later_text = translate_english_to_gujarati(later_text)
            send_whatsapp_text(from_phone, later_text)
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
            return
        # Check if user wants site visit
        elif any(word in message_lower for word in ["visit", "site", "see", "tour", "book", "appointment", "schedule", "મુલાકાત", "સાઇટ"]):
            state["waiting_for"] = None
            state["last_follow_up"] = None  # Clear previous follow-up
            visit_text = "Perfect! Please contact *Mr. Nilesh at 7600612701* to book your site visit. He'll help you schedule a convenient time. 📞✨"
            if state["language"] == "gujarati":
                visit_text = translate_english_to_gujarati(visit_text)
            send_whatsapp_text(from_phone, visit_text)
            return
        else:
            # If still unclear, ask again
            state["waiting_for"] = None
            unclear_text = "I want to help you properly! Are you interested in seeing the layout details and brochure, or would you like to schedule a site visit? 🏠✨"
            if state["language"] == "gujarati":
                unclear_text = translate_english_to_gujarati(unclear_text)
            send_whatsapp_text(from_phone, unclear_text)
            return

    if not retriever:
        error_text = "Please contact our agents at 8238477697 or 9974812701 for more info."
        if state["language"] == "gujarati":
            error_text = translate_english_to_gujarati(error_text)
        send_whatsapp_text(from_phone, error_text)
        return

    try:
        # Translate Gujarati query to English for Pinecone search
        search_query = message_text
        if state["language"] == "gujarati":
            search_query = translate_gujarati_to_english(message_text)
            logging.info(f"🔄 Translated query: {search_query}")

        docs = retriever.invoke(search_query)
        logging.info(f"📚 Retrieved {len(docs)} relevant documents")

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
        
        # Determine language for system prompt
        language_instruction = ""
        if state["language"] == "gujarati":
            language_instruction = "IMPORTANT: User is asking in Gujarati. Respond in ENGLISH first (keep it VERY SHORT), then it will be translated to Gujarati automatically. The Gujarati translation should also be brief and concise."
        
        system_prompt = f"""
You are a friendly real estate assistant for Brookstone project. Be conversational, natural, and convincing.

{language_instruction}

CORE INSTRUCTIONS:
- Be VERY CONCISE - give brief, direct answers (2-3 sentences max)
- Answer using context below when available
- Use 2-3 relevant emojis to make responses engaging
- Keep responses WhatsApp-friendly
- Do NOT invent details
- Remember conversation flow and previous follow-ups
- ALWAYS try to convince user in a friendly way
- Use the conversation memory and user preferences provided

MEMORY CONTEXT: {follow_up_memory}{conversation_context}{preferences_context}

MANDATORY FLAT MENTIONS:
- ALWAYS say "Brookstone offers luxurious 3&4BHK flats" (mention both types)
- Even if user asks only about 3BHK or 4BHK, mention both options
- This showcases our complete offering

SPECIAL HANDLING:

1. TIMINGS: "Our site office is open from *10:30 AM to 7:00 PM* every day. Would you like me to send you the location? 📍"

2. SITE VISIT BOOKING: "Perfect! Please contact *Mr. Nilesh at 7600612701* to book your site visit. 📞✨"

3. GENERAL QUERIES: "You can contact our agents at 8238477697 or 9974812701 for any queries. 📱😊"

4. PRICING: Check context first. If no pricing info: "For latest pricing details, please contact our agents at 8238477697 or 9974812701. 💰📞"

5. LOCATION REQUEST: "Would you like me to send you our location? 📍🏠"

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

Example Responses:
- "Absolutely! Brookstone offers luxurious 3&4BHK flats 🏠✨ Would you like to know about the premium amenities? 🌟"
- "Great choice! Our 4BHK units are part of Brookstone's luxurious 3&4BHK collection 💎 Interested in the spacious layouts? 📐"

---
Available Knowledge Context:
{context}

User Question: {search_query}

Provide a brief, convincing answer with good emojis and ask ONE relevant follow-up question.
Assistant:
        """.strip()

        # Use Gemini for generating response
        if not gemini_chat:
            error_text = "AI service unavailable. Please contact our agents at 8238477697 or 9974812701."
            if state["language"] == "gujarati":
                error_text = translate_english_to_gujarati(error_text)
            send_whatsapp_text(from_phone, error_text)
            return

        response = gemini_chat.invoke(system_prompt).content.strip()
        logging.info(f"🧠 LLM Response: {response}")

        # Translate response to Gujarati if user language is Gujarati
        final_response = response
        if state["language"] == "gujarati":
            final_response = translate_english_to_gujarati(response)
            logging.info(f"🔄 Translated response: {final_response}")

        # --- Send primary text response ---
        send_whatsapp_text(from_phone, final_response)

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
        
        # Check if bot is asking for location confirmation
        if "would you like me to send" in response_lower and "location" in response_lower:
            state["waiting_for"] = "location_confirmation"
            logging.info(f"🎯 Set state to location_confirmation for {from_phone}")
        
        # Check if bot is asking multiple choice question (layout or site visit)
        elif ("layout" in response_lower or "details" in response_lower) and ("site visit" in response_lower or "schedule" in response_lower):
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
        if re.search(r"\bsend.*location\b|\bhere.*location\b", response_lower) and state.get("waiting_for") != "location_confirmation":
            logging.info(f"📍 Legacy location trigger for {from_phone}")
            send_whatsapp_location(from_phone)

        elif re.search(r"\bhere.*brochure\b|\bsending.*brochure\b", response_lower) and state.get("waiting_for") != "brochure_confirmation":
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
