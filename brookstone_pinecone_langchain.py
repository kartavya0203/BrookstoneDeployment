import os
import re
import logging
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory, ConversationEntityMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import json
from datetime import datetime, timedelta

load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ================================================
# ENVIRONMENT VARIABLES
# ================================================
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "brookstone_verify_token_2024")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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
    logging.info(f"📱 Media management initialized. Use refresh_media.py for 29-day renewals.")
except Exception as e:
    logging.error(f"❌ Error initializing media: {e}")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    logging.error("❌ Missing API keys!")

# ================================================
# PINECONE SETUP
# ================================================
INDEX_NAME = "brookstone-faq-json"

def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

try:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 100})
    logging.info("✅ Pinecone vectorstore loaded successfully")
except Exception as e:
    logging.error(f"❌ Error loading Pinecone: {e}")
    retriever = None

# ================================================
# LLM SETUP WITH LANGCHAIN MEMORY
# ================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
translator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ================================================
# LANGCHAIN MEMORY SYSTEM
# ================================================
CONVERSATION_MEMORIES = {}
USER_PROFILES = {}

def get_or_create_memory(from_phone):
    """Get or create a LangChain memory instance for a user"""
    if from_phone not in CONVERSATION_MEMORIES:
        # Create a summary buffer memory that keeps recent messages and summarizes older ones
        CONVERSATION_MEMORIES[from_phone] = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=800,  # Keep around 800 tokens of recent conversation
            return_messages=True,
            memory_key="chat_history",
            human_prefix="User",
            ai_prefix="Assistant"
        )
        
        # Initialize user profile
        USER_PROFILES[from_phone] = {
            "language": "english",
            "interests": set(),
            "stage": "initial",  # initial, interested, serious, ready_to_visit
            "last_intent": None,
            "visit_scheduled": False,
            "brochure_sent": False,
            "location_sent": False,
            "first_interaction": True
        }
    
    return CONVERSATION_MEMORIES[from_phone], USER_PROFILES[from_phone]

def analyze_user_intent(message_text, user_profile):
    """Analyze user message to detect intents using semantic understanding"""
    message_lower = message_text.lower()
    
    # Dynamic intent detection without hardcoding
    intents = {
        "location_request": any(word in message_lower for word in [
            "location", "address", "where", "place", "map", "કયાં", "સરનામું", "લોકેશન", "એડ્રેસ", "સ્થળ"
        ]),
        "brochure_request": any(word in message_lower for word in [
            "brochure", "details", "information", "catalog", "બ્રોશર", "વિગત", "માહિતી", "ડિટેલ્સ"
        ]),
        "pricing_inquiry": any(word in message_lower for word in [
            "price", "cost", "budget", "rate", "amount", "કિંમત", "ભાવ", "દર", "રેટ"
        ]),
        "amenities_inquiry": any(word in message_lower for word in [
            "amenities", "facilities", "features", "gym", "pool", "સુવિધા", "સુવિધાઓ", "જીમ", "પૂલ"
        ]),
        "visit_request": any(word in message_lower for word in [
            "visit", "see", "tour", "show", "book", "appointment", "મુલાકાત", "જોવા", "દેખાડો", "બતાવો"
        ]),
        "positive_response": any(word in message_lower for word in [
            "yes", "okay", "sure", "good", "fine", "right", "હા", "જોઈએ", "બરાબર", "સારું"
        ]),
        "unit_inquiry": any(word in message_lower for word in [
            "3bhk", "4bhk", "bedroom", "bhk", "flat", "apartment", "બેડરૂમ", "ફ્લેટ"
        ])
    }
    
    # Update user profile with detected intents
    detected_intents = [intent for intent, detected in intents.items() if detected]
    for intent in detected_intents:
        user_profile["interests"].add(intent)
    
    # Update user stage based on interaction patterns
    interest_count = len(user_profile["interests"])
    if "visit_request" in user_profile["interests"]:
        user_profile["stage"] = "ready_to_visit"
    elif interest_count >= 3:
        user_profile["stage"] = "serious"
    elif interest_count >= 1:
        user_profile["stage"] = "interested"
    
    # Store last intent for context
    if detected_intents:
        user_profile["last_intent"] = detected_intents[0]
    
    return detected_intents

def create_dynamic_prompt_template():
    """Create a dynamic prompt template based on conversation context"""
    template = """You are a friendly, professional real estate assistant for Brookstone - a luxury residential project in Ahmedabad.

CONVERSATION CONTEXT:
{chat_history}

USER PROFILE INSIGHTS:
- User Stage: {user_stage}
- Interests: {user_interests}
- Language: {language}
- Last Intent: {last_intent}

CORE PRINCIPLES:
- Be conversational, warm, and professional
- Keep responses concise (2-3 sentences max)
- Always mention "Brookstone offers luxurious 3&4BHK flats" when discussing units
- Use 2-3 relevant emojis per response
- Ask ONE engaging follow-up question
- Be helpful but not pushy

INTENT-SPECIFIC ACTIONS:
{intent_actions}

STAGE-SPECIFIC APPROACH:
{stage_approach}

STANDARD INFO:
- Office Hours: 10:30 AM to 7:00 PM daily
- Site Visits: Mr. Nilesh at 7600612701
- General Queries: 8238477697 or 9974812701

RELEVANT KNOWLEDGE:
{context}

Current User Message: {input}

Assistant Response (respond in English - will be auto-translated if needed):"""
    
    return PromptTemplate(
        input_variables=["chat_history", "user_stage", "user_interests", "language", 
                        "last_intent", "intent_actions", "stage_approach", "context", "input"],
        template=template
    )

def build_context_variables(user_profile, detected_intents, context):
    """Build dynamic context variables for the prompt"""
    
    # Build intent-specific actions
    intent_actions = []
    if "location_request" in detected_intents:
        intent_actions.append("🎯 USER WANTS LOCATION: Include 'SEND_LOCATION_NOW' and provide address")
    if "brochure_request" in detected_intents:
        intent_actions.append("🎯 USER WANTS BROCHURE: Include 'SEND_BROCHURE_NOW' and describe brochure")
    if "visit_request" in detected_intents:
        intent_actions.append("🎯 USER WANTS VISIT: Provide Mr. Nilesh's contact (7600612701)")
    if "pricing_inquiry" in detected_intents:
        intent_actions.append("🎯 PRICING QUERY: Direct to agents (8238477697/9974812701)")
    if "positive_response" in detected_intents:
        intent_actions.append("🎯 POSITIVE RESPONSE: Continue conversation naturally based on context")
    
    # Build stage-specific approach
    stage_approaches = {
        "initial": "Be welcoming and introduce key features. Ask about preferences.",
        "interested": "Build on their interest. Provide specific details they seek.",
        "serious": "Focus on benefits and move towards scheduling a visit.",
        "ready_to_visit": "Facilitate visit booking and maintain excitement."
    }
    
    return {
        "user_stage": user_profile["stage"],
        "user_interests": ", ".join(user_profile["interests"]) or "None yet",
        "language": user_profile["language"],
        "last_intent": user_profile.get("last_intent", "None"),
        "intent_actions": "\n".join(intent_actions) or "No specific actions needed",
        "stage_approach": stage_approaches.get(user_profile["stage"], "Be helpful and engaging"),
        "context": context
    }

# ================================================
# TRANSLATION FUNCTIONS
# ================================================
def translate_gujarati_to_english(text):
    """Translate Gujarati text to English"""
    try:
        translation_prompt = f"""
Translate the following Gujarati text to English. Provide only the English translation, nothing else.

Gujarati text: {text}

English translation:
        """
        response = translator_llm.invoke(translation_prompt)
        return response.content.strip()
    except Exception as e:
        logging.error(f"❌ Error translating Gujarati to English: {e}")
        return text  # Return original text if translation fails

def translate_english_to_gujarati(text):
    """Translate English text to Gujarati"""
    try:
        translation_prompt = f"""
Translate the following English text to Gujarati. Keep the same tone, style, and LENGTH - make it brief and concise like the original. Provide only the Gujarati translation, nothing else.

English text: {text}

Gujarati translation (keep it brief and concise):
        """
        response = translator_llm.invoke(translation_prompt)
        return response.content.strip()
    except Exception as e:
        logging.error(f"❌ Error translating English to Gujarati: {e}")
        return text  # Return original text if translation fails

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
# MESSAGE PROCESSING WITH LANGCHAIN MEMORY
# ================================================
def process_incoming_message(from_phone, message_text, message_id):
    # Get or create LangChain memory and user profile
    memory, user_profile = get_or_create_memory(from_phone)
    
    # Detect language
    gujarati_chars = any("\u0A80" <= c <= "\u0AFF" for c in message_text)
    user_profile["language"] = "gujarati" if gujarati_chars else "english"
    
    # Handle first interaction
    if user_profile.get("first_interaction", True):
        user_profile["first_interaction"] = False
        welcome_text = "Hello! Welcome to Brookstone 🏠✨ How can I assist you with our luxurious 3&4BHK flats today?"
        
        if user_profile["language"] == "gujarati":
            welcome_text = translate_english_to_gujarati(welcome_text)
        
        send_whatsapp_text(from_phone, welcome_text)
        # Add welcome to memory
        memory.save_context({"input": message_text}, {"output": welcome_text})
        return

    # Analyze user intent
    detected_intents = analyze_user_intent(message_text, user_profile)
    
    logging.info(f"📱 Processing: {from_phone} | Message: {message_text} | Language: {user_profile['language']} | Intents: {detected_intents} | Stage: {user_profile['stage']}")

    if not retriever:
        error_text = "I'm experiencing technical difficulties. Please contact our team at 8238477697 or 9974812701."
        if user_profile["language"] == "gujarati":
            error_text = translate_english_to_gujarati(error_text)
        send_whatsapp_text(from_phone, error_text)
        return

    try:
        # Translate Gujarati to English for Pinecone search
        search_query = message_text
        if user_profile["language"] == "gujarati":
            search_query = translate_gujarati_to_english(message_text)
            logging.info(f"🔄 Translated query: {search_query}")

        # Retrieve relevant context from Pinecone
        docs = retriever.invoke(search_query)
        logging.info(f"📚 Retrieved {len(docs)} documents")

        # Build context from retrieved documents
        context = "\n\n".join([
            (d.page_content or "") + ("\n" + "\n".join(f"{k}: {v}" for k, v in (d.metadata or {}).items()))
            for d in docs
        ])

        # Build dynamic context variables
        context_vars = build_context_variables(user_profile, detected_intents, context)
        
        # Create conversation chain with dynamic prompt
        prompt_template = create_dynamic_prompt_template()
        conversation_chain = ConversationChain(
            llm=llm,
            prompt=prompt_template,
            memory=memory,
            verbose=False
        )

        # Get response from LangChain conversation chain
        response = conversation_chain.predict(
            input=message_text,
            **context_vars
        )
        
        logging.info(f"🧠 LLM Response: {response}")

        # Translate response if needed
        final_response = response
        if user_profile["language"] == "gujarati":
            final_response = translate_english_to_gujarati(response)
            logging.info(f"🔄 Translated response: {final_response}")

        # Clean response from action triggers
        clean_response = final_response.replace("SEND_LOCATION_NOW", "").replace("SEND_BROCHURE_NOW", "").strip()
        
        # Send main response
        send_whatsapp_text(from_phone, clean_response)

        # Handle automatic actions based on triggers or intents
        if "SEND_LOCATION_NOW" in response or "location_request" in detected_intents:
            send_whatsapp_location(from_phone)
            user_profile["location_sent"] = True
            logging.info(f"📍 Location sent to {from_phone}")
            
        if "SEND_BROCHURE_NOW" in response or "brochure_request" in detected_intents:
            send_whatsapp_document(from_phone)
            user_profile["brochure_sent"] = True
            logging.info(f"📄 Brochure sent to {from_phone}")

        logging.info(f"✅ Conversation processed successfully for {from_phone}")

    except Exception as e:
        logging.error(f"❌ Error processing message: {e}")
        error_text = "I encountered an issue. Please contact our team at 8238477697 or 9974812701."
        if user_profile["language"] == "gujarati":
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
        "openai_configured": bool(OPENAI_API_KEY),
        "pinecone_configured": bool(PINECONE_API_KEY)
    }), 200

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Brookstone WhatsApp RAG Bot with LangChain Memory is running! 🚀",
        "memory_type": "ConversationSummaryBufferMemory",
        "features": ["Dynamic intent detection", "User profiling", "Conversation context", "Auto-translation"],
        "brochure_url": BROCHURE_URL,
        "endpoints": {"webhook": "/webhook", "health": "/health"}
    }), 200

# ================================================
# RUN APP
# ================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Starting Brookstone WhatsApp Bot with LangChain Memory on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)