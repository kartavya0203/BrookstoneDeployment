"""
Brookstone WhatsApp RAG Bot with LangChain Memory
A conversational AI assistant for real estate inquiries with memory and context awareness.
"""

import os
import re
import logging
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Set, Any, Optional

# Web Framework
from flask import Flask, request, jsonify

# HTTP Requests
import requests

# Environment Variables
from dotenv import load_dotenv

# LangChain Core Components
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

# LangChain OpenAI Integration
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Pinecone Vector Store
from langchain_community.vectorstores import Pinecone as LangchainPinecone

# LangChain Memory Components
from langchain.memory import (
    ConversationSummaryBufferMemory,
    ConversationEntityMemory,
)
from langchain.chains import ConversationChain
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ================================================
# ENVIRONMENT VARIABLES
# ================================================
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "brookstone_verify_token_2024")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
BROCHURE_URL = os.getenv(
    "BROCHURE_URL",
    "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/BROOKSTONE.pdf"
)

# Media state configuration
MEDIA_STATE_FILE = os.path.join(os.path.dirname(__file__), "media_state.json")
MEDIA_ID: Optional[str] = None
MEDIA_EXPIRY_DAYS = 29  # Refresh media every 29 days

# Pinecone configuration
INDEX_NAME = "brookstone-faq-json"

# ================================================
# TYPE DEFINITIONS
# ================================================
UserProfile = Dict[str, Any]
MediaState = Dict[str, str]

# ================================================
# MEDIA MANAGEMENT FUNCTIONS
# ================================================
def load_media_state() -> MediaState:
    """Load media state (media_id and uploaded_at) from file."""
    try:
        if os.path.exists(MEDIA_STATE_FILE):
            with open(MEDIA_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"❌ Error loading media state: {e}")
    return {}


def save_media_state(state: MediaState) -> None:
    """Save media state to file."""
    try:
        with open(MEDIA_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"❌ Error saving media state: {e}")


def _find_brochure_file() -> Optional[str]:
    """Find brochure file in common locations."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "static", "brochure", "BROOKSTONE.pdf"),
        os.path.join(os.path.dirname(__file__), "static", "BROCHURE.pdf"),
        os.path.join(os.path.dirname(__file__), "BROOKSTONE.pdf"),
    ]
    for candidate_path in candidates:
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def upload_brochure_media() -> Optional[str]:
    """Upload brochure PDF to WhatsApp Cloud and store media id."""
    global MEDIA_ID
    
    file_path = _find_brochure_file()
    if not file_path:
        logger.error("❌ Brochure file not found in static folders. Skipping media upload.")
        return None

    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/media"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    data = {"messaging_product": "whatsapp"}
    
    try:
        with open(file_path, "rb") as fh:
            files = {"file": (os.path.basename(file_path), fh, "application/pdf")}
            resp = requests.post(url, headers=headers, data=data, files=files, timeout=60)
        
        if resp.status_code == 200:
            response_data = resp.json()
            new_media_id = response_data.get("id")
            if new_media_id:
                MEDIA_ID = new_media_id
                state = {
                    "media_id": MEDIA_ID,
                    "uploaded_at": datetime.now(timezone.utc).isoformat()
                }
                save_media_state(state)
                logger.info(f"✅ Uploaded brochure media, id={MEDIA_ID}")
                return MEDIA_ID
            else:
                logger.error(f"❌ Upload succeeded but no media id returned: {resp.text}")
        else:
            logger.error(f"❌ Failed to upload media: {resp.status_code} - {resp.text}")
    except Exception as e:
        logger.error(f"❌ Exception uploading media: {e}")
    
    return None


def ensure_media_up_to_date() -> None:
    """Ensure we have a media_id and it's not expired."""
    global MEDIA_ID
    
    state = load_media_state()
    media_id = state.get("media_id")
    uploaded_at = state.get("uploaded_at")
    need_upload = True
    
    if media_id and uploaded_at:
        try:
            uploaded_dt = datetime.fromisoformat(uploaded_at)
            if datetime.now(timezone.utc) - uploaded_dt < timedelta(days=MEDIA_EXPIRY_DAYS):
                MEDIA_ID = media_id
                need_upload = False
                logger.info(f"ℹ️ Using existing media_id (uploaded {uploaded_at})")
        except Exception:
            need_upload = True

    if need_upload:
        logger.info("ℹ️ Uploading brochure media to WhatsApp Cloud (initial/refresh)")
        upload_brochure_media()


# Initialize media state at startup
try:
    ensure_media_up_to_date()
    logger.info("📱 Media management initialized. Use refresh_media.py for 29-day renewals.")
except Exception as e:
    logger.error(f"❌ Error initializing media: {e}")

# Validate API keys
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    logger.error("❌ Missing API keys! Check your .env file.")

# ================================================
# PINECONE SETUP
# ================================================
def load_vectorstore() -> LangchainPinecone:
    """Initialize Pinecone vector store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Initialize Pinecone with the modern API
    from pinecone import Pinecone as PineconeClient
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    # Use LangChain's Pinecone vectorstore
    return LangchainPinecone(index, embeddings, text_key="text")


# Initialize vectorstore and retriever
retriever = None
try:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 100})
    logger.info("✅ Pinecone vectorstore loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading Pinecone: {e}")

# ================================================
# LLM SETUP
# ================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
translator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ================================================
# MEMORY SYSTEM
# ================================================
CONVERSATION_MEMORIES: Dict[str, ConversationSummaryBufferMemory] = {}
USER_PROFILES: Dict[str, UserProfile] = {}


def get_or_create_memory(from_phone: str) -> tuple[ConversationSummaryBufferMemory, UserProfile]:
    """Get or create a LangChain memory instance for a user."""
    if from_phone not in CONVERSATION_MEMORIES:
        # Create summary buffer memory
        CONVERSATION_MEMORIES[from_phone] = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=800,
            return_messages=True,
            memory_key="chat_history",
            human_prefix="User",
            ai_prefix="Assistant"
        )
        
        # Initialize user profile
        USER_PROFILES[from_phone] = {
            "language": "english",
            "interests": set(),
            "stage": "initial",
            "last_intent": None,
            "visit_scheduled": False,
            "brochure_sent": False,
            "location_sent": False,
            "first_interaction": True
        }
    
    return CONVERSATION_MEMORIES[from_phone], USER_PROFILES[from_phone]


def analyze_user_intent(message_text: str, user_profile: UserProfile) -> List[str]:
    """Analyze user message to detect intents using semantic understanding."""
    message_lower = message_text.lower()
    
    # Intent detection keywords
    intents = {
        "location_request": any(word in message_lower for word in [
            "location", "address", "where", "place", "map",
            "કયાં", "સરનામું", "લોકેશન", "એડ્રેસ", "સ્થળ"
        ]),
        "brochure_request": any(word in message_lower for word in [
            "brochure", "details", "information", "catalog",
            "બ્રોશર", "વિગત", "માહિતી", "ડિટેલ્સ"
        ]),
        "pricing_inquiry": any(word in message_lower for word in [
            "price", "cost", "budget", "rate", "amount",
            "કિંમત", "ભાવ", "દર", "રેટ"
        ]),
        "amenities_inquiry": any(word in message_lower for word in [
            "amenities", "facilities", "features", "gym", "pool",
            "સુવિધા", "સુવિધાઓ", "જીમ", "પૂલ"
        ]),
        "visit_request": any(word in message_lower for word in [
            "visit", "see", "tour", "show", "book", "appointment",
            "મુલાકાત", "જોવા", "દેખાડો", "બતાવો"
        ]),
        "positive_response": any(word in message_lower for word in [
            "yes", "okay", "sure", "good", "fine", "right",
            "હા", "જોઈએ", "બરાબર", "સારું"
        ]),
        "unit_inquiry": any(word in message_lower for word in [
            "3bhk", "4bhk", "bedroom", "bhk", "flat", "apartment",
            "બેડરૂમ", "ફ્લેટ"
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


def create_dynamic_prompt_template() -> PromptTemplate:
    """Create a dynamic prompt template based on conversation context."""
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
        input_variables=[
            "chat_history", "user_stage", "user_interests", "language",
            "last_intent", "intent_actions", "stage_approach", "context", "input"
        ],
        template=template
    )


def build_context_variables(
    user_profile: UserProfile,
    detected_intents: List[str],
    context: str
) -> Dict[str, str]:
    """Build dynamic context variables for the prompt."""
    
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
def translate_gujarati_to_english(text: str) -> str:
    """Translate Gujarati text to English."""
    try:
        translation_prompt = f"""Translate the following Gujarati text to English. Provide only the English translation, nothing else.

Gujarati text: {text}

English translation:"""
        
        response = translator_llm.invoke(translation_prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"❌ Error translating Gujarati to English: {e}")
        return text


def translate_english_to_gujarati(text: str) -> str:
    """Translate English text to Gujarati."""
    try:
        translation_prompt = f"""Translate the following English text to Gujarati. Keep the same tone, style, and LENGTH - make it brief and concise like the original. Provide only the Gujarati translation, nothing else.

English text: {text}

Gujarati translation (keep it brief and concise):"""
        
        response = translator_llm.invoke(translation_prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"❌ Error translating English to Gujarati: {e}")
        return text


# ================================================
# WHATSAPP FUNCTIONS
# ================================================
def send_whatsapp_text(to_phone: str, message: str) -> None:
    """Send text message via WhatsApp."""
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "text",
        "text": {"body": message}
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            logger.info(f"✅ Message sent to {to_phone}")
        else:
            logger.error(f"❌ Failed to send message: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"❌ Error sending message: {e}")


def send_whatsapp_location(to_phone: str) -> None:
    """Send location via WhatsApp."""
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
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
            logger.info(f"✅ Location sent to {to_phone}")
        else:
            logger.error(f"❌ Failed to send location: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"❌ Error sending location: {e}")


def send_whatsapp_document(to_phone: str, caption: str = "Here is your Brookstone Brochure 📄") -> None:
    """Send document via WhatsApp."""
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }

    if MEDIA_ID:
        payload = {
            "messaging_product": "whatsapp",
            "to": to_phone,
            "type": "document",
            "document": {
                "id": MEDIA_ID,
                "caption": caption,
                "filename": "Brookstone_Brochure.pdf"
            }
        }
    else:
        payload = {
            "messaging_product": "whatsapp",
            "to": to_phone,
            "type": "document",
            "document": {
                "link": BROCHURE_URL,
                "caption": caption,
                "filename": "Brookstone_Brochure.pdf"
            }
        }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            logger.info(f"✅ Document sent to {to_phone}")
        else:
            logger.error(f"❌ Failed to send document: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"❌ Error sending document: {e}")


def mark_message_as_read(message_id: str) -> None:
    """Mark message as read on WhatsApp."""
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id
    }
    
    try:
        requests.post(url, headers=headers, json=payload, timeout=10)
    except Exception as e:
        logger.error(f"❌ Error marking message as read: {e}")


# ================================================
# MESSAGE PROCESSING
# ================================================
def process_incoming_message(from_phone: str, message_text: str, message_id: str) -> None:
    """Process incoming message with memory and context."""
    # Get or create memory and user profile
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
        memory.save_context({"input": message_text}, {"output": welcome_text})
        return

    # Analyze user intent
    detected_intents = analyze_user_intent(message_text, user_profile)
    
    logger.info(
        f"📱 Processing: {from_phone} | Message: {message_text} | "
        f"Language: {user_profile['language']} | Intents: {detected_intents} | "
        f"Stage: {user_profile['stage']}"
    )

    if not retriever:
        error_text = "I'm experiencing technical difficulties. Please contact our team at 8238477697 or 9974812701."
        if user_profile["language"] == "gujarati":
            error_text = translate_english_to_gujarati(error_text)
        send_whatsapp_text(from_phone, error_text)
        return

    try:
        # Translate Gujarati to English for search
        search_query = message_text
        if user_profile["language"] == "gujarati":
            search_query = translate_gujarati_to_english(message_text)
            logger.info(f"🔄 Translated query: {search_query}")

        # Retrieve relevant context
        docs = retriever.invoke(search_query)
        logger.info(f"📚 Retrieved {len(docs)} documents")

        # Build context from documents
        context = "\n\n".join([
            (d.page_content or "") + 
            ("\n" + "\n".join(f"{k}: {v}" for k, v in (d.metadata or {}).items()))
            for d in docs
        ])

        # Build context variables
        context_vars = build_context_variables(user_profile, detected_intents, context)
        
        # Create conversation chain
        prompt_template = create_dynamic_prompt_template()
        conversation_chain = ConversationChain(
            llm=llm,
            prompt=prompt_template,
            memory=memory,
            verbose=False
        )

        # Get response
        response = conversation_chain.predict(input=message_text, **context_vars)
        logger.info(f"🧠 LLM Response: {response}")

        # Translate if needed
        final_response = response
        if user_profile["language"] == "gujarati":
            final_response = translate_english_to_gujarati(response)
            logger.info(f"🔄 Translated response: {final_response}")

        # Clean response
        clean_response = (
            final_response
            .replace("SEND_LOCATION_NOW", "")
            .replace("SEND_BROCHURE_NOW", "")
            .strip()
        )
        
        # Send main response
        send_whatsapp_text(from_phone, clean_response)

        # Handle automatic actions
        if "SEND_LOCATION_NOW" in response or "location_request" in detected_intents:
            send_whatsapp_location(from_phone)
            user_profile["location_sent"] = True
            logger.info(f"📍 Location sent to {from_phone}")
            
        if "SEND_BROCHURE_NOW" in response or "brochure_request" in detected_intents:
            send_whatsapp_document(from_phone)
            user_profile["brochure_sent"] = True
            logger.info(f"📄 Brochure sent to {from_phone}")

        logger.info(f"✅ Conversation processed successfully for {from_phone}")

    except Exception as e:
        logger.error(f"❌ Error processing message: {e}")
        error_text = "I encountered an issue. Please contact our team at 8238477697 or 9974812701."
        if user_profile["language"] == "gujarati":
            error_text = translate_english_to_gujarati(error_text)
        send_whatsapp_text(from_phone, error_text)


# ================================================
# WEBHOOK ROUTES
# ================================================
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """Verify webhook for WhatsApp."""
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logger.info("✅ WEBHOOK VERIFIED")
        return challenge, 200
    else:
        return "Forbidden", 403


@app.route("/webhook", methods=["POST"])
def webhook():
    """Handle incoming webhook from WhatsApp."""
    data = request.get_json()
    logger.info("Incoming webhook data")

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
        logger.exception("❌ Error processing webhook")

    return jsonify({"status": "ok"}), 200


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "whatsapp_configured": bool(WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID),
        "openai_configured": bool(OPENAI_API_KEY),
        "pinecone_configured": bool(PINECONE_API_KEY)
    }), 200


@app.route("/", methods=["GET"])
def home():
    """Home endpoint with bot information."""
    return jsonify({
        "message": "Brookstone WhatsApp RAG Bot with LangChain Memory is running! 🚀",
        "memory_type": "ConversationSummaryBufferMemory",
        "features": [
            "Dynamic intent detection",
            "User profiling",
            "Conversation context",
            "Auto-translation"
        ],
        "brochure_url": BROCHURE_URL,
        "endpoints": {
            "webhook": "/webhook",
            "health": "/health"
        }
    }), 200


# ================================================
# RUN APP
# ================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info(f"🚀 Starting Brookstone WhatsApp Bot with LangChain Memory on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)