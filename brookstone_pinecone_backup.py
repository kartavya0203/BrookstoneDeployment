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
    logging.info(f"� Media management initialized. Use refresh_media.py for 29-day renewals.")
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
# CONVERSATION STATE & CONTEXT ANALYSIS
# ================================================
CONV_STATE = {}

def ensure_conversation_state(from_phone):
    """Ensure conversation state has all required fields"""
    if from_phone not in CONV_STATE:
        CONV_STATE[from_phone] = {
            "chat_history": [], 
            "language": "english", 
            "conversation_context": "",  # Summary of ongoing conversation topic
            "last_bot_question": "",     # Last question asked by bot
            "user_profile": {            # Build user profile over time
                "interests": set(),
                "preferences": {},
                "stage": "initial"       # initial, interested, serious, ready_to_visit
            },
            "conversation_flow": [],     # Track the flow of conversation
            "is_first_message": True
        }
    else:
        # Ensure all required fields exist for backwards compatibility
        if "conversation_context" not in CONV_STATE[from_phone]:
            CONV_STATE[from_phone]["conversation_context"] = ""
        if "last_bot_question" not in CONV_STATE[from_phone]:
            CONV_STATE[from_phone]["last_bot_question"] = ""
        if "user_profile" not in CONV_STATE[from_phone]:
            CONV_STATE[from_phone]["user_profile"] = {
                "interests": set(),
                "preferences": {},
                "stage": "initial"
            }
        if "conversation_flow" not in CONV_STATE[from_phone]:
            CONV_STATE[from_phone]["conversation_flow"] = []

def analyze_conversation_intent(message_text, state):
    """Analyze user message to understand intent and context"""
    # Convert message to lowercase for analysis
    message_lower = message_text.lower()
    
    # Define intent categories dynamically based on message content
    intents = {
        "location_request": False,
        "brochure_request": False,
        "pricing_inquiry": False,
        "amenities_inquiry": False,
        "visit_request": False,
        "general_inquiry": False,
        "positive_response": False,
        "negative_response": False,
        "specific_unit_inquiry": False
    }
    
    # Simple intent detection without hardcoding specific keywords
    # This uses semantic understanding rather than exact keyword matching
    if any(word in message_lower for word in ["location", "address", "where", "place", "કયાં", "સરનામું", "લોકેશન", "એડ્રેસ", "સ્થળ"]):
        intents["location_request"] = True
    
    if any(word in message_lower for word in ["brochure", "details", "information", "બ્રોશર", "વિગત", "માહિતી", "ડિટેલ્સ"]):
        intents["brochure_request"] = True
    
    if any(word in message_lower for word in ["price", "cost", "budget", "rate", "કિંમત", "ભાવ", "દર", "રેટ"]):
        intents["pricing_inquiry"] = True
    
    if any(word in message_lower for word in ["amenities", "facilities", "gym", "pool", "સુવિધા", "સુવિધાઓ", "જીમ", "પૂલ"]):
        intents["amenities_inquiry"] = True
    
    if any(word in message_lower for word in ["visit", "see", "tour", "show", "મુલાકાત", "જોવા", "દેખાડો", "બતાવો"]):
        intents["visit_request"] = True
    
    if any(word in message_lower for word in ["yes", "okay", "sure", "good", "હા", "જોઈએ", "બરાબર", "સારું"]):
        intents["positive_response"] = True
    
    if any(word in message_lower for word in ["no", "not", "later", "ના", "નહીં", "પછી"]):
        intents["negative_response"] = True
    
    if any(word in message_lower for word in ["3bhk", "4bhk", "bedroom", "bhk", "બેડરૂમ"]):
        intents["specific_unit_inquiry"] = True
    
    # If no specific intent detected, it's a general inquiry
    if not any(intents.values()):
        intents["general_inquiry"] = True
    
    # Update user profile based on intents
    for intent, detected in intents.items():
        if detected:
            state["user_profile"]["interests"].add(intent)
    
    return intents

def update_conversation_context(state, user_message, bot_response, intents):
    """Update conversation context and flow"""
    # Update conversation flow
    flow_entry = {
        "user_message": user_message,
        "intents": [k for k, v in intents.items() if v],
        "bot_response": bot_response[:100] + "..." if len(bot_response) > 100 else bot_response
    }
    state["conversation_flow"].append(flow_entry)
    
    # Keep only last 5 exchanges to manage memory
    if len(state["conversation_flow"]) > 5:
        state["conversation_flow"] = state["conversation_flow"][-5:]
    
    # Update conversation context summary
    primary_intents = [k for k, v in intents.items() if v]
    if primary_intents:
        state["conversation_context"] = f"User is interested in: {', '.join(primary_intents)}. Last discussed: {primary_intents[0]}"
    
    # Update user stage based on behavior
    interests_count = len(state["user_profile"]["interests"])
    if interests_count >= 3:
        state["user_profile"]["stage"] = "serious"
    elif interests_count >= 1:
        state["user_profile"]["stage"] = "interested"
    
    if "visit_request" in state["user_profile"]["interests"]:
        state["user_profile"]["stage"] = "ready_to_visit"

def build_conversation_memory(state):
    """Build a smart conversation memory summary"""
    if not state["conversation_flow"]:
        return "First conversation with user."
    
    # Get recent conversation flow
    recent_flow = state["conversation_flow"][-3:]  # Last 3 exchanges
    memory_parts = []
    
    # Add conversation stage context
    stage = state["user_profile"]["stage"]
    stage_context = {
        "initial": "User is just starting to explore",
        "interested": "User has shown interest in specific aspects",
        "serious": "User is seriously considering the property",
        "ready_to_visit": "User is ready for a site visit"
    }
    memory_parts.append(f"User Stage: {stage_context.get(stage, stage)}")
    
    # Add recent conversation topics
    if state["conversation_context"]:
        memory_parts.append(f"Current Context: {state['conversation_context']}")
    
    # Add last bot question if there was one
    if state["last_bot_question"]:
        memory_parts.append(f"Last Question Asked: {state['last_bot_question']}")
    
    # Add interest summary
    interests = list(state["user_profile"]["interests"])
    if interests:
        memory_parts.append(f"User has shown interest in: {', '.join(interests)}")
    
    return " | ".join(memory_parts)

def build_smart_system_prompt(state, context, search_query, conversation_memory, intents):
    """Build an intelligent system prompt based on conversation state"""
    
    # Language specific instructions
    language_instruction = ""
    if state["language"] == "gujarati":
        language_instruction = """
LANGUAGE: User is asking in Gujarati. Respond in ENGLISH first (keep it concise), it will be translated automatically.
"""
    
    # Intent-specific handling
    intent_handling = ""
    active_intents = [k for k, v in intents.items() if v]
    
    if "location_request" in active_intents:
        intent_handling += "\n🎯 USER WANTS LOCATION: Include 'SEND_LOCATION_NOW' in your response and provide address details."
    
    if "brochure_request" in active_intents:
        intent_handling += "\n🎯 USER WANTS BROCHURE: Include 'SEND_BROCHURE_NOW' in your response and mention brochure details."
    
    if "positive_response" in active_intents and state["last_bot_question"]:
        intent_handling += f"\n🎯 USER RESPONDED POSITIVELY to your question: '{state['last_bot_question']}' - Continue naturally based on what they agreed to."
    
    if "visit_request" in active_intents:
        intent_handling += "\n🎯 USER WANTS TO VISIT: Provide contact details for booking - Mr. Nilesh at 7600612701."
    
    if "pricing_inquiry" in active_intents:
        intent_handling += "\n🎯 USER ASKING ABOUT PRICING: Direct them to agents at 8238477697 or 9974812701 for latest rates."
    
    # Conversation stage specific guidance
    stage_guidance = ""
    user_stage = state["user_profile"]["stage"]
    
    if user_stage == "initial":
        stage_guidance = "\n📋 APPROACH: Be welcoming and introduce key features. Ask about their preferences."
    elif user_stage == "interested": 
        stage_guidance = "\n📋 APPROACH: Build on their interest. Provide specific details they're looking for."
    elif user_stage == "serious":
        stage_guidance = "\n📋 APPROACH: User is seriously interested. Focus on convincing and moving towards visit."
    elif user_stage == "ready_to_visit":
        stage_guidance = "\n📋 APPROACH: User is ready to visit. Facilitate the visit booking and maintain excitement."
    
    return f"""You are a friendly, professional real estate assistant for Brookstone - a luxury residential project.

CONVERSATION MEMORY: {conversation_memory}

{language_instruction}

CORE PRINCIPLES:
- Be conversational and natural
- Keep responses concise (2-3 sentences max)
- Always mention "Brookstone offers luxurious 3&4BHK flats" when discussing units
- Use 2-3 relevant emojis per response
- Ask ONE follow-up question to continue conversation
- Be convincing but not pushy

{intent_handling}

{stage_guidance}

SPECIAL ACTIONS:
- Include "SEND_LOCATION_NOW" when user asks for location/address
- Include "SEND_BROCHURE_NOW" when user asks for brochure/details
- These triggers will automatically send the actual location/brochure

STANDARD INFORMATION:
- Office Hours: 10:30 AM to 7:00 PM daily
- Site Visit Booking: Mr. Nilesh at 7600612701
- General Queries: 8238477697 or 9974812701

KNOWLEDGE BASE:
{context}

USER QUERY: {search_query}

Provide a natural, contextual response that continues the conversation flow:"""

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
    
    # Detect language
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

    # Analyze conversation intent for better context understanding
    intents = analyze_conversation_intent(message_text, state)
    
    logging.info(f"📱 Processing message from {from_phone}: {message_text} [Language: {state['language']}] [Intents: {[k for k,v in intents.items() if v]}] [Stage: {state['user_profile']['stage']}]")

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

        # Retrieve relevant documents
        docs = retriever.invoke(search_query)
        logging.info(f"📚 Retrieved {len(docs)} relevant documents")

        context = "\n\n".join(
            [(d.page_content or "") + ("\n" + "\n".join(f"{k}: {v}" for k, v in (d.metadata or {}).items())) for d in docs]
        )

        # Build conversation memory for better context
        conversation_memory = build_conversation_memory(state)
        
        # Build smart system prompt based on conversation state
        system_prompt = build_smart_system_prompt(state, context, search_query, conversation_memory, intents)

        # Get LLM response
        response = llm.invoke(system_prompt).content.strip()
        logging.info(f"🧠 LLM Response: {response}")

        # Translate response to Gujarati if needed
        final_response = response
        if state["language"] == "gujarati":
            final_response = translate_english_to_gujarati(response)
            logging.info(f"🔄 Translated response: {final_response}")

        # Remove action triggers from response before sending to user
        clean_response = final_response.replace("SEND_LOCATION_NOW", "").replace("SEND_BROCHURE_NOW", "").strip()
        
        # Send text response
        send_whatsapp_text(from_phone, clean_response)

        # Handle automatic actions based on intents or LLM triggers
        if "SEND_LOCATION_NOW" in response or intents.get("location_request", False):
            send_whatsapp_location(from_phone)
            logging.info(f"📍 Location sent to {from_phone} - Intent: {intents.get('location_request', False)}, LLM: {'SEND_LOCATION_NOW' in response}")
            
        if "SEND_BROCHURE_NOW" in response or intents.get("brochure_request", False):
            send_whatsapp_document(from_phone)
            logging.info(f"📄 Brochure sent to {from_phone} - Intent: {intents.get('brochure_request', False)}, LLM: {'SEND_BROCHURE_NOW' in response}")

        # Extract and store the follow-up question for memory
        sentences = clean_response.split('.')
        follow_up_question = None
        for sentence in sentences:
            if '?' in sentence:
                follow_up_question = sentence.strip()
                break
        
        if follow_up_question:
            state["last_bot_question"] = follow_up_question
            logging.info(f"🧠 Stored follow-up question: {follow_up_question}")

        # Update conversation context and memory
        update_conversation_context(state, message_text, clean_response, intents)
        
        # Add to chat history
        state["chat_history"].append({"role": "assistant", "content": clean_response})

        # Keep chat history manageable (last 10 messages)
        if len(state["chat_history"]) > 10:
            state["chat_history"] = state["chat_history"][-10:]

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
        "openai_configured": bool(OPENAI_API_KEY),
        "pinecone_configured": bool(PINECONE_API_KEY)
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
