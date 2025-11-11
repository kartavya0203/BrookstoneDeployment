import os
import re
import logging
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
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
# LLM SETUP
# ================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
translator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
            "last_context_topics": [],
            "user_interests": [],
            "last_follow_up": None,  # Store the last follow-up question asked
            "follow_up_context": None,  # Store context for the follow-up
            "is_first_message": True  # Track if this is the first interaction
        }
    else:
        # Ensure all required fields exist
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

    # Analyze user interests for better follow-up questions
    current_interests = analyze_user_interests(message_text, state)
    
    logging.info(f"📱 Processing message from {from_phone}: {message_text} [Language: {state['language']}] [Interests: {current_interests}]")

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

        # Pre-check for direct Gujarati action keywords to ensure triggers are activated
        message_lower = message_text.lower()
        force_location = False
        force_brochure = False
        
        # Check for location keywords in both English and Gujarati
        location_keywords = [
            "location", "address", "where", "कहाँ", "कयाँ", 
            "કયાં", "સરનામું", "લોકેશન", "એડ્રેસ", "સ્થળ", "જગ્યા", 
            "કયાં આવેલું", "ક્યાં છે", "send me address", "મોકલો સરનામું"
        ]
        
        # Check for brochure keywords in both English and Gujarati  
        brochure_keywords = [
            "brochure", "details", "pdf", "document", "info", "information",
            "બ્રોશર", "મોકલો", "મોકલજો", "આપો", "મળશે", "વિગત", "માહિતી", 
            "ડિટેલ્સ", "કાગળ", "need brochure", "send brochure",
            "વિગત આપો", "માહિતી આપો", "બ્રોશર જોઈએ"
        ]
        
        if any(keyword in message_lower for keyword in location_keywords):
            force_location = True
            logging.info(f"🎯 Direct location keyword detected: {message_text}")
            
        if any(keyword in message_lower for keyword in brochure_keywords):
            force_brochure = True
            logging.info(f"🎯 Direct brochure keyword detected: {message_text}")

        docs = retriever.invoke(search_query)
        logging.info(f"📚 Retrieved {len(docs)} relevant documents")

        context = "\n\n".join(
            [(d.page_content or "") + ("\n" + "\n".join(f"{k}: {v}" for k, v in (d.metadata or {}).items())) for d in docs]
        )

        # Store current context topics for future reference
        state["last_context_topics"] = [d.metadata.get("topic", "") for d in docs if d.metadata.get("topic")]

        # Enhanced system prompt with user context and memory
        user_context = f"User's previous interests: {', '.join(state['user_interests'])}" if state['user_interests'] else "New conversation"
        
        # Include memory of last follow-up question and enhance for Gujarati handling
        follow_up_memory = ""
        if state.get("last_follow_up"):
            follow_up_memory = f"\nRECENT FOLLOW-UP: I recently asked '{state['last_follow_up']}' and user is now responding to that question. IMPORTANT: Understand their response in both English and Gujarati, and continue the conversation naturally based on their intent."
        
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

MEMORY CONTEXT: {follow_up_memory}

        GUJARATI FOLLOW-UP RESPONSE HANDLING:
- If user asks for location in Gujarati (કયાં આવેલું છે, સરનામું આપો, એડ્રેસ આપો), ALWAYS include SEND_LOCATION_NOW in response
- If user asks for brochure in Gujarati (બ્રોશર મોકલો, વિગત આપો, માહિતી આપો, ડિટેલ્સ આપો), ALWAYS include SEND_BROCHURE_NOW in response
- If user says "need brochure" or "બ્રોશર જોઈએ", ALWAYS include SEND_BROCHURE_NOW in response
- If user says "send me address" or "સરનામું મોકલો", ALWAYS include SEND_LOCATION_NOW in response
- If user responds positively (હા, જોઈએ, મોકલો, ભેજો, આપો), check previous context and send appropriate item
- Always continue the conversation after addressing their response with a new relevant questionFOLLOW-UP CONVERSATION EXAMPLES:
- After location: "Great! Our office is open 10:30 AM to 7:00 PM. Would you like to know about our luxury amenities? 🌟"
- After brochure: "Perfect! The brochure has all details. Any specific questions about our 3&4BHK flats? 🏠"
- After amenities: "Wonderful! We have premium facilities. Would you like to schedule a site visit? 📅"
- After pricing: "Excellent! Our agents will provide the best rates. Interested in seeing the actual flats? 🏠✨"

MANDATORY FLAT MENTIONS:
- ALWAYS say "Brookstone offers luxurious 3&4BHK flats" (mention both types)
- Even if user asks only about 3BHK or 4BHK, mention both options
- This showcases our complete offering

SPECIAL HANDLING (ENGLISH & GUJARATI):

1. TIMINGS: When user asks about office hours/timings in English or Gujarati (સમય, ખુલ્લું, ઓફિસ, કયારે):
   "Our site office is open from *10:30 AM to 7:00 PM* every day. Would you like me to send you the location? 📍"

2. SITE VISIT BOOKING: When user wants to visit/see property in English or Gujarati (મુલાકાત, જોવાનું, સાઇટ, વિઝિટ):
   "Perfect! Please contact *Mr. Nilesh at 7600612701* to book your site visit. 📞✨"

3. GENERAL QUERIES: When user asks general questions or needs help in English or Gujarati (મદદ, પ્રશ્ન, માહિતી):
   "You can contact our agents at 8238477697 or 9974812701 for any queries. 📱😊"

4. PRICING: When user asks about rates/cost in English or Gujarati (કિંમત, ભાવ, રેટ, દર, પૈસા):
   Check context first. If no pricing info: "For latest pricing details, please contact our agents at 8238477697 or 9974812701. 💰📞"

        INTELLIGENT ACTION TRIGGERS:
When user asks for location/address in any language, include: "SEND_LOCATION_NOW" in your response
When user asks for brochure/details in any language, include: "SEND_BROCHURE_NOW" in your response

GUJARATI KEYWORDS TO DETECT:
- Location requests: કયાં, સરનામું, લોકેશન, એડ્રેસ, સ્થળ, જગ્યા, કયાં આવેલું, ક્યાં છે
- Brochure requests: બ્રોશર, મોકલો, મોકલજો, આપો, મળશે, વિગત, માહિતી, ડિટેલ્સ, કાગળ
- General positive responses: હા, હા જી, જોઈએ, બતાવો, જણાવો, ભેજો, મળે, આપશો

EXAMPLES:
- User: "Where is it located?" or "કયાં આવેલું છે?" or "સરનામું આપો" → Response: "SEND_LOCATION_NOW Great! Here's our location in Bopal, Ahmedabad. Would you like to know our office timings? 📍"
- User: "Send me brochure" or "બ્રોશર મોકલો" or "વિગત આપો" → Response: "SEND_BROCHURE_NOW Perfect! Here's our detailed brochure with all information about luxury 3&4BHK flats. Any specific questions? 📄✨"
- User: "need brochure" → Response: "SEND_BROCHURE_NOW Perfect! Here's our detailed brochure with all information about luxury 3&4BHK flats. Any specific questions? 📄✨"IMPORTANT: The system will automatically detect these triggers and send the actual location/brochure via WhatsApp API, so you just need to include the trigger words in your response.

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

CONVERSATION FLOW (ENHANCED FOR GUJARATI):
- If user is answering my previous question in English or Gujarati, understand their intent and provide relevant info
- For Gujarati responses like "હા" (yes), "જોઈએ" (want), "બતાવો" (show me), understand they're agreeing and continue accordingly
- For Gujarati responses like "ના" (no), "પછી" (later), "નહીં" (not now), understand they're declining and offer alternatives
- After addressing their response, naturally continue with another relevant question
- Keep the conversation engaging and helpful in both languages
- Always sound excited about Brookstone!

GUJARATI CONVERSATION EXAMPLES:
User says "હા" (yes) to location → Send location + "Great! Our office timings are 10:30 AM to 7:00 PM. Would you like to know about our amenities? 🌟"
User says "જોઈએ" (want) to brochure → Send brochure + "Perfect! Any specific questions about our luxury 3&4BHK flats? 🏠✨"
User says "મુલાકાત" (visit) → "Wonderful! Please contact Mr. Nilesh at 7600612701 to schedule. What interests you most about our flats? 💎"

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

        response = llm.invoke(system_prompt).content.strip()
        logging.info(f"🧠 LLM Response: {response}")

        # Translate response to Gujarati if user language is Gujarati
        final_response = response
        if state["language"] == "gujarati":
            final_response = translate_english_to_gujarati(response)
            logging.info(f"🔄 Translated response: {final_response}")

        # --- Send primary text response ---
        # Remove action triggers from the response before sending to user
        clean_response = final_response.replace("SEND_LOCATION_NOW", "").replace("SEND_BROCHURE_NOW", "").strip()
        send_whatsapp_text(from_phone, clean_response)

        # --- Handle intelligent actions based on LLM response triggers OR forced detection ---
        if "SEND_LOCATION_NOW" in final_response or force_location:
            send_whatsapp_location(from_phone)
            logging.info(f"📍 Location sent to {from_phone} - LLM trigger: {'SEND_LOCATION_NOW' in final_response}, Force: {force_location}")
            
        if "SEND_BROCHURE_NOW" in final_response or force_brochure:
            send_whatsapp_document(from_phone)
            logging.info(f"📄 Brochure sent to {from_phone} - LLM trigger: {'SEND_BROCHURE_NOW' in final_response}, Force: {force_brochure}")

        # Store the follow-up question asked by the bot for memory
        # Extract follow-up question from response (look for question marks)
        sentences = clean_response.split('.')
        follow_up_question = None
        for sentence in sentences:
            if '?' in sentence:
                follow_up_question = sentence.strip()
                break
        
        if follow_up_question:
            state["last_follow_up"] = follow_up_question
            state["follow_up_context"] = context[:500]  # Store some context for reference
            logging.info(f"🧠 Stored follow-up: {follow_up_question}")

        state["chat_history"].append({"role": "assistant", "content": clean_response})

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
