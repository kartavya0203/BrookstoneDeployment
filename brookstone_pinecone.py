import os
import re
import logging
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================================================
# BROCHURE URL (public link to your PDF)
# ================================================
BROCHURE_URL = os.getenv("BROCHURE_URL", "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/BROOKSTONE.pdf")

# ================================================
# ENVIRONMENT VARIABLES
# ================================================
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "brookstone_verify_token_2024")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

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

# ================================================
# CONVERSATION STATE (In-memory)
# ================================================
CONV_STATE = {}

# ================================================
# STATIC RESPONSES
# ================================================
STATIC = {
    "timing_en": "Our site office is open from **10:30 AM to 7:00 PM** every day. Would you like me to send you the **location**?",
    "timing_gu": "અમારું સાઇટ ઑફિસ દરરોજ **સવારે 10:30 થી સાંજે 7:00 વાગ્યા સુધી** ખુલ્લું રહે છે. શું હું તમને લોકેશન મોકલું?",
    "location_en": "📍 *Brookstone Location*\n\nBrookstone, Vaikunth Bungalows,\nBeside DPS Bopal Rd, next to A. Shridhar Oxygen Park,\nBopal, Shilaj, Ahmedabad, Gujarat 380058\n\n⏰ *Site Office Hours*: 10:30 AM to 7:00 PM",
    "location_gu": "📍 *બ્રૂકસ્ટોન લોકેશન*\n\nબ્રૂકસ્ટોન, વૈકુંઠ બંગલોઝ,\nડીપીએસ બોપલ રોડ બાજુમાં, એ. શ્રીધર ઓક્સિજન પાર્ક પાસે,\nબોપલ, શિલજ, અમદાવાદ, ગુજરાત 380058\n\n⏰ *સાઇટ ઓફિસ સમય*: સવારે 10:30 થી સાંજે 7:00",
    "appointment_en": "Please contact **Mr. Nilesh at 7600612701** to book your site visit.",
    "appointment_gu": "તમારી સાઇટ વિઝિટ બુક કરવા માટે **શ્રી નિલેશ (7600612701)** ને સંપર્ક કરો.",
    "agent_en": "You can contact our agents directly on **8238477697** or **9974812701** for any in-person discussion.",
    "agent_gu": "વ્યક્તિગત ચર્ચા માટે કૃપા કરી **8238477697** અથવા **9974812701** પર અમારા એજન્ટ્સને સંપર્ક કરો.",
    "pricing_en": "For the latest pricing details, please contact our agents directly on **8238477697** or **9974812701**.",
    "pricing_gu": "તાજેતરના ભાવ માટે કૃપા કરી **8238477697** અથવા **9974812701** પર અમારા એજન્ટ્સને સંપર્ક કરો.",
}

# ================================================
# HELPER FUNCTIONS
# ================================================
def is_gujarati(txt):
    return any("\u0A80" <= c <= "\u0AFF" for c in txt)

def detect_static(txt):
    guj = is_gujarati(txt)
    t = txt.lower()

    if re.search(r"(site.*time|office.*hour)|સમય", t):
        return STATIC["timing_gu" if guj else "timing_en"]

    if re.search(r"(location|address|map)|લોકેશન|સરનામું", t):
        return STATIC["location_gu" if guj else "location_en"], "location"

    if re.search(r"(brochure|pdf|floor\s*plan|details)|બ્રોશર", t):
        return f"{'આ રહ્યો બ્રોશર' if guj else 'Here is the brochure:'}", "brochure"

    if re.search(r"(book|appointment|visit)|બુક|અપોઇન્ટમેન્ટ", t):
        return STATIC["appointment_gu" if guj else "appointment_en"]

    if re.search(r"(talk|speak|meet|discuss|call).*(person|agent)|વાત|સંપર્ક", t):
        return STATIC["agent_gu" if guj else "agent_en"]

    return None

def is_pricing(txt):
    return bool(re.search(r"(price|pricing|cost|rate|charges|sq\.?ft)|ભાવ|કિંમત|દર", txt.lower()))

# ================================================
# WHATSAPP API FUNCTIONS
# ================================================
def send_whatsapp_text(to_phone, message):
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
            logging.info(f"✅ Message sent to {to_phone}")
            return True
        else:
            logging.error(f"❌ Failed to send message: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logging.error(f"❌ Error sending message: {e}")
        return False

def send_whatsapp_location(to_phone):
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
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
            return True
        else:
            logging.error(f"❌ Failed to send location: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logging.error(f"❌ Error sending location: {e}")
        return False

def send_whatsapp_document(to_phone, caption="Here is your Brookstone Brochure 📄"):
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
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
            return True
        else:
            logging.error(f"❌ Failed to send document: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logging.error(f"❌ Error sending document: {e}")
        return False

def mark_message_as_read(message_id):
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "status": "read", "message_id": message_id}
    try:
        requests.post(url, headers=headers, json=payload, timeout=10)
    except Exception as e:
        logging.error(f"Error marking message as read: {e}")

# ================================================
# MESSAGE PROCESSING WITH RAG
# ================================================
def process_incoming_message(from_phone, message_text, message_id):
    if from_phone not in CONV_STATE:
        CONV_STATE[from_phone] = {'chat_history': [], 'language': 'english'}

    state = CONV_STATE[from_phone]
    user_lower = message_text.lower().strip()
    guj = is_gujarati(message_text)
    state['language'] = 'gujarati' if guj else 'english'
    state['chat_history'].append({"role": "user", "content": message_text})

    logging.info(f"📱 Processing message from {from_phone}: {message_text} [Language: {state['language']}]")

    # --- STATIC RESPONSES ---
    static_result = detect_static(message_text)
    if static_result:
        if isinstance(static_result, tuple):
            response, action = static_result
            if action == "location":
                send_whatsapp_location(from_phone)
                send_whatsapp_text(from_phone, response)
                state['chat_history'].append({"role": "assistant", "content": response})
                return
            elif action == "brochure":
                send_whatsapp_document(from_phone)
                state['chat_history'].append({"role": "assistant", "content": response})
                return
        else:
            send_whatsapp_text(from_phone, static_result)
            state['chat_history'].append({"role": "assistant", "content": static_result})
            return

    # --- RETRIEVE FROM PINECONE ---
    if not retriever:
        fallback = STATIC["agent_gu" if guj else "agent_en"]
        send_whatsapp_text(from_phone, fallback)
        state['chat_history'].append({"role": "assistant", "content": fallback})
        return

    try:
        docs = retriever.invoke(message_text)
        logging.info(f"📚 Retrieved {len(docs)} relevant documents")

        if not docs:
            msg = STATIC["pricing_gu" if guj else "pricing_en"] if is_pricing(message_text) else STATIC["agent_gu" if guj else "agent_en"]
            send_whatsapp_text(from_phone, msg)
            state['chat_history'].append({"role": "assistant", "content": msg})
            return

        # Build context
        context_parts = []
        for d in docs:
            text = d.page_content or ""
            if d.metadata:
                text += "\n" + "\n".join(f"{k}: {v}" for k, v in d.metadata.items())
            context_parts.append(text)
        context = "\n\n".join(context_parts)

        system_prompt = f"""
You are a friendly real estate assistant for Brookstone project. Be conversational and natural like a helpful friend.

Answer **only** using the context below.
If something is not mentioned, say you don't have that information and suggest contacting the agent.
Ask follow-up questions and try to convince the user.
Be concise and direct - don't give overly detailed explanations, but include all relevant facts
For general BHK interest: "Brookstone has luxury 3&4BHK flats 🏠 What would you like to know - size, location, or amenities?"
Use 1-2 emojis maximum
End with short, natural follow-up
Always provide complete information when asked - don't cut off important details to make responses shorter
ELEVATOR/LIFT RESPONSES:
    - For structure/material questions: "KONE/SCHINDLER or equivalent"
    - For ground floor lift questions: Only mention Block A and Block B lifts
    - For "Are lifts available in all towers?": "Yes, each tower is equipped with premium elevators ensuring smooth mobility"
    - Use the specific elevator_response from PROJECT DATA when available

Examples:
- If a user asks "Do you have 4BHK flats?", reply:
  "Sure! Brookstone offers luxurious 3 & 4BHK flats. Would you like to know more about sizes, amenities, or availability?"
- If a user asks about sizes or amenities, answer from context and then ask if they'd like the brochure.
- If user asks about timings to visit: "10:30 AM to 7:00 PM. Would you like me to send the location?"
- If user wants to book a visit: "Please contact Mr. Nilesh at 7600612701 to book your site visit."
- If user wants to contact someone: "You can contact our agents directly on 8238477697 or 9974812701."
- If user asks about pricing and it's not in context: "For latest pricing, please contact our agents directly."

Carry out a friendly, conversational flow.
Do **NOT** invent or guess details.
Keep responses concise and WhatsApp-friendly (avoid markdown formatting).
---
Context:
{context}

User ({'Gujarati' if guj else 'English'}): {message_text}
Assistant:
        """.strip()

        response = llm.invoke(system_prompt).content.strip()

        if is_pricing(message_text) and not re.search(r"\d", response):
            response = STATIC["pricing_gu" if guj else "pricing_en"]

        send_whatsapp_text(from_phone, response)
        state['chat_history'].append({"role": "assistant", "content": response})

    except Exception as e:
        logging.error(f"❌ Error in RAG processing: {e}")
        fallback = "Sorry, I'm having trouble right now. Please contact our agents at 8238477697 / 9974812701."
        send_whatsapp_text(from_phone, fallback)
        state['chat_history'].append({"role": "assistant", "content": fallback})

# ================================================
# WEBHOOK ROUTES
# ================================================
@app.route('/webhook', methods=['GET'])
def verify_webhook():
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')

    logging.info(f"Webhook verification: mode={mode}, token={token}")

    if mode == 'subscribe' and token == VERIFY_TOKEN:
        logging.info('✅ WEBHOOK VERIFIED')
        return challenge, 200
    else:
        logging.warning('❌ WEBHOOK VERIFICATION FAILED')
        return 'Forbidden', 403

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    logging.info("Incoming webhook data")

    try:
        for entry in data.get('entry', []):
            for change in entry.get('changes', []):
                value = change.get('value', {})
                messages = value.get('messages', [])

                for message in messages:
                    from_phone = message.get('from')
                    message_id = message.get('id')
                    msg_type = message.get('type')

                    text = ''
                    if msg_type == 'text':
                        text = message.get('text', {}).get('body', '')
                    elif msg_type == 'button':
                        text = message.get('button', {}).get('text', '')
                    elif msg_type == 'interactive':
                        interactive = message.get('interactive', {})
                        if 'button_reply' in interactive:
                            text = interactive['button_reply'].get('title', '')
                        elif 'list_reply' in interactive:
                            text = interactive['list_reply'].get('title', '')

                    if not text:
                        logging.warning(f"No text in message type: {msg_type}")
                        continue

                    mark_message_as_read(message_id)
                    process_incoming_message(from_phone, text, message_id)

    except Exception as e:
        logging.exception('❌ Error processing webhook')

    return jsonify({'status': 'ok'}), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'whatsapp_configured': bool(WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID),
        'openai_configured': bool(OPENAI_API_KEY),
        'pinecone_configured': bool(PINECONE_API_KEY)
    }), 200

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Brookstone WhatsApp RAG Bot is running!',
        'brochure_url': BROCHURE_URL,
        'endpoints': {'webhook': '/webhook', 'health': '/health'}
    }), 200

# ================================================
# RUN APP
# ================================================
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    logging.info(f"🚀 Starting Brookstone WhatsApp RAG Bot on port {port}")
    logging.info(f"WhatsApp configured: {bool(WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID)}")
    logging.info(f"OpenAI configured: {bool(OPENAI_API_KEY)}")
    logging.info(f"Pinecone configured: {bool(PINECONE_API_KEY)}")

    app.run(host='0.0.0.0', port=port, debug=False)
