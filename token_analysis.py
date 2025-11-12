#!/usr/bin/env python3
"""
Token Usage Analysis for Brookstone WhatsApp Bot
Analyzes token consumption for different API calls
"""

def estimate_tokens(text):
    """Rough token estimation: ~1 token per 4 characters for English, ~1.5x for Gujarati"""
    return len(text) / 4

def analyze_gemini_requests():
    """Analyze token usage for different Gemini API calls"""
    
    print("🔍 BROOKSTONE BOT TOKEN USAGE ANALYSIS")
    print("="*60)
    
    # 1. Translation Requests
    print("\n1. 📝 TRANSLATION REQUESTS:")
    print("-" * 30)
    
    gujarati_to_english_prompt = """
Translate the following Gujarati text to English. Provide only the English translation, nothing else.

Gujarati text: {text}

English translation:
    """
    
    english_to_gujarati_prompt = """
Translate the following English text to Gujarati. Keep the same tone, style, and LENGTH - make it brief and concise like the original. Provide only the Gujarati translation, nothing else.

English text: {text}

Gujarati translation (keep it brief and concise):
    """
    
    sample_text = "બ્રોશર મોકલો"  # ~20 chars
    
    gu_to_en_tokens = estimate_tokens(gujarati_to_english_prompt.format(text=sample_text))
    en_to_gu_tokens = estimate_tokens(english_to_gujarati_prompt.format(text="Send brochure"))
    
    print(f"   Gujarati → English: ~{gu_to_en_tokens:.0f} tokens")
    print(f"   English → Gujarati: ~{en_to_gu_tokens:.0f} tokens")
    
    # 2. Location Detection
    print("\n2. 🗺️ LOCATION DETECTION:")
    print("-" * 30)
    
    location_prompt = """
Analyze this message and determine if the user is asking for location, address, or directions to a property/site.

User Message: "{message}"

Consider these as location requests:
- Asking for address, location, directions
- "Where is it located?"
- "Can you share the location?"
- "What's the address?"
- "How to reach there?"
- Similar location-related queries in any language

Respond with only "YES" if it's a location request, or "NO" if it's not.
    """
    
    location_tokens = estimate_tokens(location_prompt.format(message="Where is Brookstone located?"))
    print(f"   Location Detection: ~{location_tokens:.0f} tokens")
    
    # 3. Interest Analysis
    print("\n3. 🧠 INTEREST ANALYSIS:")
    print("-" * 30)
    
    interest_prompt = """
Analyze this real estate inquiry and identify the user's interests/intent:

User Message: "{message}"
Previous Interests: {interests}

Categorize interests from: pricing, size, amenities, location, availability, visit, brochure, general_info

Return only a comma-separated list of relevant categories. Example: "pricing, size, visit"
    """
    
    interest_tokens = estimate_tokens(interest_prompt.format(
        message="I want to know about 3BHK flats and their price",
        interests="['general_info']"
    ))
    print(f"   Interest Analysis: ~{interest_tokens:.0f} tokens")
    
    # 4. Main RAG System Prompt
    print("\n4. 🤖 MAIN RAG SYSTEM PROMPT:")
    print("-" * 30)
    
    # This is the big one - the main system prompt
    main_system_prompt_size = """
You are a friendly real estate assistant for Brookstone project. Be conversational, natural, and convincing.

IMPORTANT: User is asking in Gujarati. Respond in ENGLISH first (keep it VERY SHORT), then it will be translated to Gujarati automatically.

CORE INSTRUCTIONS:
- Be VERY CONCISE - give brief, direct answers (2-3 sentences max)
- Answer using context below when available
- Use 2-3 relevant emojis to make responses engaging
- Keep responses WhatsApp-friendly
- Do NOT invent details
- Remember conversation flow and previous follow-ups
- ALWAYS try to convince user in a friendly way
- Use the conversation memory and user preferences provided
- Be NATURAL and CONTEXTUAL - don't repeat the same phrases in every response
- Only mention flat types (3&4BHK) when user specifically asks about them

MEMORY CONTEXT: [Previous conversation context and follow-ups]

SMART FLAT MENTIONS:
- ONLY mention "Brookstone offers luxurious 3&4BHK flats" when user specifically asks about:
  * Flat types/configurations (3BHK, 4BHK)
  * "What do you have?" / "What's available?"
  * Property types or unit options
  * First-time inquiries about the project
- Whether user asks about 3BHK or whether user asks about 4BHK, mention both types
- Do NOT force this phrase into every response - be natural and contextual

[... continues with all the detailed instructions ...]

Available Knowledge Context:
[Retrieved context from Pinecone - typically 2-5 documents with metadata]

User Question: {query}

Provide a brief, convincing answer with good emojis and ask ONE relevant follow-up question.
    """
    
    # Estimate with context
    sample_context = "Brookstone offers luxury 3&4BHK flats in Bopal, Ahmedabad. Amenities include gym, swimming pool, children's play area..." * 5  # Typical context
    
    main_prompt_tokens = estimate_tokens(main_system_prompt_size + sample_context)
    print(f"   Main RAG Request: ~{main_prompt_tokens:.0f} tokens")
    
    # 5. Memory Update
    print("\n5. 🧠 CONVERSATION MEMORY UPDATE:")
    print("-" * 30)
    
    memory_prompt = """
Analyze this real estate conversation and extract:
1. User's key interests and preferences
2. Important conversation points to remember
3. User's likely budget range or property requirements
4. Any specific questions or concerns raised

Conversation History:
[Last 8 messages from conversation]

Current Summary: [Previous conversation summary]

Provide a concise updated summary (max 100 words) and key user preferences in JSON format:
{
    "summary": "brief conversation summary",
    "preferences": {
        "budget_range": "inferred budget or 'unknown'",
        "preferred_bhk": "3BHK/4BHK/both/unknown",
        "key_interests": ["list", "of", "interests"],
        "concerns": ["any", "concerns", "raised"],
        "visit_intent": "high/medium/low/unknown"
    }
}
    """
    
    memory_tokens = estimate_tokens(memory_prompt + "user: નમસ્તે\nassistant: નમસ્કાર!" * 4)  # Sample conversation
    print(f"   Memory Update: ~{memory_tokens:.0f} tokens")
    
    # 6. Summary
    print("\n" + "="*60)
    print("📊 TOTAL TOKEN USAGE PER MESSAGE:")
    print("="*60)
    
    typical_message_tokens = (
        gu_to_en_tokens +  # Translation of user message
        location_tokens +  # Location detection
        interest_tokens +  # Interest analysis
        main_prompt_tokens +  # Main RAG response
        en_to_gu_tokens +  # Translation of response
        (memory_tokens / 4)  # Memory update (every 4 messages)
    )
    
    print(f"🔹 Simple query (location/brochure): ~{location_tokens + gu_to_en_tokens + en_to_gu_tokens:.0f} tokens")
    print(f"🔹 Typical RAG query: ~{typical_message_tokens:.0f} tokens")
    print(f"🔹 Complex query with memory: ~{typical_message_tokens + memory_tokens:.0f} tokens")
    
    print("\n💡 TOKEN OPTIMIZATION OPPORTUNITIES:")
    print("-" * 40)
    print("✅ Location detection is efficient (~60 tokens)")
    print("✅ Interest analysis is lightweight (~80 tokens)")
    print("⚠️ Main system prompt is large (~800-1200 tokens)")
    print("⚠️ Memory updates add ~300 tokens every 4 messages")
    print("⚠️ Translations add ~60-80 tokens each")
    
    print("\n🎯 COST ESTIMATION (Gemini 2.5 Flash):")
    print("-" * 40)
    # Gemini 2.5 Flash pricing: $0.075 per 1M input tokens, $0.30 per 1M output tokens
    input_cost_per_1k = 0.075 / 1000
    output_cost_per_1k = 0.30 / 1000
    
    avg_input_tokens = typical_message_tokens
    avg_output_tokens = 150  # Typical response length
    
    cost_per_message = (avg_input_tokens * input_cost_per_1k) + (avg_output_tokens * output_cost_per_1k)
    
    print(f"💰 Cost per typical message: ~${cost_per_message:.4f}")
    print(f"💰 Cost per 100 messages: ~${cost_per_message * 100:.2f}")
    print(f"💰 Cost per 1000 messages: ~${cost_per_message * 1000:.2f}")

if __name__ == "__main__":
    analyze_gemini_requests()