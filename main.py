from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import openai
import google.generativeai as genai
import base64
from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph
import asyncio
import io
import json
from pydantic import BaseModel
import hashlib
from datetime import datetime
import traceback
from uuid import uuid4
from neo4j import GraphDatabase

from langchain_neo4j import Neo4jChatMessageHistory, Neo4jGraph
from langchain_core.messages import HumanMessage, AIMessage

# Telegram Bot imports
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes
import logging

# TTS imports
import edge_tts
import tempfile

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in your .env file.")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY must be set in your .env file.")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN must be set in your .env file.")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)  # ADD this closing parenthesis
logger = logging.getLogger(__name__)

neo4j_graph = Neo4jGraph(
    url="bolt://neo4j:7687",
    username="neo4j",
    password="reform-william-center-vibrate-press-5829",
    database="neo4j"
)

neo4j_driver = GraphDatabase.driver(
    "bolt://neo4j:7687", 
    auth=("neo4j", "reform-william-center-vibrate-press-5829")
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserRegistration(BaseModel):
    user_name: str  
    age: int = None
    district: str = None
    crops_grown: str = None
    farm_size: str = None
    contact: str = None

registered_users = {}
user_sessions = {}
user_states = {}  # Track user registration states for Telegram

# ADD: canonical user key helper (below registered_users / user_sessions)
def get_user_key(telegram_user_id: str = None, api_user_name: str = None) -> str:
    # Telegram flows: always use numeric Telegram user_id as the storage key
    if telegram_user_id is not None:
        return str(telegram_user_id)
    # HTTP API flows: registration uses user_name as user_id, keep that
    return str(api_user_name) if api_user_name is not None else "unknown_user"


def get_or_create_session_id(user_name: str) -> str:
    """Get existing session ID or create new one for user"""
    if user_name not in user_sessions:
        user_sessions[user_name] = str(uuid4())
        print(f"Created session ID {user_sessions[user_name]} for user {user_name}")
    return user_sessions[user_name]

# def get_chat_memory(user_name: str) -> Neo4jChatMessageHistory:
#     """Get Neo4j chat memory for specific user"""
#     session_id = get_or_create_session_id(user_name)
#     return Neo4jChatMessageHistory(
#         session_id=session_id,
#         graph=neo4j_graph
#     )

# REPLACE the function signature and body of get_chat_memory
def get_chat_memory(user_key: str) -> Neo4jChatMessageHistory:
    """Get Neo4j chat memory for specific user (canonical key)"""
    session_id = get_or_create_session_id(user_key)
    return Neo4jChatMessageHistory(
        session_id=session_id,
        graph=neo4j_graph
    )


def store_user_profile(user_id: str, profile_data: Dict[str, Any]):
    """Store user profile in Neo4j"""
    try:
        with neo4j_driver.session() as session:
            session.run("""
                MERGE (u:UserProfile {user_id: $user_id})
                SET u.user_name = $user_name,
                    u.age = $age,
                    u.district = $district,
                    u.crops_grown = $crops_grown,
                    u.farm_size = $farm_size,
                    u.contact = $contact,
                    u.registration_date = $registration_date,
                    u.session_id = $session_id,
                    u.last_updated = $last_updated
                RETURN u
            """, 
                user_id=user_id,
                user_name=profile_data.get("user_name"),
                age=profile_data.get("age"),
                district=profile_data.get("district"),
                crops_grown=profile_data.get("crops_grown"),
                farm_size=profile_data.get("farm_size"),
                contact=profile_data.get("contact"),
                registration_date=profile_data.get("registration_date"),
                session_id=get_or_create_session_id(user_id),
                last_updated=datetime.now().isoformat()
            )
        return True
    except Exception as e:
        print(f"Error storing user profile: {e}")
        return False

def get_user_profile(user_id: str) -> Dict[str, Any]:
    """Get user profile from Neo4j"""
    try:
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (u:UserProfile {user_id: $user_id})
                RETURN u.user_name as user_name,
                       u.age as age,
                       u.district as district,
                       u.crops_grown as crops_grown,
                       u.farm_size as farm_size,
                       u.contact as contact,
                       u.registration_date as registration_date,
                       u.session_id as session_id,
                       u.last_updated as last_updated
            """, user_id=user_id)
            
            record = result.single()
            if record:
                if record["session_id"]:
                    user_sessions[user_id] = record["session_id"]
                
                return {
                    "user_id": user_id,
                    "user_name": record["user_name"],
                    "age": record["age"],
                    "district": record["district"],
                    "crops_grown": record["crops_grown"],
                    "farm_size": record["farm_size"],
                    "contact": record["contact"],
                    "registration_date": record["registration_date"],
                    "session_id": record["session_id"],
                    "last_updated": record["last_updated"]
                }
            return None
    except Exception as e:
        print(f"Error getting user profile: {e}")
        return None

# NEW HELPER FUNCTION - This is the key fix!
def get_user_name_from_telegram_id(telegram_user_id: str) -> str:
    """Get the registered user_name from telegram user_id"""
    if telegram_user_id in registered_users:
        return registered_users[telegram_user_id].get("user_name", telegram_user_id)
    return telegram_user_id

def get_conversation_history(user_key: str) -> list:
    """Get conversation history using LangChain Neo4j memory"""
    try:
        chat_memory = get_chat_memory(user_key)
        messages = chat_memory.messages
        
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({
                    "type": "human",
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat()  
                })
            elif isinstance(msg, AIMessage):
                history.append({
                    "type": "ai", 
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat()
                })
        
        return history
    except Exception as e:
        print(f"Error getting conversation history: {e}")
        return []

# def add_to_conversation(user_name: str, human_message: str = None, ai_message: str = None):
#     """Add message to conversation history"""
#     try:
#         chat_memory = get_chat_memory(user_name)
        
#         if human_message:
#             chat_memory.add_user_message(human_message)
#             print(f"Added human message for {user_name}")
        
#         if ai_message:
#             chat_memory.add_ai_message(ai_message)
#             print(f"Added AI message for {user_name}")
        
#         return True
#     except Exception as e:
#         print(f"Error adding to conversation: {e}")
#         return False

# REPLACE add_to_conversation
def add_to_conversation(user_key: str, human_message: str = None, ai_message: str = None):
    """Add message to conversation history"""
    try:
        chat_memory = get_chat_memory(user_key)
        if human_message:
            chat_memory.add_user_message(human_message)
            print(f"Added human message for {user_key}")
        if ai_message:
            chat_memory.add_ai_message(ai_message)
            print(f"Added AI message for {user_key}")
        return True
    except Exception as e:
        logger.exception(f"Error adding to conversation for {user_key}: {e}")
        return False

def get_relevant_context(user_key: str, query: str) -> str:
    """Get relevant context from user's conversation history"""
    try:
        history = get_conversation_history(user_key)
        
        profile = get_user_profile(user_key)
        context_parts = []
        
        if profile:
            profile_context = f"User: {profile.get('user_name')} from {profile.get('district', 'Unknown')} grows {profile.get('crops_grown', 'various crops')}"
            context_parts.append(profile_context)
        
        for msg in history[-6:]:
            if msg["type"] == "human":
                context_parts.append(f"User asked: {msg['content'][:150]}...")
            elif msg["type"] == "ai":
                context_parts.append(f"AI responded: {msg['content'][:150]}...")
        
        return "\n".join(context_parts) if context_parts else ""
        
    except Exception as e:
        print(f"Error getting relevant context: {e}")
        return ""

class PestState(TypedDict):
    image_b64: Optional[str]
    user_description: Optional[str]
    description_source: str 
    description: str
    diagnosis: str
    user_id: str

class QueryState(TypedDict):
    query_text: str
    llm_response: str
    user_id: str

async def transcribe_audio_with_gemini(audio_data: bytes, filename: str) -> str:
    """Transcribe audio using Gemini for user descriptions"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        audio_file = {
            "mime_type": "audio/wav", 
            "data": audio_data
        }
        
        prompt = (
            "Please transcribe the following audio file. The audio contains a farmer describing symptoms they observe on their plant leaves. "
            "The description may be in Malayalam, English, or other languages. "
            "Provide the transcription in the original language spoken. If Malayalam is detected, provide transcription in Malayalam script. "
            "Focus on plant symptoms, disease signs, and agricultural observations."
        )
        
        response = model.generate_content([prompt, audio_file])
        
        if response.text:
            return response.text.strip()
        else:
            raise Exception("No transcription returned from Gemini")
            
    except Exception as e:
        print(f"Gemini transcription error: {e}")
        raise Exception(f"Gemini transcription failed: {str(e)}")

async def text_to_speech_malayalam(text: str) -> bytes:
    """Convert text to Malayalam speech using edge-tts"""
    try:
        # Use Malayalam voice
        voice = "ml-IN-SobhanaNeural"  # Malayalam female voice
        
        communicate = edge_tts.Communicate(text, voice)
        
        # Create temporary file to store audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            await communicate.save(tmp_file.name)
            with open(tmp_file.name, 'rb') as f:
                audio_data = f.read()
        
        # Clean up temp file
        os.unlink(tmp_file.name)
        
        return audio_data
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None

def extract_malayalam_response(response_text: str) -> str:
    """Extract Malayalam text from bilingual response"""
    lines = response_text.split('\n')
    malayalam_lines = []
    
    for line in lines:
        # Simple check for Malayalam characters
        if any(ord(char) >= 0x0D00 and ord(char) <= 0x0D7F for char in line):
            malayalam_lines.append(line)
    
    malayalam_response = '\n'.join(malayalam_lines)
    
    # If no Malayalam found, return first few lines as fallback
    if not malayalam_response.strip():
        malayalam_response = '\n'.join(lines[:3])
    
    return malayalam_response.strip()

def process_description(state: PestState) -> Dict[str, Any]:
    """Process description - either from image analysis or user-provided description"""
    user_id = state.get("user_id", "farmer_system")
    user_description = state.get("user_description", "")
    image_b64 = state.get("image_b64", "")
    description_source = state.get("description_source", "image")
    
    # If user provided description, use it directly
    if user_description and user_description.strip():
        print(f"Using user-provided description from {description_source}")
        return {
            "description": user_description.strip(),
            "description_source": description_source,
            "user_id": user_id
        }

    if not image_b64:
        return {"error": "No image or user description provided.", "user_id": user_id}
    
    print("Using AI image analysis for description")
    
    prompt = (
        """You are a senior plant pathologist with 15 years of experience specializing in tropical crop diseases, particularly those affecting Kerala's major agricultural crops including coconut, rubber, pepper, cardamom, rice, banana, tea, coffee, ginger, turmeric, and spices.
        ANALYSIS INSTRUCTIONS:
        Carefully examine this leaf image and provide a detailed description focusing on:

        1. VISUAL SYMPTOMS:
        - Leaf discoloration patterns (yellowing, browning, blackening, reddening)
        - Spot characteristics (size, shape, color, borders, concentric rings)
        - Physical damage (holes, wilting, curling, distortion)
        - Surface abnormalities (powdery coating, fuzzy growth, sticky residue)
        - Vein patterns and leaf texture changes

        2. DISTRIBUTION PATTERN:
        - Location of symptoms (leaf tips, edges, center, base)
        - Progression pattern (scattered, clustered, systematic)
        - Age-related symptoms (young vs. old leaves)

        3. ENVIRONMENTAL CONTEXT:
        - Consider Kerala's tropical monsoon climate
        - High humidity and temperature effects
        - Seasonal disease patterns common in the region

        Describe ONLY what you observe - avoid making diagnostic conclusions at this stage. Use precise botanical and pathological terminology. Consider that this may be from crops commonly grown in Kerala's diverse agro-climatic zones.
        give response in malayalam and english both languages"""
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }
    ]
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=800
        )
        desc = response.choices[0].message.content or ""
        return {
            "description": desc,
            "description_source": "image",
            "user_id": user_id
        }
    except Exception as e:
        print(f"ERROR in process_description: {e}")
        return {"error": f"Vision model error: {str(e)}", "user_id": user_id}

def diagnose_leaf(state: PestState) -> Dict[str, Any]:
    """Diagnose based on description (from image or user-provided)"""
    description = state.get("description", "")
    description_source = state.get("description_source", "unknown")
    user_id = state.get("user_id", "farmer_system")
    
    if not description:
        return {"error": "No description available for diagnosis.", "user_id": user_id}
    
    # FIXED: Get the actual user_name for context retrieval
    # user_name = get_user_name_from_telegram_id(user_id)
    # context = get_relevant_context(user_name, description)
    user_key = get_user_key(telegram_user_id=user_id)
    context = get_relevant_context(user_key, description)
    
    memory_context = ""
    if context:
        memory_context = f"""
        Your profile and conversation history:
        {context}
        
        Use this context to provide personalized advice:
        """
    description_note = ""
    if description_source == "user_text":
        description_note = "Note: This description was provided by the farmer in text format."
    elif description_source == "user_voice":
        description_note = "Note: This description was provided by the farmer via voice recording."
    elif description_source == "image":
        description_note = "Note: This description was generated from AI analysis of the leaf image."
    
    prompt = (
        f"""You are a senior plant pathology expert specializing in tropical crop diseases, particularly those affecting Kerala's major agricultural crops including coconut, rubber, pepper, cardamom, rice, banana, tea, coffee, ginger, turmeric, and spices.
        
        {memory_context}
        
        {description_note}
        
        Given the following description of a plant leaf, provide a comprehensive analysis:
        1. What symptoms are visible?
        2. What is the most likely disease or condition?
        3. What pest or pathogen might cause it?
        4. What are recommended organic treatments?
        5. What are recommended chemical treatments?
        6. What preventive measures should be taken by the farmer?
        
        Leaf description: {description}
        
        Provide practical, actionable advice suitable for Kerala's agricultural conditions.
        Give response PRIMARILY in Malayalam with English terms in brackets where needed. Make sure Malayalam response comes first.
        """
    )
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        diag = response.choices[0].message.content or ""
        
        return {
            "diagnosis": diag,
            "description_source": description_source,
            "user_id": user_id
        }
    except Exception as e:
        print(f"ERROR in diagnose_leaf: {e}")
        return {"error": f"Diagnosis model error: {str(e)}", "user_id": user_id}

def process_query(state: QueryState) -> Dict[str, Any]:
    query_text = state.get("query_text", "")
    user_id = state.get("user_id", "farmer_system")
    
    if not query_text:
        return {"error": "No query text provided.", "user_id": user_id}
    
    # FIXED: Get the actual user_name for context retrieval
    # user_name = get_user_name_from_telegram_id(user_id)
    # context = get_relevant_context(user_name, query_text)

    user_key = get_user_key(telegram_user_id=user_id)
    context = get_relevant_context(user_key, query_text)
    
    memory_context = ""
    if context:
        memory_context = f"""
        Your profile and conversation history:
        {context}
        
        Use this context to provide informed agricultural advice:
        """
    
    prompt = (
        f"""You are Kerala's leading agricultural consultant and plant pathologist with 15+ years of field experience. You've worked extensively with farmers across Kerala's diverse agro-climatic zones - from the coastal plains to the Western Ghats. Your expertise covers all major Kerala crops and you understand the unique challenges posed by the state's tropical monsoon climate.

        {memory_context}

        KERALA AGRICULTURAL CONTEXT:
        Major Crops: Coconut, rubber, pepper, cardamom, rice, banana, tea, coffee, ginger, turmeric, vegetables
        Climate: Tropical monsoon with heavy rains (June-September), high humidity year-round
        Regions: Coastal plains, midlands, high ranges - each with specific challenges
        Farmers: Mix of traditional knowledge and modern techniques

        RESPONSE GUIDELINES:
        - Provide practical, actionable advice suitable for Kerala's climate
        - Include both modern scientific methods and traditional Kerala practices when relevant
        - Consider monsoon timing in your recommendations
        - Mention locally available materials and resources
        - Address economic aspects - cost-effective solutions for small farmers
        - Include preventive measures specific to high humidity conditions
        - Reference Kerala's agricultural calendar when timing is important

        FARMER'S QUERY:
        {query_text}

        Provide a comprehensive, empathetic response as if you're directly consulting with a Kerala farmer. Structure your answer with clear sections and practical steps they can implement immediately.
        Give response PRIMARILY in Malayalam with English terms in brackets where needed. Make sure Malayalam response comes first.
        """
    )
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        answer = response.choices[0].message.content or ""
        
        return {"llm_response": answer, "user_id": user_id}
    except Exception as e:
        print(f"ERROR in process_query: {e}")
        return {"error": f"Query processing error: {str(e)}", "user_id": user_id}

def create_pest_workflow():
    """Create enhanced pest detection workflow that can handle user descriptions"""
    pest_graph = StateGraph(PestState)
    pest_graph.add_node("process_description", process_description)
    pest_graph.add_node("diagnose_leaf", diagnose_leaf)
    pest_graph.add_edge("process_description", "diagnose_leaf")
    pest_graph.set_entry_point("process_description")
    pest_graph.set_finish_point("diagnose_leaf")
    return pest_graph.compile()

def create_query_workflow():
    query_graph = StateGraph(QueryState)
    query_graph.add_node("process_query", process_query)
    query_graph.set_entry_point("process_query")
    query_graph.set_finish_point("process_query")
    return query_graph.compile()

pest_flow = create_pest_workflow()
query_flow = create_query_workflow()

# TELEGRAM BOT FUNCTIONS
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user_id = str(update.effective_user.id)
    user_name = update.effective_user.first_name or "Farmer"
    
    # Check if user is already registered
    if user_id in registered_users:
        welcome_text = f"സ്വാഗതം {user_name}! നിങ്ങൾ ഇതിനകം രജിസ്റ്റർ ചെയ്തിട്ടുണ്ട്.\n\nWelcome back {user_name}! You are already registered."
        
        keyboard = [
            [InlineKeyboardButton("🌱 ചെടിയുടെ രോഗം പരിശോധിക്കുക / Plant Disease Check", callback_data="pest_detection")],
            [InlineKeyboardButton("❓ ചോദ്യം ചോദിക്കുക / Ask Question", callback_data="ask_question")],
            [InlineKeyboardButton("👤 പ്രൊഫൈൽ കാണുക / View Profile", callback_data="view_profile")],
        ]
    else:
        welcome_text = (
            f"സ്വാഗതം {user_name}! കേരള കാർഷിക സഹായ ബോട്ടിലേക്ക് സ്വാഗതം.\n\n"
            f"Welcome {user_name}! Welcome to Kerala Agricultural Assistant Bot.\n\n"
            "ദയവായി ആദ്യം രജിസ്റ്റർ ചെയ്യുക:\nPlease register first:"
        )
        
        keyboard = [
            [InlineKeyboardButton("📝 രജിസ്റ്റർ ചെയ്യുക / Register", callback_data="register")],
        ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(welcome_text, reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks"""
    query = update.callback_query
    await query.answer()
    
    user_id = str(query.from_user.id)
    
    if query.data == "register":
        user_states[user_id] = {"state": "awaiting_name"}
        await query.edit_message_text(
            text="നിങ്ങളുടെ പേര് എന്താണ്?\nWhat is your name?",
        )
    
    elif query.data == "pest_detection":
        if user_id not in registered_users:
            await query.edit_message_text(text="ദയവായി ആദ്യം രജിസ്റ്റർ ചെയ്യുക!\nPlease register first!")
            return
        
        keyboard = [
            [InlineKeyboardButton("📷 ചിത്രം അയക്കുക / Send Image", callback_data="send_image")],
            [InlineKeyboardButton("📝 വിവരണം ടൈപ്പ് ചെയ്യുക / Type Description", callback_data="type_description")],
            [InlineKeyboardButton("🎤 ശബ്ദം റെക്കോർഡ് ചെയ്യുക / Record Voice", callback_data="record_voice")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            text="ചെടിയുടെ രോഗം പരിശോധിക്കാൻ എന്ത് ചെയ്യാൻ ആഗ്രഹിക്കുന്നു?\nHow would you like to check plant disease?",
            reply_markup=reply_markup
        )
    
    elif query.data == "send_image":
        user_states[user_id] = {"state": "awaiting_image"}
        await query.edit_message_text(
            text="ചെടിയുടെ ഇലയുടെ ചിത്രം അയക്കുക\nPlease send a photo of the plant leaf"
        )
    
    elif query.data == "type_description":
        user_states[user_id] = {"state": "awaiting_text_description"}
        await query.edit_message_text(
            text="ചെടിയുടെ രോഗത്തിന്റെ വിവരണം ടൈപ്പ് ചെയ്യുക\nType the description of plant disease"
        )
    
    elif query.data == "record_voice":
        user_states[user_id] = {"state": "awaiting_voice"}
        await query.edit_message_text(
            text="ശബ്ദ സന്ദേശം അയക്കുക\nSend voice message"
        )
    
    elif query.data == "ask_question":
        if user_id not in registered_users:
            await query.edit_message_text(text="ദയവായി ആദ്യം രജിസ്റ്റർ ചെയ്യുക!\nPlease register first!")
            return
        
        keyboard = [
            [InlineKeyboardButton("📝 ടെക്സ്റ്റ് ചോദ്യം / Text Question", callback_data="text_question")],
            [InlineKeyboardButton("🎤 ശബ്ദ ചോദ്യം / Voice Question", callback_data="voice_question")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            text="എങ്ങനെ ചോദ്യം ചോദിക്കാൻ ആഗ്രഹിക്കുന്നു?\nHow would you like to ask your question?",
            reply_markup=reply_markup
        )
    
    elif query.data == "text_question":
        user_states[user_id] = {"state": "awaiting_question"}
        await query.edit_message_text(
            text="നിങ്ങളുടെ കാർഷിക ചോദ്യം ടൈപ്പ് ചെയ്യുക\nType your agricultural question"
        )
    
    elif query.data == "voice_question":
        user_states[user_id] = {"state": "awaiting_voice_question"}
        await query.edit_message_text(
            text="നിങ്ങളുടെ ചോദ്യം ശബ്ദ സന്ദേശമായി അയക്കുക\nSend your question as voice message"
        )
    
    elif query.data == "view_profile":
        if user_id not in registered_users:
            await query.edit_message_text(text="ദയവായി ആദ്യം രജിസ്റ്റർ ചെയ്യുക!\nPlease register first!")
            return
        
        profile = registered_users[user_id]
        profile_text = f"""
👤 പ്രൊഫൈൽ വിവരങ്ങൾ / Profile Information:

പേര് / Name: {profile.get('user_name', 'N/A')}
പ്രായം / Age: {profile.get('age', 'N/A')}
ജില്ല / District: {profile.get('district', 'N/A')}
വിളകൾ / Crops: {profile.get('crops_grown', 'N/A')}
കൃഷിയിടത്തിന്റെ വലുപ്പം / Farm Size: {profile.get('farm_size', 'N/A')}
ബന്ധപ്പെടാൻ / Contact: {profile.get('contact', 'N/A')}
        """
        await query.edit_message_text(text=profile_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages based on user state"""
    user_id = str(update.effective_user.id)
    user_state = user_states.get(user_id, {})
    current_state = user_state.get("state")
    
    if current_state == "awaiting_name":
        user_states[user_id] = {"state": "awaiting_age", "user_name": update.message.text}
        await update.message.reply_text("നിങ്ങളുടെ പ്രായം എത്രയാണ്?\nWhat is your age?")
    
    elif current_state == "awaiting_age":
        try:
            age = int(update.message.text)
            user_states[user_id].update({"age": age, "state": "awaiting_district"})
            await update.message.reply_text("നിങ്ങൾ ഏത് ജില്ലയിൽ നിന്നാണ്?\nWhich district are you from?")
        except ValueError:
            await update.message.reply_text("ദയവായി സാധുവായ പ്രായം നൽകുക\nPlease provide a valid age")
    
    elif current_state == "awaiting_district":
        user_states[user_id].update({"district": update.message.text, "state": "awaiting_crops"})
        await update.message.reply_text("നിങ്ങൾ എന്ത് വിളകൾ കൃഷി ചെയ്യുന്നു?\nWhat crops do you grow?")
    
    elif current_state == "awaiting_crops":
        user_states[user_id].update({"crops_grown": update.message.text, "state": "awaiting_farm_size"})
        await update.message.reply_text("നിങ്ങളുടെ കൃഷിയിടത്തിന്റെ വലുപ്പം എത്രയാണ്?\nWhat is the size of your farm?")
    
    elif current_state == "awaiting_farm_size":
        user_states[user_id].update({"farm_size": update.message.text, "state": "awaiting_contact"})
        await update.message.reply_text("നിങ്ങളുടെ ഫോൺ നമ്പർ എന്താണ്?\nWhat is your phone number?")
    
    elif current_state == "awaiting_contact":
        user_data = user_states[user_id]
        user_data["contact"] = update.message.text
        
        # Register the user
        user_info = {
            "user_id": user_id,
            "user_name": user_data["user_name"],
            "age": user_data.get("age"),
            "district": user_data.get("district"),
            "crops_grown": user_data.get("crops_grown"),
            "farm_size": user_data.get("farm_size"),
            "contact": user_data.get("contact"),
            "registration_date": datetime.now().isoformat()
        }
        
        registered_users[user_id] = user_info
        store_user_profile(user_id, user_info)
        
        # FIXED: Use the actual user_name for chat history
        # user_name = user_data["user_name"]
        # registration_msg = f"User {user_name} registered via Telegram from {user_data.get('district', 'Unknown')} district"
        # add_to_conversation(user_name, human_message=registration_msg)
        registration_msg = f"User {user_data['user_name']} registered via Telegram from {user_data.get('district', 'Unknown')} district"
        add_to_conversation(get_user_key(telegram_user_id=user_id), human_message=registration_msg) 
        
        # Clear state
        del user_states[user_id]
        
        success_text = (
            f"രജിസ്ട്രേഷൻ വിജയകരമായി പൂർത്തിയാക്കി! 🎉\n"
            f"Registration completed successfully! 🎉\n\n"
            f"സ്വാഗതം {user_data['user_name']}!\n"
            f"Welcome {user_data['user_name']}!"
        )
        
        keyboard = [
            [InlineKeyboardButton("🌱 ചെടിയുടെ രോഗം പരിശോധിക്കുക / Plant Disease Check", callback_data="pest_detection")],
            [InlineKeyboardButton("❓ ചോദ്യം ചോദിക്കുക / Ask Question", callback_data="ask_question")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(success_text, reply_markup=reply_markup)
    
    elif current_state == "awaiting_text_description":
        # Process text description for plant disease
        await process_text_description(update, context, update.message.text)
    
    elif current_state == "awaiting_question":
        # Process text question
        await process_text_question(update, context, update.message.text)
    
    else:
        # Default response for unregistered or no state users
        if user_id not in registered_users:
            await update.message.reply_text(
                "ദയവായി /start കമാൻഡ് ഉപയോഗിച്ച് ആരംഭിക്കുക\n"
                "Please use /start command to begin"
            )
        else:
            # Treat as general question for registered users
            await process_text_question(update, context, update.message.text)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo uploads"""
    user_id = str(update.effective_user.id)
    user_state = user_states.get(user_id, {})
    
    if user_id not in registered_users:
        await update.message.reply_text("ദയവായി ആദ്യം രജിസ്റ്റർ ചെയ്യുക!\nPlease register first!")
        return
    
    if user_state.get("state") != "awaiting_image":
        await update.message.reply_text("ദയവായി ആദ്യം 'ചിത്രം അയക്കുക' ഓപ്ഷൻ തിരഞ്ഞെടുക്കുക\nPlease select 'Send Image' option first")
        return
    
    try:
        await update.message.reply_text("ചിത്രം പ്രോസസ് ചെയ്യുന്നു... ദയവായി കാത്തിരിക്കുക\nProcessing image... Please wait")
        
        # Get the photo
        photo = update.message.photo[-1]  # Get the highest resolution photo
        file = await context.bot.get_file(photo.file_id)
        
        # Download image data
        image_bytes = await file.download_as_bytearray()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Process with pest detection workflow
        initial_state: PestState = {
            "image_b64": image_b64,
            "user_description": None,
            "description_source": "image",
            "description": "",
            "diagnosis": "",
            "user_id": user_id
        }
        
        result = pest_flow.invoke(initial_state)
        
        description = result.get("description", "")
        diagnosis = result.get("diagnosis", "")
        error = result.get("error", "")
        
        if error:
            await update.message.reply_text(f"പിശക്: {error}\nError: {error}")
            return
        
        # Extract Malayalam response
        malayalam_diagnosis = extract_malayalam_response(diagnosis)
        
        # FIXED: Use the actual user_name for chat history
        # user_name = get_user_name_from_telegram_id(user_id)
        # human_msg = f"Analyzed leaf image via Telegram"
        # ai_msg = f"Image analysis and diagnosis provided: {malayalam_diagnosis[:100]}..."
        # add_to_conversation(user_name, human_message=human_msg, ai_message=ai_msg)

        human_msg = "Analyzed leaf image via Telegram"
        ai_msg = f"Image analysis and diagnosis provided: {malayalam_diagnosis[:100]}..."
        add_to_conversation(get_user_key(telegram_user_id=user_id), human_message=human_msg, ai_message=ai_msg)
        
        # Clear user state
        if user_id in user_states:
            del user_states[user_id]
        
        # Send text response
        await update.message.reply_text(malayalam_diagnosis)
        
        # Generate and send voice response
        audio_data = await text_to_speech_malayalam(malayalam_diagnosis[:500])  # Limit length for TTS
        if audio_data:
            await context.bot.send_voice(
                chat_id=update.effective_chat.id,
                voice=audio_data,
                caption="ശബ്ദ മറുപടി / Voice Response"
            )
        
        # Show main menu
        keyboard = [
            [InlineKeyboardButton("🌱 മറ്റൊരു രോഗം പരിശോധിക്കുക / Check Another Disease", callback_data="pest_detection")],
            [InlineKeyboardButton("❓ ചോദ്യം ചോദിക്കുക / Ask Question", callback_data="ask_question")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("മറ്റെന്തെങ്കിലും സഹായം വേണോ?\nNeed any other help?", reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Photo processing error: {e}")
        await update.message.reply_text("ചിത്രം പ്രോസസ് ചെയ്യുന്നതിൽ പിശക്\nError processing image")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages"""
    user_id = str(update.effective_user.id)
    user_state = user_states.get(user_id, {})
    current_state = user_state.get("state")
    
    if user_id not in registered_users:
        await update.message.reply_text("ദയവായി ആദ്യം രജിസ്റ്റർ ചെയ്യുക!\nPlease register first!")
        return
    
    if current_state not in ["awaiting_voice", "awaiting_voice_question"]:
        await update.message.reply_text("ദയവായി ആദ്യം ശബ്ദ ഓപ്ഷൻ തിരഞ്ഞെടുക്കുക\nPlease select voice option first")
        return
    
    try:
        await update.message.reply_text("ശബ്ദ സന്ദേശം പ്രോസസ് ചെയ്യുന്നു...\nProcessing voice message...")
        
        # Get voice file
        voice = update.message.voice
        file = await context.bot.get_file(voice.file_id)
        
        # Download voice data
        voice_bytes = await file.download_as_bytearray()
        
        # Transcribe audio
        transcribed_text = await transcribe_audio_with_gemini(bytes(voice_bytes), "voice.ogg")
        
        if current_state == "awaiting_voice":
            # This is for plant disease description
            await process_voice_description(update, context, transcribed_text)
        elif current_state == "awaiting_voice_question":
            # This is for general questions
            await process_voice_question(update, context, transcribed_text)
        
    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        await update.message.reply_text("ശബ്ദ സന്ദേശം പ്രോസസ് ചെയ്യുന്നതിൽ പിശക്\nError processing voice message")

async def process_text_description(update: Update, context: ContextTypes.DEFAULT_TYPE, description: str):
    """Process text description for plant disease"""
    user_id = str(update.effective_user.id)
    
    try:
        await update.message.reply_text("വിവരണം പ്രോസസ് ചെയ്യുന്നു...\nProcessing description...")
        
        initial_state: PestState = {
            "image_b64": None,
            "user_description": description,
            "description_source": "user_text",
            "description": "",
            "diagnosis": "",
            "user_id": user_id
        }
        
        result = pest_flow.invoke(initial_state)
        
        diagnosis = result.get("diagnosis", "")
        error = result.get("error", "")
        
        if error:
            await update.message.reply_text(f"പിശക്: {error}\nError: {error}")
            return
        
        # Extract Malayalam response
        malayalam_diagnosis = extract_malayalam_response(diagnosis)
        
        # FIXED: Use the actual user_name for chat history
        # user_name = get_user_name_from_telegram_id(user_id)
        # add_to_conversation(user_name, human_message=f"Text description: {description}", ai_message=diagnosis)
        add_to_conversation(get_user_key(telegram_user_id=user_id), human_message=f"Text description: {description}", ai_message=diagnosis)
        
        # Clear user state
        if user_id in user_states:
            del user_states[user_id]
        
        # Send response
        await send_malayalam_response(update, context, malayalam_diagnosis)
        
    except Exception as e:
        logger.error(f"Text description processing error: {e}")
        await update.message.reply_text("വിവരണം പ്രോസസ് ചെയ്യുന്നതിൽ പിശക്\nError processing description")

async def process_voice_description(update: Update, context: ContextTypes.DEFAULT_TYPE, description: str):
    """Process voice description for plant disease"""
    user_id = str(update.effective_user.id)
    
    try:
        await update.message.reply_text(f"ട്രാൻസ്ക്രിപ്ഷൻ: {description}\n\nപ്രോസസ് ചെയ്യുന്നു...\nTranscription: {description}\n\nProcessing...")
        
        initial_state: PestState = {
            "image_b64": None,
            "user_description": description,
            "description_source": "user_voice",
            "description": "",
            "diagnosis": "",
            "user_id": user_id
        }
        
        result = pest_flow.invoke(initial_state)
        
        diagnosis = result.get("diagnosis", "")
        error = result.get("error", "")
        
        if error:
            await update.message.reply_text(f"പിശക്: {error}\nError: {error}")
            return
        
        # Extract Malayalam response
        malayalam_diagnosis = extract_malayalam_response(diagnosis)
        
        # FIXED: Use the actual user_name for chat history
        #user_name = get_user_name_from_telegram_id(user_id)
        # add_to_conversation(user_name, human_message=f"Voice description: {description}", ai_message=diagnosis)
        add_to_conversation(get_user_key(telegram_user_id=user_id), human_message=f"Voice description: {description}", ai_message=diagnosis)
        
        # Clear user state
        if user_id in user_states:
            del user_states[user_id]
        
        # Send response
        await send_malayalam_response(update, context, malayalam_diagnosis)
        
    except Exception as e:
        logger.error(f"Voice description processing error: {e}")
        await update.message.reply_text("ശബ്ദ വിവരണം പ്രോസസ് ചെയ്യുന്നതിൽ പിശക്\nError processing voice description")

async def process_text_question(update: Update, context: ContextTypes.DEFAULT_TYPE, question: str):
    """Process text question"""
    user_id = str(update.effective_user.id)
    
    try:
        await update.message.reply_text("ചോദ്യം പ്രോസസ് ചെയ്യുന്നു...\nProcessing question...")
        
        initial_state: QueryState = {
            "query_text": question,
            "llm_response": "",
            "user_id": user_id
        }
        
        result = query_flow.invoke(initial_state)
        
        response = result.get("llm_response", "")
        error = result.get("error", "")
        
        if error:
            await update.message.reply_text(f"പിശക്: {error}\nError: {error}")
            return
        
        # Extract Malayalam response
        malayalam_response = extract_malayalam_response(response)
        
        # FIXED: Use the actual user_name for chat history
        #user_name = get_user_name_from_telegram_id(user_id)
        # add_to_conversation(user_name, human_message=question, ai_message=response)
        add_to_conversation(get_user_key(telegram_user_id=user_id), human_message=question, ai_message=response)
        
        # Clear user state if it exists
        if user_id in user_states:
            del user_states[user_id]
        
        # Send response
        await send_malayalam_response(update, context, malayalam_response)
        
    except Exception as e:
        logger.error(f"Text question processing error: {e}")
        await update.message.reply_text("ചോദ്യം പ്രോസസ് ചെയ്യുന്നതിൽ പിശക്\nError processing question")

async def process_voice_question(update: Update, context: ContextTypes.DEFAULT_TYPE, question: str):
    """Process voice question"""
    user_id = str(update.effective_user.id)
    
    try:
        await update.message.reply_text(f"ട്രാൻസ്ക്രിപ്ഷൻ: {question}\n\nപ്രോസസ് ചെയ്യുന്നു...\nTranscription: {question}\n\nProcessing...")
        
        initial_state: QueryState = {
            "query_text": question,
            "llm_response": "",
            "user_id": user_id
        }
        
        result = query_flow.invoke(initial_state)
        
        response = result.get("llm_response", "")
        error = result.get("error", "")
        
        if error:
            await update.message.reply_text(f"പിശക്: {error}\nError: {error}")
            return
        
        # Extract Malayalam response
        malayalam_response = extract_malayalam_response(response)
        
        # FIXED: Use the actual user_name for chat history
        #user_name = get_user_name_from_telegram_id(user_id)
        # add_to_conversation(user_name, human_message=question, ai_message=response)
        add_to_conversation(get_user_key(telegram_user_id=user_id), human_message=question, ai_message=response)
        
        # Clear user state
        if user_id in user_states:
            del user_states[user_id]
        
        # Send response
        await send_malayalam_response(update, context, malayalam_response)
        
    except Exception as e:
        logger.error(f"Voice question processing error: {e}")
        await update.message.reply_text("ശബ്ദ ചോദ്യം പ്രോസസ് ചെയ്യുന്നതിൽ പിശക്\nError processing voice question")

async def send_malayalam_response(update: Update, context: ContextTypes.DEFAULT_TYPE, response_text: str):
    """Send Malayalam response in both text and voice format"""
    try:
        # Send text response
        await update.message.reply_text(response_text)
        
        # Generate and send voice response
        # Limit text length for TTS (max ~500 characters for good performance)
        tts_text = response_text[:500] + "..." if len(response_text) > 500 else response_text
        
        audio_data = await text_to_speech_malayalam(tts_text)
        if audio_data:
            await context.bot.send_voice(
                chat_id=update.effective_chat.id,
                voice=audio_data,
                caption="ശബ്ദ മറുപടി / Voice Response"
            )
        
        # Show main menu
        keyboard = [
            [InlineKeyboardButton("🌱 ചെടിയുടെ രോഗം പരിശോധിക്കുക / Plant Disease Check", callback_data="pest_detection")],
            [InlineKeyboardButton("❓ മറ്റൊരു ചോദ്യം / Another Question", callback_data="ask_question")],
            [InlineKeyboardButton("👤 പ്രൊഫൈൽ കാണുക / View Profile", callback_data="view_profile")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("മറ്റെന്തെങ്കിലും സഹായം വേണോ?\nNeed any other help?", reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Error sending Malayalam response: {e}")
        await update.message.reply_text("മറുപടി അയക്കുന്നതിൽ പിശക്\nError sending response")

def create_telegram_app():
    """Create and configure Telegram application"""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    
    return application

# Initialize Telegram bot
telegram_app = create_telegram_app()

# FastAPI endpoints remain the same
@app.get("/")
def root():
    return {
        "message": "Enhanced Plant Disease Diagnosis API with Telegram Bot Integration",
        "endpoints": {
            "/register": "POST - Register user",
            "/analyze": "POST - Upload leaf image (with optional text description)", 
            "/analyze-with-voice": "POST - Upload leaf image with voice description",
            "/analyze-text-only": "POST - Analyze with text description only",
            "/query": "POST - Upload audio file",
            "/user/{user_name}": "GET - Get user info",
            "/user/{user_name}/history": "GET - Get chat history",
            "/webhook": "POST - Telegram webhook endpoint"
        },
        "features": {
            "telegram_bot": "Farmers can use the bot directly on Telegram",
            "malayalam_responses": "AI responds in Malayalam with voice output",
            "flexible_input": "Users can provide descriptions via text, voice, or rely on AI image analysis",
            "langchain_memory": "Uses Neo4jChatMessageHistory for conversation persistence",
            "session_management": "Each user gets unique session ID",
            "graph_storage": "Messages stored as connected nodes with NEXT relationships"
        },
        "telegram_features": {
            "registration": "In-chat registration process",
            "plant_disease_detection": "Photo analysis, text/voice descriptions",
            "agricultural_questions": "Text and voice Q&A",
            "malayalam_tts": "Voice responses in Malayalam",
            "user_profiles": "Profile management through chat"
        },
        "status": "Running with Telegram Bot Integration - Chat History Fixed"
    }

@app.post("/webhook")
async def telegram_webhook(request):
    """Webhook endpoint for Telegram bot"""
    try:
        json_data = await request.json()
        update = Update.de_json(json_data, telegram_app.bot)
        await telegram_app.process_update(update)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error", "message": str(e)}

# All existing FastAPI endpoints remain the same
@app.post("/register")
async def register_user(user_data: UserRegistration):
    try:
        user_name = user_data.user_name.strip()
        if not user_name:
            raise HTTPException(status_code=400, detail="User name cannot be empty")
        
        user_id = user_name
        
        user_info = {
            "user_id": user_id,
            "user_name": user_name,
            "age": user_data.age,
            "district": user_data.district,
            "crops_grown": user_data.crops_grown,
            "farm_size": user_data.farm_size,
            "contact": user_data.contact,
            "registration_date": datetime.now().isoformat()
        }
        
        registered_users[user_id] = user_info
        
        profile_stored = store_user_profile(user_id, user_info)
        
        session_id = get_or_create_session_id(user_name)
        registration_msg = f"User {user_name} registered from {user_data.district or 'Unknown district'}"
        chat_stored = add_to_conversation(user_name, human_message=registration_msg)
        
        return {
            "success": True,
            "message": f"User {user_name} registered with chat memory",
            "user_id": user_id,
            "user_name": user_name,
            "session_id": session_id,
            "profile_stored": profile_stored,
            "chat_history_started": chat_stored,
            "storage_info": f"Profile and chat history stored in Neo4j for: {user_name}"
        }
        
    except Exception as e:
        print(f"Registration error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/analyze")
async def analyze_leaf(
    user_name: str = Form(...), 
    file: UploadFile = File(...),
    user_description: Optional[str] = Form(None)
):
    """Enhanced analyze endpoint - image required, text description optional"""
    if user_name not in registered_users:
        raise HTTPException(status_code=404, detail=f"User '{user_name}' not found.")
    
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Only JPEG and PNG images supported.")
    
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        b64_image = base64.b64encode(contents).decode('utf-8')

        description_source = "user_text" if user_description and user_description.strip() else "image"
        
        initial_state: PestState = {
            "image_b64": b64_image,
            "user_description": user_description.strip() if user_description else None,
            "description_source": description_source,
            "description": "",
            "diagnosis": "",
            "user_id": user_name
        }
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, pest_flow.invoke, initial_state)
        
        description = result.get("description", "")
        diagnosis = result.get("diagnosis", "")
        final_source = result.get("description_source", description_source)
        error = result.get("error", "")
        
        if error:
            raise HTTPException(status_code=500, detail=error)
        
        if final_source == "user_text":
            human_msg = f"Analyzed leaf with user text description: {user_description[:100]}..."
        else:
            human_msg = f"Analyzed leaf image: {file.filename}"
        
        ai_msg = f"Description ({final_source}): {description[:100]}... Diagnosis: {diagnosis[:100]}..."
        
        chat_stored = add_to_conversation(
            user_name, 
            human_message=human_msg,
            ai_message=ai_msg
        )
        
        if not chat_stored:
            logger.error(f"Failed to write chat history for {user_name}")
            # Optional: Surface the error to the user
            raise HTTPException(status_code=500, detail="Analysis completed but failed to save chat history")

        return JSONResponse({
            "success": True,
            "message": f"Analysis completed using {final_source} description",
            "user_name": user_name,
            "description_source": final_source,
            "chat_history_updated": chat_stored,
            "description": description,
            "diagnosis": diagnosis,
            "storage_info": f"Added to Neo4j chat memory for: {user_name}"
        })
        
    except Exception as e:
        print(f"Error in analyze_leaf: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/analyze-with-voice")
async def analyze_leaf_with_voice(
    user_name: str = Form(...), 
    file: UploadFile = File(...),
    description_audio: UploadFile = File(...)
):
    """Analyze leaf with image and voice description"""
    if user_name not in registered_users:
        raise HTTPException(status_code=404, detail=f"User '{user_name}' not found.")
    
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Only JPEG and PNG images supported.")
    
    try:
       
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded image file is empty.")
        
        b64_image = base64.b64encode(contents).decode('utf-8')
        
       
        audio_contents = await description_audio.read()
        if not audio_contents:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
        
        user_voice_description = await transcribe_audio_with_gemini(
            audio_contents, 
            description_audio.filename or "description.wav"
        )
        
        initial_state: PestState = {
            "image_b64": b64_image,
            "user_description": user_voice_description,
            "description_source": "user_voice",
            "description": "",
            "diagnosis": "",
            "user_id": user_name
        }
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, pest_flow.invoke, initial_state)
        
        description = result.get("description", "")
        diagnosis = result.get("diagnosis", "")
        error = result.get("error", "")
        
        if error:
            raise HTTPException(status_code=500, detail=error)
        
        human_msg = f"Analyzed leaf with voice description: {user_voice_description[:100]}..."
        ai_msg = f"Description (user_voice): {description[:100]}... Diagnosis: {diagnosis[:100]}..."
        
        chat_stored = add_to_conversation(
            user_name, 
            human_message=human_msg,
            ai_message=ai_msg
        )
        if not chat_stored:
            logger.error(f"Failed to write chat history for {user_name}")
            # Optional: Surface the error to the user
            raise HTTPException(status_code=500, detail="Analysis completed but failed to save chat history")
        
        return JSONResponse({
            "success": True,
            "message": f"Analysis completed using user voice description",
            "user_name": user_name,
            "description_source": "user_voice",
            "transcribed_description": user_voice_description,
            "chat_history_updated": chat_stored,
            "description": description,
            "diagnosis": diagnosis,
            "storage_info": f"Added to Neo4j chat memory for: {user_name}"
        })
        
    except Exception as e:
        print(f"Error in analyze_leaf_with_voice: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/analyze-text-only")
async def analyze_text_only(
    user_name: str = Form(...), 
    user_description: str = Form(...)
):
    """Analyze using only text description (no image required)"""
    if user_name not in registered_users:
        raise HTTPException(status_code=404, detail=f"User '{user_name}' not found.")
    
    if not user_description or not user_description.strip():
        raise HTTPException(status_code=400, detail="User description cannot be empty.")
    
    try:
        initial_state: PestState = {
            "image_b64": None,
            "user_description": user_description.strip(),
            "description_source": "user_text",
            "description": "",
            "diagnosis": "",
            "user_id": user_name
        }
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, pest_flow.invoke, initial_state)
        
        description = result.get("description", "")
        diagnosis = result.get("diagnosis", "")
        error = result.get("error", "")
        
        if error:
            raise HTTPException(status_code=500, detail=error)
        
        human_msg = f"Provided text description: {user_description[:100]}..."
        ai_msg = f"Analysis based on user description: {diagnosis[:100]}..."
        
        chat_stored = add_to_conversation(
            user_name, 
            human_message=human_msg,
            ai_message=ai_msg
        )

        if not chat_stored:
            logger.error(f"Failed to write chat history for {user_name}")
            # Optional: Surface the error to the user
            raise HTTPException(status_code=500, detail="Analysis completed but failed to save chat history")
        
        return JSONResponse({
            "success": True,
            "message": f"Analysis completed using text description only",
            "user_name": user_name,
            "description_source": "user_text",
            "chat_history_updated": chat_stored,
            "description": description,
            "diagnosis": diagnosis,
            "storage_info": f"Added to Neo4j chat memory for: {user_name}"
        })
        
    except Exception as e:
        print(f"Error in analyze_text_only: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/query")
async def handle_query(user_name: str = Form(...), audio: UploadFile = File(...)):
    if user_name not in registered_users:
        raise HTTPException(status_code=404, detail=f"User '{user_name}' not found.")
    
    try:
        audio_contents = await audio.read()
        if not audio_contents:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
        
        query_text = await transcribe_audio_with_gemini(audio_contents, audio.filename or "audio.wav")
        
        if not query_text or query_text.strip() == "":
            raise HTTPException(status_code=400, detail="Could not transcribe audio.")

        initial_state: QueryState = {
            "query_text": query_text.strip(),
            "llm_response": "",
            "user_id": user_name
        }
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, query_flow.invoke, initial_state)
        
        llm_response = result.get("llm_response", "")
        error = result.get("error", "")
        
        if error:
            raise HTTPException(status_code=500, detail=error)
        
        chat_stored = add_to_conversation(
            user_name,
            human_message=query_text.strip(),
            ai_message=llm_response
        )
        
        if not chat_stored:
            logger.error(f"Failed to write chat history for {user_name}")
            raise HTTPException(status_code=500, detail="Query processed but failed to save to chat history")

        return JSONResponse({
            "success": True,
            "message": f"Query processed and added to chat history for {user_name}",
            "user_name": user_name,
            "chat_history_updated": chat_stored,
            "transcribed_text": query_text.strip(),
            "response": llm_response,
            "storage_info": f"Added to Neo4j chat memory for: {user_name}"
        })
        
    except Exception as e:
        print(f"Error in handle_query: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/user/{user_name}")
async def get_user_info(user_name: str):
    if user_name not in registered_users:
        raise HTTPException(status_code=404, detail=f"User '{user_name}' not found")
    
    profile = get_user_profile(user_name)
    
    history = get_conversation_history(user_name)
    
    return {
        "success": True,
        "user": profile,
        "session_id": user_sessions.get(user_name),
        "total_messages": len(history),
        "recent_history": history[-10:] if history else [],
        "storage_info": f"Profile and chat history from Neo4j for: {user_name}"
    }

@app.get("/user/{user_name}/history")
async def get_complete_chat_history(user_name: str, limit: int = 100):
    if user_name not in registered_users:
        raise HTTPException(status_code=404, detail=f"User '{user_name}' not found")
    
    history = get_conversation_history(user_name)
    
    return {
        "success": True,
        "user_name": user_name,
        "session_id": user_sessions.get(user_name),
        "total_messages": len(history),
        "chat_history": history[-limit:] if history else [],
        "storage_info": f"Complete chat history from Neo4j for: {user_name}"
    }

async def setup_webhook():
    """Set up webhook for Telegram bot"""
    webhook_url = "YOUR_DOMAIN/webhook" 
    await telegram_app.bot.set_webhook(url=webhook_url)
    print(f"Webhook set up at: {webhook_url}")

async def run_telegram_bot():
    """Run Telegram bot in polling mode"""
    try:
        print("🤖 Starting Telegram bot...")
        await telegram_app.initialize()
        await telegram_app.start()
        await telegram_app.updater.start_polling(drop_pending_updates=True)
        print("✅ Telegram bot started successfully!")
        print(f"Bot username: @{telegram_app.bot.username}")
        
        # Keep running forever until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Bot interrupted by user")
    except Exception as e:
        print(f"❌ Bot error: {e}")
    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        if telegram_app.updater.running:
            await telegram_app.updater.stop()
        await telegram_app.stop()
        await telegram_app.shutdown()

if __name__ == "__main__":
    import uvicorn
    import threading
    import time
    
    print("🚀 Starting application...")
    
    # Start FastAPI in background
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    time.sleep(2)
    print("✅ FastAPI running on http://localhost:8000")
    
    # Run bot (blocks here)
    try:
        asyncio.run(run_telegram_bot())
    except KeyboardInterrupt:
        print("👋 Goodbye!")
