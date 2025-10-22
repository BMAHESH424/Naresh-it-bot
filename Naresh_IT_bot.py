import os
import base64
import json
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# Optional features (voice input / TTS / translation)
try:
    import speech_recognition as sr  # type: ignore
except Exception:  # pragma: no cover
    sr = None  # type: ignore

try:
    from gtts import gTTS  # type: ignore
except Exception:  # pragma: no cover
    gTTS = None  # type: ignore

try:
    from deep_translator import GoogleTranslator  # type: ignore
except Exception:  # pragma: no cover
    GoogleTranslator = None  # type: ignore

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ===== Hero Section =====
st.markdown("""
<div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin: 0; padding: 0;">
  <span style="font-size: 3.2rem; padding: 0;">🎓</span>
  <h1 style="font-size: 3.2rem; font-weight: 800; margin: 0; padding: 10; line-height: 1;
             background: linear-gradient(135deg, #6366f1, #ec4899, #06b6d4);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
    NareshIT Course Assistant
  </h1>
</div>
""", unsafe_allow_html=True)

# ===== Styles =====
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
      --primary: #6366f1; /* indigo-500 */
      --primary-dark: #4f46e5; /* indigo-600 */
      --primary-light: #a5b4fc; /* indigo-300 */
      --secondary: #ec4899; /* pink-500 */
      --accent: #06b6d4; /* cyan-500 */
      --success: #10b981; /* emerald-500 */
      --warning: #f59e0b; /* amber-500 */
      --danger: #ef4444; /* red-500 */
      
      --bg-primary: #0a0a0a; /* near black */
      --bg-secondary: #111111; /* dark gray */
      --bg-tertiary: #1a1a1a; /* lighter dark */
      --bg-card: #1e1e1e; /* card background */
      --bg-glass: rgba(30, 30, 30, 0.8); /* glass effect */
      
      --text-primary: #ffffff;
      --text-secondary: #a1a1aa; /* zinc-400 */
      --text-muted: #71717a; /* zinc-500 */
      --text-accent: #e4e4e7; /* zinc-200 */
      
      --border: #27272a; /* zinc-800 */
      --border-light: #3f3f46; /* zinc-700 */
      --shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
      --shadow-lg: 0 35px 60px -12px rgba(0, 0, 0, 0.6);
    }

    * { box-sizing: border-box; }
    
    .stApp { 
      background: linear-gradient(135deg, var(--bg-primary) 0%, #0f0f23 50%, var(--bg-primary) 100%);
      color: var(--text-primary);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      min-height: 100vh;
    }

    /* Animated Background */
    .stApp::before {
      content: '';
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: 
        radial-gradient(circle at 20% 80%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(236, 72, 153, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 40% 40%, rgba(6, 182, 212, 0.1) 0%, transparent 50%);
      z-index: -1;
      animation: float 20s ease-in-out infinite;
    }

    @keyframes float {
      0%, 100% { transform: translateY(0px) rotate(0deg); }
      50% { transform: translateY(-20px) rotate(180deg); }
    }


    /* Premium Cards */
    .card { 
      background: var(--bg-card); 
      border: 1px solid var(--border); 
      border-radius: 20px; 
      padding: 24px; 
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      transition: all 0.3s ease;
    }
    
    .card:hover {
      transform: translateY(-2px);
      box-shadow: var(--shadow-lg);
      border-color: var(--border-light);
    }
    
    .card-tonal { 
      background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-tertiary) 100%);
      border: 1px solid var(--border); 
      border-radius: 20px; 
      padding: 28px; 
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }

    /* Modern Chat Interface */
    .chat-container {
      font-size: 1.2rem;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      position: relative;
      overflow: hidden;
    }

    .course-container {
      font-size: 1.2rem;
      background: #8D8D9E;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      position: relative;
      text-align: center;
      overflow: hidden;
    }
    
    .chat-container::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, var(--primary), transparent);
    }

    .stChatMessage { 
      border-radius: 20px; 
      padding: 16px 20px; 
      margin: 12px 0; 
      max-width: 85%;
      backdrop-filter: blur(10px);
      transition: all 0.3s ease;
    }
    
    .stChatMessage:hover {
      transform: translateX(4px);
    }
    
    .user-msg { 
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
      border: 1px solid rgba(99, 102, 241, 0.3);
      color: white; 
      margin-left: auto;
      box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
    }
    
    .bot-msg { 
      background: linear-gradient(135deg, var(--bg-tertiary) 0%, var(--bg-card) 100%);
      border: 1px solid var(--border);
      color: var(--text-primary);
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] { 
      background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
      border-right: 1px solid var(--border); 
      color: var(--text-primary);
    }
    .sidebar-banner {
      background: radial-gradient(1200px 200px at 0% -10%, rgba(99,102,241,0.25), transparent),
                  radial-gradient(1000px 200px at 100% -20%, rgba(236,72,153,0.25), transparent),
                  linear-gradient(135deg, rgba(30,30,30,0.9), rgba(20,20,30,0.9));
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      margin: 0 0 16px 0;
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }
    .sidebar-banner::after {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(90deg, transparent, rgba(99,102,241,0.15), transparent);
      height: 1px;
      top: 0;
    }
    .sidebar-header {
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 0 0 16px 0;
    }
    .sidebar-header h2 {
      font-size: 1.4rem;
      font-weight: 800;
      margin: 0;
      background: linear-gradient(135deg, var(--primary), var(--secondary), var(--accent));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      letter-spacing: 0.3px;
    }
    .sidebar-icon {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      display: grid;
      place-items: center;
      background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(6,182,212,0.15));
      border: 1px solid var(--border);
    }
    .sidebar-subtext {
      font-size: 0.9rem;
      color: var(--text-secondary);
      margin: -4px 0 12px 0;
    }
    .sidebar-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      margin-bottom: 14px;
    }
    .sidebar-card:hover {
      border-color: var(--border-light);
      transform: translateY(-1px);
      transition: all 0.2s ease;
    }
    
    .section-title { 
      font-weight: 700; 
      font-size: 1.125rem;
      color: var(--text-primary); 
      margin: 0 0 1rem 0;
      padding-bottom: 0.5rem;
      border-bottom: 2px solid var(--primary);
    }
    .section-subtitle {
      font-weight: 700;
      font-size: 0.95rem;
      color: var(--text-accent);
      margin: 0 0 10px 0;
    }
    .divider { height: 1px; background: var(--border); margin: 10px 0; }
    .muted { color: var(--text-muted); font-size: 0.9rem; }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--bg-tertiary);
      font-size: 0.8rem;
      color: var(--text-accent);
    }
    .status-pill { padding: 4px 10px; border-radius: 999px; font-size: 0.75rem; border: 1px solid var(--border); }
    .status-ok { background: rgba(16,185,129,0.12); color: #10b981; border-color: rgba(16,185,129,0.35); }
    .status-warn { background: rgba(245,158,11,0.12); color: #f59e0b; border-color: rgba(245,158,11,0.35); }
    .status-err { background: rgba(239,68,68,0.12); color: #ef4444; border-color: rgba(239,68,68,0.35); }
    

    /* Buttons */
    .stButton > button {
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
      color: white;
      border: none;
      border-radius: 12px;
      padding: 0.75rem 1.5rem;
      font-weight: 600;
      transition: all 0.3s ease;
      box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
    }

    /* Form Elements */
    .stTextInput > div > div > input {
      background: var(--bg-tertiary);
      border: 1px solid var(--border);
      border-radius: 12px;
      color: var(--text-primary);
      padding: 0.75rem 1rem;
      font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
      gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
      background: var(--bg-tertiary);
      border: 1px solid var(--border);
      border-radius: 12px;
      color: var(--text-secondary);
      font-weight: 600;
      padding: 0.75rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
      color: white;
      border-color: var(--primary);
    }

    /* Links */
    a, .stMarkdown a { 
      color: var(--primary-light);
      text-decoration: none;
      transition: color 0.3s ease;
    }
    
    a:hover, .stMarkdown a:hover {
      color: var(--primary);
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
      width: 8px;
    }
    
    ::-webkit-scrollbar-track {
      background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
      background: var(--primary);
      border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
      background: var(--primary-dark);
    }

    /* Hide default Streamlit elements */
    .stApp > header { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
    .stStatusWidget { display: none; }
    
    /* Remove default container padding */
    .main .block-container { 
      padding-top: 0 !important; 
      padding-bottom: 0 !important; 
    }
    /* Floating contact */
    .floating-contact {
      position: fixed;
      right: 16px;
      bottom: 16px;
      background: linear-gradient(135deg, var(--primary), var(--primary-dark));
      color: #fff;
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 999px;
      padding: 10px 14px;
      box-shadow: var(--shadow-lg);
      z-index: 9999;
      font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===== Env / Setup =====
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
contact_number = os.getenv("CONTACT_PHONE", "+91-90000-00000").strip()


# ===== Helpers =====
LANGUAGE_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Punjabi": "pa",
    "Urdu": "ur",
}


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def load_and_split_from_url(url: str) -> List[Any]:
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    return splitter.split_documents(docs)


@st.cache_resource(show_spinner=False)
def build_vectordb_for_url(url: str, persist_dir: str = "") -> FAISS:
    texts = load_and_split_from_url(url)
    embeddings = get_embeddings()
    # FAISS is in-memory; 'persist_dir' kept for API compatibility but not used
    return FAISS.from_documents(texts, embedding=embeddings)


def tts_to_audio_tag(text: str, lang_code: str) -> str:
    if not gTTS:
        return ""
    try:
        tts = gTTS(text, lang=lang_code)
        tmp_path = "response.mp3"
        tts.save(tmp_path)
        with open(tmp_path, "rb") as f:
            b64_audio = base64.b64encode(f.read()).decode()
        return f"""
        <audio controls>
            <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3" />
        </audio>
        """
    except Exception:
        return ""


def maybe_translate(text: str, target_lang_code: str) -> str:
    if not GoogleTranslator or target_lang_code == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang_code).translate(text)
    except Exception:
        return text



# ===== Sidebar =====
with st.sidebar:
    st.markdown('<div class="sidebar-card">🌐 Language</div>', unsafe_allow_html=True)
    target_language_name = st.selectbox("Response Language", list(LANGUAGE_MAP.keys()), index=0)
    target_lang_code = LANGUAGE_MAP[target_language_name]
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">🎛️ I/O Options</div>', unsafe_allow_html=True)
    enable_voice = st.checkbox("Microphone input", value=False, disabled=(sr is None))
    enable_tts = st.checkbox("Text-to-speech", value=False, disabled=(gTTS is None))
    st.markdown('</div>', unsafe_allow_html=True)

    # Export & Share
    st.markdown('<div class="sidebar-card">🔗 Export & Share</div>', unsafe_allow_html=True)
    has_msgs = bool(st.session_state.get("messages"))
    try:
        history_json_str = json.dumps(st.session_state.get("messages", []), ensure_ascii=False, indent=2)
    except Exception:
        history_json_str = "[]"

    lines = ["# Conversation Transcript\n"]
    for i, msg in enumerate(st.session_state.get("messages", []), start=1):
        who = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").replace("\r", "")
        lines.append(f"## {i}. {who}\n\n{content}\n")
    transcript_md = "\n".join(lines)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.download_button(
            label="⬇️ JSON",
            data=history_json_str.encode("utf-8"),
            file_name=f"chat_history_{timestamp}.json",
            mime="application/json",
            use_container_width=True,
            disabled=not has_msgs,
        )
    with col_e2:
        st.download_button(
            label="⬇️ Markdown",
            data=transcript_md.encode("utf-8"),
            file_name=f"chat_history_{timestamp}.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=not has_msgs,
        )
    if not has_msgs:
        st.markdown('<div class="muted">No messages yet. Start a chat to enable export.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ===== Session State =====
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of dicts {role: "user"|"assistant", content: str}

if "retriever_ready" not in st.session_state:
    st.session_state["retriever_ready"] = False

# ===== Actions =====
action_clear, action_adv = st.columns([1,1])

with action_clear:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"].clear()
        st.toast("💬 Chat history cleared")
    st.markdown('</div>', unsafe_allow_html=True)

with action_adv:
        course_options: Dict[str, str] = {
        "Select Course":"",
        "Full Stack Python Online Training": "https://nareshit.com/courses/full-stack-python-online-training",
        "Full Stack Data Science & AI": "https://nareshit.com/courses/full-stack-data-science-ai-online-training",
        "Full Stack Software Testing" : "https://nareshit.com/courses/full-stack-software-testing-online-training",
        "UI Full Stack Web Development With React":"https://nareshit.com/courses/ui-full-stack-web-development-with-react-online-training",
        "Full Stack Dot Net Core":"https://nareshit.com/courses/full-stack-dot-net-core-online-training",
        "Full Stack Python":"https://nareshit.com/courses/full-stack-python-online-training",
        "Full Stack Java":"https://nareshit.com/courses/full-stack-java-online-training",
        "Spring Boot MicroServices":"https://nareshit.com/courses/spring-boot-microservices-online-training",
        "Django":"https://nareshit.com/courses/django-online-training",
        "Tableau":"https://nareshit.com/courses/tableau-online-training",
        "Power BI":"https://nareshit.com/courses/power-bi-online-training",
        "MySQL":"https://nareshit.com/courses/mysql-online-training"
    }

        def _on_course_change():
            name = st.session_state.get("selected_course_name")
            url = course_options.get(name, "")
            st.session_state["active_url"] = url
            if not url:
                st.session_state["retriever_ready"] = False
                return
            with st.spinner("🤖 Processing course content with AI..."):
                try:
                    persist_dir = os.path.join("./vectordb", "web")
                    vectordb = build_vectordb_for_url(url, persist_dir=persist_dir)
                    st.session_state["vectordb"] = vectordb
                    st.session_state["retriever_ready"] = True
                    st.toast("✅ Course content loaded and indexed successfully!")
                except Exception as err:
                    st.session_state["retriever_ready"] = False
                    st.error(f"❌ Failed to process course content: {err}")
        selected_course_name = st.selectbox("", list(course_options.keys()), key="selected_course_name", on_change=_on_course_change)
        selected_course_url = course_options[selected_course_name]
        active_url = st.session_state.get("active_url", selected_course_url)
        if not st.session_state.get("retriever_ready"):
          _on_course_change()
        st.markdown('</div>', unsafe_allow_html=True)


# ===== Chat =====
chat_tab, history_tab = st.tabs(["💬 Chat", "📜 History"])

with chat_tab:
    st.markdown('<div class="chat-container"> 💬 AI Course Assistant</div>', unsafe_allow_html=True)
    st.markdown("Ask intelligent questions about your course content")
    
    # Show message history
    for msg in st.session_state["messages"]:
        css_cls = "user-msg" if msg["role"] == "user" else "bot-msg"
        st.markdown(f"<div class='stChatMessage {css_cls}'>{msg['content']}</div>", unsafe_allow_html=True)

    # Process any pending query first (so answer appears above input)
    pending_query = st.session_state.get("pending_query")
    if pending_query:
        # Display user query immediately
        st.markdown(f"<div class='stChatMessage user-msg'>{pending_query}</div>", unsafe_allow_html=True)
        
        if not st.session_state.get("retriever_ready"):
            st.info("⏳ Loading course content automatically. Please wait a moment and try again.")
        elif not groq_api_key:
            st.stop()
        else:
            # Prepare RAG chain
            embeddings = get_embeddings()
            vectordb = st.session_state.get("vectordb")
            retriever = vectordb.as_retriever(search_kwargs={"k": 5}) if vectordb else None

            prompt_template = (
                "You are a helpful course assistant. Answer using only the provided context.\n"
                "If unsure, say 'I am not certain; please check the course page.'\n\n"
                "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:" \
                "Instructions : Write a friendly, one-sentence popup message reminding users to save their work" \
                "auto generate the text related to the questions in search bar to reduce the users time  " \
                "popup the text related to the loaded data while searching"
                                
            )
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

            llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile", temperature=0.4, max_tokens=512)

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt},
            )

            # Add user message to history
            st.session_state["messages"].append({"role": "user", "content": pending_query})
            
            with st.spinner("Thinking..."):
                try:
                    result = qa.invoke({"query": pending_query})
                    answer = result.get("result", "")
                    sources = result.get("source_documents", [])
                except Exception as run_err:
                    answer = f"There was an error answering the question: {run_err}"
                    sources = []

            # Translate if needed
            final_answer = maybe_translate(answer, target_lang_code)
            st.session_state["messages"].append({"role": "assistant", "content": final_answer})

            # Render AI response
            st.markdown(f"<div class='stChatMessage bot-msg'>{final_answer}</div>", unsafe_allow_html=True)

            # TTS
            if enable_tts and final_answer:
                audio_tag = tts_to_audio_tag(final_answer, target_lang_code)
                if audio_tag:
                    st.markdown(audio_tag, unsafe_allow_html=True)
        
        # Clear pending query
        st.session_state["pending_query"] = None

    if not groq_api_key:
        st.warning("⚠️ GROQ_API_KEY is missing. Add it to your environment to use the AI assistant.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Input row (moved to bottom)
    with st.form(key="chat_form", clear_on_submit=True):
        user_query = st.text_input("💭 Ask about the course content", placeholder="e.g., What are the prerequisites for this course?")
        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button("🚀 Send", type="primary", use_container_width=True)
        with col2:
            if enable_voice and sr is not None:
                if st.form_submit_button("🎤 Voice", use_container_width=True):
                    recognizer = sr.Recognizer()
                    try:
                        with st.spinner("🎙️ Listening..."):
                            with sr.Microphone() as source:
                                recognizer.adjust_for_ambient_noise(source, duration=1)
                                audio = recognizer.listen(source, timeout=5)
                            voice_text = recognizer.recognize_google(audio)
                            st.info(f"🎤 Captured: {voice_text}")
                    except Exception as mic_err:
                        st.error(f"❌ Microphone error: {mic_err}")

    if submitted and user_query:
        st.session_state["pending_query"] = user_query
        st.rerun()


with history_tab:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("### 📜 Conversation History")
    st.markdown("Review your previous interactions with the AI assistant")
    
    if st.session_state["messages"]:
        for i, msg in enumerate(st.session_state["messages"], start=1):
            role = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
            css_cls = "user-msg" if msg["role"] == "user" else "bot-msg"
            st.markdown(f"<div class='stChatMessage {css_cls}'><strong>{i}. {role}:</strong> {msg['content']}</div>", unsafe_allow_html=True)

    else:
        st.info("💬 No conversation yet. Start chatting in the Chat tab!")
    st.markdown('</div>', unsafe_allow_html=True)

# Floating contact number
st.markdown(f'<div class="floating-contact">📞 +91 8179191999</div>', unsafe_allow_html=True)


