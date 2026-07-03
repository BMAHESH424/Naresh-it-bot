# 🎓 AI-Powered Course Assistant Chatbot

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot built using **Google Gemini, LangChain, FAISS, and Streamlit** to help students explore NareshIT course information through natural conversations.

Instead of relying on hardcoded responses, the application dynamically retrieves course content from official NareshIT course pages, indexes it into a vector database, and generates context-aware answers using Google's Gemini Large Language Model.

---

## 🚀 Features

- 🤖 AI-powered conversational assistant using Gemini 2.5 Flash
- 📚 Retrieval-Augmented Generation (RAG)
- 🔍 Dynamic retrieval from official NareshIT course pages
- 🧠 FAISS vector database for semantic search
- 🌍 Multilingual responses
- 🎤 Voice input using SpeechRecognition
- 🔊 Text-to-Speech using Google TTS
- 📜 Chat history management
- 💾 Export conversations as JSON and Markdown
- 🎨 Modern responsive Streamlit UI
- ⚡ Cached embeddings and vector database for faster performance
- 📋 Quick course registration integration

---

# Problem Statement

Students often have questions about:

- Course syllabus
- Curriculum
- Prerequisites
- Batch information
- Technologies covered
- Duration
- Learning roadmap

Searching through multiple webpages can be time-consuming.

This chatbot solves the problem by providing an AI assistant that understands natural language questions and answers them using the official course content.

---

# Tech Stack

| Category | Technology |
|------------|------------|
| Language | Python |
| Framework | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Gemini Embedding 001 |
| RAG Framework | LangChain |
| Vector Database | FAISS |
| Web Scraping | WebBaseLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| Voice Input | SpeechRecognition |
| Text-to-Speech | Google TTS |
| Environment | python-dotenv |

---

# Architecture

```
                User Question
                      │
                      ▼
              Streamlit Interface
                      │
                      ▼
        Course Selection (NareshIT URL)
                      │
                      ▼
            WebBaseLoader loads webpage
                      │
                      ▼
      Recursive Character Text Splitter
                      │
                      ▼
       Gemini Embedding Model (Embedding-001)
                      │
                      ▼
            FAISS Vector Database
                      │
                      ▼
          Similarity Search (Retriever)
                      │
                      ▼
      LangChain Prompt + Gemini 2.5 Flash
                      │
                      ▼
            Context-aware AI Response
                      │
          ┌───────────┴────────────┐
          ▼                        ▼
   Multilingual Output       Text-to-Speech
```

---

# Workflow

### Step 1

User selects a course.

Example:

- Full Stack Python
- Data Science & AI
- Java
- Django
- Power BI

---

### Step 2

The application loads the official NareshIT course webpage.

---

### Step 3

The webpage content is split into semantic chunks using RecursiveCharacterTextSplitter.

---

### Step 4

Each chunk is converted into vector embeddings using Gemini Embedding-001.

---

### Step 5

Embeddings are stored inside a FAISS vector database.

---

### Step 6

When a user asks a question:

- Similar chunks are retrieved
- Relevant context is injected into the prompt
- Gemini generates an accurate response grounded on retrieved information

---

### Step 7

The response can optionally be:

- translated into multiple languages
- converted into speech
- exported as Markdown or JSON

---

# Supported Courses

- Full Stack Python
- Full Stack Data Science & AI
- Java Full Stack
- .NET Full Stack
- Django
- Spring Boot
- React UI Full Stack
- Tableau
- Power BI
- MySQL
- Software Testing

---

# Key Functionalities

## Retrieval-Augmented Generation (RAG)

Instead of asking Gemini directly, the chatbot first retrieves the most relevant course content and then generates answers using retrieved context.

This significantly reduces hallucinations and improves answer accuracy.

---

## Dynamic Course Loading

The chatbot does not use static datasets.

It automatically fetches the latest course content from NareshIT webpages.

---

## Multilingual Support

Users can receive answers in multiple languages including:

- English
- Hindi
- Telugu
- Tamil
- Kannada
- Malayalam
- Bengali
- Gujarati
- Marathi
- Punjabi
- Urdu

---

## Voice Features

- Speech-to-Text using SpeechRecognition
- Text-to-Speech using Google TTS

---

## Conversation Export

Users can download chat history as:

- JSON
- Markdown

---

## Session Management

Each course maintains its own independent conversation history using Streamlit Session State.

---

# Project Structure

```
project/
│
├── naresh_bot.py
├── .env
├── requirements.txt
├── README.md
└── assets/
```

---

# Environment Variables

```
Gemini_Gcp=YOUR_GEMINI_API_KEY
CONTACT_PHONE=+91XXXXXXXXXX
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/nareshit-course-chatbot.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run naresh_bot.py
```

---

# Future Improvements

- PDF knowledge base support
- YouTube lecture indexing
- Persistent FAISS storage
- User authentication
- Analytics dashboard
- Batch schedule integration
- WhatsApp chatbot
- Instructor-specific assistants
- Feedback collection
- Conversation memory using LangGraph

---

# Learning Outcomes

This project demonstrates practical experience with:

- Generative AI
- Retrieval-Augmented Generation (RAG)
- LangChain
- Prompt Engineering
- Vector Databases
- Semantic Search
- Google Gemini APIs
- Streamlit
- AI-powered Web Applications
- REST-ready AI architecture

---

# Author

**Mahesh Bonthala**

AI Engineer | Python | Generative AI | RAG | LangChain | FastAPI

GitHub: https://github.com/yourusername

LinkedIn: https://linkedin.com/in/yourprofile
