"""
Medibot API backend powered by FastAPI + LangGraph.
"""

import os
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional

import PyPDF2
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langgraph.graph import END, StateGraph

    LANGCHAIN_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    LANGCHAIN_AVAILABLE = False
    LANGCHAIN_ERROR = str(exc)

from dotenv import load_dotenv

load_dotenv()

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "").strip()
CHROMA_DB_PATH = "chroma_db"
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "ta": "Tamil",
    "ml": "Malayalam",
    "te": "Telugu",
    "hi": "Hindi",
}

EMERGENCY_KEYWORDS = [
    "chest pain",
    "heart attack",
    "stroke",
    "breathless",
    "suicide",
    "severe bleeding",
    "unconscious",
    "seizure",
    "anaphylaxis",
    "மார்பு வலி",
    "గుండె నొప్పి",
    "ഹൃദയാഘാതം",
    "दिल का दौरा",
]

AGENT_GRAPH = None
RETRIEVER = None


class AgentState(dict):
    """Mutable state carried by LangGraph nodes."""

    messages: List[Dict[str, str]]
    current_query: str
    language: str
    uploaded_files: List[str]
    uploaded_report_text: str
    context: str
    retrieval_results: List[str]
    web_search_results: str
    agent_decision: str
    final_response: str
    conversation_history: List[Dict[str, str]]
    user_location: str
    model_used: Optional[str]


def call_perplexity_api(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.2,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    model: str = "sonar-pro",
) -> Dict[str, Any]:
    if not PERPLEXITY_API_KEY:
        return {
            "success": False,
            "content": "",
            "error": "PERPLEXITY_API_KEY missing",
            "model_used": model,
            "citations": [],
        }

    try:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if conversation_history:
            filtered: List[Dict[str, str]] = []
            last_role: Optional[str] = None
            for msg in conversation_history[-6:]:
                role = msg.get("role", "user")
                content = msg.get("content", "").strip()
                if role in ("user", "assistant") and content and role != last_role:
                    filtered.append({"role": role, "content": content[:400]})
                    last_role = role
            messages.extend(filtered)

        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += f"\n\nFollow-up: {prompt}"
        else:
            messages.append({"role": "user", "content": prompt})

        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2000,
            },
            timeout=60,
        )

        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            return {
                "success": True,
                "content": content,
                "citations": data.get("citations", [])[:3],
                "error": None,
                "model_used": model,
            }

        error_detail = response.json().get("error", {}).get("message", response.text)
        return {
            "success": False,
            "content": "",
            "citations": [],
            "error": f"API Error {response.status_code}: {error_detail}",
            "model_used": model,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "success": False,
            "content": "",
            "citations": [],
            "error": str(exc),
            "model_used": model,
        }


def setup_advanced_rag():
    if not LANGCHAIN_AVAILABLE:
        return None, f"LangChain unavailable: {LANGCHAIN_ERROR}"

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH, embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        return retriever, "RAG ready"
    except Exception as exc:  # pragma: no cover
        return None, f"RAG setup error: {exc}"


def agent_router(state: AgentState) -> str:
    query = state["current_query"].lower()
    route = "hybrid_medical"

    if any(keyword in query for keyword in EMERGENCY_KEYWORDS):
        route = "emergency"
    elif state.get("uploaded_report_text"):
        route = "file_analysis"
    elif any(kw in query for kw in ["doctor", "hospital", "clinic", "find"]):
        route = "doctor_search"

    state["agent_decision"] = route
    return route


def retrieve_from_documents(state: AgentState) -> AgentState:
    if RETRIEVER:
        try:
            docs = RETRIEVER.get_relevant_documents(state["current_query"])
            state["retrieval_results"] = [doc.page_content for doc in docs[:4]]
        except Exception as exc:  # pragma: no cover
            state["retrieval_results"] = [f"Retriever error: {exc}"]
    else:
        state["retrieval_results"] = []
    return state


def web_search_node(state: AgentState) -> AgentState:
    conv_context = "\n".join(
        f"{m['role']}: {m['content'][:200]}"
        for m in state.get("conversation_history", [])[-3:]
    )
    enhanced_query = (
        f"Context: {conv_context}\n\nQuestion: {state['current_query']}"
        if conv_context
        else state["current_query"]
    )

    result = call_perplexity_api(
        enhanced_query,
        system_prompt=(
            "You are a medical research assistant with real-time access. "
            "Provide evidence-based medical information. Use bullet points."
        ),
        temperature=0.3,
        conversation_history=state.get("conversation_history"),
    )
    state["web_search_results"] = (
        f"{result['content']}\n\n🌐 Sources:\n" + "\n".join(result.get("citations", []))
        if result["success"]
        else f"Search error: {result['error']}"
    )
    return state


def synthesize_response(state: AgentState) -> AgentState:
    context_parts: List[str] = []
    if state.get("retrieval_results"):
        context_parts.append(
            "📄 Document Context:\n" + "\n".join(state["retrieval_results"][:2])
        )
    if state.get("web_search_results"):
        context_parts.append("🌐 Live Medical Information:\n" + state["web_search_results"])

    composed_context = "\n\n".join(context_parts) if context_parts else "No external context"

    synthesis_prompt = (
        f"Context:\n{composed_context}\n\nConversation history:\n"
        + "\n".join(
            f"{m['role'].upper()}: {m['content'][:300]}"
            for m in state.get("conversation_history", [])[-4:]
        )
        + f"\n\nCurrent query: {state['current_query']}\n"
        "Provide a comprehensive, user-friendly response with headings, bullets, and disclaimer."
    )

    result = call_perplexity_api(
        synthesis_prompt,
        system_prompt="You are Medibot, an AI medical assistant.",
        temperature=0.3,
        conversation_history=state.get("conversation_history"),
    )

    if result["success"]:
        response = result["content"]
        if state.get("retrieval_results"):
            response += "\n\n📄 Sources: Internal medical documents"
        if state.get("web_search_results"):
            response += "\n\n🌐 Live sources: Trusted medical databases"
        response += (
            f"\n\n🤖 Model: {result.get('model_used', 'sonar-pro')}"
            "\n---\n⚕️ Disclaimer: Educational information only."
        )
        state["final_response"] = response
        state["model_used"] = result.get("model_used")
    else:
        state["final_response"] = f"Unable to generate response: {result['error']}"
        state["model_used"] = result.get("model_used")
    return state


def emergency_response_node(state: AgentState) -> AgentState:
    state["final_response"] = (
        "🚨 MEDICAL EMERGENCY DETECTED 🚨\n\n"
        "Call your local emergency number immediately (e.g., 911 / 112 / 999).\n"
        "Stay with the person, keep them calm, and follow operator instructions."
    )
    state["model_used"] = None
    return state


def doctor_search_node(state: AgentState) -> AgentState:
    location = state.get("user_location", "")
    query = state["current_query"]
    if location:
        query += f" in {location}"

    prompt = (
        f"Find verified doctors/clinics for: {query}\n"
        "Return name, specialty, location, contact info, and website if available."
    )
    result = call_perplexity_api(
        prompt,
        system_prompt="You are a medical directory assistant. Provide factual, recent data.",
        temperature=0.1,
    )
    if result["success"]:
        state["final_response"] = (
            "🏥 Verified Doctor Search Results\n\n"
            f"{result['content']}\n\n🤖 Model: {result.get('model_used', 'sonar-pro')}"
        )
        state["model_used"] = result.get("model_used")
    else:
        state["final_response"] = f"Error searching for doctors: {result['error']}"
        state["model_used"] = result.get("model_used")
    return state


def file_analysis_node(state: AgentState) -> AgentState:
    report_text = state.get("uploaded_report_text") or state.get("context") or ""
    if not report_text:
        state["final_response"] = "No report content available for analysis."
        return state

    prompt = (
        "You are analyzing a medical report. Explain it in clear language for the patient.\n\n"
        "Report:\n"
        f"{report_text}\n\n"
        "Patient's question:\n"
        f"{state['current_query']}\n\n"
        "Provide:\n"
        "1. Report summary\n"
        "2. Key findings\n"
        "3. What this means\n"
        "4. Next steps\n"
        "5. Questions to ask the doctor\n"
    )

    result = call_perplexity_api(
        prompt,
        system_prompt="You are a medical report explainer for patients.",
        temperature=0.2,
        conversation_history=state.get("conversation_history"),
    )

    if result["success"]:
        state["final_response"] = (
            "📄 Medical Report Analysis\n\n"
            f"{result['content']}\n\n🤖 Model: {result.get('model_used', 'sonar-pro')}"
        )
        state["model_used"] = result.get("model_used")
    else:
        state["final_response"] = f"Error analyzing report: {result['error']}"
        state["model_used"] = result.get("model_used")
    return state


def build_agent_graph():
    if not LANGCHAIN_AVAILABLE:
        return None

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_from_documents)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("synthesize", synthesize_response)
    workflow.add_node("emergency", emergency_response_node)
    workflow.add_node("doctor_search", doctor_search_node)
    workflow.add_node("file_analysis", file_analysis_node)

    workflow.set_conditional_entry_point(
        agent_router,
        {
            "emergency": "emergency",
            "doctor_search": "doctor_search",
            "file_analysis": "file_analysis",
            "hybrid_medical": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "web_search")
    workflow.add_edge("web_search", "synthesize")
    workflow.add_edge("synthesize", END)
    workflow.add_edge("emergency", END)
    workflow.add_edge("doctor_search", END)
    workflow.add_edge("file_analysis", END)

    return workflow.compile()


def detect_language(text: str) -> str:
    if any(0x0B80 <= ord(c) <= 0x0BFF for c in text):
        return "ta"
    if any(0x0D00 <= ord(c) <= 0x0D7F for c in text):
        return "ml"
    if any(0x0C00 <= ord(c) <= 0x0C7F for c in text):
        return "te"
    if any(0x0900 <= ord(c) <= 0x097F for c in text):
        return "hi"
    return "en"


def translate_text(text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text
    lang_name = SUPPORTED_LANGUAGES.get(target_lang, "target language")
    prompt = f"Translate the following medical text to {lang_name}:\n\n{text}"
    result = call_perplexity_api(
        prompt,
        system_prompt="You are a careful medical translator.",
        temperature=0.1,
        model="sonar",
    )
    return result["content"] if result["success"] else text


def extract_pdf_text(data: bytes, limit_chars: int = 4000) -> str:
    try:
        reader = PyPDF2.PdfReader(BytesIO(data))
        text_parts: List[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
            if sum(len(t) for t in text_parts) > limit_chars:
                break
        return "\n".join(text_parts)[:limit_chars]
    except Exception as exc:
        return f"Error processing PDF: {exc}"


class ChatMessage(BaseModel):
  role: Literal["user", "assistant"]
  content: str


class ChatRequest(BaseModel):
  message: str = Field(..., description="User's latest message")
  history: List[ChatMessage] = Field(default_factory=list)
  language: str = Field("en", description="Preferred response language code")
  location: Optional[str] = Field(None, description="User location for doctor search")
  uploaded_report_text: Optional[str] = Field(
      None, description="Previously uploaded report text (if any)"
  )


class ChatResponse(BaseModel):
  reply: str
  language: str
  route: str
  model_used: Optional[str] = None


class FileAnalysisResponse(BaseModel):
  text: str
  filename: str


app = FastAPI(title="Medibot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    global RETRIEVER, AGENT_GRAPH
    RETRIEVER, _ = setup_advanced_rag()
    AGENT_GRAPH = build_agent_graph()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "langchain_available": LANGCHAIN_AVAILABLE,
        "retriever_ready": RETRIEVER is not None,
    }


@app.post("/files/report-text", response_model=FileAnalysisResponse)
async def upload_report(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    data = await file.read()
    text = await run_in_threadpool(extract_pdf_text, data)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    return FileAnalysisResponse(text=text, filename=file.filename)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if AGENT_GRAPH is None:
        base_lang = req.language or detect_language(req.message)
        result = await run_in_threadpool(
            call_perplexity_api,
            req.message,
            "You are a medical assistant. Provide safe, general information.",
            0.3,
            [m.dict() for m in req.history],
        )
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"] or "LLM error")

        reply = result["content"]
        if base_lang != "en":
            reply = await run_in_threadpool(translate_text, reply, base_lang)

        return ChatResponse(
            reply=reply,
            language=base_lang,
            route="direct_llm",
            model_used=result.get("model_used"),
        )

    conversation_history = [m.dict() for m in req.history]
    conversation_history.append({"role": "user", "content": req.message})

    state = AgentState(
        messages=[],
        current_query=req.message,
        language=req.language or detect_language(req.message),
        uploaded_files=[],
        uploaded_report_text=req.uploaded_report_text or "",
        context=req.uploaded_report_text or "",
        retrieval_results=[],
        web_search_results="",
        agent_decision="",
        final_response="",
        conversation_history=conversation_history,
        user_location=req.location or "",
        model_used=None,
    )

    result_state: AgentState = await run_in_threadpool(AGENT_GRAPH.invoke, state)
    reply: str = result_state.get("final_response", "No response generated")
    route: str = result_state.get("agent_decision", "unknown")
    model_used: Optional[str] = result_state.get("model_used")

    target_lang = req.language or detect_language(req.message)
    if target_lang != "en":
        reply = await run_in_threadpool(translate_text, reply, target_lang)

    return ChatResponse(
        reply=reply,
        language=target_lang,
        route=route,
        model_used=model_used,
    )
