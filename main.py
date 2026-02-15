"""
Vivek Council - LLM Council App using Venice API
A modern, single-file implementation of karpathy/llm-council concept
Deploys easily on Railway with Venice API backend
"""

import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import httpx


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Application configuration"""
    # Venice API Configuration
    VENICE_API_KEY: str = os.getenv("VENICE_API_KEY", "")
    VENICE_BASE_URL: str = "https://api.venice.ai/api/v1"
    
    # Council Models - Customize your council here
    COUNCIL_MODELS: List[str] = [
        "claude-opus-4-6",
        "deepseek-v3.2", 
        "grok-41-fast",
        "kimi-k2-5",
    ]
    
    # Chairman Model - Produces final response
    CHAIRMAN_MODEL: str = "claude-opus-4-6"
    
    # App Settings
    APP_NAME: str = "Vivek Council"
    APP_DESCRIPTION: str = "Multiple AI models working together to answer your hardest questions"
    
    # Storage
    DATA_DIR: str = "data"
    CONVERSATIONS_DIR: str = f"{DATA_DIR}/conversations"


# ============================================================================
# Venice API Client
# ============================================================================

class VeniceClient:
    """Venice API client for chat completions"""
    
    def __init__(self, api_key: str, base_url: str = Config.VENICE_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Send a chat completion request to Venice API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Venice API error: {response.status_code} - {response.text}")
        
        return response.json()
    
    async def close(self):
        await self.client.aclose()


# ============================================================================
# Data Models
# ============================================================================

class Message(BaseModel):
    role: str
    content: str


class CouncilQuery(BaseModel):
    message: str
    model_ids: Optional[List[str]] = None
    chairman_model: Optional[str] = None


class CouncilResponse(BaseModel):
    conversation_id: str
    stage: str  # "opinions", "review", "final", "complete"
    opinions: Optional[List[Dict[str, Any]]] = None
    reviews: Optional[List[Dict[str, Any]]] = None
    final_response: Optional[str] = None
    progress: int = 0  # 0-100


class Conversation:
    """Conversation storage"""
    
    def __init__(self, conversation_id: str):
        self.id = conversation_id
        self.created_at = datetime.now()
        self.messages: List[Dict[str, Any]] = []
        self.opinions: List[Dict[str, Any]] = []
        self.reviews: List[Dict[str, Any]] = []
        self.final_response: str = ""
        self.model_ids: List[str] = []
        self.chairman_model: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "messages": self.messages,
            "opinions": self.opinions,
            "reviews": self.reviews,
            "final_response": self.final_response,
            "model_ids": self.model_ids,
            "chairman_model": self.chairman_model
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        conv = cls(data["id"])
        conv.created_at = datetime.fromisoformat(data["created_at"])
        conv.messages = data.get("messages", [])
        conv.opinions = data.get("opinions", [])
        conv.reviews = data.get("reviews", [])
        conv.final_response = data.get("final_response", "")
        conv.model_ids = data.get("model_ids", [])
        conv.chairman_model = data.get("chairman_model", "")
        return conv


# ============================================================================
# Council Logic
# ============================================================================

class CouncilEngine:
    """LLM Council processing engine"""
    
    def __init__(self, venice_client: VeniceClient):
        self.client = venice_client
    
    async def get_opinions(
        self, 
        query: str, 
        model_ids: List[str],
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        """Stage 1: Get first opinions from all council members"""
        opinions = []
        
        system_prompt = """You are a member of an LLM Council. 
Provide a thoughtful, detailed response to the user's question.
Be accurate, insightful, and consider multiple perspectives.
Respond in a clear, well-structured manner."""
        
        for i, model_id in enumerate(model_ids):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            try:
                response = await self.client.chat_completion(
                    model=model_id,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                content = response["choices"][0]["message"]["content"]
                opinions.append({
                    "model_id": model_id,
                    "model_name": self._get_model_display_name(model_id),
                    "content": content,
                    "rank": i + 1
                })
                
                if progress_callback:
                    await progress_callback(int((i + 1) / len(model_ids) * 30))
                    
            except Exception as e:
                opinions.append({
                    "model_id": model_id,
                    "model_name": self._get_model_display_name(model_id),
                    "content": f"Error: {str(e)}",
                    "error": True
                })
        
        return opinions
    
    async def get_reviews(
        self,
        query: str,
        opinions: List[Dict[str, Any]],
        model_ids: List[str],
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        """Stage 2: Have each model review all other responses (anonymized)"""
        reviews = []
        
        # Prepare anonymized opinions (remove model identifiers)
        anonymized_opinions = []
        for i, op in enumerate(opinions):
            if not op.get("error"):
                anonymized_opinions.append({
                    "response_id": f"Response {i + 1}",
                    "content": op["content"]
                })
        
        review_prompt = f"""You are evaluating responses to: "{query}"

Below are the responses from different AI models (anonymized). 
Read each response carefully and rank them by:
1. Accuracy - Is the information correct?
2. Insight - Does it provide deep understanding?
3. Clarity - Is it well-structured and easy to understand?

Provide your evaluation in the following format:
- Best response: [Response #]
- Worst response: [Response #]
- Brief explanation of your rankings

Then provide your own improved answer that combines the best elements from all responses.

---
RESPONSES TO EVALUATE:
{self._format_opinions(anonymized_opinions)}

---
YOUR EVALUATION:"""
        
        for i, model_id in enumerate(model_ids):
            # Skip if this model had an error
            if opinions[i].get("error"):
                continue
            
            messages = [
                {"role": "system", "content": "You are an expert evaluator. Be critical but fair."},
                {"role": "user", "content": review_prompt}
            ]
            
            try:
                response = await self.client.chat_completion(
                    model=model_id,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=1500
                )
                
                content = response["choices"][0]["message"]["content"]
                reviews.append({
                    "model_id": model_id,
                    "model_name": self._get_model_display_name(model_id),
                    "content": content,
                    "response_id": i + 1
                })
                
                if progress_callback:
                    await progress_callback(30 + int((i + 1) / len(model_ids) * 30))
                    
            except Exception as e:
                reviews.append({
                    "model_id": model_id,
                    "model_name": self._get_model_display_name(model_id),
                    "content": f"Error: {str(e)}",
                    "error": True
                })
        
        return reviews
    
    async def get_final_response(
        self,
        query: str,
        opinions: List[Dict[str, Any]],
        reviews: List[Dict[str, Any]],
        chairman_model: str,
        progress_callback=None
    ) -> str:
        """Stage 3: Chairman produces final response"""
        
        # Compile all opinions and reviews
        summary = f"""User Query: {query}

---
COUNCIL MEMBERS' INITIAL RESPONSES:
{self._format_opinions(opinions)}

---
CROSS-REVIEWS FROM COUNCIL:
{self._format_reviews(reviews)}

---
Based on the diverse perspectives and critiques above, please produce a final, polished response that:
1. Incorporates the strongest insights from all council members
2. Addresses any criticisms raised in the reviews
3. Provides a clear, comprehensive answer
4. Notes any areas of disagreement or uncertainty where appropriate

FINAL RESPONSE:"""
        
        messages = [
            {"role": "system", "content": "You are the Chairman of the LLM Council. Produce a final, authoritative response that synthesizes all perspectives."},
            {"role": "user", "content": summary}
        ]
        
        try:
            response = await self.client.chat_completion(
                model=chairman_model,
                messages=messages,
                temperature=0.6,
                max_tokens=2500
            )
            
            content = response["choices"][0]["message"]["content"]
            
            if progress_callback:
                await progress_callback(100)
            
            return content
            
        except Exception as e:
            return f"Error generating final response: {str(e)}"
    
    def _get_model_display_name(self, model_id: str) -> str:
        """Convert model ID to display name"""
        names = {
            "claude-opus-4-6": "Claude Opus 4.6",
            "claude-opus-45": "Claude Opus 4.5",
            "claude-sonnet-45": "Claude Sonnet 4.5",
            "deepseek-v3.2": "DeepSeek V3.2",
            "grok-41-fast": "Grok 4.1 Fast",
            "kimi-k2-5": "Kimi K2.5",
            "openai-gpt-52": "GPT-5.2",
            "llama-3.3-70b": "Llama 3.3 70B",
            "minimax-m25": "MiniMax M2.5",
        }
        return names.get(model_id, model_id.replace("-", " ").title())
    
    def _format_opinions(self, opinions: List[Dict[str, Any]]) -> str:
        """Format opinions for prompts"""
        lines = []
        for op in opinions:
            resp_id = op.get("response_id", "Response")
            content = op.get("content", "")
            lines.append(f"\n=== {resp_id} ===\n{content}\n")
        return "\n".join(lines)
    
    def _format_reviews(self, reviews: List[Dict[str, Any]]) -> str:
        """Format reviews for prompts"""
        lines = []
        for r in reviews:
            model_name = r.get("model_name", "Model")
            content = r.get("content", "")
            lines.append(f"\n=== Review by {model_name} ===\n{content}\n")
        return "\n".join(lines)


# ============================================================================
# Application State
# ============================================================================

app_state = {
    "venice_client": None,
    "council_engine": None,
    "conversations": {},
    "current_stage": {},
    "current_progress": {}
}


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    api_key = Config.VENICE_API_KEY
    if not api_key:
        print("WARNING: VENICE_API_KEY not set. Set it via environment variable.")
    
    app_state["venice_client"] = VeniceClient(api_key)
    app_state["council_engine"] = CouncilEngine(app_state["venice_client"])
    
    # Create data directories
    os.makedirs(Config.CONVERSATIONS_DIR, exist_ok=True)
    
    yield
    
    # Shutdown
    if app_state["venice_client"]:
        await app_state["ven()


app = Fastice_client"].closeAPI(
    title=Config.APP_NAME,
    description=Config.APP_DESCRIPTION,
    lifespan=lifespan
)

# Mount static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# ============================================================================
# API Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "app_name": Config.APP_NAME,
        "models": Config.COUNCIL_MODELS,
        "chairman": Config.CHAIRMAN_MODEL,
        "model_names": [CouncilEngine(None)._get_model_display_name(m) for m in Config.COUNCIL_MODELS]
    })


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok", "api_configured": bool(Config.VENICE_API_KEY)}


@app.post("/api/council")
async def council_query(query: CouncilQuery):
    """Process a council query through all stages"""
    if not app_state["council_engine"]:
        raise HTTPException(status_code=500, detail="Council engine not initialized")
    
    conversation_id = str(uuid.uuid4())
    model_ids = query.model_ids or Config.COUNCIL_MODELS
    chairman = query.chairman_model or Config.CHAIRMAN_MODEL
    
    # Store conversation
    conv = Conversation(conversation_id)
    conv.model_ids = model_ids
    conv.chairman_model = chairman
    conv.messages.append({"role": "user", "content": query.message})
    app_state["conversations"][conversation_id] = conv
    
    try:
        # Stage 1: Get opinions
        opinions = await app_state["council_engine"].get_opinions(
            query.message, 
            model_ids
        )
        conv.opinions = opinions
        
        # Save after stage 1
        _save_conversation(conv)
        
        # Stage 2: Get reviews
        reviews = await app_state["council_engine"].get_reviews(
            query.message,
            opinions,
            model_ids
        )
        conv.reviews = reviews
        
        # Save after stage 2
        _save_conversation(conv)
        
        # Stage 3: Get final response
        final_response = await app_state["council_engine"].get_final_response(
            query.message,
            opinions,
            reviews,
            chairman
        )
        conv.final_response = final_response
        
        # Save final
        _save_conversation(conv)
        
        return {
            "conversation_id": conversation_id,
            "stage": "complete",
            "opinions": opinions,
            "reviews": reviews,
            "final_response": final_response,
            "progress": 100
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/council/stream")
async def council_query_stream(query: CouncilQuery, request: Request):
    """Process council query with streaming updates via SSE"""
    if not app_state["council_engine"]:
        raise HTTPException(status_code=500, detail="Council engine not initialized")
    
    conversation_id = str(uuid.uuid4())
    model_ids = query.model_ids or Config.COUNCIL_MODELS
    chairman = query.chairman_model or Config.CHAIRMAN_MODEL
    
    # Store conversation
    conv = Conversation(conversation_id)
    conv.model_ids = model_ids
    conv.chairman_model = chairman
    conv.messages.append({"role": "user", "content": query.message})
    app_state["conversations"][conversation_id] = conv
    
    async def generate():
        try:
            # Stage 1: Get opinions
            yield f"data: {json.dumps({'stage': 'opinions', 'progress': 0})}\n\n"
            
            for i, model_id in enumerate(model_ids):
                messages = [
                    {"role": "system", "content": """You are a member of an LLM Council. 
Provide a thoughtful, detailed response to the user's question.
Be accurate, insightful, and consider multiple perspectives."""},
                    {"role": "user", "content": query.message}
                ]
                
                response = await app_state["venice_client"].chat_completion(
                    model=model_id,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                content = response["choices"][0]["message"]["content"]
                opinion = {
                    "model_id": model_id,
                    "model_name": app_state["council_engine"]._get_model_display_name(model_id),
                    "content": content
                }
                conv.opinions.append(opinion)
                
                yield f"data: {json.dumps({'stage': 'opinions', 'progress': int((i+1)/len(model_ids)*30), 'opinion': opinion})}\n\n"
            
            _save_conversation(conv)
            
            # Stage 2: Get reviews
            yield f"data: {json.dumps({'stage': 'review', 'progress': 30})}\n\n"
            
            anonymized = [{"response_id": f"Response {i+1}", "content": op["content"]} 
                          for i, op in enumerate(conv.opinions)]
            
            review_prompt = f"""You are evaluating responses to: "{query.message}"

Read each response and rank them by accuracy, insight, and clarity.
Then provide your own improved answer.

---
RESPONSES:
{app_state["council_engine"]._format_opinions(anonymized)}

---
YOUR EVALUATION:"""
            
            for i, model_id in enumerate(model_ids):
                messages = [
                    {"role": "system", "content": "You are an expert evaluator."},
                    {"role": "user", "content": review_prompt}
                ]
                
                response = await app_state["venice_client"].chat_completion(
                    model=model_id,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=1500
                )
                
                content = response["choices"][0]["message"]["content"]
                review = {
                    "model_id": model_id,
                    "model_name": app_state["council_engine"]._get_model_display_name(model_id),
                    "content": content
                }
                conv.reviews.append(review)
                
                yield f"data: {json.dumps({'stage': 'review', 'progress': 30 + int((i+1)/len(model_ids)*30), 'review': review})}\n\n"
            
            _save_conversation(conv)
            
            # Stage 3: Final response
            yield f"data: {json.dumps({'stage': 'final', 'progress': 60})}\n\n"
            
            summary = f"""User Query: {query.message}

---
COUNCIL RESPONSES:
{app_state["council_engine"]._format_opinions(conv.opinions)}

---
REVIEWS:
{app_state["council_engine"]._format_reviews(conv.reviews)}

---
Produce a final response synthesizing all perspectives."""

            messages = [
                {"role": "system", "content": "You are the Chairman. Produce a final, polished response."},
                {"role": "user", "content": summary}
            ]
            
            response = await app_state["venice_client"].chat_completion(
                model=chairman,
                messages=messages,
                temperature=0.6,
                max_tokens=2500
            )
            
            content = response["choices"][0]["message"]["content"]
            conv.final_response = content
            _save_conversation(conv)
            
            yield f"data: {json.dumps({'stage': 'complete', 'progress': 100, 'final': content, 'conversation_id': conversation_id})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return generate()


@app.get("/api/conversations")
async def list_conversations():
    """List all conversations"""
    convs = []
    for cid, conv in app_state["conversations"].items():
        convs.append({
            "id": cid,
            "created_at": conv.created_at.isoformat(),
            "message": conv.messages[0]["content"] if conv.messages else "",
            "has_final": bool(conv.final_response)
        })
    return sorted(convs, key=lambda x: x["created_at"], reverse=True)[:20]


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation"""
    conv = app_state["conversations"].get(conversation_id)
    if not conv:
        # Try to load from disk
        conv = _load_conversation(conversation_id)
        if conv:
            app_state["conversations"][conversation_id] = conv
    
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conv.to_dict()


# ============================================================================
# Storage Helpers
# ============================================================================

def _save_conversation(conv: Conversation):
    """Save conversation to disk"""
    os.makedirs(Config.CONVERSATIONS_DIR, exist_ok=True)
    filepath = os.path.join(Config.CONVERSATIONS_DIR, f"{conv.id}.json")
    with open(filepath, "w") as f:
        json.dump(conv.to_dict(), f, indent=2)


def _load_conversation(conversation_id: str) -> Optional[Conversation]:
    """Load conversation from disk"""
    filepath = os.path.join(Config.CONVERSATIONS_DIR, f"{conversation_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            return Conversation.from_dict(data)
    return None


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
