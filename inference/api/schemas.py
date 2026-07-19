from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    message: str = Field(..., description="Customer query in Bangla", example="আমার অর্ডার কখন পাব?")
    mode: str = Field("direct", description="Mode: direct | rag | agent")
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)

class ChatResponse(BaseModel):
    response: str
    mode: str
    retrieved_context: Optional[List[str]] = None
    tool_called: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
