from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main_system import LegalChatbotSystem, ChatbotConfig
import os

app = FastAPI(title="Legal Chatbot API")

# Initialize system
config = ChatbotConfig(
    mongodb_uri=os.getenv("MONGODB_URI"),
    generation_api_key=os.getenv("OPENAI_API_KEY")
)
system = LegalChatbotSystem(config)

class QueryRequest(BaseModel):
    query: str
    user_id: str = "default"
    stream: bool = False

class IndexRequest(BaseModel):
    file_path: str
    van_ban_id: str
    format: str = "pdf"

@app.post("/query")
async def query(request: QueryRequest):
    try:
        response = system.query(
            query=request.query,
            user_id=request.user_id,
            stream=request.stream
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
async def index_document(request: IndexRequest):
    try:
        result = system.index_document(
            file_path=request.file_path,
            van_ban_id=request.van_ban_id,
            format=request.format
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    return system.get_system_stats()

@app.delete("/conversation/{user_id}")
async def clear_conversation(user_id: str):
    system.clear_conversation(user_id)
    return {"message": f"Cleared conversation for {user_id}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)