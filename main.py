from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import datetime
import json
import os

# Define the file path for your local storage
# It will be created in the same directory where you run main.py
BEHAVIOR_LOG_FILE = "behavior_logs.jsonl" # .jsonl for JSON Lines format

# Initialize FastAPI app
app = FastAPI(
    title="BEACON Behavioral Backend",
    description="API for receiving and processing behavioral data from the BEACON web app.",
    version="0.1.0"
)

# Pydantic model for incoming behavior log entries
class BehaviorLogEntry(BaseModel):
    timestamp: str
    user_email: Optional[str] = None # Assuming you might get user email from a header or token later
    data: Dict[str, Any]

# --- Helper functions for file-based storage ---

def load_behavior_logs() -> List[BehaviorLogEntry]:
    """Loads behavior logs from the JSONL file."""
    logs = []
    if os.path.exists(BEHAVIOR_LOG_FILE):
        with open(BEHAVIOR_LOG_FILE, 'r') as f:
            for line in f:
                try:
                    # Each line is a JSON object
                    log_data = json.loads(line.strip())
                    logs.append(BehaviorLogEntry(**log_data))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from log file: {line.strip()} - {e}")
    return logs

def save_behavior_log(log_entry: BehaviorLogEntry):
    """Appends a new behavior log entry to the JSONL file."""
    with open(BEHAVIOR_LOG_FILE, 'a') as f: # 'a' for append mode
        f.write(json.dumps(log_entry.model_dump()) + '\n') # .model_dump() is for Pydantic v2

# --- Initialize storage when the app starts ---
# This will load existing logs into memory, though we will primarily append to the file
# For very large datasets, you'd directly query the file or a proper database.
# For now, we'll keep `temporary_behavior_storage` for easy retrieval via `/api/behavior_logs`
temporary_behavior_storage: List[BehaviorLogEntry] = load_behavior_logs()
print(f"Loaded {len(temporary_behavior_storage)} existing behavior logs from {BEHAVIOR_LOG_FILE}")


@app.get("/")
async def root():
    return {"message": "BEACON Backend is running!"}

@app.post("/api/behavior")
async def receive_behavior_data(request: Request):
    """
    Receives behavioral data from the frontend and stores it in a local file.
    """
    try:
        data = await request.json()
        
        # You might want to get the actual user email from authentication (e.g., JWT token)
        # For now, let's use a placeholder or extract from the data if available
        user_email = "anonymous@example.com" # Default placeholder
        if 'currentUser' in data and data['currentUser'] is not None: 
             if 'email' in data['currentUser']:
                 user_email = data['currentUser']['email']
        
        log_entry = BehaviorLogEntry(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            user_email=user_email,
            data=data
        )
        
        save_behavior_log(log_entry) # Save to file
        temporary_behavior_storage.append(log_entry) # Also add to in-memory for immediate retrieval
        
        print(f"Received behavior data for {user_email}. Saved to {BEHAVIOR_LOG_FILE}. Total entries in memory: {len(temporary_behavior_storage)}")
        
        return JSONResponse(content={"message": "Behavior data received successfully", "status": "success"}, status_code=200)
    except Exception as e:
        print(f"Error receiving behavior data: {e}")
        return JSONResponse(content={"message": f"Error processing data: {e}", "status": "error"}, status_code=400)

@app.get("/api/behavior_logs")
async def get_behavior_logs():
    """
    Retrieves all temporarily stored behavior logs from memory.
    (For demonstration/debugging purposes)
    """
    return temporary_behavior_storage

# To run the application directly from this script
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)