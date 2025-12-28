import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai


from google.genai import types  # Required for strict data validation
from supabase_client import supabase

# 1. Configuration
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-image-galleryui-otukgngn0-rohans-projects-28969eb7.vercel.app"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Initialize Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 3. AI Analysis Function

def analyze_image_with_ai(image_url: str):
    
    try:
        # Fetch the image bytes from the URL
        image_response = requests.get(image_url, timeout=15)
        image_response.raise_for_status()
        image_bytes = image_response.content
        
        # Detect MIME type (default to jpeg if not found)
        mime_type = image_response.headers.get("Content-Type", "image/jpeg")

        # Call Gemini using the Part helper to avoid Pydantic validation errors
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
                ),
                "Describe this image in one sentence.\n"
            "Tags: comma-separated list of 5 tags.\n"
            "Colors: comma-separated list of 3 hex colors."
            ]
        )
        
        return response.text if response.text else "No description available."

    except Exception as e:
        print(f"AI Processing Error: {e}")
        return f"Processing failed: {str(e)}"
    


    
def parse_ai_output(ai_text: str):
    description = ""
    tags = []
    colors = []

    for line in ai_text.split("\n"):
        line = line.strip()

        if not description:
            description = line

        elif line.lower().startswith("tags"):
            tags = [
                tag.strip()
                for tag in line.split(":", 1)[1].split(",")
                if tag.strip()
            ]

        elif line.lower().startswith("colors"):
            colors = [
                color.strip()
                for color in line.split(":", 1)[1].split(",")
                if color.strip()
            ]

    return description, tags, colors
    

# 4. API Route
@app.post("/analyze-image")
def analyze_image(payload: dict):
    image_id = payload.get("image_id")
    image_url = payload.get("image_url")
    user_id = payload.get("user_id")

    if not image_url:
        raise HTTPException(status_code=400, detail="Missing image_url")

    # Perform Analysis
    ai_raw_text = analyze_image_with_ai(image_url)

    # Simple Parsing (Assumes first line is description)
    description, tags, colors = parse_ai_output(ai_raw_text)


    # Save to Supabase
    try:
        supabase.table("image_metadata").insert({
            "image_id": image_id,
            "user_id": user_id,
            "description": description,
            "tags": tags,
            "colors": colors,
            "ai_processing_status": "completed",
        }).execute()
    except Exception as e:
        print(f"Supabase Error: {e}")

    return {
        "status": "completed",
        "description": description,
        "raw_ai_output": ai_raw_text
    }