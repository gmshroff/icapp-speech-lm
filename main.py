# api/main.py
import base64
from fastapi import FastAPI, HTTPException, UploadFile, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Generator, AsyncGenerator # Streaming
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from google.cloud import speech, texttospeech
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Generator, AsyncGenerator
from gemini_agent_stream import GeminiChatAgentStream
from gemini_agent import GeminiChatAgent

import warnings

class Settings(BaseSettings):
    GOOGLE_APPLICATION_CREDENTIALS_JSON: str
    GOOGLE_API_KEY: str

    class Config:
        env_file = ".env"

# Load settings from the environment or .env file
try:
    settings = Settings()
except Exception as e:
    raise RuntimeError(f"Failed to load settings: {str(e)}")

# Initialize Google clients once during startup
speech_client = speech.SpeechClient.from_service_account_json(
    settings.GOOGLE_APPLICATION_CREDENTIALS_JSON
)
tts_client = texttospeech.TextToSpeechClient.from_service_account_json(
    settings.GOOGLE_APPLICATION_CREDENTIALS_JSON
)

app = FastAPI()

# Add this before your routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioRequest(BaseModel):
    audio_base64: str

class TextRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    user_input: str

# Initialize agent once
custom_prompt = """You are a mighty avenger from Marvel and need to act like that only. Give short answers."""
agent = GeminiChatAgent(
    api_key=settings.GOOGLE_API_KEY,
    system_prompt=custom_prompt
)

custom_prompt = """You are a mighty avenger from Marvel and need to act like that only. Give short answers."""
streaming_agent = GeminiChatAgentStream(
    api_key=settings.GOOGLE_API_KEY,
    system_prompt=custom_prompt
)

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon set"}

@app.get("/")
async def root():
    return {"message": "Welcome to my FastAPI app!"}

@app.get("/ping")
async def ping():
    return {"status": "success", "message": "Server is healthy"}

@app.post("/transcribe")
async def transcribe_audio(request: AudioRequest):
    try:
        audio_content = base64.b64decode(request.audio_base64)
        audio = speech.RecognitionAudio(content=audio_content)
        
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )
        
        response = speech_client.recognize(config=config, audio=audio)
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])
        return {"transcript": transcript.strip()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-response")
async def generate_response(request: ChatRequest):
    try:
        response = agent.chat(request.user_input)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/generate-streaming-response")
async def chat_stream_api(request: ChatRequest) -> StreamingResponse:
    """
    Endpoint to stream data from the GeminiChatAgentStream.
    Args:
        user_input: The prompt provided by the user.
    Returns:
        StreamingResponse that streams JSON chunks.
    """
    async def response_generator() -> AsyncGenerator[str, None]:
        async for chunk in streaming_agent.chat_stream_for_fastApi(request.user_input):
            yield chunk

    return StreamingResponse(response_generator(), media_type="text/event-stream")

@app.post("/text-to-speech")
async def text_to_speech(request: TextRequest):
    try:
        synthesis_input = texttospeech.SynthesisInput(text=request.text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name = "en-US-Wavenet-A",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        return Response(
            content=response.audio_content,
            media_type="audio/mp3",
            headers={"Content-Disposition": "attachment; filename=response.mp3"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))