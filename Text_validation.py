
Text Validation Code : 

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
import nltk
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from langdetect import detect
import re
import emoji
from detoxify import Detoxify
import asyncio
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis.asyncio as redis
import json
import numpy as np
from googletrans import Translator
import nltk

# Download required resources
nltk.download('punkt_tab')
translator = Translator()

# Initialize FastAPI app
app = FastAPI()

model = Detoxify('unbiased')  # More accurate toxicity model
tokenizer = RegexpTokenizer(r'\w+')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize rate limiter (10 requests per minute)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize Redis for caching
redis_client = redis.from_url("redis://localhost")

# Lock for deduplication
lock = asyncio.Lock()

# Maximum text length before chunking
MAX_TEXT_LENGTH = 500
MAX_CHUNKS = 10

# JSON serialization helper
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Text preprocessing: cleans text before analysis
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = emoji.demojize(text)  # Convert emojis to text
    text = ' '.join(text.split())  # Remove extra spaces
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # Reduce repeated characters
    text = re.sub(r'([!?.]){2,}', r'\1', text)  # Reduce repeated punctuation
    text = re.sub(r'[^a-zA-Z0-9 !?.]', '', text)  # Remove special characters
    tokens = tokenizer.tokenize(text)
    return ' '.join(tokens)

# Splits text into meaningful chunks
def split_into_chunks(text: str, max_length: int = MAX_TEXT_LENGTH) -> list:
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks[:MAX_CHUNKS]

# Analyze toxicity using Detoxify
def analyze_text(text: str) -> dict:
    try:
        results = model.predict(text)
        return {k: float(v) for k, v in results.items()}
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        return {'toxicity': 0.0, 'severe_toxicity': 0.0, 'obscene': 0.0, 'threat': 0.0, 'insult': 0.0, 'identity_attack': 0.0}

# Cache results in Redis
async def cache_result(key: str, result: dict, expiration: int = 300):
    await redis_client.set(key, json.dumps(result, default=convert_numpy_types), ex=expiration)

# Retrieve cached results from Redis
async def get_cached_result(key: str):
    result = await redis_client.get(key)
    return json.loads(result) if result else None

# Determine content action based on toxicity score
def determine_action(toxicity_score: float) -> str:
    if toxicity_score > 0.6:
        return "block"
    elif toxicity_score > 0.4:
        return "warn"
    else:
        return "allow"

# Analyze multiple chunks and return max toxicity score
async def analyze_chunks(chunks: list) -> dict:
    all_results = []
    for chunk in chunks:
        preprocessed_chunk = preprocess_text(chunk)
        cache_key = f"toxicity_{preprocessed_chunk[:50]}"
        cached_result = await get_cached_result(cache_key)
        
        if cached_result:
            results = cached_result
        else:
            async with lock:
                results = analyze_text(preprocessed_chunk)
                await cache_result(cache_key, results)
        
        all_results.append(results)
    
    if not all_results:
        return analyze_text("empty text")
    
    combined_results = {key: max(res[key] for res in all_results) for key in all_results[0].keys()}
    return combined_results

# Toxicity detection endpoint
@app.post("/detect-toxicity/")
@limiter.limit("10/minute")
async def detect_toxicity(request: Request, request_data: dict) -> dict:
    try:
        text = request_data.get('text', '')
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")

        try:
            language = detect(text[:1000])
            logger.info(f"Detected language: {language}")
        except Exception:
            language = "unknown"

        # Translate if the text is not in English
        if language != "en":
            text = translator.translate(text, src=language, dest="en").text

        # Process text
        if len(text) > MAX_TEXT_LENGTH:
            chunks = split_into_chunks(text)
            results = await analyze_chunks(chunks)
        else:
            preprocessed_text = preprocess_text(text)
            cache_key = f"toxicity_{preprocessed_text[:50]}"
            cached_result = await get_cached_result(cache_key)
            
            if cached_result:
                results = cached_result
            else:
                async with lock:
                    results = analyze_text(preprocessed_text)
                    await cache_result(cache_key, results)

        toxicity_score = round(float(results['toxicity']), 3)
        action = determine_action(toxicity_score)
        
        return {
            "action": action,
            "message": f"Content has been {action}ed based on toxicity analysis.",
            "scores": results
        }

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health/")
async def health_check():
    return {"status": "healthy"}

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    try:
        await redis_client.ping()
        logger.info("Connected to Redis successfully")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    await redis_client.close()
    logger.info("Closed Redis connection")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

