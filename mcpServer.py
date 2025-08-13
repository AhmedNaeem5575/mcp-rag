import pandas as pd
import faiss
import numpy as np
import requests
import openai
from mcp.server.fastmcp import FastMCP
import os
import json
import time
import pickle
from dotenv import load_dotenv

load_dotenv()

# ====== CONFIG ======
API_KEY = os.getenv("OPEN_AI_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK')
CSV_FILE = "./data/cars.csv"
EMBEDDINGS_CACHE = "./data/embeddings_cache.pkl"
TEXTS_CACHE = "./data/texts_cache.pkl"
# ====================

model_name = "gpt-4.1-mini"
deployment = "gpt-4.1-mini"

# Memory store
last_answer = None

# ==== Load CSV ====
df = pd.read_csv(CSV_FILE)

def load_or_create_embeddings():
    """Load cached embeddings or create them with rate limiting"""
    global texts, index, client
    
    # Check if cache exists and is newer than CSV
    cache_exists = os.path.exists(EMBEDDINGS_CACHE) and os.path.exists(TEXTS_CACHE)
    if cache_exists:
        csv_mtime = os.path.getmtime(CSV_FILE)
        cache_mtime = os.path.getmtime(EMBEDDINGS_CACHE)
        if cache_mtime > csv_mtime:
            print("📦 Loading cached embeddings...")
            with open(TEXTS_CACHE, 'rb') as f:
                texts = pickle.load(f)
            with open(EMBEDDINGS_CACHE, 'rb') as f:
                emb_np = pickle.load(f)
            dimension = len(emb_np[0])
            index = faiss.IndexFlatL2(dimension)
            index.add(emb_np)
            print("✅ Cached embeddings loaded!")
            return
    
    # Create texts from CSV
    print("🔄 Processing CSV data...")
    texts = []
    for _, row in df.iterrows():
        row_text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
        texts.append(row_text)
    
    # Create embeddings with rate limiting
    print(f"🚀 Creating embeddings for {len(texts)} rows...")
    
    embeddings = []
    for i, text in enumerate(texts):
        try:
            emb = client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding
            embeddings.append(emb)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(texts)} rows...")
            # Rate limiting - wait between requests
            time.sleep(0.1)  # 100ms delay between requests
        except Exception as e:
            if "429" in str(e):
                print(f"  Rate limit hit at row {i + 1}, waiting 2 seconds...")
                time.sleep(2)
                # Retry the request
                emb = client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding
                embeddings.append(emb)
            else:
                raise e
    
    emb_np = np.array(embeddings).astype('float32')
    dimension = len(emb_np[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(emb_np)
    
    # Cache the results
    print("💾 Caching embeddings...")
    os.makedirs(os.path.dirname(EMBEDDINGS_CACHE), exist_ok=True)
    with open(TEXTS_CACHE, 'wb') as f:
        pickle.dump(texts, f)
    with open(EMBEDDINGS_CACHE, 'wb') as f:
        pickle.dump(emb_np, f)
    print("✅ Embeddings cached!")

# ==== Build FAISS index ====
client = openai.AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://test-ofmp-sellerhub-ssr6495.openai.azure.com/",
    api_key=API_KEY,
)
texts = []
index = None
load_or_create_embeddings()

# ==== RAG search ====
def search_csv(query, top_k=3):
    q_emb = client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
    q_emb_np = np.array([q_emb]).astype('float32')
    _, indices = index.search(q_emb_np, top_k)
    return [texts[i] for i in indices[0]]

# ==== Tools ====
def get_car_info(query: str):
    """RAG query on cars CSV"""
    global last_answer
    context = "\n".join(search_csv(query))
    prompt = f"Answer the question based only on this car data:\n{context}\n\nQuestion: {query}"
    
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a car data assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    answer = resp.choices[0].message.content
    last_answer = answer
    return answer

def send_to_discord(message: str = None):
    """Send last answer or a custom message to Discord"""
    global last_answer
    if not message and not last_answer:
        return "No message to send."
    msg = message if message else last_answer
    r = requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
    if r.status_code == 204:
        return "✅ Sent to Discord!"
    else:
        return f"❌ Error sending to Discord: {r.text}"

def send_to_slack(message: str = None):
    """Send last answer or a custom message to Slack"""
    global last_answer
    if not message and not last_answer:
        return "No message to send."
    msg = message if message else last_answer
    r = requests.post(SLACK_WEBHOOK_URL, json={"text": msg})
    if r.status_code == 200:
        return "✅ Sent to Slack!"
    else:
        return f"❌ Error sending to Slack: {r.text}"

# ==== Register MCP server ====
mcp = FastMCP("CarRAGBot")

@mcp.tool()
def get_car_info_tool(query: str) -> str:
    """Get car information using CSV RAG"""
    return get_car_info(query)

@mcp.tool()
def send_to_discord_tool(message: str = None) -> str:
    """Send last answer or a custom message to Discord"""
    return send_to_discord(message)

@mcp.tool()
def send_to_slack_tool(message: str = None) -> str:
    """Send last answer or a custom message to Slack"""
    return send_to_slack(message)

if __name__ == "__main__":
    mcp.run()
