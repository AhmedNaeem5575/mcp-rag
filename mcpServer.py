import pandas as pd
import faiss
import numpy as np
import requests
import openai
from mcp.server import Server
from mcp import Tool
import os
from dotenv import load_dotenv

load_dotenv()

# ====== CONFIG ======
API_KEY = os.getenv("OPEN_AI_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK')
CSV_FILE = "./data/cars.csv"
# ====================

# Memory store
last_answer = None

# ==== Load CSV ====
df = pd.read_csv(CSV_FILE)
texts = []
for _, row in df.iterrows():
    row_text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
    texts.append(row_text)

# ==== Build FAISS index ====
client = openai.OpenAI(api_key=API_KEY)
embeddings = [
    client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding
    for text in texts
]
emb_np = np.array(embeddings).astype('float32')
dimension = len(emb_np[0])
index = faiss.IndexFlatL2(dimension)
index.add(emb_np)

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
        model="gpt-4o-mini",
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
server = Server("CarRAGBot")

server.add_tool(Tool(
    name="get_car_info",
    description="Get car information using CSV RAG",
    parameters={"query": {"type": "string", "description": "User's car-related question"}},
    function=get_car_info
))

server.add_tool(Tool(
    name="send_to_discord",
    description="Send last answer or a custom message to Discord",
    parameters={"message": {"type": "string", "description": "Optional custom message"},},
    function=send_to_discord
))

server.add_tool(Tool(
    name="send_to_slack",
    description="Send last answer or a custom message to Slack",
    parameters={"message": {"type": "string", "description": "Optional custom message"},},
    function=send_to_slack
))

if __name__ == "__main__":
    server.run()
