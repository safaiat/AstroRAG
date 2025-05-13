# multi_source_rag_chatbot.py

import os
import faiss
import requests
import xml.etree.ElementTree as ET
import numpy as np
import warnings
import re
import sys
import time
import threading
import random
from transformers import Qwen2Tokenizer, Qwen2ForCausalLM, GenerationConfig
from transformers.utils import logging
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import ads

# === SUPPRESS WARNINGS ===
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# === CONFIG ===
HF_CACHE = "/gpfs/wolf2/olcf/trn040/scratch/8mn/hf_cache"
MODEL_PATH = "/gpfs/wolf2/olcf/trn040/scratch/8mn/project1/qwen2.5-7b"
ads.config.token = "rWD23vPXVZzKB0TeSzEfcnfZKwYUBUmPxKPYwGO3"
os.environ.update({
    "HF_HOME": HF_CACHE,
    "TRANSFORMERS_CACHE": HF_CACHE,
    "SENTENCE_TRANSFORMERS_HOME": HF_CACHE
})

# === LOAD MODELS ===
tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = Qwen2ForCausalLM.from_pretrained(
    MODEL_PATH, local_files_only=True, torch_dtype="auto", device_map="auto"
)
model.eval()
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=HF_CACHE)

# === THINKING ANIMATION ===
def thinking_animation(stop_event):
    animations = ["dots", "spinner", "bounce"]
    chosen_animation = random.choice(animations)

    messages = [
        "Searching the universe for your answer",
        "Consulting the galactic library",
        "Tuning into deep space frequencies",
        "Aligning the virtual telescope",
        "Analyzing interstellar signals"
    ]
    base_message = random.choice(messages)

    # Animate base message once
    sys.stdout.write("\r")
    for c in base_message:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(0.033)

    if chosen_animation == "dots":
        dots = ["   ", ".  ", ".. ", "..."]
        i = 0
        while not stop_event.is_set():
            sys.stdout.write(f"\r{base_message}{dots[i % 4]}")
            sys.stdout.flush()
            time.sleep(0.4)
            i += 1

    elif chosen_animation == "spinner":
        spinner = ["|", "/", "-", "\\"]
        i = 0
        while not stop_event.is_set():
            sys.stdout.write(f"\r{base_message} {spinner[i % 4]}")
            sys.stdout.flush()
            time.sleep(0.2)
            i += 1

    elif chosen_animation == "bounce":
        frames = ["(*   )", "( *  )", "(  * )", "(   *)","(  * )","( *  )","(*   )"]
        i = 0
        while not stop_event.is_set():
            sys.stdout.write(f"\r{base_message} {frames[i % len(frames)]}")
            sys.stdout.flush()
            time.sleep(0.3)
            i += 1

    sys.stdout.write("\r" + " " * 80 + "\r")


# === KEYWORD EXTRACTION ===
def extract_keywords(query):
    query_clean = re.sub(r"[^a-zA-Z0-9\s]", "", query)
    words = query_clean.lower().split()
    return sorted(set([w for w in words if w not in ENGLISH_STOP_WORDS and len(w) >= 2]))

# === FETCH DOCUMENTS ===
def fetch_ads_docs(keywords, max_docs=10):
    docs = []
    for kw in keywords:
        results = list(ads.SearchQuery(q=kw, fl=["title", "abstract"], rows=max_docs))
        for r in results:
            if r.abstract:
                docs.append(f"[ADS] {r.title[0]}. {r.abstract}")
    return docs

def fetch_arxiv_docs(keywords, max_docs=10):
    docs = []
    for kw in keywords:
        url = f"http://export.arxiv.org/api/query?search_query=cat:astro-ph.*+AND+{kw}&max_results={max_docs}"
        r = requests.get(url)
        if r.status_code == 200:
            root = ET.fromstring(r.text)
            for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
                summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
                docs.append(f"[arXiv] {title}. {summary}")
    return docs

# === VECTOR RETRIEVAL ===
def build_context_vector_store(all_docs):
    vectors = embedder.encode(all_docs, convert_to_numpy=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, vectors

def retrieve_top_k(query, all_docs, index, k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    _, I = index.search(q_emb.reshape(1, -1), k)
    return [all_docs[i] for i in I[0]]

# === CONTEXT TRUNCATION ===
def truncate_context(context, max_tokens=950):
    words = context.split()
    return ' '.join(words[:max_tokens]) if len(words) > max_tokens else context

# === ANSWER GENERATION ===
def generate_answer(query, context):
    prompt = f"""You are an expert astrophysicist assistant. Respond concisely in English based on the context if relevant. If context lacks relevant data, answer based on scientific consensus. Use complete sentences. Avoid lists or bullet points.

Context:
{context}

Question: {query}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
    outputs = model.generate(**inputs, generation_config=GenerationConfig(
        do_sample=False,
        max_new_tokens=350,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    ))
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("Answer:")[1].split("Human:")[0].strip() if "Answer:" in decoded else decoded.strip()

# === MAIN LOOP ===
if __name__ == "__main__":
    character_name = "AstroRAG"

    greeting = (
        f"\n{character_name}: Hello! I'm your astrophysics assistant.\n"
        "Ask a question about stars, planets, galaxies, or any cosmic topic.\n"
        "Type 'exit' to quit.\n"
    )
    for char in greeting:
        print(char, end="", flush=True)
        time.sleep(0.033)

    user_name = input("\nBefore we start, what's your name?\n> ").strip()
    welcome_lines = [
        f"Nice to meet you, {user_name}! Let's explore the universe together.",
        f"Welcome aboard, {user_name}! Our cosmic journey begins.",
        f"Great to have you here, {user_name}. Let's dive into space mysteries.",
        f"{user_name}, ready to unlock the secrets of the stars? Let’s go!",
    ]
    print(f"\n{character_name}: ", end="")
    for c in random.choice(welcome_lines):
        print(c, end="", flush=True)
        time.sleep(0.033)

    last_sources = []

    while True:
        prompt_text = random.choice([
            "What astrophysics concept are you curious about today?",
            "Got a question about stars, planets, or galaxies? Ask away:",
            "Ready to explore the cosmos? Type your question here:",
            "Need help with a space-related topic? I'm all ears:",
            "Shoot me your astrophysics question, cadet:",
            "Let's study the universe! What's your question?"
        ])

        print(f"\n\n{character_name}: ", end="", flush=True)
        for char in prompt_text:
            print(char, end="", flush=True)
            time.sleep(0.033)
        print()

        query = input(f"{user_name}: ").strip()

        if query.lower() == "exit":
            print(f"{character_name}: Goodbye {user_name}! Keep exploring the stars.")
            break

        elif query.lower().startswith("nn "):
            new_name = query[3:].strip()
            if new_name:
                user_name = new_name
                print(f"{character_name}: Got it! I'll call you {user_name} now.")
            continue

        elif query.lower() == "help":
            print(f"""\n{character_name}: Here’s what you can do:
- Ask questions about space, astronomy, or astrophysics.
- To change your name, type: nn NewName
- To see sources used in your last question, type: source
- To exit, type: exit\n""")
            continue

        elif query.lower() == "source":
            if last_sources:
                print(f"\n{character_name}: Full sources from your last question:\n")
                for src in last_sources:
                    print(f"{src}\n{'-'*60}")
            else:
                print(f"{character_name}: No sources yet. Ask something first.")
            continue

        keywords = extract_keywords(query)
        if not keywords:
            print("Sorry, I couldn't find meaningful keywords in your question.")
            continue

        stop_event = threading.Event()
        anim_thread = threading.Thread(target=thinking_animation, args=(stop_event,))
        anim_thread.start()

        ads_docs = fetch_ads_docs(keywords)
        arxiv_docs = fetch_arxiv_docs(keywords)
        all_docs = list(set(ads_docs + arxiv_docs))
        last_sources = all_docs.copy()

        if not all_docs:
            stop_event.set()
            anim_thread.join()
            print("[INFO] No relevant documents retrieved.")
            continue

        index, _ = build_context_vector_store(all_docs)
        top_chunks = retrieve_top_k(query, all_docs, index)
        context = truncate_context("\n\n".join(top_chunks))
        answer = generate_answer(query, context)

        stop_event.set()
        anim_thread.join()

        print(f"\n{character_name}: ", end="", flush=True)
        for char in answer:
            print(char, end="", flush=True)
            time.sleep(0.033)
        print("\n")
  
