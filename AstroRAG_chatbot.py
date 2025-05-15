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
from collections import Counter
from transformers import Qwen2Tokenizer, Qwen2ForCausalLM, GenerationConfig
from transformers.utils import logging
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import ads

# === CONFIGURATION ===
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

HF_CACHE = "/gpfs/wolf2/olcf/trn040/scratch/8mn/hf_cache"
MODEL_PATH = "/gpfs/wolf2/olcf/trn040/scratch/8mn/project1/qwen2.5-7b"
ads.config.token = "rWD23vPXVZzKB0TeSzEfcnfZKwYUBUmPxKPYwGO3"
os.environ.update({
    "HF_HOME": HF_CACHE,
    "TRANSFORMERS_CACHE": HF_CACHE,
    "SENTENCE_TRANSFORMERS_HOME": HF_CACHE
})

tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = Qwen2ForCausalLM.from_pretrained(
    MODEL_PATH, local_files_only=True, torch_dtype="auto", device_map="auto"
)
model.eval()
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=HF_CACHE)

# === UTILITIES ===
def animated_typing(text, delay=0.03):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def thinking_animation(stop_event):
    animations = ["dots", "spinner", "bounce"]
    chosen = random.choice(animations)
    base = random.choice([
        "Searching the universe for your answer",
        "Consulting the galactic library",
        "Tuning into deep space frequencies",
        "Aligning the virtual telescope",
        "Analyzing interstellar signals"
    ])
    sys.stdout.write("\r")
    for c in base:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(0.033)

    if chosen == "dots":
        frames = ["   ", ".  ", ".. ", "..."]
    elif chosen == "spinner":
        frames = ["|", "/", "-", "\\"]
    elif chosen == "bounce":
        frames = ["(*   )", "( *  )", "(  * )", "(   *)", "(  * )", "( *  )", "(*   )"]
    i = 0
    while not stop_event.is_set():
        frame = frames[i % len(frames)]
        sys.stdout.write(f"\r{base} {frame}")
        sys.stdout.flush()
        time.sleep(0.3)
        i += 1
    sys.stdout.write("\r" + " " * 80 + "\r")

def extract_keywords(text, top_n=5):
    text = re.sub(r"[^a-zA-Z0-9\s\-]", "", text.lower())
    words = text.split()
    filtered = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    freq = Counter(filtered)
    return sorted(set([word for word, _ in freq.most_common(top_n)]))

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

def build_context_vector_store(all_docs):
    vectors = embedder.encode(all_docs, convert_to_numpy=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, vectors

def retrieve_top_k(query, all_docs, index, k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    _, I = index.search(q_emb.reshape(1, -1), k)
    return [all_docs[i] for i in I[0]]

def truncate_context(context, max_tokens=950):
    words = context.split()
    return ' '.join(words[:max_tokens]) if len(words) > max_tokens else context

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

# === MAIN CHAT LOOP ===
if __name__ == "__main__":
    character_name = "AstroBot"

    greetings = [
        f"{character_name}: Ready to explore galaxies and gravitational waves?",
        f"{character_name}: Welcome aboard! Let’s navigate the universe together.",
        f"{character_name}: Ready to dive into black holes, stars, and strange new worlds?",
        f"{character_name}: Curious about the cosmos? You’ve come to the right place!",
        f"{character_name}: Let’s unfold the mysteries of the universe — ask away!",
        f"{character_name}: I’ve just calibrated the telescope. What shall we observe?",
        f"{character_name}: Whether it’s quasars or quantum foam — I’m here to help!",
        f"{character_name}: Welcome, fellow stargazer! The universe is full of answers."
    ]
    animated_typing("\n" + random.choice(greetings))
    animated_typing(f"{character_name}: Ask a question about stars, planets, galaxies, or any cosmic topic.")
    animated_typing(f"{character_name}: Type 'exit' to quit.")

    user_name = input(f"\n{character_name}: Let's get to know each other! I'm AstroBot, and you are?\n> ").strip()

    name_greetings = [
        f"{character_name}: {user_name}, ready to unlock the secrets of the stars? Let's go!",
        f"{character_name}: Great to meet you, {user_name}! Let’s chase some cosmic mysteries together.",
        f"{character_name}: Awesome, {user_name}. Fire away your questions and I’ll beam up the answers!",
        f"{character_name}: Buckle up, {user_name}! The universe is calling.",
        f"{character_name}: {user_name}, shall we surf some gravitational waves?",
        f"{character_name}: Star charts are open, {user_name}. What shall we explore?",
        f"{character_name}: {user_name}, cosmic knowledge is just a question away!",
        f"{character_name}: Let’s map the Milky Way together, {user_name}. What’s on your mind?"
    ]
    animated_typing("\n" + random.choice(name_greetings))

    last_sources = {}
    last_keywords = []
    follow_up = ""

    while True:
        if follow_up:
            temp_query = follow_up
            follow_up = ""
        else:
            temp_query = input(f"\n{user_name}: ").strip()

        normalized = temp_query.lower()

        if normalized == "exit":
            print(f"{character_name}: Goodbye {user_name}! Keep exploring the stars.")
            break

        if normalized.startswith("nn "):
            new_name = temp_query[3:].strip()
            if new_name:
                user_name = new_name
                animated_typing(f"{character_name}: Noted! I’ll call you {user_name} from now on.")
                animated_typing("\n" + random.choice(name_greetings).replace(user_name, new_name))
            else:
                animated_typing(f"{character_name}: Hmm... you didn’t give me a name. Try again with: nn YourName")
            continue

        if normalized == "source":
            if last_sources:
                ads_count = len(last_sources["ads"])
                arxiv_count = len(last_sources["arxiv"])
                total_count = len(last_sources["all"])
                print(f"\n{character_name}: Last used keywords -> {', '.join(last_keywords)}")
                print(f"{character_name}: Retrieved documents - NASA ADS: {ads_count}, arXiv: {arxiv_count}, Total: {total_count}\n")
                for i, src in enumerate(last_sources["all"], 1):
                    print(f"[DOC {i}]: {src}\n{'-'*60}")
                animated_typing(random.choice([
                    f"{character_name}: Want to ask another big question about the universe? I'm all ears.",
                    f"{character_name}: Wondering how something works in space? Try me.",
                    f"{character_name}: I can help you dive deeper into any cosmic idea. What are you curious about next?",
                    f"{character_name}: Any topic you want to unravel? I can help you explore it.",
                    f"{character_name}: Curious minds make the best astronomers. Got another question?",
                    f"{character_name}: Whether it's stars, space-time, or galaxies — just ask!",
                    f"{character_name}: There's always more to discover. What shall we look into now?",
                    f"{character_name}: You guide the telescope. What topic should we zoom in on next?"
                ]))
            else:
                print(f"{character_name}: No sources retrieved yet. Ask a question first.")
            continue

        query = temp_query
        stop_event = threading.Event()
        anim_thread = threading.Thread(target=thinking_animation, args=(stop_event,))
        anim_thread.start()

        keywords = extract_keywords(query)
        last_keywords = keywords

        ads_docs = fetch_ads_docs(keywords)
        arxiv_docs = fetch_arxiv_docs(keywords)
        all_docs = list(set(ads_docs + arxiv_docs))
        last_sources = {"ads": ads_docs, "arxiv": arxiv_docs, "all": all_docs}

        if not all_docs:
            stop_event.set()
            anim_thread.join()
            print(f"{character_name}: No relevant documents retrieved.")
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

        keywords = extract_keywords(answer)
        last_keywords = keywords

        if keywords:
            followup_templates = [
                f"{character_name}: Fascinating discovery, don’t you think? Perhaps you'd like to explore more about: {', '.join(keywords)}.",
                f"{character_name}: These stellar terms popped up - {', '.join(keywords)}. Want to go deeper?",
                f"{character_name}: The cosmos whispers secrets about: {', '.join(keywords)}. Curious to learn more?",
                f"{character_name}: We just skimmed the surface! You could dive into topics like: {', '.join(keywords)}.",
                f"{character_name}: The galactic trail leads us toward: {', '.join(keywords)}. Shall we follow it?"
            ]

            followup_prompt_templates = [
                f"{character_name}: Or we could hop to an entirely different constellation of ideas, {user_name}. What’s your next curiosity?",
                f"{character_name}: Or... is there another cosmic puzzle you're itching to solve, {user_name}?",
                f"{character_name}: Or we could zoom out and tackle something new across the galaxy, {user_name}. Your call!",
                f"{character_name}: Or shall we reroute to a brand new quadrant of the universe, {user_name}?",
                f"{character_name}: Or… we could shift course toward something completely unexpected, {user_name}. Just say the word!"
            ]

            followup_text = random.choice(followup_templates) + "\n" + random.choice(followup_prompt_templates)
            animated_typing(followup_text)
