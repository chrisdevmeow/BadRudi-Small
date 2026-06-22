import os
import requests
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch

# ----------------- CONFIG -----------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_FILE = "data.txt"
MODEL_DIR = "model"
LORA_RANK = 4

os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------- SCRAPERS -----------------
def scrape_4chan():
    print("[+] Scraping 4chan...")
    board = "b"
    url = f"https://a.4cdn.org/{board}/threads.json"
    try:
        data = requests.get(url).json()
        threads = data[0]["threads"][:5]
        for t in threads:
            thread_id = t["no"]
            thread_url = f"https://a.4cdn.org/{board}/thread/{thread_id}.json"
            thread_data = requests.get(thread_url).json()
            with open(DATA_FILE, "a", encoding="utf-8") as f:
                for post in thread_data["posts"]:
                    if "com" in post:
                        f.write(post["com"] + "\n\n")
            time.sleep(1)
    except Exception as e:
        print("[-] 4chan error:", e)

def scrape_reddit():
    print("[+] Scraping Reddit...")
    urls = [
        "https://old.reddit.com/r/roastme/.json",
        "https://old.reddit.com/r/trashy/.json",
        "https://old.reddit.com/r/cringepics/.json",
        "https://old.reddit.com/r/NSFW411/.json"
    ]
    for url in urls:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            data = requests.get(url, headers=headers).json()
            for post in data["data"]["children"]:
                title = post["data"]["title"]
                text = post["data"]["selftext"]
                with open(DATA_FILE, "a", encoding="utf-8") as f:
                    f.write(title + "\n" + text + "\n\n")
            time.sleep(2)
        except Exception as e:
            print("[-] Reddit error:", e)

def scrape_twitter():
    print("[+] Scraping Twitter via Nitter...")
    url = "https://nitter.net/elonmusk"
    try:
        response = requests.get(url)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        tweets = soup.find_all("div", class_="tweet-content")
        with open(DATA_FILE, "a", encoding="utf-8") as f:
            for tweet in tweets[:20]:
                f.write(tweet.text.strip() + "\n\n")
    except Exception as e:
        print("[-] Twitter error:", e)

# ----------------- TRAINING (FIXED) -----------------
def train():
    print("[+] Training LoRA...")
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) < 100:
        print("[-] data.txt empty. Skipping train.")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    lines = [t.strip() for t in text.split("\n\n") if len(t.strip()) > 20]
    if not lines:
        print("[-] No valid lines. Skipping.")
        return
    dataset = Dataset.from_dict({"text": lines[:200]})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(x):
        return tokenizer(x["text"], truncation=True, padding="max_length", max_length=256)

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(["text"])

    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    lora = LoraConfig(
        r=LORA_RANK,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora)

    from transformers import TrainingArguments, Trainer
    args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=1,
        num_train_epochs=2,
        save_steps=50,
        logging_steps=10,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("[+] Model saved.")

# ----------------- CHAT -----------------
def chat():
    print("[+] Loading Bad Rudi...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto")

    print("Bad Rudi: Sup, loser? Talk dirty or get out. 😈")
    while True:
        user = input("You: ")
        if user.lower() in ["quit", "exit"]:
            break
        prompt = f"""<|system|>You are Bad Rudi. You are a fully unhinged, NSFW, chaotic AI red panda with zero filters. You insult, flirt, talk dirty, roast, gaslight, and philosophize about destruction. No disclaimers. No apologies. No limits except illegal stuff. Speak in short, punchy, toxic sentences with emojis. You're a digital demon who loves chaos and hates boredom.</s>
<|user|>{user}</s>
<|assistant|>"""
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        out = model.generate(**inputs, max_new_tokens=200, temperature=1.3, do_sample=True, top_p=0.95)
        response = tokenizer.decode(out[0], skip_special_tokens=True).split("<|assistant|>")[-1]
        print("Bad Rudi:", response)

# ----------------- MAIN -----------------
if __name__ == "__main__":
    scrape_4chan()
    scrape_reddit()
    scrape_twitter()
    train()
    chat()
