import requests
import json
import random
import logging
import time
from datetime import datetime
from dotenv import load_dotenv
from os import getenv

logging.basicConfig(filename='tinynews_training_data_generation.log', encoding='utf-8', level=logging.ERROR, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

load_dotenv()
OPENROUTER_API_KEY = getenv("OPENROUTER_API_KEY")

with open("vocabulary.json", "r") as f:
    vocabulary = json.load(f)

def generate(prompt, response_start="", temperature=0, llama_size=8, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}"
                },
                data=json.dumps({
                    "model": f"meta-llama/llama-3-{llama_size}b-instruct",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response_start}
                    ],
                    "provider": {
                        "order": [
                            "DeepInfra",
                            "Novita"
                        ]
                    },
                    "temperature": temperature
                })
            )
            response = response.json()['choices'][0]['message']['content']
            return response
        except (requests.exceptions.ChunkedEncodingError, KeyError, requests.exceptions.JSONDecodeError) as e:
            retries += 1
            if retries < max_retries:
                print(f"Retry {retries}/{max_retries} due to error: {str(e)}")
                time.sleep(2 ** retries)  # Exponential backoff
            else:
                logging.error(f"Failed after {max_retries} retries: {str(e)}")
                return ""

def generate_news(words, response_start="<h1>", temperature=0, llama_size=8):
    
    story_count = len(words)

    prompt_beginning = f"""Generate {story_count+3} news stories with headlines
Each story should be 5 sentences long
Headlines should be descriptive
Write simple, direct prose
Use a vocabulary that a typical 6th grader would understand
Do not use proper nouns
Do not make up new words
Each headline Must contain a specified seed word:
"""
    
    prompt_middle = ""
    for i in range(story_count):
        prompt_middle += f"Headline {i+1}: {words[i]}\n"
        
    prompt_end = """Format your response as HTML:
<h1>Headline1</h1><p>Story1</p><h1>Headline2</h1><p>Story2</p>
YOUR REPLY WILL BE PARSED AS HTML SO DO NOT ADD ANY ADDITIONAL COMMENTS OUTSIDE OF THE HTML"""

    prompt = prompt_beginning + prompt_middle + prompt_end
    
    return generate(prompt, response_start, temperature, llama_size)

with open("few_shot_examples.json", "r") as f:
    few_shot_examples = json.load(f)
epoch = 0
batch = 0
new_epoch = True
seed_words = []
raw_training_data = []
print(f'Started Batch {batch} of Epoch {epoch} at {datetime.now()}')

while True:

    if new_epoch:
        random.shuffle(vocabulary)
        seed_words = [random.sample(word_list, 1)[0] for word_list in vocabulary]
        raw_training_data = []
        batch = 0
        paused_timer = 5
        new_epoch = False

    while (batch*10) < len(seed_words):

        sampled_examples = random.sample(sorted(few_shot_examples), 3)
        few_shot = "</p>\n\n".join([few_shot_examples[example] for example in sampled_examples]) + "</p>\n\n<h1>"
        seed_input = sampled_examples + seed_words[batch*10:(batch*10)+10]
        
        news = generate_news(seed_input, few_shot, temperature=0.5, llama_size=8)
        
        if news:
            try:
                raw_training_data.append(news)
                with open(f"training_data/raw_training_data_{epoch}.json", "w") as f:
                    json.dump(raw_training_data, f)
                batch += 1
                paused_timer = 5
            except:
                print("unknown error",end="\n\n")
                logging.error(news)
                print(news, end="\n\n")
                time.sleep(paused_timer)
                paused_timer *= 2
        else:
            print(f'ERROR | batch {batch} failed to generate', flush=True)
            time.sleep(paused_timer)
            paused_timer *= 2
        print(f'Generating Batch {batch} of Epoch {epoch}', flush=True, end="\r")
    epoch += 1
    new_epoch = True
