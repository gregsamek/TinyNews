import requests
import json
import logging
import time
from dotenv import load_dotenv
from os import getenv
from tqdm import tqdm

logging.basicConfig(filename='self_prompt_samples_temp_1_topk_32.log', encoding='utf-8', level=logging.ERROR, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

load_dotenv()
OPENROUTER_API_KEY = getenv("OPENROUTER_API_KEY")

def generate(prompt, bulletin, response_start="", temperature=0, llama_size=70, max_retries=3):
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
                        {"role": "assistant", "content": """{"notes": "The bulletin matches the headline and is coherent. The prose is logical and easy to understand. Grammar and punctuation are correct. The language is simple and direct, as instructed. The bulletin does not repeat phrases and provides some additional information to support the headline.", "evaluation": 5}"""},
                        {"role": "user", "content": """Input Headline:\ncity council decides on scrapping old building\n\nOutput Bulletin:\nthe city council has decided to dart its old building over ceats and properly ceases. the old building, which was once a popular tourist attraction, has had a transformation over the years. the council boxer found s together just to help address the city's history and built beautiful gardens. the final decision to build the building has been given off at midst!"""},
                        {"role": "assistant", "content": """{"notes": "The bulletin does not match the headline, as it mentions 'dart its old building', 'ceats', 'properly ceases', 'boxer found s', and 'given off at midst', which are nonsensical phrases. The prose is illogical and words are used incorrectly. Grammar and punctuation are also incorrect. The language is not coherent and does not provide a clear understanding of the topic.", "evaluation": 1}"""},
                        {"role": "user", "content": bulletin},
                        {"role": "assistant", "content": response_start}
                    ],
                    "provider": {
                        "order": [
                            "Together",
                            "OctoAI",
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

def generate_evaluation(bulletin, temperature=0, llama_size=70):
    
    prompt = f"""INSTRUCTIONS
You are evaluating an AI model that was trained to generate fictional news bulletins \
based on a headline that was provided as input. Use the following criteria to evaluate the bulletin:
1. Does the bulletin match the headline?
2. Is the bulletin coherent? Is the prose logical or is it nonsensical? Are words used correctly?
3. Does the bulletin use correct grammar and punctuation?
4. Does the bulletin repeat the same phrases over and over? (it's OK if the first sentence of the bulletin restates the headline)
5. The model was specifically instructed to generate simple, direct prose at a 6th grade reading level and to \
avoid using proper nouns. Take this into consideration by not penalizing simplistic language.

You will first write a few notes about the bulletin, observing how well it meets the above criteria. \
You will then grade the bulletin on a scale of 1-5, with 1 being the worst and 5 being the best. \

You will respond in JSON using this format:
{{"notes": "Your evaluation notes here", "evaluation": (a number from 1-5 here)}}

Your response will be parsed by a JSON script, so please make sure your response is valid JSON. \
Do not make any additional comments outside of the JSON format. \
---
Input Headline:
statistics show rise in volunteer work

Output Bulletin:
new statistics show that more people are volunteering their time to help others. \
according to the report, volunteer work has increased by 20% in the past year. \
this is great news for the community, as volunteers are essential in helping those in need. \
the statistics also show that young people are more likely to volunteer than ever before. \
this trend is expected to continue in the coming years.
"""
    
    headline_bulletin = f"""Input Headline:\n{bulletin.split("</h>")[0]}\n\nOutput Bulletin:\n{bulletin.split("</h>")[1]}"""
    
    response_start='{"notes": "'
        
    return generate(prompt, headline_bulletin, response_start, temperature, llama_size)

with open("self_prompt_samples_temp_1_topk_32.json", "r") as f:
    self_prompt_samples_temp_1_topk_32 = json.load(f)

evaluations = {}
paused_timer = 5

for params in self_prompt_samples_temp_1_topk_32.keys():

    evaluations[params] = []

    for bulletin in tqdm(self_prompt_samples_temp_1_topk_32[params]):

        evaluation = generate_evaluation(bulletin, temperature=0.5, llama_size=70)

        if evaluation:
            try:
                evaluations[params].append(evaluation)
                with open(f"results/70b_as_judge/self_prompt_samples_temp_1_topk_32_evaluation.json", "w") as f:
                    json.dump(evaluations, f)
                paused_timer = 5
            except:
                print("unknown error",end="\n\n")
                logging.error(evaluation)
                print(evaluation, end="\n\n")
                time.sleep(paused_timer)
                paused_timer *= 2
        else:
            print(f'ERROR | failed to generate', flush=True)
            time.sleep(paused_timer)
            paused_timer *= 2