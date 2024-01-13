import os
import re
import json
from bs4 import BeautifulSoup


def clean_text(text):
    text = text.lower()
    text = re.sub(r" |-", " ", text)
    text = re.sub(r"[^a-z0-9\40]", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()
    return text


def file_to_prompt_pairs(fd):
    soup = BeautifulSoup(fd, "html.parser")
    conversation_turn_divs = soup.find_all(attrs={"data-testid": re.compile(r"conversation-turn")})

    prompt_answer_pairs = []
    for i in range(0, len(conversation_turn_divs), 2):
        prompt = clean_text(conversation_turn_divs[i].find(class_="flex-grow").text)
        answer = clean_text(conversation_turn_divs[i + 1].find(class_="flex-grow").text)

        prompt_answer_pairs.append((prompt, answer))
    return prompt_answer_pairs


raw_data = {}
file_names = os.listdir('./materials/dataset/')
for file_name in file_names:
    fd = open(f"./materials/dataset/{file_name}")
    key = file_name.removesuffix('.html')
    raw_data[key] = file_to_prompt_pairs(fd)
    fd.close()

raw_data_fd = open('raw_data.json', 'w')
json.dump(raw_data, raw_data_fd)
