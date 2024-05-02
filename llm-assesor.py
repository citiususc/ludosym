import json
import openai
import tiktoken
import csv

openai.api_key = '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'
model = 'gpt-4'
enc = tiktoken.encoding_for_model(model)

def load_json(fn:str)->dict:
    with open(fn) as user_file:
        parsed_json = json.load(user_file)
    return parsed_json

def load_prompt(fn:str)->str:
    with open(fn) as prompt_file:
        return prompt_file.read()

def assess(prompt:str, question:str, sentence:str)->str:
    prompt = prompt + f'\n\n\nPregunta: {question}\nOración a evaluar como relevante para la pregunta anterior: {sentence}\nEtiqueta:'
    rel_token, irrel_token = enc.encode('Relevante')[0], enc.encode('No relevante')[0]
    print(prompt)
    response = openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                logit_bias= {rel_token:0.5, irrel_token:0.5},
                max_tokens=3,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

if __name__=="__main__":
    pool = load_json('./dataset/k30/k30_pools.json')
    collection = load_json('./dataset/k30/k30_collection.json')
    questions = load_json('./res/dsm-v.json')
    prompt = load_prompt('./res/labelling_prompt.txt')

    for question_id in pool.keys():
        if question_id=="8" or question_id=="9":
            for sentence_id in pool[question_id]:
                label = assess(prompt, questions[question_id], collection[str(sentence_id)])
                with open('./outputs/'+model+'-2.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([question_id, sentence_id, label])
