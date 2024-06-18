from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import openai
import google.generativeai as genai
import os

class Gemini:
    def __init__(self, clazz):
        self.clazz = clazz
        self.api_key = os.environ["GEMINI_KEY"]

        genai.configure(api_key=self.api_key)
        
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        self.model = genai.GenerativeModel(model_name=self.clazz, generation_config=generation_config)
        

    def send(self, messages):
        response = self.model.generate_content(
            list(map(lambda x: {'role': 'user' if  x['role'] in ['user', 'system'] else 'model', 'parts':[x['content']]}, messages))
        )
        messages.append({"role":"assistant", "content":response.text})
        return messages

        

class Mistral:
    def __init__(self, clazz):
        self.clazz = clazz
        self.api_key = os.environ["MISTRAL_KEY"]

    def send(self, messages):
        client = MistralClient(api_key=self.api_key)
        #ChatMessage(role="user", content="What is the best French cheese?")
        chat_response = client.chat(
            model=self.clazz,
            messages=messages
        )

        reply = chat_response.choices[0].message.content

        print(chat_response.choices[0])
        messages.append({"role":"assistant", "content":reply})
        
        return messages

class HuggingFlanT5:
    def __init__(self, clazz):
        self.clazz = clazz
        self.model = AutoModelForSeq2SeqLM.from_pretrained(clazz)
        self.tokenizer = AutoTokenizer.from_pretrained(clazz)
    
    def send(self, messages, *args, **kwargs):
        in_text = " ".join(list(map(lambda x: x["content"] if x["role"] == 'user' else '\n', messages)))
        inputs = self.tokenizer(in_text, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for resp in response:
            messages.append({"role":"assistant", "content":resp.lower()})
        return messages
    
    
class GPTApiModel:
    def __init__(self, clazz):
        self.clazz = clazz
        self.api_key = os.environ["OPENAI_KEY"]


    def send(self, messages, *args, **kwargs):
        response = openai.ChatCompletion.create(
                    api_key=self.api_key,
                    model=self.clazz,
                    messages=messages
                )

        reply = dict(response["choices"][0]["message"])
        messages.append(reply)
        
        return messages