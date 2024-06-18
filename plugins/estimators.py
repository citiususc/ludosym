
from plugins.models import GPTApiModel, HuggingFlanT5, Mistral, Gemini as GeminiMe


class FlanT5:
    model = HuggingFlanT5("google/flan-t5-small")

class GPT4:
    model = GPTApiModel("gpt-4")

class GPT4_turbo:
    model = GPTApiModel("gpt-4-turbo")

class GPT3:
    model = GPTApiModel("gpt-3.5")

class GPT3_turbo:
    model = GPTApiModel("gpt-3.5-turbo")

class Mistral_7b:
    model = Mistral('open-mistral-7b')

class Mistral_large:
    model = Mistral('mistral-large-latest')

class Mistral_small:
    model = Mistral('mistral-small-latest')

class Mistral_code:
    model = Mistral('codestral-latest')

class Gemini_flash15:
    model = GeminiMe('gemini-1.5-flash-latest')

class Gemini_pro15:
    model = GeminiMe('gemini-1.5-pro-latest')

class Gemini_pro10:
    model = GeminiMe('gemini-1.0-pro-latest')