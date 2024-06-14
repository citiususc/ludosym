import json
import math
import torch
import adapters
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
from transformers import pipeline
# from adapters import AutoAdapterModel, AdapterConfig

class CustomTextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.texts = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for l in data:
                dd = json.loads(l)
                self.texts.append(dd['text'])
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

class CustomDataCollatorForLanguageModeling:
    def __init__(self, model, tokenizer, mask_probability=0.15, lexicon=None):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability
        self.lexicon = lexicon

    def __call__(self, batch):
        input_ids = [item['input_ids'].squeeze() for item in batch]
        attention_mask = [item['attention_mask'].squeeze() for item in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = input_ids.clone()

        lexicon_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self.lexicon is not None:
            input_ids, lexicon_mask = self._apply_lexicon_masking(input_ids)

        probability_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self.mask_probability > 0:
            input_ids, probability_mask = self._apply_probability_masking(input_ids, lexicon_mask)

        labels[(~lexicon_mask) & (~probability_mask)] = -100
        # self._print_masked_tokens_and_predictions(input_ids, lexicon_mask)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    def _apply_lexicon_masking(self, input_ids):
        lexicon_token_sequences = [self.tokenizer.encode(word, add_special_tokens=False) for word in self.lexicon]
        # print(f"Tokens del léxico: {lexicon_token_sequences}")  # Verifica la tokenización del léxico
        mask_token_id = self.tokenizer.mask_token_id
        lexicon_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for seq in lexicon_token_sequences:
            seq_len = len(seq)
            for i in range(len(input_ids)):
                for j in range(len(input_ids[i]) - seq_len + 1):
                    if input_ids[i][j:j + seq_len].tolist() == seq:
                        input_ids[i][j:j + seq_len] = torch.tensor([mask_token_id] * seq_len, dtype=input_ids.dtype)
                        lexicon_mask[i][j:j + seq_len] = True

        return input_ids, lexicon_mask

    def _apply_probability_masking(self, input_ids, lexicon_mask):
        total_tokens = input_ids.numel()
        max_masked_tokens = int(total_tokens * self.mask_probability)
        remaining_masked_tokens = max_masked_tokens - lexicon_mask.sum().item()

        if remaining_masked_tokens <= 0:
            return input_ids, torch.zeros_like(input_ids, dtype=torch.bool)

        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_token_id in self.tokenizer.all_special_ids:
            special_tokens_mask[input_ids == special_token_id] = True

        probability_mask = torch.rand(input_ids.shape) < (remaining_masked_tokens / total_tokens)
        mask_token_id = self.tokenizer.mask_token_id
        probability_mask[input_ids == self.tokenizer.pad_token_id] = False
        probability_mask[lexicon_mask] = False  # No volver a enmascarar los tokens del léxico
        input_ids[probability_mask] = mask_token_id

        return input_ids, probability_mask
    
    def _print_masked_tokens_and_predictions(self, input_ids, labels):
        sentence = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        masked_sentence = self.tokenizer.decode([token_id if token_id != -100 else self.tokenizer.pad_token_id for token_id in labels[0]], skip_special_tokens=True)

        print("Oración original:")
        print(sentence)
        print("\nOración enmascarada:")
        print(masked_sentence)

        fill_mask = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer)

        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        masked_indices = (input_ids[0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]

        for idx in masked_indices:
            masked_token_list = input_tokens.copy()
            masked_token_list[idx] = self.tokenizer.mask_token
            text_with_mask = self.tokenizer.convert_tokens_to_string(masked_token_list)
            print(text_with_mask)
            result = fill_mask(text_with_mask)
            print(f"\nPredicciones para la máscara en la posición {idx}:")
            print(result)
            #for res in result:
            #    for pred in res:
            #        print(f"Token predicho: {pred['token_str']}, Score: {pred['score']:.4f}")
            #    print()

# Cargar el modelo y el tokenizer BERT multilingüe
model_name = "dccuchile/bert-base-spanish-wwm-cased"
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

lexico_ludopatia = [
    "ludopatía", "enfermedad", "cura", "parar", "jugar", "apuestas", "juego", "resultados",
    "desastrosos", "autoexcluir", "tarjetas", "créditos", "salón", "cuenta", "dinero", "mentir",
    "situación", "adicción", "imposible", "ayuda", "familia", "problemas", "financieros", "relativamente",
    "controlarte", "abandonar", "mundo", "ludopata", "frustración", "ansiedad", "estres", "recuperar",
    "ganar", "perder", "solución", "controlar", "voluntad", "solución", "temporal", "deudas",
    "problemas", "ilegales", "videojuegos", "rutina", "maquina", "tragaperras", "terapia", "apoyo",
    "rehabilitación", "emociones", "depresión", "relaciones", "mentira", "riesgo", "consecuencias",
    "preocupación", "empeorar", "desesperación", "culpa", "ira", "miedo", "recuperación", "sobrecarga",
    "luchas", "familiares", "amigos", "trabajo", "abandono", "recaída", "rechazo", "autoestima",
    "comportamiento", "necesidades", "económicas", "compulsivo", "obsesivo", "tratamiento", "terapia",
    "confianza", "honestidad", "responsabilidad", "dependencia", "recuperar", "apoyo", "ayuda",
    "consumo", "descontrol", "obsesión", "control", "esperanza", "rehabilitar", "desintoxicación",
    "resiliencia", "adaptación", "motivación", "superación"
]


data_collator = CustomDataCollatorForLanguageModeling(model=model, tokenizer=tokenizer, lexicon=lexico_ludopatia)

### Splits train y validación
#split_train_test("../dataset/k50/k50_collection.json")
train_dataset = CustomTextDataset(file_path='../dataset/corpus_train.jsonl', tokenizer=tokenizer)
test_dataset = CustomTextDataset(file_path='../dataset/corpus_test.jsonl', tokenizer=tokenizer)
print(len(train_dataset))

# Especificación de los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=4,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    report_to="none",
    save_strategy="no"
)

####### TRAIN
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
trainer.train()
trainer.save_model('./models/mixed-full-beto-mlm-gambling')

##### VALIDATION
data_collator_random = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15  # Puedes ajustar la probabilidad de enmascaramiento
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator_random,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

eval_results = trainer.evaluate()
print(f">>> Perplexity Custom: {math.exp(eval_results['eval_loss']):.2f}")

model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator_random,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
eval_results = trainer.evaluate()
print(f">>> Perplexity Base: {math.exp(eval_results['eval_loss']):.2f}")