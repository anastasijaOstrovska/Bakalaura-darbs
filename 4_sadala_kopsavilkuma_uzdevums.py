#Sākumā tiek importētas nepieciešamas pākotnes
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    LlamaTokenizer,
    LlamaForCausalLM,
    pipeline
)
from tqdm import tqdm
from trl import SFTTrainer
from functools import partial
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time, psutil
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datetime import datetime
import threading


#Tiek pārbaudīts vai ir CUDA atbalsts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#Tiek lejupielādēts piekļuves pilnvara (angļu val. access token)
with open("hf_token.txt", "r") as f:
    auth_token = f.read().strip()

#Ir nepieciešams izvēlēties atbilstošo modeli

#model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name = "microsoft/phi-2"
#model_name = "mistralai/Mistral-7B-v0.1"

#4-bitu konfigurācija, efektīvai modeļa lejuplādei
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

#Ir nepieciešams izvelēties modelim atbilstošo teksta vienību dalitāju (angļu val. tokenizer)
#  priekš TinyLLaMA - LlamaTokenizer
#  priekš Phi-2 - AutoTokenizer model_max_length=2048
#  priekš Mistral - AutoTokenizer model_max_length=8192

#tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True,model_max_length=2048, use_auth_token=auth_token, padding_side="left")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,model_max_length=2048, use_auth_token=auth_token, padding_side="left")
#tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,model_max_length=8192, use_auth_token=auth_token, padding_side="left")

#Teksta vienību dalitāju konfigurācija
tokenizer.pad_token = tokenizer.eos_token

#Modeļa lejuplāde ar 4-bitu konfigurāciju un resursu ietaupīšanu
#Ir nepieciešams izvelēties, kāda vieda tiek lejuplādēts modelis
#  priekš TinyLLaMA - LlamaForCausalLM
#  priekš Mistral un Phi-2 - AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    quantization_config=bnb_config,
    use_auth_token=auth_token
)

#Datu kopas lejuplāde
train_df = pd.read_csv("samsum-train.csv", engine="python", on_bad_lines='skip')
test_df = pd.read_csv("samsum-test.csv", engine="python", on_bad_lines='skip')
val_df = pd.read_csv("samsum-validation.csv", engine="python", on_bad_lines='skip')

#Apmācību kopas samazināšana
train_df = train_df.sample(n=5600, random_state=42)

#Teksta pārveidošanas uzvendes veidā formātu definēšana
def generate_prompt(data_point):
    return f"""
            Summarize the following dialog enclosed in square brackets:\n\n

            [{data_point["dialogue"]}]
            Summary: {data_point["summary"]}
            """.strip()

def generate_test_prompt(data_point):
    return f"""
            Summarize the following dialog enclosed in square brackets:\n\n

            [{data_point["dialogue"]}] Summary: """.strip()

#Datu kopas teksta kolonnas pārveidošana uzvednes formatā
train_df["text"] = train_df.apply(generate_prompt, axis=1)
train_df = train_df[["text", "summary"]]

test_df["text"] = test_df.apply(generate_test_prompt, axis=1)
test_df = test_df[["text", "summary"]]

val_df["text"] = val_df.apply(generate_prompt, axis=1)
val_df = val_df[["text", "summary"]]

#Atbilžu saraksta definēšana
test_answers = test_df["summary"].tolist()

#Datu kopas pārveidošana apmācībai nepieciešamā formātā
train = Dataset.from_pandas(train_df)
eval = Dataset.from_pandas(val_df)
test = Dataset.from_pandas(test_df)

#Validācijas pakotne priekš ROUGE metrikas izmantošanas
import evaluate  

#Metriks ROUGE definēšana
rouge = evaluate.load("rouge")

#Negatīvo zīmes (angļu val.token) izdzēšana
def clear_negative(batch, tokenizer):
    cleaned = [[id for id in seq if id >= 0] for seq in batch]
    return tokenizer.batch_decode(cleaned, skip_special_tokens=True)

#Funkcijas prikš validācijas izstrāde, lai izmantotu ROUGE-L metriku
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    decoded_preds = clear_negative(preds, tokenizer)

    labels = [[token if token != -100 else tokenizer.pad_token_id for token in seq] for seq in labels]
    decoded_labels = clear_negative(labels, tokenizer)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: v * 100 for k, v in result.items()}

    return {"rougeL": result["rougeL"]}

#Funkcija priekš ģenerētas testa atbildes sallidzināšanai ar pareizām
def evaluate(y_true, y_pred):
    results = rouge.compute(predictions=y_pred, references=y_true)

    print("\n ROUGE rezultāts:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    with open(train_result, "a", encoding="utf-8") as f:
      f.write("ROUGE rezultāts:\n")
      for k, v in results.items():
        f.write(f"{k}: {v:.4f}\n")

#Konfigurācija priekš aprēķinu veikšanai 16-bitu formātā
compute_dtype = getattr(torch, "float16")

#Funkcija priekš ievadteksta saisināšana 
def short_input(text, tokenizer, max_tokens=1978):
    tokenized = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokenized, skip_special_tokens=True)

#Funkcija priekš testa kopas atbilžu ģenerēšanai
def predict(test_df, model, tokenizer, batch_size=4, max_input_tokens=1978, max_new_tokens=70):

    test_df = test_df.to_pandas()
    test_df["text"] = test_df["text"].apply(lambda x: short_input(x, tokenizer, max_input_tokens))

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        batch_size=batch_size
    )

    dataset = Dataset.from_pandas(test_df)
    results = pipe(dataset["text"])
    summaries = []

    for input_text, result in zip(dataset["text"], results):
      full_output = result[0]["generated_text"].strip()
      answer_only = full_output[len(input_text):].strip()
      summaries.append(answer_only)

    return summaries

#Faila priekš rezultātiem definēšana
train_result = "train_result.txt"

#Pirms apmācības modeļa veikstējas pārbaude
y_pred = predict(test, model, tokenizer)
evaluate(test_answers, y_pred)

#Svaru saglabšanas mapes definēšana  
output_dir="trained_weigths"

#Gradienta aktivācija
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

#LoRA konfigurācija
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8, 
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

#Maksimāla ģenerācijas garuma definēšana
max_seq_length = 1024

#Šadalīšanas zīmes konfigurācija 
def tokenize(examples):
    model_inputs = tokenizer(
        examples["text"],
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"],
            max_length=max_seq_length,    
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
#Apmācību un validācijas kopas sadalīšana zīmēs
tokenized_train = train.map(
    tokenize,
    batched=True,
    batch_size=4,  
    remove_columns=train.column_names
)

tokenized_eval = eval.map(
    tokenize,
    batched=True,
    batch_size=4,
    remove_columns=eval.column_names
)

#Apmācības parametru konfigurācija
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    weight_decay=0.001,
    predict_with_generate=True,
    logging_strategy="steps",
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=100,  
    save_strategy="steps",
    save_steps=100,  
    save_total_limit=2,
    fp16=True,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    max_grad_norm=0.3, 
    gradient_checkpointing=True
)

model.config.use_cache = False
#Peft konfigurācija (LoRA) 
peft_model = get_peft_model(model, peft_config)

#Ampācības modeļa, datu kopas un modeļa veida konfigurācija
trainer = Seq2SeqTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

#Resursu izmantošanas izvades faila definēšana
resource_file = "gpu_usage_log.txt"

#Funkcijas, kas ierakstīs izmantotus resursus katras 60 sekundēs
def trac_resources():
    with open(resource_file, 'w') as f:
        f.write("Laiks,CPU, RAM, Pieskirta_atmina_MB,Rezerveta_atmina_MB\n")
        while training_running:
            allocated = torch.cuda.memory_allocated(device) / 1e6
            reserved = torch.cuda.memory_reserved(device) / 1e6

            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent

            timestamp = datetime.now().strftime("%H:%M:%S")
            f.write(f"{timestamp},{cpu:.1f},{ram:.1f},{allocated:.2f},{reserved:.2f}\n")
            time.sleep(60)

#Resursu skaitīšanas sākums
training_running = True
logging= threading.Thread(target=trac_resources)
logging.start()

#Modeļa apmācība
start = time.time()
trainer.train()
end = time.time()

#Resursu skaitīšanas beigums
training_running = False
logging.join()

#Modeļa saglābšana
trainer.save_model()
tokenizer.save_pretrained(output_dir)

#Saglabāta modeļa lejupielāde
trained_model_name = "trained_weigths/checkpoint-435"
trained_model = AutoModelForCausalLM.from_pretrained(
    trained_model_name,
    device_map=device,
    torch_dtype=compute_dtype,
    quantization_config=bnb_config,
)

trained_model.config.use_cache = False
trained_model.config.pretraining_tp = 1

#Apmācības laika izvade failā
print(f"Apmacības laiks: {(end - start)/60:.2f} min")
with open(train_result, "a", encoding="utf-8") as f:
    f.write(f"Apmācības laiks: {(end - start)/60:.2f} min\n\n")

#Apmācīta modeļa veikstējas pārbaude
y_trained_pred = predict(test, trained_model, tokenizer)
evaluate(test_answers, y_trained_pred)