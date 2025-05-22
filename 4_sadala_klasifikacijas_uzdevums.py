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
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, accuracy_score, confusion_matrix
import time, psutil
from peft import LoraConfig
from datasets import Dataset
from datetime import datetime
import threading


#Tiek pārbaudīts vai ir CUDA atbalsts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#Tiek lejuplādēts piekļuves pilnvara (angļu val. access token)
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
#  priekš Mistral un Phi-2 - AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token=auth_token, padding_side="left")
#tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token=auth_token, padding_side="left")

#Teksta vienību dalitāju konfigurācija
tokenizer.pad_token = tokenizer.eos_token

#Modeļa lejuplāde ar 4-bitu konfigurāciju un resursu ietaupīšanu
#Ir nepieciešams izvelēties, kāda vieda tiek lejuplādēts modelis
#  priekš TinyLLaMA - LlamaForCausalLM
#  priekš Mistral un Phi-2 - AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    use_auth_token=auth_token
)

#Tiek definēts rezultātu izvades fails
train_result = "train_result.txt"

#Datu kopas lejuplāde
df = pd.read_csv('spam.csv')
df.columns = ["label", "text"]

#Datu kopas sadalīšana apmācību, testa un validācijas (angļu val. evaluation) kopas 
train_list, test_list  = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
train_list, eval_list  = train_test_split(train_list, test_size=0.2, random_state=42, stratify=train_list["label"])

#Teksta pārveidošanas uzvendes veidā formātu definēšana
def gen_train_prompt(row):
    return f"""
            Analyze the message enclosed in square brackets,
            determine if it is spam or ham (not spam), and return the answer as
            the corresponding label "spam" or "ham".

            [{row["text"]}] = {row["label"]}
            """.strip()

def gen_test_prompt(row):
    return f"""
            Analyze the message enclosed in square brackets,
            determine if it is spam or ham (not spam), and return the answer as
            the corresponding label "spam" or "ham".

            [{row["text"]}] = """.strip()

#Atbilžu saraksta definēšana
test_answers = test_list.label

#Datu kopas teksta kolonnas pārveidošana uzvednes formatā
train_list = pd.DataFrame(train_list.apply(gen_train_prompt, axis=1), columns=["text"])
test_list = pd.DataFrame(test_list.apply(gen_test_prompt, axis=1), columns=["text"])
eval_list = pd.DataFrame(eval_list.apply(gen_train_prompt, axis=1), columns=["text"])

#Datu kopas pārveidošana apmācībai nepieciešamā formātā
train_dataset = Dataset.from_pandas(train_list)
eval_dataset = Dataset.from_pandas(eval_list)

#Funkicjas, kas salīdzinās ģenerētas atbildes ar pareizām
def evaluate(test_answers, predicted):
    labels = ['spam', 'ham', 'unpredicted']
    mapping = {'spam': 1, 'ham': 0, 'unpredicted': 2}

    def map_func(x):
        return mapping.get(x, 1)

    test_answers = np.vectorize(map_func)(test_answers)
    predicted = np.vectorize(map_func)(predicted)

    class_labels = ['Surogātpasts', 'Informatīva vēstule', 'Neprognozēts']

    accuracy = accuracy_score(y_true=test_answers, y_pred=predicted)
    with open(train_result, "a", encoding="utf-8") as f:
          f.write(f'Precizitāte - `{accuracy:.4f}`\n')

    accuracy_labels = set(test_answers)

    for label in accuracy_labels:
        label_index = [i for i in range(len(test_answers))
                         if test_answers[i] == label]
        label_answers = [test_answers[i] for i in label_index]
        label_predicted = [predicted[i] for i in label_index]
        accuracy = accuracy_score(label_answers, label_predicted)
        with open(train_result, "a", encoding="utf-8") as f:
          f.write(f'Precizitāte priekš `{class_labels[label]}`: {accuracy:.4f}\n')

    conf_matrix = confusion_matrix(y_true=test_answers, y_pred=predicted, labels=[0, 1, 2])
    with open(train_result, "a", encoding="utf-8") as f:
          f.write(f"Pārpratumu matrica:\n{conf_matrix}\n")

#Konfigurācija priekš aprēķinu veikšanai 16-bitu formātā
compute_dtype = getattr(torch, "float16")

#Funkcija priekš testa kopas atbilžu ģenerēšanai
def predict(predictor_model, tokenizer, batch_size=4):
    predicted = []

    pipe = pipeline(task="text-generation",
            model=predictor_model,
            tokenizer=tokenizer,
            max_new_tokens = 15,
            batch_size = batch_size
    )
    for i in tqdm(range(len(test_list))):
        prompt = test_list.iloc[i]["text"]
        result = pipe(prompt)
        answer = result[0]['generated_text']
        answer = answer.lower()
        position_spam = answer.find('spam', len(prompt))
        position_ham = answer.find('ham', len(prompt))
        if position_spam != -1 and (position_spam < position_ham or position_ham == -1):
            predicted.append("spam")
        elif position_ham != -1 and (position_ham < position_spam or position_spam == -1):
            predicted.append("ham")
        else:
            predicted.append('unpredicted')
    return predicted

#Pirms apmācības modeļa veikstējas pārbaude
predicted = predict(model, tokenizer)
evaluate(test_answers, predicted)

#Svaru saglabšanas mapes definēšana  
output_dir="weigths"

#LoRA konfigurācija
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
)

#Apmācības parametru konfigurācija
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=50,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=False,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    metric_for_best_model='eval_loss',
    load_best_model_at_end=True
)

#Ampācības modeļa, datu kopas un modeļa veida konfigurācija
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)
#Resursu izmantošanas izvades faila definēšana
log_file = "gpu_usage_log.txt"

#Funkcijas, kas ierakstīs izmantotus resursus katras 60 sekundēs
def log_gpu_memory():
    with open(log_file, 'w') as f:
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
log_thread = threading.Thread(target=log_gpu_memory)
log_thread.start()

#Modeļa apmācība
start = time.time()
trainer.train()
end = time.time()

# Resursu skaitīšanas beigums
training_running = False
log_thread.join()

#Modeļa saglābšana
trainer.save_model()
tokenizer.save_pretrained(output_dir)

#Saglabāta modeļa lejuplāde
trained_model_name = "weigths/checkpoint-550"
trained_model = AutoModelForCausalLM.from_pretrained(
    trained_model_name,
    device_map=device,
    torch_dtype=compute_dtype,
    quantization_config=bnb_config,
)

#Apmācības laika izvade failā
with open(train_result, "a", encoding="utf-8") as f:
      f.write(f"Apmācības laiks: {(end - start)/60:.2f} min\n")

#Apmācīta modeļa veikstējas pārbaude
trained_predictions = predict(trained_model, tokenizer)
evaluate(test_answers, trained_predictions)