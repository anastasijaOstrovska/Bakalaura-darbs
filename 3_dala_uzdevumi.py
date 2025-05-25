#Sākumā tiek importētas nepieciešamas pākotnes
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import time
import torch

#Tiek lejuplādēts piekļuves pilnvara (angļu val. access token)
with open("hf_token.txt", "r") as f:
    auth_token = f.read().strip()

#Ir nepieciešams izvēlēties atbilstošo modeli

#model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#model_name = "microsoft/phi-2"
model_name = "mistralai/Mistral-7B-v0.1"

#Ir nepieciešams izvelēties modelim atbilstošo teksta vienību dalitāju (angļu val. tokenizer)

#tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True,model_max_length=2048, use_auth_token=auth_token, padding_side="left")
#tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,model_max_length=2048, use_auth_token=auth_token, padding_side="left")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,model_max_length=2048, use_auth_token=auth_token, padding_side="left")

#Teksta vienību dalitāju konfigurācija
tokenizer.pad_token = tokenizer.eos_token

#Ir nepieciešams izvelēties, kāda vieda tiek lejuplādēts modelis
#  priekš TinyLLaMA - LlamaForCausalLM
#  priekš Mistral un Phi-2 - AutoModelForCausalLM

#Modeļa lejupielāde, kas izmanto CPU
model = model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=None,
    low_cpu_mem_usage=False,
    trust_remote_code=True,
    use_auth_token=auth_token
).to("cpu")


#Uzvednes definējums - "Kas ir pretējs baltajam?"
prompt = "What is opposite to white?"

#Uzvednes sadaļīšana zīmes (angļu val. token)
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

#Tiek uzsākta ģenerācijas laika skaitīšana 
start = time.time()

#Atbildes ģenerēšana 
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
# Uzģeneēta teksta pārveidošana no zīmēm uz tekstu
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#Tiek pabeigta ģenerācijas laika skaitīšana 
end = time.time()

#Ģenerācijs laika un modeļa atbildi izvade
print(f"Modeļa ģenerācijas laiks:\n {end - start:.4f} sekundes")
print("Modeļa atbilde:\n")
print(response)
