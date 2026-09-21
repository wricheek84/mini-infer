import time 
import threading 
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM,TextIteratorStreamer
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_KEY")
login(token=hf_token)
model_id="meta-llama/Llama-3.2-1B-Instruct"
print("loading model")
tokenizer=AutoTokenizer.from_pretrained(model_id)
model=AutoModelForCausalLM.from_pretrained(model_id,device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
print("model loaded")
def cold(model,tokenizer):
    prompt=tokenizer("hello",return_tensors="pt").to(model.device)
    _=model.generate(**prompt,max_new_tokens=3)
    torch.cuda.reset_peak_memory_stats()
print ("fixing cold start")
cold(model,tokenizer)
print("cold start fixed")

