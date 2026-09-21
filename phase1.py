import time 
import threading 
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM,TextIteratorStreamer
model_id="meta-llama/Llama-3.2-1B-Instruct"
print("loading model")
tokenizer=AutoTokenizer.from_pretrained(model_id)
model=AutoModelForCausalLM.from_pretrained(model_id,device_map="cuda", torch_dtype=torch.float16, trust_remote_code=True)
print("model loaded")
