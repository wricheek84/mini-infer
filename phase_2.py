import time
from dataclasses import dataclass, field
from typing import Optional, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_KEY")
login(token=hf_token)

model_name= "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
model.eval()
@dataclass
class request:
    req_id:str
    prompt:str
    max_tokens:int
    intput_id:torch.Tensor
    generated_token_ids:List[int]=field(default_factory=list)
    past_kv:Any=None
    is_prefill:bool=False
    is_finished:bool=False
    arrival_time:float=field(default_factory=time.perf_counter)
    start_time:Optional[float]=None
    ttft:Optional[float]=None
    finish_time:Optional[float]=None
    
