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

def prefill(req:request)->none:
    if req.start_time is None:
        req.start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model(input_ids=req.intput_id, use_cache=True)

    next_token_id=torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
    req.generated_token_ids.append(next_token_id)
    req.past_kv=outputs.past_key_values
    req.is_prefill=True
    req.ttft=time.perf_counter()-req.arrival_time

    if next_token_id==tokenizer.eos_token_id or len(req.generated_token_ids)>=req.max_tokens:
        req.is_finished=True
        req.finish_time=time.perf_counter()

def decode(req:request)->None:
    if req.is_finished:
        return
    last_token_id=req.generated_token_ids[-1]
    input_tensor=torch.tensor([[last_token_id]], device=model.device)
    with torch.inference_mode():
        outputs=model(input_ids=input_tensor, past_key_values=req.past_kv, use_cache=True)
    next_token_id=torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
    req.past_kv=outputs.past_key_values
    req.generated_token_ids.append(next_token_id)
    if next_token_id==tokenizer.eos_token_id or len(req.generated_token_ids)>=req.max_tokens:
        req.is_finished=True
        req.finish_time=time.perf_counter()



    
