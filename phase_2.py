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
    input_id:torch.Tensor
    generated_token_ids:List[int]=field(default_factory=list)
    past_kv:Any=None
    is_prefill:bool=False
    is_finished:bool=False
    arrival_time:float=field(default_factory=time.perf_counter)
    start_time:Optional[float]=None
    ttft:Optional[float]=None
    finish_time:Optional[float]=None

def prefill(req:request)->None:
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

def run_continous_batching(req_to_run: List[request]) -> List[request]:
    waiting_queue: List[request] = list(req_to_run)
    active_batch: List[request] = []
    finished_requests: List[request] = []

    while waiting_queue or active_batch:
        while waiting_queue:
            req = waiting_queue.pop(0)
            prefill(req)

            if req.is_finished:
                finished_requests.append(req)
            else:
                active_batch.append(req)

        for req in active_batch:
            decode(req)

        remaining_active_batch = [req for req in active_batch if not req.is_finished]

        for req in active_batch:
            if req.is_finished:
                finished_requests.append(req)

        active_batch = remaining_active_batch

    return finished_requests

prompt_a_text = "Explain the history of Kolkata in detail, including its cultural and economic growth:"
prompt_b_text = "What is the capital of France? Answer in one word:"

input_ids_a = tokenizer(prompt_a_text, return_tensors="pt").input_ids.to(model.device)
input_ids_b = tokenizer(prompt_b_text, return_tensors="pt").input_ids.to(model.device)

req_a = request(
    req_id="Req-A (Long)",
    prompt=prompt_a_text,
    max_tokens=256,
    input_ids=input_ids_a
)

req_b = request(
    req_id="Req-B (Short)",
    prompt=prompt_b_text,
    max_tokens=15,
    input_ids=input_ids_b
)

print(" Starting Continuous Batching Engine ")

benchmark_start = time.perf_counter()
completed_requests = run_continous_batching([req_a, req_b])
benchmark_total = time.perf_counter() - benchmark_start

print(f" All Requests Finished in {benchmark_total:.2f}s")

for req in completed_requests:
    output_text = tokenizer.decode(
        req.generated_token_ids,
        skip_special_tokens=True
    )
    total_latency = req.finish_time - req.arrival_time

    print(f"[{req.req_id}]")
    print(f"  Tokens Produced: {len(req.generated_token_ids)}")
    print(f"  TTFT:            {req.ttft:.4f}s")
    print(f"  Total Latency:   {total_latency:.2f}s")
    print(f"  Output Preview:  {output_text.strip()[:80]}\n")