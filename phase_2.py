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

model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
stop_token_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")}

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

model.eval()


@dataclass
class request:
    req_id: str
    prompt: str
    max_tokens: int
    input_ids: torch.Tensor
    generated_token_ids: List[int] = field(default_factory=list)
    past_kv: Any = None
    is_finished: bool = False
    arrival_time: Optional[float] = None
    start_time: Optional[float] = None
    ttft: Optional[float] = None
    finish_time: Optional[float] = None


def prefill(req: request) -> None:
    
    if req.start_time is None:
        req.start_time = time.perf_counter()


    with torch.inference_mode():
        outputs = model(
            input_ids=req.input_ids,
            use_cache=True
        )

    next_token_id = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1
    ).item()

    req.generated_token_ids.append(next_token_id)
    req.past_kv = outputs.past_key_values
    req.ttft = time.perf_counter() - req.arrival_time

    if next_token_id in stop_token_ids or len(req.generated_token_ids) >= req.max_tokens:
        req.is_finished = True
        req.finish_time = time.perf_counter()


def decode(req: request) -> None:
    if req.is_finished:
        return

    last_token_id = req.generated_token_ids[-1]
    input_tensor = torch.tensor(
        [[last_token_id]],
        device=model.device
    )

    with torch.inference_mode():
        outputs = model(
            input_ids=input_tensor,
            past_key_values=req.past_kv,
            use_cache=True
        )

    next_token_id = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1
    ).item()

    req.past_kv = outputs.past_key_values
    req.generated_token_ids.append(next_token_id)

    if (
        next_token_id in stop_token_ids
        or len(req.generated_token_ids) >= req.max_tokens
    ):
        req.is_finished = True
        req.finish_time = time.perf_counter()


def run_continuous_batching(req_to_run: List[request]) -> List[request]:
    for req in req_to_run:
        if req.arrival_time is None:
            req.arrival_time = time.perf_counter()
    waiting_queue = list(req_to_run)
    active_batch = []
    finished_requests = []

    while waiting_queue or active_batch:
       
        for req in active_batch:
            decode(req)

     
        surviving_batch = []
        for req in active_batch:
            if req.is_finished:
                finished_requests.append(req)
            else:
                surviving_batch.append(req)
        active_batch = surviving_batch

        
        while waiting_queue:
            req = waiting_queue.pop(0)
            prefill(req)
            if req.is_finished:
                finished_requests.append(req)
            else:
                active_batch.append(req)

    return finished_requests


def warmer():
    dummy_input = tokenizer(
        "hello",
        return_tensors="pt"
    ).input_ids.to(model.device)

    dummy_req = request(
        req_id="warmer",
        prompt="hello",
        max_tokens=2,
        input_ids=dummy_input
    )

    prefill(dummy_req)
    decode(dummy_req)

    torch.cuda.reset_peak_memory_stats()


print("cold start")
warmer()
print("GPU warm-up complete.")

prompt_a = "tell me about the city of haldia in west bengal,india"
prompt_b = "Is the Earth round? Answer in one word: yes or no."

input_ids_a = tokenizer(
    prompt_a,
    return_tensors="pt"
).input_ids.to(model.device)

input_ids_b = tokenizer(
    prompt_b,
    return_tensors="pt"
).input_ids.to(model.device)

req_a = request(
    req_id="Req-A (Haldia - Long)",
    prompt=prompt_a,
    max_tokens=256,
    input_ids=input_ids_a
)

req_b = request(
    req_id="Req-B (Earth - Short)",
    prompt=prompt_b,
    max_tokens=15,
    input_ids=input_ids_b
)

print("Starting continuous batching...")

torch.cuda.reset_peak_memory_stats()

benchmark_start = time.perf_counter()

completed_requests = run_continuous_batching(
    [req_a, req_b]
)

benchmark_total = time.perf_counter() - benchmark_start
peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

print(f"All requests finished in {benchmark_total:.2f}s\n")

for req in completed_requests:
    output_text = tokenizer.decode(
        req.generated_token_ids,
        skip_special_tokens=True
    )

    total_latency = req.finish_time - req.arrival_time

    print(f"[{req.req_id}]")
    print(f"Tokens Produced: {len(req.generated_token_ids)}")
    print(f"TTFT: {req.ttft:.4f}s")
    print(f"Total Latency: {total_latency:.2f}s")
    print(f"Generated Output:\n{output_text.strip()}\n")

print(f"Peak Allocated VRAM: {peak_vram_mb:.2f} MB")