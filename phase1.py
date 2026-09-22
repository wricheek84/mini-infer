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
def warmer(model,tokenizer):
    prompt=tokenizer("hello",return_tensors="pt").to(model.device)
    _=model.generate(**prompt,max_new_tokens=3)
    torch.cuda.reset_peak_memory_stats()
print ("fixing cold start")
warmer(model,tokenizer)
print("cold start fixed")
def timed(prompt,max_tokens=100):
    inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
    input_length=inputs.input_ids.shape[-1]
    streamer=TextIteratorStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True)
    start_time=torch.cuda.Event(enable_timing=True)
    end_time=torch.cuda.Event(enable_timing=True)
    container={}
    def worker():
        start_time.record()
        outputs=model.generate(**inputs,max_new_tokens=max_tokens,streamer=streamer)
        end_time.record()
        container["output"]=outputs

    t_submit=time.perf_counter()
    thread=threading.Thread(target=worker)
    thread.start()
    ttft=None
    chunk=[]
    for ch in streamer:
        if ttft is None:
            ttft=time.perf_counter()-t_submit
        chunk.append(ch)
    thread.join()
    torch.cuda.synchronize()
    t_finish=time.perf_counter()
    gpu_time=start_time.elapsed_time(end_time)/1000
    real_time=t_finish-t_submit
    output_length=container["output"].shape[-1]-input_length
    gen_tokens=output_length
    client_thoroughput=gen_tokens/real_time if real_time>0 else 0
    gpu_throughput=gen_tokens/gpu_time if gpu_time>0 else 0

    return {
        "text":"".join(chunk),
        "new_token_count":gen_tokens,
        "gpu_time":gpu_time,
        "real_time":real_time,
        "client_ttft":ttft

    }


