import torch
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_KEY")
login(token=hf_token)
model_name="meta-llama/Llama-3.2-1B-Instruct"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForCausalLM.from_pretrained(model_name,device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
test_question ="what is the world's strongest top 10 military powers?"
inputs=tokenizer(test_question, return_tensors="pt").to(model.device)
outputs=model.generate(**inputs, max_new_tokens=50)
answer=tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)


