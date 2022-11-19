# import galai as gal
# import joblib
import torch
from transformers import AutoTokenizer, OPTForCausalLM
# from pprint import pretty_print
model_path="/data1/pretrained_weight/Glatica/galactica-6.7b"

#model_path="/data1/pretrained_weight/Glatica/galactica-6.7b"
# tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-125m",cache_dir=model_path)
# model = OPTForCausalLM.from_pretrained("facebook/galactica-125m",cache_dir=model_path, device_map="auto",torch_dtype=torch.float16 ) ## [load_in_8bit=True, torch_dtype=torch.float16

tokenizer = AutoTokenizer.from_pretrained(model_path,)
model = OPTForCausalLM.from_pretrained(model_path, device_map="auto",) ## [load_in_8bit=True, torch_dtype=torch.float16
input_text = "Multi-modality self-supervised learning [START_REF]"
#input_text="In this paper, we focus on an approach termed Multi-Head Attention with Feed-Forward Filter [START_REF]"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
### -------- Generate with Greedy search ------------------- 
outputs = model.generate(input_ids, max_length=300, do_sample=True,  top_p=0.95, temperature=0.9, num_return_sequences=10)
print(tokenizer.decode(outputs[0]))

# ###---------------  Using Top-K sampling -----------------
# generated_text_samples = model.generate(
#     input_ids,
#     max_length= 50,  
#     do_sample=True,  
#     top_k=25,
#     num_return_sequences= 5
# )

# for i, beam in enumerate(generated_text_samples):
#   print(f"{i}: {tokenizer.decode(beam, skip_special_tokens=True)}")
#   print()
###---------------  Using Beam Search sampling -----------------

