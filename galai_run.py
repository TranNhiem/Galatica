# import Galatica.galai_run as gal
import galai as gal

model = gal.load_model("large")
output=model.generate("multi-modality self-supervised learning survey", new_doc=True, top_p=0.7, max_length=200)
print(output)
# from transformers import AutoTokenizer, OPTForCausalLM
# import torch
# # from pprint import pretty_print
# model_path="/data1/pretrained_weight/Glactica30b/galactica-30b/"

# #model_path="/data1/pretrained_weight/Glatica/galactica-6.7b"
# tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b",cache_dir=model_path)
# model = OPTForCausalLM.from_pretrained("facebook/galactica-30b",cache_dir=model_path, device_map="auto",torch_dtype=torch.float16 ) ## [load_in_8bit=True, torch_dtype=torch.float16
