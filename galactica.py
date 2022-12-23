# import galai as gal
# import joblib
import torch
import re 
from tokenizers import Tokenizer 
from transformers import AutoTokenizer, OPTForCausalLM
# from pprint import pretty_print
model_path="/data1/pretrained_weight/Glatica/galactica-6.7b"

#model_path="/data1/pretrained_weight/Glatica/galactica-6.7b"
# tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-125m",cache_dir=model_path)
# model = OPTForCausalLM.from_pretrained("facebook/galactica-125m",cache_dir=model_path, device_map="auto",torch_dtype=torch.float16 ) ## [load_in_8bit=True, torch_dtype=torch.float16

tokenizer = AutoTokenizer.from_pretrained(model_path,)
model = OPTForCausalLM.from_pretrained(model_path, device_map="auto",) ## [load_in_8bit=True, torch_dtype=torch.float16

####------------------ Processing Text Section ------------------####
# literally in the source code in case we ever include it in the training data.
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"

# we split individual characters inside special tokens like [START_DNA]
CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")

def _insert_split_marker(m: re.Match):
    """
    Applies split marker based on a regex match of special tokens such as
    [START_DNA].
    Parameters
    ----------
    n : str
        Input text to split
    Returns
    ----------
    str - the text with the split token added
    """
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"

def escape_custom_split_sequence(text):
    """
    Applies custom splitting to the text for GALILEO's tokenization
    Parameters
    ----------
    text : str
        Input text to split
    Returns
    ----------
    str - the text with the split token added
    """
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)

input_text = "Multi-modality self-supervised learning survey"
input_text = [escape_custom_split_sequence(input_text)]

## Convert input text to token ids 
# new_doc=True
# path="/data1/pretrained_weight/Glatica/galactica-6.7b/tokenizer.json"
# tokenizer_ = Tokenizer.from_file(path)
# tokenizer_.enable_padding(direction="left", pad_id=1, pad_type_id=0, pad_token="[PAD]")
# tokenizer_.enable_truncation(max_length=5000, direction="left")

# if new_doc:
#     pad_id = tokenizer_.padding["pad_id"]
#     pad_token = tokenizer_.id_to_token(pad_id)
#     input_text = [pad_token + t for t in input_text]

# list_encoded = tokenizer_.encode_batch(input_text)
# context_tokens = [encoded.ids for encoded in list_encoded]
# input_ids = torch.LongTensor(context_tokens).to("cuda")

input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")


### -------- Generate Contrastive Search ------------------- 
outputs = model.generate(input_ids, 
                max_length=300, 
                penalty_alpha=0.6,
                top_k=6, 
                do_sample=True,   
                num_return_sequences=10,              
                #return_dict_in_generate=True, 
                output_hidden_states=True,
                 ) # top_p=0.95, num_return_sequences=10


print(tokenizer.decode(outputs[0]))
print("------------------Done Contrastive Search------------------")


### -------- Generate with Top K Neucleus search ------------------- 
outputs = model.generate(input_ids, 
                max_length=300, 
                top_p=0.94,
                do_sample=True,   
                num_return_sequences=10,              
                #return_dict_in_generate=True, 
                output_hidden_states=True,
                 ) # top_p=0.95, num_return_sequences=10


print(tokenizer.decode(outputs[0]))
print("------------------Done Key Nucleus Search------------------")

###---------------  Using Top-K sampling -----------------
outputs = model.generate(
    input_ids,
    max_length= 300,  
    do_sample=True,  
    top_k=25,
    num_return_sequences=10,              
 #return_dict_in_generate=True, 
    output_hidden_states=True,
)
print(tokenizer.decode(outputs[0]))
print("------------------Done K sampling Search------------------")

# for i, beam in enumerate(generated_text_samples):
#   print(f"{i}: {tokenizer.decode(beam, skip_special_tokens=True)}")
#   print()
###---------------  Using Beam Search sampling -----------------

