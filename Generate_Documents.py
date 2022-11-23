# import galai as gal
# import joblib
import torch
import gradio as gr
import re 
from tokenizers import Tokenizer 
from transformers import pipeline , StoppingCriteriaList, MaxLengthCriteria
from transformers import AutoTokenizer, OPTForCausalLM, AutoModelForCausalLM
from functools import lru_cache
import time 

## -----------------  Galactica ----------------- ##
## 1. Load the model and tokenizer
#model_path="/data1/pretrained_weight/Glatica/galactica-6.7b"
# tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-125m",cache_dir=model_path)
# model = OPTForCausalLM.from_pretrained("facebook/galactica-125m",cache_dir=model_path, device_map="auto",torch_dtype=torch.float16 ) ## [load_in_8bit=True, torch_dtype=torch.float16
#model = AutoModelForCausalLM.from_pretrained(model_path,) ## [load_in_8bit=True, torch_dtype=torch.float16 


##------------------ Processing Text Helper Function ------------------####

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


###--------------- Inference Functions  -----------------
model_size={
    "mmini-125M": "facebook/galactica-125m",
    "base-1.3B": "facebook/galactica-1.3b",
    "standard-6.7B": "facebook/galactica-6.7b",
    "large-30B": "facebook/galactica-30b",
    "huge-120B": "facebook/galactica-120b",
}

@lru_cache(maxsize=1)  # only cache the latest model
def get_model_and_tokenizer(model_id):
    if "opt"  in model_id:
        cache_dir= "/data1/pretrained_weight/OPT/"
    elif "galactica" in model_id:
        cache_dir= "/data1/pretrained_weight/Glatica/"

    model = OPTForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, device_map="auto",) ## [load_in_8bit=True, torch_dtype=torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=cache_dir)
    

    return model, tokenizer

@lru_cache(maxsize=32768)  # cache up to 32k examples
def run_generation(
    text,
    model_id_,
    max_new_tokens,
    alpha=0.0,
    temperature=0.0,
    top_k=0,
    num_beams=1,
    do_sample=False,
    top_p=0.0,
    seed=0,
    new_doc=False
):
    model_id=model_size[model_id_]
    model, tokenizer = get_model_and_tokenizer(model_id)
    text= text + str("\n\n")
    
    
    
    input_texts = [escape_custom_split_sequence(text)]
    

    if new_doc:
        if "125m" in model_id:
            path="/data1/pretrained_weight/Glatica/models--facebook--galactica-125m/snapshots/f1d2d54b29cd566df7ccb5325706a39f0f2e794c/tokenizer.json"
        elif "1.3b" in model_id:
            path="/data1/pretrained_weight/Glatica/models--facebook--galactica-1.3b/snapshots/6055184ff908fcfdb53034c098160ed1b185ba1e/tokenizer.json"
        elif "6.7b" in model_id:
            path="/data1/pretrained_weight/Glatica/models--facebook--galactica-6.7b/snapshots/4537aa5d374675cc26dc52219f070a589e4574e4/tokenizer.json"
        elif "30b" in model_id:
            path= "/data1/pretrained_weight/Glatica/models--facebook--galactica-3Ob/snapshots/508480d2cab1ac3a320067a6d25dd6552e118809/tokenizer.json"
            #path="/data1/pretrained_weight/OPT/opt-tokenizer.json"
        elif "120b" in model_id:
            raise ValueError("120b model ONLY Support on VM .203")
            #path="/data1/pretrained_weight/OPT/opt-6.7b-tokenizer.json"
        tokenizer_ = Tokenizer.from_file(path)
        tokenizer_.enable_padding(direction="left", pad_id=1, pad_type_id=0, pad_token="[PAD]")
        tokenizer_.enable_truncation(max_length=5000, direction="left")
        # if new_doc:
        pad_id = tokenizer_.padding["pad_id"]
        pad_token = tokenizer_.id_to_token(pad_id)
        input_text = [pad_token + t for t in input_text]
        list_encoded = tokenizer_.encode_batch(input_text)
        context_tokens = [encoded.ids for encoded in list_encoded]
        input_ids = torch.LongTensor(context_tokens).to("cuda")

    else: 

        input_ids = tokenizer(input_texts, return_tensors='pt').input_ids.to("cuda")
    
    
    if seed:
        torch.manual_seed(seed)

    start = time.time_ns()
    contrastive_ids = model.generate(
        # from the tokenizer
        input_ids,
        # fixed arguments
        #num_return_sequences=1,
        early_stopping=True,
        # variable arguments
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
        penalty_alpha=alpha or None,
        top_k=top_k or None,
        temperature=temperature or None,
        top_p=top_p or None,
    )
    end = time.time_ns()

    contrastive_time = (end - start) / 1e6
    contrastive_text = tokenizer.decode(contrastive_ids[0], skip_special_tokens=True)
    return contrastive_text, contrastive_time


def generate_contrastive_beam_search(text, model_id, max_new_tokens, alpha, k, num_beams):
    contrastive_text, contrastive_time = run_generation(text, model_id, max_new_tokens, alpha=alpha, top_k=k)
    beam_search_text, beam_search_time = run_generation(text, model_id, max_new_tokens, num_beams=num_beams)
    return contrastive_text, beam_search_text


def generate_sampling_search(text, model_id, max_new_tokens, top_p, top_k,temperature, seed):
    nucleus_text, nucleus_time = run_generation(text, model_id, max_new_tokens, top_p=top_p, seed=seed, do_sample=True)
    top_k_text, top_k_time = run_generation(text, model_id, max_new_tokens, top_k=top_k, seed=seed, do_sample=True)
    return nucleus_text, top_k_text


##------------------ Gradio App ------------------####
demo = gr.Blocks()

with demo:
    with gr.Tabs():
        with gr.TabItem("📄 Generate Documents 1:"):
            with gr.Row():
                with gr.Column():
                   
                    gr.Markdown("## Inputs ✍️")
                    gr.Markdown("General options:")
                    
                    with gr.Box():
                        with gr.Row(mobile_collapse=False, equal_height=True):
                            with gr.Column(scale=1, min_width=100, min_height=100):
                                #model_id = gr.Text(value="facebook/galactica-6.7b", label="Model Repository")
                                model_id = gr.Dropdown( ["mmini-125M","base-1.3B", "standard-6.7B","large-30B", "huge-120B"], value="standard-6.7B", label="Model Size")
                            with gr.Column(scale=1, min_width=100, min_height=600):
                                new_doc= gr.Checkbox(value=False, label="Generateion New Document")
                            with gr.Column(scale=1, min_width=100, min_height=100):
                                token_output = gr.Slider(value=400, minimum=100, maximum=3000,step=100, label="Maximum length")
                    
                    with gr.Row():
                        input_text = gr.Textbox(value="Self-Supervised learning is", lines=5, label="Input Text").style(height=600)
                    
                    gr.Markdown("Contrastive Search options:")
                    with gr.Row(mobile_collapse=False, equal_height=True):
                        
                        with gr.Column(scale=1, min_width=100, min_height=100):
                            alpha = gr.Slider(value=0.6, minimum=0.01, maximum=1.0, step=0.01, label="Alpha")
                        with gr.Column(scale=1, min_width=100, min_height=100):
                            k = gr.Slider(value=6, minimum=1, maximum=20, step=1, label="K")
                    
                    gr.Markdown("Beam Search options:")
                    with gr.Row(mobile_collapse=False, equal_height=True):
                       
                        num_beams = gr.Slider(value=4, minimum=1, maximum=16, step=1, label="Number of beams")
                    generate_button = gr.Button(value="Generate", label="Generate")

                with gr.Column():
                    gr.Markdown("## Outputs 🤖 Contrastive and Beam Search")
                    gr.Markdown("Contrastive Search generation:")
                    text_contrastive = gr.Textbox(value="", label="")
                    #time_contrastive = gr.Number(value=0.0, precision=1, label="Generation time (ms)")
                    gr.Markdown("Beam Search generation:")
                    text_beam_search = gr.Textbox(value="", label="")
                    #time_beam_search = gr.Number(value=0.0, precision=1, label="Generation time (ms)")

            # actions
            generate_button.click(
                fn=generate_contrastive_beam_search,
                inputs=[input_text, model_id, token_output, alpha, k, num_beams],
                outputs=[text_contrastive, text_beam_search]
            )

        with gr.TabItem("📄 Generate Documents 2:"):
       
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Inputs ✍️")
                    gr.Markdown("General options:")
                    #model_id = gr.Text(value="facebook/galactica-6.7b", label="Model Repository")
                    with gr.Box():
                        with gr.Row(mobile_collapse=False, equal_height=True):
                            with gr.Column(scale=1, min_width=100, min_height=100):
                                #model_id = gr.Text(value="facebook/galactica-6.7b", label="Model Repository")
                                model_id = gr.Dropdown( ["mmini-125M","base-1.3B", "standard-6.7B","large-30B", "huge-120B"], value="standard-6.7B", label="Model Size")
                            with gr.Column(scale=1, min_width=100, min_height=600):
                                new_doc= gr.Checkbox(value=False, label="Generateion New Document")
                            with gr.Column(scale=1, min_width=100, min_height=100):
                                token_output = gr.Slider(value=400, minimum=100, maximum=3000,step=100, label="Maximum length")
                    with gr.Row():
                        input_text = gr.Textbox(value="Self-Supervised learning is", lines=5, label="Input Text").style(height=600)
                    
                    gr.Markdown("Neucleus Search options:")
                    with gr.Row(mobile_collapse=False, equal_height=True):
                        with gr.Column(scale=1, min_width=100, min_height=600):
                            seed = gr.Slider(value=42, minimum=10, maximum=200,step=10, label="New tokens to generate")
                        with gr.Column(scale=1, min_width=100, min_height=600):
                            top_p = gr.Slider(value=0.95, minimum=0.01, maximum=1.0, step=0.01, label="Top P")
                        #seed = gr.Number(value=42, precision=0, label="Seed")
                    
                    gr.Markdown("Top K Sampling options:")
                    with gr.Row(mobile_collapse=False, equal_height=True):
                        with gr.Column(scale=1, min_width=100, min_height=600):
                            temperature = gr.Slider(value=0.7, minimum=0.1, maximum=1, step=0.05, label="temperature")
                        with gr.Column(scale=1, min_width=100, min_height=600):
                            top_k = gr.Slider(value=50, minimum=1, maximum=100, step=10, label="Top K")
                    #seed = gr.Number(value=42, precision=0, label="Seed")
                    generate_button = gr.Button(value="Generate", label="Generate")

                with gr.Column():
                    gr.Markdown("## Outputs 🤖  Top K Sampling & Nucleus Sampling: ")
                    gr.Markdown("Nucleus generation:")
                    text_contrastive = gr.Textbox(value="", label="")
                    #time_contrastive = gr.Number(value=0.0, precision=1, label="Generation time (ms)")
                    gr.Markdown("Top-K Sampling generation:")
                    text_top_k = gr.Textbox(value="", label="")
                    #time_top_k = gr.Number(value=0.0, precision=1, label="Generation time (ms)")

            # actions
            generate_button.click(
                fn=generate_sampling_search,
                inputs=[input_text, model_id, token_output, top_p, top_k, temperature, seed],
                outputs=[text_contrastive, text_top_k]
            )


demo.launch()
