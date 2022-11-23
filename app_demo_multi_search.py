import time
from functools import lru_cache

import torch
import gradio as gr
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 

model_path="/data1/pretrained_weight/Glatica/galactica-6.7b"

@lru_cache(maxsize=1)  # only cache the latest model
def get_model_and_tokenizer(model_id):
    if "opt"  in model_id:
        cache_dir= "/data1/pretrained_weight/OPT/"
    elif "galactica" in model_id:
        cache_dir= "/data1/pretrained_weight/Glatica/"

    config = AutoConfig.from_pretrained(model_id)
    if config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=cache_dir )
    breakpoint()
    return model, tokenizer


@lru_cache(maxsize=32768)  # cache up to 32k examples
def run_generation(
    text,
    model_id,
    max_new_tokens,
    alpha=0.0,
    top_k=0,
    num_beams=1,
    do_sample=False,
    top_p=0.0,
    seed=0
):
    model, tokenizer = get_model_and_tokenizer(model_id)

    inputs = tokenizer(text, return_tensors='pt')
    if seed:
        torch.manual_seed(seed)

    start = time.time_ns()
    contrastive_ids = model.generate(
        # from the tokenizer
        **inputs,
        # fixed arguments
        num_return_sequences=1,
        early_stopping=True,
        # variable arguments
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
        penalty_alpha=alpha or None,
        top_k=top_k or None,
        top_p=top_p or None,
    )
    end = time.time_ns()

    contrastive_time = (end - start) / 1e6
    contrastive_text = tokenizer.decode(contrastive_ids[0], skip_special_tokens=True)
    return contrastive_text, contrastive_time


def generate_beam_search(text, model_id, max_new_tokens, alpha, k, num_beams):
    contrastive_text, contrastive_time = run_generation(text, model_id, max_new_tokens, alpha=alpha, top_k=k)
    beam_search_text, beam_search_time = run_generation(text, model_id, max_new_tokens, num_beams=num_beams)
    return contrastive_text, contrastive_time, beam_search_text, beam_search_time


def generate_top_k(text, model_id, max_new_tokens, alpha, k, top_k, seed):
    contrastive_text, contrastive_time = run_generation(text, model_id, max_new_tokens, alpha=alpha, top_k=k)
    top_k_text, top_k_time = run_generation(
        text, model_id, max_new_tokens, top_k=top_k, seed=seed, do_sample=True
    )
    return contrastive_text, contrastive_time, top_k_text, top_k_time


def generate_top_p(text, model_id, max_new_tokens, alpha, k, top_p, seed):
    contrastive_text, contrastive_time = run_generation(text, model_id, max_new_tokens, alpha=alpha, top_k=k)
    top_p_text, top_p_time = run_generation(
        text, model_id, max_new_tokens, top_p=top_p, seed=seed, do_sample=True
    )
    return contrastive_text, contrastive_time, top_p_text, top_p_time


demo = gr.Blocks()

with demo:
    gr.Markdown(
        """
        # Contrastive Search Generation comparison
        Credits to the contrastive search generation [paper](https://arxiv.org/abs/2202.06417) authors, including
        @[pangpang666](https://huggingface.co/pangpang666) and @[GMFTBY](https://huggingface.co/GMFTBY). Check out the
        follow-up [work](https://arxiv.org/abs/2210.14140), which demonstrates the usefulness of the technique with
        off-the-shelf LLMs, as well as their [HF guest blog post](https://huggingface.co/blog/introducing-csearch).
        From the paper:
        "At each decoding step, the key ideas of contrastive search are (i) the generated output should be selected
        from the set of most probable candidates predicted by the model; and (ii) the generated output should be
        discriminative enough with respect to the previous context. In this way, the generated text can (i) better
        maintain the semantic coherence with respect to the prefix while (ii) avoiding model degeneration."
        🚨 Warnings: 🚨
        - Avoid using large models (> 1GB) in this demo. It will take a long time to load the model and generate text.
        - Too slow/long queue? Check our
        [colab](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/115_introducing_contrastive_search.ipynb)
        instead.
        """
    )
    with gr.Tabs():
        with gr.TabItem("vs. Beam Search"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Inputs ✍️")
                    gr.Markdown("General options:")
                    model_id = gr.Text(value="facebook/galactica-125m", label="Model Repository")
                    input_text = gr.Textbox(value="DeepMind Company is", lines=5, label="Input Text")
                    max_new_tokens = gr.Slider(value=50, minimum=1, maximum=256, label="New tokens to generate")
                    gr.Markdown("Contrastive Search options:")
                    alpha = gr.Slider(value=0.6, minimum=0.01, maximum=1.0, step=0.01, label="Alpha")
                    k = gr.Slider(value=6, minimum=1, maximum=20, step=1, label="K")
                    gr.Markdown("Beam Search options:")
                    num_beams = gr.Slider(value=4, minimum=1, maximum=16, step=1, label="Number of beams")
                    generate_button = gr.Button(value="Generate", label="Generate")

                with gr.Column():
                    gr.Markdown("## Outputs 🤖")
                    gr.Markdown("Contrastive Search generation:")
                    text_contrastive = gr.Textbox(value="", label="")
                    time_contrastive = gr.Number(value=0.0, precision=1, label="Generation time (ms)")
                    gr.Markdown("Beam Search generation:")
                    text_beam_search = gr.Textbox(value="", label="")
                    time_beam_search = gr.Number(value=0.0, precision=1, label="Generation time (ms)")

            # actions
            generate_button.click(
                fn=generate_beam_search,
                inputs=[input_text, model_id, max_new_tokens, alpha, k, num_beams],
                outputs=[text_contrastive, time_contrastive, text_beam_search, time_beam_search]
            )

        with gr.TabItem("vs. Top K Sampling"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Inputs ✍️")
                    gr.Markdown("General options:")
                    model_id = gr.Text(value="facebook/opt-125m", label="Model Repository")
                    input_text = gr.Textbox(value="DeepMind Company is", lines=5, label="Input Text")
                    max_new_tokens = gr.Slider(value=50, minimum=1, maximum=256, label="New tokens to generate")
                    gr.Markdown("Contrastive Search options:")
                    alpha = gr.Slider(value=0.6, minimum=0.01, maximum=1.0, step=0.01, label="Alpha")
                    k = gr.Slider(value=6, minimum=1, maximum=20, step=1, label="K")
                    gr.Markdown("Sampling options:")
                    top_k = gr.Slider(value=50, minimum=1, maximum=100, step=1, label="Top K")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    generate_button = gr.Button(value="Generate", label="Generate")

                with gr.Column():
                    gr.Markdown("## Outputs 🤖")
                    gr.Markdown("Contrastive Search generation:")
                    text_contrastive = gr.Textbox(value="", label="")
                    time_contrastive = gr.Number(value=0.0, precision=1, label="Generation time (ms)")
                    gr.Markdown("Top K Sampling generation:")
                    text_top_k = gr.Textbox(value="", label="")
                    time_top_k = gr.Number(value=0.0, precision=1, label="Generation time (ms)")

            # actions
            generate_button.click(
                fn=generate_top_k,
                inputs=[input_text, model_id, max_new_tokens, alpha, k, top_k, seed],
                outputs=[text_contrastive, time_contrastive, text_top_k, time_top_k]
            )

        with gr.TabItem("vs. Nucleus Sampling"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Inputs ✍️")
                    gr.Markdown("General options:")
                    model_id = gr.Text(value="facebook/opt-125m", label="Model Repository")
                    input_text = gr.Textbox(value="DeepMind Company is", lines=5, label="Input Text")
                    max_new_tokens = gr.Slider(value=50, minimum=1, maximum=256, label="New tokens to generate")
                    gr.Markdown("Contrastive Search options:")
                    alpha = gr.Slider(value=0.6, minimum=0.01, maximum=1.0, step=0.01, label="Alpha")
                    k = gr.Slider(value=6, minimum=1, maximum=20, step=1, label="K")
                    gr.Markdown("Sampling options:")
                    top_p = gr.Slider(value=0.95, minimum=0.01, maximum=1.0, step=0.01, label="Top P")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    generate_button = gr.Button(value="Generate", label="Generate")

                with gr.Column():
                    gr.Markdown("## Outputs 🤖")
                    gr.Markdown("Contrastive Search generation:")
                    text_contrastive = gr.Textbox(value="", label="")
                    time_contrastive = gr.Number(value=0.0, precision=1, label="Generation time (ms)")
                    gr.Markdown("Nucleus Sampling generation:")
                    text_top_p = gr.Textbox(value="", label="")
                    time_top_p = gr.Number(value=0.0, precision=1, label="Generation time (ms)")

            # actions
            generate_button.click(
                fn=generate_top_p,
                inputs=[input_text, model_id, max_new_tokens, alpha, k, top_p, seed],
                outputs=[text_contrastive, time_contrastive, text_top_p, time_top_p]
            )

demo.launch()