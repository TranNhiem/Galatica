import gradio as gr
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import time 

# cache_dir= "/data1/pretrained_weight/Glatica/"

def get_model_tokenizer(model_id):
    if "opt"  in model_id:
        cache_dir= "/data1/pretrained_weight/OPT/"
    elif "galactica" in model_id:
        cache_dir= "/data1/pretrained_weight/Glatica/"

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)
    text2text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, num_workers=10)

    return text2text_generator, tokenizer


def run_generation(text, model_id, 
                        max_new_tokens=100, 
                        temperature=0.0, 
                        do_sample=False, 
                        alpha=0.0,
                        top_k=0,
                        num_beams=1,
                        top_p=0.0,
                        seed=0 ):

    text2text_generator, tokenizer=get_model_tokenizer(model_id)
    text = text.strip()
    start = time.time_ns()
    out_text = text2text_generator(
                            text, max_length=max_new_tokens, 
                            temperature=temperature, 
                            do_sample=do_sample,
                            eos_token_id = tokenizer.eos_token_id,
                            bos_token_id = tokenizer.bos_token_id,
                            pad_token_id = tokenizer.pad_token_id,
                            ## Fixed Arugment
                            early_stopping=True,
                            num_return_sequences=1,
                            num_beams=num_beams,
                            penalty_alpha=alpha or None,
                            top_k=top_k or None,
                            top_p=top_p or None,
                         )[0]['generated_text']
    end = time.time_ns()
    contrastive_time = (end - start) / 1e6
    out_text = "<p>" + out_text + "</p>"
    out_text = out_text.replace(text, text + "<b><span style='background-color: #ffffcc;'>")
    out_text = out_text +  "</span></b>"
    out_text = out_text.replace("\n", "<br>")

    return out_text, contrastive_time


def generate_beam_search(text, model_id, max_new_tokens, alpha, k, num_beams):
    contrastive_text, contrastive_time = run_generation(text, model_id, max_new_tokens, alpha=alpha, top_k=k)
    beam_search_text, beam_search_time = run_generation(text, model_id, max_new_tokens, num_beams=num_beams)
    return contrastive_text, contrastive_time, beam_search_text, beam_search_time


def generate_top_k(text, model_id, max_new_tokens, alpha, k, top_k,temperature, seed):
    contrastive_text, contrastive_time = run_generation(text, model_id, max_new_tokens, alpha=alpha, top_k=k)
    top_k_text, top_k_time = run_generation(
        text, model_id, max_new_tokens, top_k=top_k, seed=seed, do_sample=True, temperature=temperature
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
    with gr.Tabs():
        with gr.TabItem("vs. Beam Search"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Inputs ✍️")
                    gr.Markdown("General options:")
                    model_id = gr.Text(value="facebook/galactica-6.7b", label="Model Repository")
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
                    model_id = gr.Text(value="facebook/galactica-6.7b", label="Model Repository")
                    input_text = gr.Textbox(value="DeepMind Company is", lines=5, label="Input Text")
                    max_new_tokens = gr.Slider(value=50, minimum=1, maximum=256, label="New tokens to generate")
                    gr.Markdown("Contrastive Search options:")
                    alpha = gr.Slider(value=0.6, minimum=0.01, maximum=1.0, step=0.01, label="Alpha")
                    k = gr.Slider(value=6, minimum=1, maximum=20, step=1, label="K")
                    gr.Markdown("Sampling options:")
                    temperature = gr.Slider(value=0.7, minimum=0,1, maximum=1, step=1, label="K")
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
                inputs=[input_text, model_id, max_new_tokens, alpha, k, top_k, temperature, seed],
                outputs=[text_contrastive, time_contrastive, text_top_k, time_top_k]
            )

        with gr.TabItem("vs. Nucleus Sampling"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Inputs ✍️")
                    gr.Markdown("General options:")
                    model_id = gr.Text(value="facebook/galactica-6.7b", label="Model Repository")
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
