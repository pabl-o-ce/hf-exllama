import spaces
import json
import subprocess
import os
import sys
import torch
import gradio as gr

from huggingface_hub import snapshot_download
from jinja2 import Template, Environment, BaseLoader
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler

import flash_attn

model = None
cache = None

snapshot_download(
    repo_id="bartowski/Mistral-7B-Instruct-v0.3-exl2",
    revision="8_0",
    local_dir = "./models/Mistral-7B-instruct-exl2"
)
snapshot_download(
    repo_id="turboderp/Llama-3-70B-Instruct-exl2",
    revision="4.0bpw",
    local_dir = "./models/Llama-3-70B-Instruct-exl2"
)

css = """
.bubble-wrap {
    padding-top: calc(var(--spacing-xl) * 3) !important;
}
.message-row {
    justify-content: space-evenly !important;
    width: 100% !important;
    margin: calc(var(--spacing-xl)) 0 !important;
    padding: 0 calc(var(--spacing-xl) * 3) !important;
}
.message.user {
    border-bottom-right-radius: var(--radius-xl) !important;
}
.message.bot{
    text-align: right;
    width: 100%;
    padding: 10px;
    border-radius: 10px;
}
.message-bubble-border {
    border-radius: 6px !important;
}
.message-buttons {
    justify-content: flex-end !important;
}
.message-buttons-bot, .message-buttons-user {
    right: 10px !important;
    left: auto !important;
    bottom: 2px !important;
}
.dark.message-bubble-border {
    border-color: #343140 !important;
}
.dark.user {
    background: #1e1c26 !important;
}
.dark.assistant.dark, .dark.pending.dark {
    background: #16141c !important;
}
"""

# Jinja2 template for conversation formatting
CHAT_TEMPLATE_MISTAL = """{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] | trim + '\n\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{{ system_message}}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] | trim + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] | trim + eos_token }}{% endif %}{% endfor %}"""
CHAT_TEMPLATE_LLAMA_3 = """{% if messages[0]['role'] == 'system' %}{% set offset = 1 %}{% else %}{% set offset = 0 %}{% endif %}{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n' }}{% endif %}"""

def load_tokenizer_config(model_path):
    config_path = os.path.join(model_path, "tokenizer_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_jinja_environment():
    def raise_exception(message):
        raise ValueError(message)
    
    env = Environment(loader=BaseLoader())
    env.globals['raise_exception'] = raise_exception
    return env

def format_conversation(system_message, history, new_message, tokenizer):
    env = create_jinja_environment()
    template = env.from_string(CHAT_TEMPLATE_MISTAL)

    messages = []
    if system_message:
        messages.append({"role": "system", "content": f'{system_message}'})
    for msn in history:
        messages.append({"role": "user", "content": msn[0]})
        messages.append({"role": "assistant", "content": msn[1]})
    messages.append({"role": "user", "content": new_message})
    print(messages)
    # try:
    conversation = template.render(
        messages=messages, 
        bos_token=tokenizer.get('bos_token'),
        eos_token=tokenizer.get('eos_token'),
        add_generation_prompt=True
    )
    # except ValueError as e:
    #     print(f"Error in template rendering: {str(e)}")
    #     # Fallback to a simple format if template rendering fails
    #     conversation = f"{tokenizer.get('bos_token', '<s>')}" + "".join([f"[INST] {msg['content']} [/INST]" if msg['role'] == 'user' else msg['content'] for msg in messages])
    
    return conversation

@spaces.GPU(duration=120)
def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    top_k,
    repeat_penalty,
):
    global model
    global cache
    model_path = "models/Mistral-7B-instruct-exl2/"
    # Set up the model configuration
    config = ExLlamaV2Config(model_path)
    if model is None or cache is None:
        # Initialize the model with the configuration
        model = ExLlamaV2(config)
        # Create a cache for the model, with lazy initialization
        cache = ExLlamaV2Cache(model, lazy = True)
        # Load the model weights, automatically splitting them if necessary
        model.load_autosplit(cache)
        
    # Initialize the tokenizer with the model configuration
    tokenizer = ExLlamaV2Tokenizer(config)
    # Create a generator for text generation
    generator = ExLlamaV2DynamicGenerator(model, cache, tokenizer)
    # Load the full tokenizer config
    tokenizer_config = load_tokenizer_config(model_path)
    # Initialize an empty context tensor to store the conversation history
    context_ids = torch.empty((1, 0), dtype = torch.long)
     # Format the entire conversation
    conversation = format_conversation(system_message, history, message, tokenizer_config)
    ## Tokenize the conversation
    instruction_ids = tokenizer.encode(conversation, add_bos = True, encode_special_tokens=True)
    context_ids = torch.cat([context_ids, instruction_ids], dim = -1)
    
    print(conversation)
    # Create and enqueue a new generation job
    generator.enqueue(
        ExLlamaV2DynamicJob(
            input_ids=context_ids,
            max_new_tokens=max_tokens,
            gen_settings=ExLlamaV2Sampler.Settings(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                token_repetition_penalty=repeat_penalty,
            ),
            stop_conditions=[tokenizer.eos_token_id],
        )
    )
    
    outputs = ""
    eos = False
    # Generate and stream the response
    while not eos:
        results = generator.iterate()
        for result in results:
            if result["stage"] == "streaming":
                eos = result["eos"]
                if "text" in result:
                    # Print the generated text in real-time
                    outputs += result["text"]
                    yield outputs
                if "token_ids" in result:
                    # Add the generated tokens to the context
                    context_ids = torch.cat([context_ids, result["token_ids"]], dim = -1)

PLACEHOLDER = """
<div class="message-bubble-border" style="display:flex; max-width: 600px; border-radius: 6px; border-width: 1px; border-color: #e5e7eb; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); backdrop-filter: blur(10px);">
    <figure style="margin: 0;max-width: 200px;min-height: 300px;">
        <img src="/gradio_api/file=https://huggingface.co/spaces/pabloce/llama-cpp-agent/resolve/main/llama.jpg" alt="Logo" style="width: 100%; height: 100%; border-radius: 8px;">
    </figure>
    <div style="padding: .5rem 1.5rem; display: flex; flex-direction: column; justify-content: space-evenly;">
        <div>
            <h2 style="text-align: left; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">ExLlama V2</h2>
            <p style="text-align: left; font-size: 16px; line-height: 1.5; margin-bottom: 15px;">ExLlamaV2 is an inference library for running local LLMs on modern consumer GPUs. Supports paged attention via Flash Attention</p>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; flex-flow: column; justify-content: space-between;">
                <span style="display: inline-flex; align-items: center; border-radius: 0.375rem; background-color: rgba(229, 70, 77, 0.1); padding: 0.1rem 0.75rem; font-size: 0.75rem; font-weight: 500; color: #f88181; margin-bottom: 2.5px;">
                    Mistral Instruct 7B v3
                </span>
                <span style="display: inline-flex; align-items: center; border-radius: 0.375rem; background-color: rgba(79, 70, 229, 0.1); padding: 0.1rem 0.75rem; font-size: 0.75rem; font-weight: 500; color: #60a5fa; margin-top: 2.5px;">
                    Meta Llama 3 70B Instruct
                </span>
            </div>
            <div style="display: flex; justify-content: flex-end; align-items: center;">
                <a href="https://discord.gg/gmVgCk6X2x" target="_blank" rel="noreferrer" style="padding: .5rem;">
                    <svg width="24" height="24" fill="currentColor" xmlns="http://www.w3.org/2000/svg" viewBox="0 5 30.67 23.25">
                        <title>Discord</title>
                        <path d="M26.0015 6.9529C24.0021 6.03845 21.8787 5.37198 19.6623 5C19.3833 5.48048 19.0733 6.13144 18.8563 6.64292C16.4989 6.30193 14.1585 6.30193 11.8336 6.64292C11.6166 6.13144 11.2911 5.48048 11.0276 5C8.79575 5.37198 6.67235 6.03845 4.6869 6.9529C0.672601 12.8736 -0.41235 18.6548 0.130124 24.3585C2.79599 26.2959 5.36889 27.4739 7.89682 28.2489C8.51679 27.4119 9.07477 26.5129 9.55525 25.5675C8.64079 25.2265 7.77283 24.808 6.93587 24.312C7.15286 24.1571 7.36986 23.9866 7.57135 23.8161C12.6241 26.1255 18.0969 26.1255 23.0876 23.8161C23.3046 23.9866 23.5061 24.1571 23.7231 24.312C22.8861 24.808 22.0182 25.2265 21.1037 25.5675C21.5842 26.5129 22.1422 27.4119 22.7621 28.2489C25.2885 27.4739 27.8769 26.2959 30.5288 24.3585C31.1952 17.7559 29.4733 12.0212 26.0015 6.9529ZM10.2527 20.8402C8.73376 20.8402 7.49382 19.4608 7.49382 17.7714C7.49382 16.082 8.70276 14.7025 10.2527 14.7025C11.7871 14.7025 13.0425 16.082 13.0115 17.7714C13.0115 19.4608 11.7871 20.8402 10.2527 20.8402ZM20.4373 20.8402C18.9183 20.8402 17.6768 19.4608 17.6768 17.7714C17.6768 16.082 18.8873 14.7025 20.4373 14.7025C21.9717 14.7025 23.2271 16.082 23.1961 17.7714C23.1961 19.4608 21.9872 20.8402 20.4373 20.8402Z"></path>
                    </svg>
                </a>
                <a href="https://github.com/turboderp/exllamav2" target="_blank" rel="noreferrer" style="padding: .5rem;">
                    <svg width="24" height="24" fill="currentColor" viewBox="3 3 18 18">
                        <title>GitHub</title>
                        <path d="M12 3C7.0275 3 3 7.12937 3 12.2276C3 16.3109 5.57625 19.7597 9.15374 20.9824C9.60374 21.0631 9.77249 20.7863 9.77249 20.5441C9.77249 20.3249 9.76125 19.5982 9.76125 18.8254C7.5 19.2522 6.915 18.2602 6.735 17.7412C6.63375 17.4759 6.19499 16.6569 5.8125 16.4378C5.4975 16.2647 5.0475 15.838 5.80124 15.8264C6.51 15.8149 7.01625 16.4954 7.18499 16.7723C7.99499 18.1679 9.28875 17.7758 9.80625 17.5335C9.885 16.9337 10.1212 16.53 10.38 16.2993C8.3775 16.0687 6.285 15.2728 6.285 11.7432C6.285 10.7397 6.63375 9.9092 7.20749 9.26326C7.1175 9.03257 6.8025 8.08674 7.2975 6.81794C7.2975 6.81794 8.05125 6.57571 9.77249 7.76377C10.4925 7.55615 11.2575 7.45234 12.0225 7.45234C12.7875 7.45234 13.5525 7.55615 14.2725 7.76377C15.9937 6.56418 16.7475 6.81794 16.7475 6.81794C17.2424 8.08674 16.9275 9.03257 16.8375 9.26326C17.4113 9.9092 17.76 10.7281 17.76 11.7432C17.76 15.2843 15.6563 16.0687 13.6537 16.2993C13.98 16.5877 14.2613 17.1414 14.2613 18.0065C14.2613 19.2407 14.25 20.2326 14.25 20.5441C14.25 20.7863 14.4188 21.0746 14.8688 20.9824C16.6554 20.364 18.2079 19.1866 19.3078 17.6162C20.4077 16.0457 20.9995 14.1611 21 12.2276C21 7.12937 16.9725 3 12 3Z"></path>
                    </svg>
                </a>
            </div>
        </div>
    </div>
</div>
"""

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a helpful assistant.", label="System message"),
        gr.Slider(minimum=1, maximum=4096, value=2048, step=1, label="Max tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p",
        ),
        gr.Slider(
            minimum=0,
            maximum=100,
            value=40,
            step=1,
            label="Top-k",
        ),
        gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=1.1,
            step=0.1,
            label="Repetition penalty",
        ),
    ],
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="violet", neutral_hue="gray",font=[gr.themes.GoogleFont("Exo"), "ui-sans-serif", "system-ui", "sans-serif"]).set(
        body_background_fill_dark="#16141c",
        block_background_fill_dark="#16141c",
        block_border_width="1px",
        block_title_background_fill_dark="#1e1c26",
        input_background_fill_dark="#292733",
        button_secondary_background_fill_dark="#24212b",
        border_color_accent_dark="#343140",
        border_color_primary_dark="#343140",
        background_fill_secondary_dark="#16141c",
        color_accent_soft_dark="transparent",
        code_background_fill_dark="#292733",
    ),
    css=css,
    # retry_btn="Retry",
    # undo_btn="Undo",
    # clear_btn="Clear",
    # submit_btn="Send",
    description="Exllama: Chat with exl2 [repo](https://github.com/pabl-o-ce/hf-exllama)",
    chatbot=gr.Chatbot(
        scale=1, 
        placeholder=PLACEHOLDER,
        # likeable=False,
        show_copy_button=True
    )
)

if __name__ == "__main__":
    demo.launch(allowed_paths=["https://huggingface.co/spaces/pabloce/llama-cpp-agent/resolve/main/llama.jpg"])