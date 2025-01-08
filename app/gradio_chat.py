import sys
sys.path.append("..")
import gradio as gr
from umbrella.speculation.auto_engine import AutoEngine
from umbrella.utils import TextColors
from umbrella.logging_config import setup_logger
import argparse
import json
from umbrella.templates import Prompts, SysPrompts
parser = argparse.ArgumentParser()
parser.add_argument('--configuration', type=str, default="../configs/chat_config_24gb.json",help='the configuration of the chatbot')
args = parser.parse_args()
logger = setup_logger()

DEVICE = "cuda:0"
with open(args.configuration, "r") as f:
    config = json.load(f)
    

template = config.pop("template", "meta-llama3")
system_prompt = SysPrompts[template]
user_prompt = Prompts[template]
engine = AutoEngine.from_config(DEVICE, **config)
engine.initialize()




def generate_response(prompt, max_new_tokens=100, temperature=0.3, top_p=0.95, repetition_penalty=1.0):
    
    inputs = {"context": prompt, "max_new_tokens":max_new_tokens, "temperature": temperature, "top_p":top_p, "repetition_penalty":repetition_penalty} 
    print(inputs)
    
    output = engine.generate(**inputs)
    log_str = "Avg Accept Tokens {:.2f} | TPOT {:.2f} ms ".format(output["avg_accept_tokens"], output["time_per_output_token"])
    
    return output['generated_text'], log_str

def chat_fn(user_input, history, max_new_tokens, temperature, top_p, repetition_penalty):
    """
    history 是一个 [(user_message, bot_message), (user_message, bot_message), ...] 的列表
    user_input 是当前最新一轮用户输入
    """
    # 将历史对话拼成一个长prompt，或者你也可以用别的方式
    prompt = ""
    prompt = prompt + system_prompt
    for i, (old_user_input, old_bot_output) in enumerate(history):
        prompt += (user_prompt.format(old_user_input) + old_bot_output)
    prompt += user_prompt.format(user_input)

    bot_response, log_str = generate_response(prompt, max_new_tokens, temperature, top_p, repetition_penalty)
    # 返回新的对话历史
    history.append((user_input, bot_response))
    return history, history, log_str, ""

custom_css = """
body {
    background-color: #f0f0f0;
}
.btn-primary {
    background-color: #007BFF;
    border-color: #007BFF;
}
"""
with gr.Blocks(theme="monochrome", title="Infini AI Chatbot", css=custom_css) as demo:
    log_box = gr.Textbox(label="Performance")
    chatbot = gr.Chatbot(label="Infini AI Chatbot")
    msg = gr.Textbox(label="input", placeholder="Input here ...")
    
    with gr.Row():
        max_new_tokens_slider = gr.Slider(
            minimum=32,
            maximum=512,
            value=128,
            step=1,
            label="max_new_tokens",
        )
        temperature_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.6,
            step=0.05,
            label="temperature",
        )
    
    with gr.Row():
        top_p_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.9,
            step=0.05,
            label="top_p",
        )
        
        repetition_penalty_slider = gr.Slider(
            minimum=1.0,
            maximum=2.0,
            value=1.05,
            step=0.05,
            label="repetition_penalty",
        )
        
    clear = gr.Button("clear")
    
    
    state = gr.State([])  

    msg.submit(chat_fn, [msg, state, max_new_tokens_slider, temperature_slider, top_p_slider, repetition_penalty_slider], [chatbot, state, log_box, msg])
    clear.click(lambda: [], None, chatbot, queue=False)
    clear.click(lambda: "", None, log_box, queue=False)
    clear.click(lambda: [], None, state, queue=False)

demo.launch(share=True)
