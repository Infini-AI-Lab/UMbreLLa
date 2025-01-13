import gradio as gr
from umbrella.speculation.auto_engine import AutoEngine
from umbrella.logging_config import setup_logger
import argparse
import json
from umbrella.templates import Prompts, SysPrompts

parser = argparse.ArgumentParser()
parser.add_argument('--configuration', type=str, default="../configs/chat_config_24gb.json", help='the configuration of the chatbot')
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


########################
# 1) 新增：真正流式方法 #
########################
def chat_fn_stream(user_input, history, max_new_tokens, temperature, top_p, repetition_penalty):
    """
    调用 engine.generate_stream(...)，多次yield实现真正的流式输出
    """
    # 拼接 prompt
    prompt = system_prompt
    for (old_user_input, old_bot_output) in history:
        prompt += user_prompt.format(old_user_input) + old_bot_output
    prompt += user_prompt.format(user_input)
    
    # 调用 engine.generate_stream
    stream = engine.generate_stream(
        context=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    
    partial_bot_response = ""
    for new_text, perf_log in stream:
        # 假设 generate_stream 逐步返回累积文本
        partial_bot_response = new_text
        
        # 临时对话：当前已完成的 + 这一轮还未完成的回答
        temp_history = history + [(user_input, partial_bot_response)]
        
        # 这里 log_box 先返回空字符串，也可只在最后一次yield里更新
        yield temp_history, temp_history, perf_log, ""
    
    # 结束后，把完整回答加入 history
    history.append((user_input, partial_bot_response))


########################
# 2) 原UI部分不变或少改 #
########################
with gr.Blocks(theme="monochrome", title="Chatbot") as demo:
    model_name_box = gr.Textbox(
        value=config["model"],      # 这里把上面获取到的模型名称放进来
        label="Model",    # 标签
        interactive=False      # 只读展示
    )
    log_box = gr.Textbox(label="Performance")
    chatbot = gr.Chatbot(label="Infini AI Chatbot")
    msg = gr.Textbox(label="Input", placeholder="Input here ...")
    
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

    
    msg.submit(
        fn=chat_fn_stream,
        inputs=[msg, state, max_new_tokens_slider, temperature_slider, top_p_slider, repetition_penalty_slider],
        outputs=[chatbot, state, log_box, msg]
    )

    clear.click(lambda: [], None, chatbot, queue=False)
    clear.click(lambda: "", None, log_box, queue=False)
    clear.click(lambda: [], None, state, queue=False)

demo.launch(share=True)
