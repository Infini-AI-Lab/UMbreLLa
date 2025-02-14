
Prompts = {
    'meta-llama3': """\n<|start_header_id|>user<|end_header_id|>

{}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

""",
    'llama3-code':"""<|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<|python_tag|>""",
   'qwen': """<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
""",

'gemma2-it': """<start_of_turn>user
{}<end_of_turn>
<start_of_turn>model
""",

'gemma2': "{}",
'mistral': "[INST] {} [/INST]",
'glm4': """<|user|>
{}<|end|>
<|assistant|>
"""
}

SysPrompts = {
    'meta-llama3': """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user. <|eot_id|>""",
    'llama3-code': """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython<|eot_id|>""",
    'qwen': """<|im_start|>system
You are a helpful assistant.<|im_end|>
""",
    'gemma2': "",
    'gemma2-it': "",
    'mistral': "",
    'glm4': """<|system|>
You are a helpful assistant.
"""

}

ExtraPrompts = {
    'llama3-code': """\nAlways try to wrap what you write in a function."""
}
