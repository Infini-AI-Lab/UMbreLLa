
Prompts = {
    'meta-llama3': """\n<|start_header_id|>user<|end_header_id|>

{}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

""",

# <|begin_of_text|>
# <|start_header_id|>system<|end_header_id|>

# {{ system_prompt }}<|eot_id|>

# <|start_header_id|>user<|end_header_id|>
# {{ user_message_1 }}<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>

# {{ model_response_1 }}<|eot_id|>


    'llama3-code':"""<|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<|python_tag|>""",
   'qwen': """<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
""", 
    # 'mistral-7b': """<s>[INST] {}[/INST] </s>""",
    # 'mistral-7b': """<s>[INST]{}[/INST]</s>""",
    # 'mistral-7b': """<s>[INST]{}[/INST]""",
    # 'mistral-7b': """[INST] {}[/INST] """,
    # 'mistral-7b': """
    # 'mistral-7b': """<s>[INST] {} [/INST]""",
#     'mistral-7b': """<s>[INST] {} [/INST] 
# """
    # 'mistral-7b': """<s>[INST] {} [/INST]  </s>.""",
    'mistral-7b': """<s>[INST] {}[/INST]""",
    # 'mistral-7b': """<s>[INST] {}[/INST]""",
    # 'mistral-7b': """<s>[INST] {} [/INST] """

    # 'mistral-7b': """[INST] {}[/INST]""" ## current best
    # 'mistral-7b': """[INST] {} [/INST] """
    # 'mistral-7b': """{}"""
    # 'mistral-7b': """<s>[INST] {} [/INST] """
    # 'mistral-7b': """<s>[INST] {} [/INST]"""
    # 'mistral-7b': """<s>[INST] {} """
    # 'mistral-7b': """<s>[INST] {}"""
    # 'mistral-7b': """<s>[INST]{}"""
    # 'mistral-7b': """<s>[INST] {}[/INST]"""
    # 'mistral-7b': """[INST] {} [/INST]\n"""
    # 'mistral-7b': """[INST] {} [/INST]"""
    
    # # {} [/INST] """,
    # 'mistral-7b': """{}[/INST]"""
    
    # 'mistral-7b': """<s>[INST]user message[/INST]assistant message</s>[INST]new user message[/INST]""",
    
    # 'mistral-7b': """<s>[INST] {}[/INST] assistant message</s>""",
    
    
}

SysPrompts = {
    'meta-llama3': """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user. <|eot_id|>""",
    'llama3-code': """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython<|eot_id|>""",
    'qwen': """<|im_start|>system
You are a helpful assistant.<|im_end|>
""",
#     'mistral-7b': """[INST] system prompt

# new user message[/INST]""",

    # 'mistral-7b': """[INST] [/INST]""",
    # 'mistral-7b': """<s>[INST]""",
    'mistral-7b': """"""
    # 'mistral-7b': """</s>""" 
    # 'mistral-7b': """</s>\n"""

#     'mistral-7b': """<s>[INST] user message[/INST] assistant message</s>[INST] system prompt

# new user message[/INST]""", 
}

ExtraPrompts = {
    'llama3-code': """\nAlways try to wrap what you write in a function."""
}

