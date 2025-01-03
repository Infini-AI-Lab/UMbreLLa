
Prompts = {
    'meta-llama3': """\n<|start_header_id|>user<|end_header_id|>

{}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

""",
    'llama3-code':"""<|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<|python_tag|>"""

}

SysPrompts = {
    'meta-llama3': """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user. <|eot_id|>""",
    'llama3-code': """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython<|eot_id|>""",

}

ExtraPrompts = {
    'llama3-code': """\nAlways try to wrap what you write in a function."""
}