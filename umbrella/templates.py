
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
'qwq': "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
'ibm-granite': """\n<|start_of_role|>user<|end_of_role|>{}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>"""

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

    'mistral': """<s>[SYSTEM_PROMPT]You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
Your knowledge base was last updated on 2023-10-01. The current date is 2025-03-07.

When you're not sure about some information, you say that you don't have the information and don't make up anything.
If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?")[/SYSTEM_PROMPT]""",
    'qwq': "",
  
    'ibm-granite': """<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
Today's Date: March 05, 2025.
You are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>"""

}

ExtraPrompts = {
    'llama3-code': """\nAlways try to wrap what you write in a function."""
}
