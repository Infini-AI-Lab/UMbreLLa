import sys
sys.path.append("..")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset
from umbrella.sequoia_utils import measure_acceptance_rate
from umbrella.templates import SysPrompts, Prompts

system_prompt = SysPrompts['llama3-code']
prompt = Prompts['llama3-code']
data = load_dataset("openai/openai_humaneval")
processed_data = []
for d in data['test']:
    d = system_prompt + prompt.format("Help me complete the code.") + d['prompt'] + d['canonical_solution']
    processed_data.append(d)

