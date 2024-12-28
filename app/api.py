import sys
sys.path.append("..")
from umbrella.api.server import APIServer
from umbrella.api.client import APIClient
from umbrella.templates import Prompts, SysPrompts
from umbrella.logging_config import setup_logger
from umbrella.utils import TextColors
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--configuration', type=str, default="../configs/chat_config_24gb.json",help='the configuration of the chatbot')
parser.add_argument('--port', type=int, default=65432,help='port')
parser.add_argument('--max_client', type=int, default=1,help='max clients')
parser.add_argument('--server', action='store_true', help="start a server; otherwise client")
args = parser.parse_args()
logger = setup_logger()

# Server code
def server_program(port, max_client, config):
    server = APIServer(config=config, port=port, max_client=max_client)
    server.run()

# Client code
def client_program(port, config):
    
    text1 = "Tell me what you know about Reinforcement Learning in 100 words."
    text2 = "Tell me what you know about LSH in 100 words."
    
    template = config.get("template", "meta-llama3")
    system_prompt = SysPrompts[template]
    user_prompt = Prompts[template]
    
    text1 = user_prompt.format(text1)
    text1 = system_prompt + text1
    
    text2 = user_prompt.format(text2)
    text2 = system_prompt + text2
    
    
    
    client = APIClient(port=port)
    client.run()
    
    input1 = {"context": text1, "max_new_tokens": 512}
    input2 = {"context": text2, "max_new_tokens": 512}
    
    output1 = client.get_output(**input1)
    
    logger.info(TextColors.colorize(output1['generated_text'], "cyan"))

    output2 = client.get_output(**input2)
    logger.info(TextColors.colorize(output2['generated_text'], "cyan"))
    
    client.close()


if __name__ == "__main__":
    
    with open(args.configuration, "r") as f:
        config = json.load(f)
    if args.server:
        server_program(port=args.port, max_client=args.max_client, config=config)
    else:
        client_program(port=args.port, config=config)
