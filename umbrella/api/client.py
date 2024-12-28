import socket
from .api_utils import send_data, receive_data
from ..logging_config import setup_logger
from ..utils import TextColors
import time
logger = setup_logger()

class APIClient:
    def __init__(self, port: int, host: str = '127.0.0.1'):
        
        self.port = port
        self.host = host
        
    def run(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.client_socket.connect((self.host, self.port))
                break
            except ConnectionRefusedError:
                logger.info(TextColors.colorize("Server is not available, retrying in 5 seconds...", "red"))
                time.sleep(5)  # Wait before retrying
        
        confirmation_message = receive_data(self.client_socket)
        logger.info(TextColors.colorize(f"Server confirmation: {confirmation_message}", "cyan"))
    
    def get_output(self, **api_args):
        
        send_data(self.client_socket, api_args)
        response_dict = receive_data(self.client_socket)
        return response_dict
    
    
    def close(self):
        send_data(self.client_socket,{"terminate": True})
        self.client_socket.close()