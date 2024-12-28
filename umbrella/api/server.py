import socket
import threading
from queue import Queue
from .api_utils import send_data, receive_data
from ..logging_config import setup_logger
from ..utils import TextColors
from ..speculation.auto_engine import AutoEngine
logger = setup_logger()

class APIServer:
    def __init__(self, config, device: str = "cuda:0", port: int = 65432, max_client: int = 4, host: str = '127.0.0.1'):
        
        self.port = port
        self.max_client = max_client
        self.host = host
        self.device = device
        self.config = config

    def handle_client(self, conn, addr):
        
        logger.info(TextColors.colorize(f"Connection from {addr}", "cyan"))
        try:
            # Send confirmation message to client upon connection
            confirmation_message = {"status": "connected", "message": "Welcome to the server!"}
            send_data(conn, confirmation_message)

            while True:
                try:
                    received_dict = receive_data(conn)
                    logger.info(TextColors.colorize(f"Received dictionary from {addr}: {received_dict}", "cyan"))
                    terminate = received_dict.get("terminate", False)
                    if terminate:
                        break
                    # Add the received data to the queue for processing
                    self.message_queue.put((addr, conn, received_dict))
                except Exception as e:
                    logger.error(TextColors.colorize(f"Error handling data from {addr}: {e}", "red"))
                    break
        finally:
            conn.close()
            logger.info(TextColors.colorize(f"Connection with {addr} closed", "cyan"))
    
    def process_queue(self):
        while True:
            addr, conn, message = self.message_queue.get()  # Get message from the queue
            with self.queue_lock:  # Ensure only one client is processed at a time
                logger.info(TextColors.colorize(f"Processing message from {addr}: {message}", "cyan"))
                
                output = self.engine.generate(**message)
                processed_data = {**output, "processed": True, "response": "Processed successfully"}

                try:
                    send_data(conn, processed_data)
                except Exception as e:
                    logger.error(TextColors.colorize(f"Error sending data to {addr}: {e}", "red"))
                     
    def run(self):
        
        self.engine = AutoEngine.from_config(self.device, **self.config)
        self.engine.initialize()
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.max_client)
        logger.info(TextColors.colorize("Infini AI LLM server started successfully", "cyan"))
        
        self.message_queue = Queue()
        self.queue_lock = threading.Lock()
        
        threading.Thread(target=self.process_queue, daemon=True).start()

        while True:
            conn, addr = self.server_socket.accept()
            threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()
            
    