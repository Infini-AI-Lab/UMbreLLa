import pickle
# Helper function to send data in chunks
def send_data(conn, data):
    serialized_data = pickle.dumps(data)
    data_length = len(serialized_data)
    conn.sendall(data_length.to_bytes(4, 'big'))  # Send the length of the data first
    conn.sendall(serialized_data)  # Send the actual data

# Helper function to receive data in chunks
def receive_data(conn):
    data_length = int.from_bytes(conn.recv(4), 'big')  # Receive the length of the data
    data = b''
    while len(data) < data_length:
        chunk = conn.recv(min(1024, data_length - len(data)))
        if not chunk:
            raise ConnectionError("Connection lost while receiving data")
        data += chunk
    return pickle.loads(data)