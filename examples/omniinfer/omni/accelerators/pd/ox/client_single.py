import random
import threading
import time
import uuid
from collections import deque
from typing import Dict, List, Optional

import msgpack
import numpy as np
import zmq


class RouterDealerClient:
    def __init__(self, server_address="tcp://localhost:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        
        client_id = f"client_{uuid.uuid4().hex[:8]}".encode('utf-8')
        self.socket.setsockopt(zmq.IDENTITY, client_id)
        
        self.socket.connect(server_address)
        print(f"Connected to server at {server_address} with ID: {client_id.decode()}")

    def send_request(self, request_id: str, id_list: List[int]) -> bool:
        try:
            request_data = {
                'request_id': request_id,
                'table_id': 0,
                'block_ids': id_list
            }
            packed_data = msgpack.packb(request_data)
            self.socket.send(packed_data)
            # print(f"Sent request: {request_id} with {len(id_list)} IDs")
            return True
        except Exception as e:
            print(f"Error sending request {request_id}: {e}")
            return False

    def receive_response(self, timeout: int = 1000) -> Optional[Dict]:
        try:
            if self.socket.poll(timeout, zmq.POLLIN):
                response_data = self.socket.recv()
                response = msgpack.unpackb(response_data)
                return response
        except Exception as e:
            print(f"Error receiving response: {e}")
        return None

    def close(self):
        self.socket.close()
        self.context.term()
        print("Client closed")


def main():
    SERVER_ADDRESS = "tcp://localhost:5555"    
    client = RouterDealerClient(SERVER_ADDRESS)
        
    client.send_request("My-test", [0, 5, 10])
    response = client.receive_response(1000000)
    print(response["success"])
       

if __name__ == "__main__":
    main()