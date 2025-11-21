import random
import threading
import time
import uuid
from collections import deque
from typing import Dict, List, Optional

import msgpack
import zmq

MAX_NUM_BLOCK = 1024
SERVER_ADDRESS = "tcp://localhost:5555"
MAX_CONCURRENT = 32  # max concurrency
MEAN_LENGTH = 30  # average block id list length
STD_DEV = 2.0  # standard deviation
MONITOR_INTERVAL = 2.0


class RouterDealerClient:

    def __init__(self, server_address="tcp://localhost:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)

        client_id = f"client_{uuid.uuid4().hex[:8]}".encode('utf-8')
        self.socket.setsockopt(zmq.IDENTITY, client_id)

        self.socket.connect(server_address)
        print(
            f"Connected to server at {server_address} with ID: {client_id.decode()}"
        )

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


class MessageManager:

    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.pending_requests = {}
        self.response_queue = deque()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.active_requests = 0

        self.sent_count = 0
        self.sent_block_count = 0
        self.received_count = 0
        self.received_block_count = 0
        self.start_time = time.time()

    def can_send(self) -> bool:
        return self.active_requests < self.max_concurrent

    def add_request(self, request_id: str, id_list: List[int]) -> bool:
        with self.condition:
            if not self.can_send():
                return False

            self.pending_requests[request_id] = id_list
            self.active_requests += 1
            self.sent_block_count += len(id_list)
            self.sent_count += 1
            # print(f"Added request {request_id}, active: {self.active_requests}/{self.max_concurrent}")
            return True

    def wait_for_capacity(self):
        with self.condition:
            while not self.can_send():
                # print("Waiting for capacity...")
                self.condition.wait()

    def process_response(self, response: Dict):
        with self.condition:
            request_id = response.get('request_id')
            if request_id in self.pending_requests:
                id_list = self.pending_requests.pop(request_id)
                self.response_queue.append((request_id, id_list, response))
                self.active_requests -= 1
                self.received_count += 1
                self.received_block_count += len(id_list)
                # print(f"Processed response for {request_id}, active: {self.active_requests}/{self.max_concurrent}")
                self.condition.notify_all()

    def get_responses(self) -> List[tuple]:
        with self.lock:
            responses = list(self.response_queue)
            self.response_queue.clear()
            return responses

    def get_stats(self) -> Dict:
        with self.lock:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            return {
                'sent_count':
                self.sent_count,
                'received_count':
                self.received_count,
                'sent_block_count':
                self.sent_count,
                'received_block_count':
                self.received_count,
                'active_requests':
                self.active_requests,
                'pending_requests':
                len(self.pending_requests),
                'elapsed_time':
                elapsed_time,
                'send_rate':
                self.sent_count / elapsed_time if elapsed_time > 0 else 0,
                'receive_rate':
                self.received_count / elapsed_time if elapsed_time > 0 else 0,
                'send_block_rate':
                self.sent_block_count /
                elapsed_time if elapsed_time > 0 else 0,
                'receive_block_rate':
                self.received_block_count /
                elapsed_time if elapsed_time > 0 else 0
            }


def generate_id_list(mean_length: int = 10, std_dev: float = 3.0) -> List[int]:
    length = max(1, int(random.gauss(mean_length, std_dev)))
    length = min(length, MAX_NUM_BLOCK)
    return random.sample(range(MAX_NUM_BLOCK), length)


def sender_thread(client: RouterDealerClient,
                  message_manager: MessageManager,
                  mean_length: int = 10,
                  std_dev: float = 3.0):
    print("Sender thread started")

    while True:
        try:
            message_manager.wait_for_capacity()

            request_id = f"REQ_{uuid.uuid4().hex[:8]}"
            id_list = generate_id_list(mean_length, std_dev)

            if message_manager.add_request(request_id, id_list):
                if client.send_request(request_id, id_list):
                    pass
                    # print(f"Successfully sent {request_id}")
                else:
                    with message_manager.condition:
                        if request_id in message_manager.pending_requests:
                            message_manager.pending_requests.pop(request_id)
                            message_manager.active_requests -= 1
                            message_manager.condition.notify_all()

            time.sleep(0.01)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error in sender thread: {e}")
            time.sleep(1)


def receiver_thread(client: RouterDealerClient,
                    message_manager: MessageManager):
    print("Receiver thread started")

    while True:
        try:
            response = client.receive_response(100)
            if response:
                message_manager.process_response(response)

                responses = message_manager.get_responses()
                for request_id, id_list, resp in responses:
                    success = resp.get('success', False)
                    # print(f"Response for {request_id}: {'Success' if success else 'Failed'}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error in receiver thread: {e}")
            time.sleep(1)


def monitor_thread(message_manager: MessageManager, interval: float = 2.0):
    print("Monitor thread started")

    while True:
        try:
            time.sleep(interval)
            stats = message_manager.get_stats()

            print("\n" + "=" * 60)
            print("📊 PERFORMANCE STATISTICS")
            print("=" * 60)
            print(f"🕒 Running time: {stats['elapsed_time']:.2f} seconds")
            print(f"📤 Total sent: {stats['sent_count']} requests")
            print(f"📥 Total received: {stats['received_count']} responses")
            print(f"⚡ Send rate: {stats['send_rate']:.2f} req/sec")
            print(f"⚡ Receive rate: {stats['receive_rate']:.2f} resp/sec")
            print(
                f"⚡ Send block rate: {stats['send_block_rate']:.2f} block/sec")
            print(
                f"⚡ Receive block rate: {stats['receive_block_rate']:.2f} block/sec"
            )
            print(
                f"🔵 Active requests: {stats['active_requests']}/{message_manager.max_concurrent}"
            )
            print(
                f"🎯 Bandwidth: {(stats['send_block_rate'] * 8784 * 1024 / 1e6) :.2f} MB/sec"
            )
            print(f"⏳ Pending responses: {stats['pending_requests']}")

            print("=" * 60)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error in monitor thread: {e}")
            time.sleep(1)


def main():
    client = RouterDealerClient(SERVER_ADDRESS)
    message_manager = MessageManager(MAX_CONCURRENT)

    try:
        print(
            f"Starting with max concurrent: {MAX_CONCURRENT}, mean length: {MEAN_LENGTH}"
        )

        sender = threading.Thread(target=sender_thread,
                                  args=(client, message_manager, MEAN_LENGTH,
                                        STD_DEV),
                                  daemon=True,
                                  name="SenderThread")

        receiver = threading.Thread(target=receiver_thread,
                                    args=(client, message_manager),
                                    daemon=True,
                                    name="ReceiverThread")

        monitor = threading.Thread(target=monitor_thread,
                                   args=(message_manager, MONITOR_INTERVAL),
                                   daemon=True,
                                   name="MonitorThread")

        sender.start()
        receiver.start()
        monitor.start()

        print("All threads started. Press Ctrl+C to stop...")

        sender.join()
        receiver.join()
        monitor.join()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        final_stats = message_manager.get_stats()
        print("\n" + "=" * 60)
        print("🎯 FINAL STATISTICS")
        print("=" * 60)
        print(f"Total running time: {final_stats['elapsed_time']:.2f} seconds")
        print(f"Total requests sent: {final_stats['sent_count']}")
        print(f"Total responses received: {final_stats['received_count']}")
        print(f"Average send rate: {final_stats['send_rate']:.2f} req/sec")
        print(
            f"Average receive rate: {final_stats['receive_rate']:.2f} resp/sec"
        )

        if final_stats['sent_count'] > 0:
            success_rate = (final_stats['received_count'] /
                            final_stats['sent_count']) * 100
            print(f"Overall success rate: {success_rate:.2f}%")

        print("=" * 60)

        client.close()
        print("Application closed")


if __name__ == "__main__":
    main()
