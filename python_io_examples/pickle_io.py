# python_io_examples/pickle_io.py

import pickle
from datetime import datetime

PICKLE_FILE = 'data.pkl'

def manage_pickle_files():
    """Demonstrates serializing and de-serializing a Python object with pickle."""
    print("\n--- Demonstrating pickle.dump() and pickle.load() (File I/O) ---")
    data_to_pickle = {
        'timestamp': datetime.now(), 'user_id': 456,
        'is_active': True, 'tags': ('pickle', 'serialization', 'binary')
    }
    print(f"Original Python object to pickle: {data_to_pickle}")
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(data_to_pickle, f)
        print(f"Successfully pickled object to {PICKLE_FILE}")
    with open(PICKLE_FILE, 'rb') as f:
        loaded_from_pickle = pickle.load(f)
        print(f"Successfully loaded object from {PICKLE_FILE}")
    print(f"Loaded object from file: {loaded_from_pickle}")
    print(f"The type of the loaded timestamp is preserved: {type(loaded_from_pickle['timestamp'])}")
    print("\n--- Contrasting with pickle.dumps() and pickle.loads() (In-Memory) ---")
    pickled_bytes = pickle.dumps(data_to_pickle)
    print(f"Object serialized to in-memory bytes (type: {type(pickled_bytes)})")
    loaded_from_bytes = pickle.loads(pickled_bytes)
    print("Object de-serialized from in-memory bytes.")
    print(f"Loaded object from bytes: {loaded_from_bytes}")
    print(f"Type of timestamp is still preserved: {type(loaded_from_bytes['timestamp'])}")

if __name__ == '__main__':
    manage_pickle_files()
