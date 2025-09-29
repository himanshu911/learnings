# python_io_examples/json_io.py

import json
from datetime import datetime

JSON_FILE = 'data.json'

def manage_json_files():
    """Demonstrates writing and reading serializable objects to a JSON file."""
    print("\n--- Demonstrating json.dump() and json.load() ---")
    now_iso = datetime.now().isoformat()
    with open(JSON_FILE, 'w') as f:
        json.dump(now_iso, f)
        print(f"Saved current ISO timestamp to {JSON_FILE}")
    with open(JSON_FILE, 'r') as f:
        loaded_iso = json.load(f)
        dt_object = datetime.fromisoformat(loaded_iso)
        print(f"Loaded timestamp: {loaded_iso}")
        print(f"Converted back to datetime object: {dt_object} (type: {type(dt_object)})")

def demonstrate_json_data_types():
    """Shows the mapping between Python data types and JSON types."""
    print("\n--- Demonstrating Python to JSON Data Type Conversion ---")
    python_data = {
        "a_string": "Hello, World!", "an_integer": 123, "a_float": 3.14159,
        "a_boolean_true": True, "a_boolean_false": False, "a_none_value": None,
        "a_list": [1, "two", 3.0, True], "a_dictionary": {"key": "value"}
    }
    print("Original Python Dictionary:")
    print(python_data)
    json_string = json.dumps(python_data, indent=4)
    print("\nEquivalent JSON String (pretty-printed):")
    print(json_string)

def json_string_and_bytes_conversion():
    """Demonstrates converting between Python objects, JSON strings, and bytes."""
    print("\n--- Demonstrating json.dumps() and json.loads() ---")
    py_dict = {'language': 'Python', 'version': 3.10}
    print(f"Original Python dict: {py_dict}")
    json_string = json.dumps(py_dict)
    print(f"Converted to JSON string: {json_string} (type: {type(json_string)})")
    json_bytes = json_string.encode('utf-8')
    print(f"Encoded to bytes: {json_bytes} (type: {type(json_bytes)})")
    print("-" * 20)
    raw_bytes = b'{"status": "ok", "count": 10}'
    print(f"Original raw bytes: {raw_bytes}")
    decoded_string = raw_bytes.decode('utf-8')
    print(f"Decoded to string: {decoded_string} (type: {type(decoded_string)})")
    py_obj = json.loads(decoded_string)
    print(f"Loaded back to Python object: {py_obj} (type: {type(py_obj)})")

if __name__ == '__main__':
    manage_json_files()
    demonstrate_json_data_types()
    json_string_and_bytes_conversion()
