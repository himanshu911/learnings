# python_io_examples/text_io.py

TEXT_FILE = 'file.txt'

def manage_text_files():
    """Demonstrates basic file writing, reading, and appending."""
    print("--- Demonstrating Basic File I/O ---")
    with open(TEXT_FILE, 'w') as f:
        f.write('Hello, World!')
        print(f"Wrote 'Hello, World!' to {TEXT_FILE}")
    with open(TEXT_FILE, 'a') as f:
        f.write('\nAppended line.')
        print(f"Appended a new line to {TEXT_FILE}")
    with open(TEXT_FILE, 'r') as f:
        content = f.read()
        print(f"Read entire file content:\n---\n{content}\n---\n")

def read_methods_comparison():
    """Compares different methods for reading lines from a file."""
    print("\n--- Comparing Line Reading Methods ---")
    print("1. Iterating line-by-line (Best for memory):")
    with open(TEXT_FILE, 'r') as f:
        for line in f:
            print(f"  - {line.strip()}")
    print("\n2. readlines() - reads all lines into a list:")
    with open(TEXT_FILE, 'r') as f:
        all_lines = f.readlines()
        if all_lines:
            print(f"  - Row 0: {all_lines[0].strip()}")
            for line in all_lines[1:]:
                print(f"  - {line.strip()}")
    print("\n3. readline() - reads one line at a time:")
    with open(TEXT_FILE, 'r') as f:
        print(f"  - First line: {f.readline().strip()}")
        print(f"  - Second line: {f.readline().strip()}")

if __name__ == '__main__':
    manage_text_files()
    read_methods_comparison()
