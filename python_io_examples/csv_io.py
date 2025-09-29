# python_io_examples/csv_io.py

import csv

CSV_FILE = 'data.csv'

def manage_csv_files():
    """Demonstrates writing and reading CSV files."""
    print("\n--- Demonstrating CSV I/O with csv.writer and csv.reader ---")
    list_data = [
        ['Name', 'Department', 'ID'],
        ['Alice', 'Engineering', 101],
        ['Bob', 'Marketing', 102]
    ]
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(list_data)
        print(f"Wrote list data to {CSV_FILE}")
    with open(CSV_FILE, 'r') as f:
        reader = csv.reader(f)
        print(f"Reading data from {CSV_FILE} as lists:")
        for row in reader:
            print(f"  - {row}")
    print("\n--- Demonstrating CSV I/O with csv.DictWriter and csv.DictReader ---")
    dict_data = [
        {'Name': 'Charlie', 'Department': 'Sales', 'ID': '103'},
        {'Name': 'David', 'Department': 'HR', 'ID': '104'}
    ]
    field_names = ['Name', 'Department', 'ID']
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(dict_data)
        print(f"Wrote dictionary data to {CSV_FILE}")
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        print(f"Reading data from {CSV_FILE} as dictionaries:")
        for row in reader:
            print(f"  - {dict(row)}")

if __name__ == '__main__':
    
    
    
    
    
    
    manage_csv_files()
