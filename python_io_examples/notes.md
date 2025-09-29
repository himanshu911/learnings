### 1. The `with` Statement: Safe File Handling

The `with open(...) as f:` syntax is the standard way to work with files. It creates a "context" that ensures the file is automatically closed when you are done, even if errors occur.

```python
with open('file.txt', 'r') as f:
    content = f.read()
```

### 2. File Modes: Action + Format

File modes consist of an **action** character and an optional **format** character.

| Action | Name                 | Effect                                                                                             |
| :----: | :------------------- | :------------------------------------------------------------------------------------------------- |
|  `r`   | Read (Default)       | Opens a file for reading. The file must exist.                                                     |
|  `w`   | Write                | Opens for writing. **Creates** the file if it doesn't exist, and **truncates** (clears) it if it does. |
|  `a`   | Append               | Opens for writing. New data is added to the end. Creates the file if it doesn't exist.             |
|  `x`   | Exclusive Creation   | Opens for writing, but fails (raises `FileExistsError`) if the file already exists.                |
|  `+`   | Update Mode          | Allows both reading and writing (e.g., `r+` to read/write, `w+` to truncate and read/write).       |

| Format | Name                  | Effect                                                                        |
| :----: | :-------------------- | :---------------------------------------------------------------------------- |
|  `t`   | Text Mode (Default)   | Data is handled as strings (`str`). Python manages encoding/decoding (usually UTF-8). |
|  `b`   | Binary Mode           | Data is handled as raw bytes (`bytes`). No encoding/decoding is performed.    |

### 3. Reading from Files

Python offers several ways to read data, each suited for different needs.

| Method          | What it Reads                 | Return Type               | Memory Efficiency | Use Case                                      |
| :-------------- | :---------------------------- | :------------------------ | :---------------- | :-------------------------------------------- |
| `f.read()`      | All remaining content         | Single string             | Low (for large files) | Reading an entire file into one variable.     |
| `f.readline()`  | The next single line          | Single string             | High              | Reading a file line-by-line sequentially.     |
| `f.readlines()` | All remaining lines           | List of strings           | Low (for large files) | Loading all lines to access them by index.    |
| `for line in f:`| Iterates over lines one-by-one| Iterator (yields strings) | **Highest**       | The standard, most Pythonic way to process large files. |

### 4. Working with JSON

The `json` library helps encode Python objects into JSON format and decode them back.

**Key Distinction:**
- **`dump`/`load`**: Work with **file objects**. Use them to write to or read from files directly.
- **`dumps`/`loads`**: Work with **strings** (the 's' stands for "string"). Use them to convert Python objects to JSON strings or parse JSON strings into Python objects.

**JSON Data Types:**
A JSON file can contain any single valid JSON value. It is not limited to just objects (`{}`) or arrays (`[]`).

| Python Data Type | JSON Data Type | Example JSON |
| :--------------- | :------------- | :------------------ |
| `dict` | Object | `{"key": "value"}` |
| `list`, `tuple` | Array | `[10, "hello", true]` |
| `str` | String | `"simple text"` |
| `int`, `float` | Number | `12345` or `3.14` |
| `True`, `False` | Boolean | `true` or `false` |
| `None` | null | `null` |

**Encoding & Decoding:**
When you `json.dump()` to a file, the Python object is converted to a JSON **string**, which is then **encoded** (usually to UTF-8 bytes) and written to the disk. When you `json.load()`, the reverse happens. If you have raw bytes (e.g., from a network request), you must first `.decode()` them into a string before using `json.loads()`. 

### 5. Working with Pickle

The `pickle` module is used for serializing and de-serializing Python object structures into a binary format. Unlike JSON, pickle does not have its own set of distinct "data types" (Object, Array, etc.). Instead, it serializes nearly any native Python object directly, preserving its exact Python type.

#### Key Characteristics
- **Python-Specific**: Unlike JSON, pickle is designed exclusively for Python.
- **Binary Protocol**: Pickle writes a stream of bytes, which is not human-readable. Files **must** be opened in binary mode (`'wb'` for writing, `'rb'` for reading).
- **Type Preservation**: Its main advantage is that it preserves the exact Python data type. If you pickle a `datetime` object or a custom class instance, you get the same type back when you unpickle it.
- **Security Warning**: `pickle` is not secure against maliciously crafted data. It can execute arbitrary code during de-serialization. **Only unpickle data you trust.**

#### Serialization Functions (`dump`/`dumps` vs. `load`/`loads`)

Similar to the `json` library, `pickle` provides four main functions for serialization and de-serialization:

| Function              | Purpose                                                      | I/O Type                                                              | Analogy       |
| :-------------------- | :----------------------------------------------------------- | :-------------------------------------------------------------------- | :------------ |
| `pickle.dump(obj, f)` | Serializes an object and writes it to a **file object**.     | Takes a Python object and a file opened in binary mode (`'wb'`).      | `json.dump()` |
| `pickle.dumps(obj)`   | Serializes an object and returns it as a **bytes object**.   | Takes a Python object and returns `bytes`.                            | `json.dumps()`|
| `pickle.load(f)`      | De-serializes an object by reading from a **file object**.   | Takes a file opened in binary mode (`'rb'`) and returns a Python object.| `json.load()` |
| `pickle.loads(bytes)` | De-serializes an object by reading from a **bytes object**.  | Takes a `bytes` object and returns a Python object.                   | `json.loads()`|

### 6. Working with CSV

The `csv` module is designed to handle reading and writing structured data in Comma-Separated Values (CSV) format.

#### File Modes and `newline=''`
-   **Mode**: Since CSV is a text-based format, you use standard text modes: `'r'` for reading and `'w'` for writing.
-   **`newline=''`**: When *writing* a CSV file, always specify `newline=''` in the `open()` function, like `open('data.csv', 'w', newline='')`. This is crucial because the `csv` module manages line endings itself. Without this, you might get extra blank rows in your output file on certain operating systems.

#### Core Objects
The module provides two main ways to work with data: as lists or as dictionaries.

1.  **`reader` and `writer` (for lists)**
    -   `csv.reader(file)`: Reads a CSV file row by row, where each row is a **list of strings**.
    -   `csv.writer(file)`: Writes data to a CSV file. Its `writerow()` (for a single row) and `writerows()` (for multiple rows) methods expect data formatted as lists.

2.  **`DictReader` and `DictWriter` (for dictionaries)**
    -   `csv.DictReader(file)`: Reads a CSV file row by row, where each row is a **dictionary**. It automatically uses the first row as the dictionary keys (the header).
    -   `csv.DictWriter(file, fieldnames=...)`: Writes data from a list of dictionaries. You must specify the column order via the `fieldnames` argument. It has methods like `writeheader()` and `writerows()`.