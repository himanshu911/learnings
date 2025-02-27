import os
import time  # For adding pauses to make the demo clearer

print("--- Demo of Python's 'os' module commands ---")

# 1. Getting and Changing Working Directory
print("\n--- 1. Working Directory Operations ---")
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

demo_directory = "demo_os_directory"  # Directory to be created for demo purposes

if not os.path.exists(demo_directory):
    os.mkdir(demo_directory)
    print(f"Created directory: {demo_directory}")
else:
    print(f"Directory '{demo_directory}' already exists, using it.")

new_working_dir = os.path.join(current_dir, demo_directory)
os.chdir(new_working_dir)
print(f"Changed working directory to: {os.getcwd()}")
time.sleep(1)

# 2. Listing Directory Contents
print("\n--- 2. Listing Directory Contents ---")
print(f"Contents of '{os.getcwd()}': {os.listdir()}")
time.sleep(1)

# Create some files and directories for further demos
print("\nCreating example files and directories...")
os.makedirs("nested_dir/subdir", exist_ok=True)  # Using makedirs for nested structure
with open("example_file.txt", "w") as f:
    f.write("This is an example file.")
with open(os.path.join("nested_dir", "another_file.txt"), "w") as f:
    f.write("Another file in nested dir.")
print(f"Contents of '{os.getcwd()}' after creation: {os.listdir()}")
time.sleep(2)

# 3. Creating Directories (mkdir and makedirs - already used above slightly)
print("\n--- 3. Creating Directories (mkdir and makedirs) ---")
# mkdir already used to create demo_directory initially (and exist_ok handled it existing)
# makedirs used for nested_dir - let's demonstrate mkdir failing if exists

try:
    os.mkdir(
        "example_file.txt"
    )  # Trying to mkdir with same name as existing file will fail too
    print(
        "mkdir with existing name - This should NOT be printed in this example!"
    )  # Should not reach here
except FileExistsError as e:
    print(f"mkdir example demonstration: Expected FileExistsError: {e}")
except Exception as e:
    print(f"mkdir example demonstration: Unexpected Error: {e}")

print("makedirs example - creating 'another_nested/yet_another'")
os.makedirs(
    "another_nested/yet_another", exist_ok=True
)  # Demonstrating exist_ok=True for makedirs
print(f"Contents after makedirs: {os.listdir()}")
time.sleep(2)

# 4. Removing Directories (rmdir and removedirs)
print("\n--- 4. Removing Directories (rmdir and removedirs) ---")
os.rmdir("another_nested/yet_another")  # Removing empty directory 'yet_another'
print("Removed 'yet_another' using rmdir.")
print(
    f"Contents after rmdir of 'yet_another': {os.listdir()}"
)  # 'another_nested' still exists
time.sleep(1)

# Demonstrating removedirs -  will remove 'another_nested' because it's now empty
os.removedirs("another_nested")
print("Removed 'another_nested' using removedirs.")
print(
    f"Contents after removedirs of 'another_nested': {os.listdir()}"
)  # 'another_nested' is gone
time.sleep(1)

# 5. Renaming Files and Directories
print("\n--- 5. Renaming Files and Directories ---")
os.rename("example_file.txt", "renamed_file.txt")
print("Renamed 'example_file.txt' to 'renamed_file.txt'")
print(f"Contents after rename: {os.listdir()}")
os.rename("nested_dir", "renamed_nested_dir")
print("Renamed 'nested_dir' to 'renamed_nested_dir'")
print(f"Contents after directory rename: {os.listdir()}")
time.sleep(2)


# 6. Deleting Files (remove)
print("\n--- 6. Deleting Files (remove/unlink) ---")
os.remove("renamed_file.txt")  # Or os.unlink("renamed_file.txt") would work the same
print("Removed 'renamed_file.txt' using remove.")
print(f"Contents after file removal: {os.listdir()}")
time.sleep(1)


# 7. os.path operations
print("\n--- 7. os.path Operations ---")
sample_path = "renamed_nested_dir/subdir/another_file.txt"

print(f"\nAnalyzing path: '{sample_path}'")

joined_path = os.path.join("parent_dir", "child_dir", "file.txt")
print(f"\nos.path.join('parent_dir', 'child_dir', 'file.txt'): {joined_path}")

basename = os.path.basename(sample_path)
print(f"\nos.path.basename('{sample_path}'): {basename}")

dirname = os.path.dirname(sample_path)
print(f"\nos.path.dirname('{sample_path}'): {dirname}")

split_path = os.path.split(sample_path)
print(f"\nos.path.split('{sample_path}'): {split_path}")

exists = os.path.exists(sample_path)
print(f"\nos.path.exists('{sample_path}'): {exists}")

is_dir = os.path.isdir("renamed_nested_dir")
print(f"\nos.path.isdir('renamed_nested_dir'): {is_dir}")

is_file = os.path.isfile(sample_path)
print(f"\nos.path.isfile('{sample_path}'): {is_file}")

splitext_path = os.path.splitext("image.jpg")
print(f"\nos.path.splitext('image.jpg'): {splitext_path}")
splitext_no_ext = os.path.splitext("document")
print(f"\nos.path.splitext('document'): {splitext_no_ext}")


# --- Cleanup --- (Moving back to original dir and removing demo dir)
print("\n--- Cleanup ---")
os.chdir(current_dir)  # Go back to the original working directory
print(f"Changed working directory back to: {os.getcwd()}")

if os.path.exists(demo_directory):
    for root, dirs, files in os.walk(demo_directory, topdown=False):
        print(root, dirs, files)
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(demo_directory)  # Remove the main demo directory

print(
    f"Demo directory '{demo_directory}' and its contents cleaned up (if they existed)."
)


print("\n--- Demo Completed ---")
