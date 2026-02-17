import json
import re
from typing import List

def extract_json_data(text):
    try:
        pattern = r"```json\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_text = match.group(1).strip()
            json_data = json.loads(json_text)
        else:
            print("No JSON found in the response.")
            print(text)
            json_data = {}
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        json_data = {}
    return json_data

def read_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        return data

def write_json_data(file_path, data:List):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def complete_hierarchy_tags(tags):
    # Use set for deduplication and result storage
    try:
        completed_tags = set(tags)
    except TypeError:
        print("TypeError encountered, tags:", tags)
        return tags

    # Process all tags
    for tag in tags:
        if tag is None or not tag.startswith('/'):
            continue  # Skip invalid tags

        # Split tag into hierarchical parts
        parts = tag.split('/')

        # Generate all parent-level tags (from root to current level)
        for i in range(1, len(parts)):
            parent_tag = '/'.join(parts[:i + 1])
            completed_tags.add(parent_tag)

    # Return sorted result (alphabetical order)
    return sorted(completed_tags)


class SimpleTextWriter:
    def __init__(self, file_path, buffer_size=100, write_mode="w"):
        """
        Simple text writer with buffer

        Args:
            file_path: Output file path
            buffer_size: Buffer size, automatically writes to file when reached
        """
        self.file_path = file_path
        self.buffer_size = buffer_size
        self.buffer = []
        self.file_handle = open(file_path, write_mode, encoding='utf-8')
        self.total_written = 0

    def write(self, text_data):
        """
        Write text data to buffer

        Args:
            text_data: Text data to write
        """
        self.buffer.append(text_data)
        self.total_written += 1

        # Auto-write to file when buffer is full
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Write buffer data to file and clear buffer"""
        if self.buffer:
            for item in self.buffer:
                # Write text data directly with newline
                self.file_handle.write(str(item) + '\n')

            self.file_handle.flush()  # Ensure data is written to disk
            self.buffer.clear()

    def close(self):
        """Close file, ensure all data is written"""
        if self.buffer:
            self.flush()
        self.file_handle.close()
        print(f"Write completed! Saved {self.total_written} items to {self.file_path}")