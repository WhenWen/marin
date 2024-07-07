"""
file: viewer.py
----------------
simple record viewer for jsonl files in Dolma format

Minimal and lightweight for quick analysis.

Example usage:
    python viewer.py --path=/path/to/file.jsonl
"""

import os
import argparse
import json
from typing import List, Dict


def read_jsonl_file(jsonl_path) -> List[Dict[str, str]]:
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


class DolmaDataViewer:
    def __init__(self, jsonl_path, segment_max_len=500, segment_max_lines=10):
        self.jsonl_path = jsonl_path
        self.data = read_jsonl_file(jsonl_path)
        self.segment_max_len = segment_max_len
        self.segment_max_lines = segment_max_lines

    def print_intro(self):
        print(
            """
            ==================================================
            Interactive record viewer.
            
            Press enter to view the next record, or input 'q' to quit.
            Only part of the record is displayed. Press 'v' to view the full record.
            ==================================================
            """
        )

    def prompt_user(self, in_segment):
        input_string = input().strip().lower()

        valid_input = ["", "q"]
        if in_segment:
            valid_input.append("v")

        while input_string not in valid_input:
            view_str = "'v' to view full record, " if in_segment else ""
            print(
                f"Invalid key. Press {view_str}enter to view the next record, or input 'q' to quit."
            )
            input_string = input().strip().lower()

        if input_string == "q":
            os.system("clear" if os.name == "posix" else "cls")
            exit(0)

        return input_string

    def print_header(self, id):
        os.system("clear" if os.name == "posix" else "cls")

        print("record_id:", id)

        print("=" * 50)

    def print_segment(self, id, text):
        self.print_header(id)
        # print up to SEGMENT_MAX_LINES lines of text or until SEGMENT_MAX_LEN chars
        lines = text.split("\n")
        segment_len = 0
        i = 0
        while i < len(lines):
            print(lines[i])
            segment_len += len(lines[i])
            if (segment_len > self.segment_max_len) or (i > self.segment_max_lines):
                print("...\n(truncated, press 'v' to view full record.)")
                break
            i += 1

        npt = self.prompt_user(in_segment=(i < len(lines)))
        if npt == "v":
            self.print_full_record(id, text)
        elif npt == "":
            return

    def print_full_record(self, id, text):
        self.print_header(id)
        print(text)
        self.prompt_user(in_segment=False)

    def click_through_records(self):
        """
        Wrapper for easier data exploration. Prints out records one-by-one as the user prompts it.
        """
        self.print_intro()

        self.prompt_user(in_segment=False)

        for example in self.data:
            self.print_segment(example["id"], example["text"])

        os.system("clear" if os.name == "posix" else "cls")
        print("End of WARC file.")

    # def sample(self, n_samples):
    #     """
    #     Sample n records from the file using reservoir sampling.
    #     """
    #     self.record_iter = reservoir_sample(self.record_iter, n_samples)
    #     self.click_through_records()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="Path to the jsonl file to view.")
    args = parser.parse_args()

    viewer = DolmaDataViewer(args.file_path)
    viewer.click_through_records()
