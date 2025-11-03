# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""plot_host_memory"""
import re
import argparse
import matplotlib.pyplot as plt


def parse_log_file(log_file_path):
    """parse_log_file"""
    virtual_memory = []
    process_memory = []

    with open(log_file_path, "r") as file:
        for line in file:
            match = re.search(r"Virtual Memory: (\d+\.\d+) GB \| Process Memory: (\d+\.\d+) GB", line)
            if match:
                virtual_memory.append(float(match.group(1)))
                process_memory.append(float(match.group(2)))

    return virtual_memory, process_memory


def plot_memory_usage(virtual_memory, process_memory, output_file, title="Memory Usage Over Time"):
    """plot_memory_usage"""

    dangerous_paths = ["/etc", "/usr", "/bin", "/sbin", "/boot", "/proc", "/sys"]

    for dangerous_path in dangerous_paths:
        if dangerous_path in output_file:
            raise PermissionError(
                f"Security violation: Cannot use system directory '{dangerous_path}' "
                f"as dump path. Attempted path: '{output_file}'"
            )

    plt.figure(figsize=(10, 6))
    plt.plot(virtual_memory, label="Virtual Memory (GB)", marker="o")
    plt.plot(process_memory, label="Process Memory (GB)", marker="x")
    plt.xlabel("Time")
    plt.ylabel("Memory (GB)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    print(f"The chart is saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Parse log files and chart memory usage")
    parser.add_argument("--log", type=str, required=True, help="Log file path")
    parser.add_argument("--output_file", type=str, required=True, help="Log file path")
    args = parser.parse_args()
    virtual_memory, process_memory = parse_log_file(args.log)
    plot_memory_usage(virtual_memory, process_memory, args.output_file)


if __name__ == "__main__":
    main()
