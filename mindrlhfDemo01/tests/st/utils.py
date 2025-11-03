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
""" utils.py """
import re
import subprocess


def check_log(file_path, check_pairs=None, check_values=None,
              device_memory=None, host_memory=None):
    """ check log """
    with open(file_path, 'r') as f:
        content = f.read()

    # check the number of key in check_pairs in log file is equal to the value
    if check_pairs is not None:
        for key_word, value in check_pairs.items():
            log_output = subprocess.check_output(
                ["grep -r '%s' %s | wc -l" % (key_word, file_path)],
                shell=True)
            log_cnt = str(log_output, 'utf-8').strip()
            assert log_cnt == str(value), (f"Failed to find {key_word} in {file_path} or content is not correct."
                                           f"Expected occurrences: {value}, but got {log_cnt}")

    if check_values is not None:
        log_output = subprocess.check_output(
            [f"cat {file_path}"],
            shell=True)
        log_output = str(log_output, 'utf-8')
        for value in check_values:
            if value not in log_output:
                raise ValueError(f'"{value}" is not in logs, config may be not set right')

    if device_memory is not None:
        device_patterns = [
            r"MindSpore Used memory size:\s*(\d+)M",
            r"Used peak memory usage \(without fragments\):\s*(\d+)M",
            r"Actual peak memory usage \(with fragments\):\s*(\d+)M"
        ]
        cur_device_memory = []
        for pattern in device_patterns:
            matches = re.findall(pattern, content)
            if matches:
                cur_device_memory.append(int(matches[-1]))

        for i, mem in enumerate(cur_device_memory):
            if mem > device_memory[i]:
                raise ValueError(
                    f'Device memory exceeds at index {i}: '
                    f'Expected <= {device_memory[i]}, Actual: {mem}'
                )

    if host_memory is not None:
        pid_mem_pattern = r"peak pid memory: (\d+\.\d+) GB"
        vir_mem_pattern = r"peak virtual memory: (\d+\.\d+) GB"

        pid_matches = re.findall(pid_mem_pattern, content)
        vir_matches = re.findall(vir_mem_pattern, content)

        if pid_matches:
            cur_pid_memory_gb = float(pid_matches[-1])
            if cur_pid_memory_gb > host_memory[0]:
                raise ValueError(
                    f'Host memory (PID) exceeds: Expected <= {host_memory[0]} GB, Actual: {cur_pid_memory_gb} GB'
                )

        if vir_matches:
            cur_vir_memory_gb = float(vir_matches[-1])
            if cur_vir_memory_gb > host_memory[1]:
                raise ValueError(
                    f'Host virtual memory exceeds: Expected <= {host_memory[1]} GB, Actual: {cur_vir_memory_gb} GB'
                )
