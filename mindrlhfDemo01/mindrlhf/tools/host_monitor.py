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
"""Host resource monitor for tracking system resource usage during training."""
import os
import threading
import time
import psutil
from mindformers import logger

class ResourceMonitor:
    """Host resource monitor for tracking system resource usage during training.

    This class monitors system resources (primarily memory) and implements memory protection
    mechanisms to prevent excessive resource consumption. Supports both periodic monitoring
    and step-specific monitoring during training processes.

    Attributes:
        host_monitor_interval (float): Monitoring interval in seconds. Negative values disable monitoring.
        host_monitor_steps (list): Specific training steps to monitor (e.g., {1, 3, 7, 8, 9, 10}).
        host_memory_protection (bool): Enable/disable memory protection mechanism.
        host_max_memory_threshold (float): Threshold for triggering memory protection (0.0-1.0).
        peak_memory_mb (float): Peak memory usage by monitored processes (GB).
        peak_vir_memory_mb (float): Peak virtual memory usage (GB).
        pid_memory_history (list): Historical record of process memory usage.
        vir_memory_history (list): Historical record of virtual memory usage.
        last_known_processes (dict): Mapping of active process IDs to last seen timestamps.
        stop_monitor (threading.Event): Event signal to stop monitoring thread.
        monitor_thread (threading.Thread): Background thread for resource monitoring.
        current_step (int): Current training step for step-based monitoring.
        is_init (bool): Initialization status flag.
        enable_host_monitor (bool): Global monitoring enable/disable flag.
    """

    def __init__(self,
                 host_monitor_interval=-1.0,
                 host_monitor_steps=None,
                 host_memory_protection=False,
                 host_max_memory_threshold=0.95):
        """
        Initialize the resource monitor with configuration parameters.

        Args:
            host_monitor_interval (float): Monitoring interval in seconds. Negative values disable monitoring.
            host_monitor_steps (list): Training steps to monitor (integers or [start,end] ranges).
            host_memory_protection (bool): Enable memory protection mechanism. Default=False.
            host_max_memory_threshold (float): Memory usage threshold for protection (0.0-1.0). Default=0.95.
        """
        self.interval = host_monitor_interval
        self.host_monitor_steps = host_monitor_steps
        if self.host_monitor_steps:
            self.host_monitor_steps = self._parse_host_monitor_steps(self.host_monitor_steps)  # Parse specified steps
        else:
            self.host_monitor_steps = []
        self.host_memory_protection = host_memory_protection
        self.host_max_memory_threshold = host_max_memory_threshold
        if self.host_memory_protection:
            logger.info(
                f"Enable host memory protection to automatically end the "
                f"process when the host memory exceeds {self.host_max_memory_threshold * 100:.2f}"
            )
            self.interval = 1.0
        self.enable_host_monitor = False
        self.is_init = False
        if self.interval > 0:
            self.enable_host_monitor = True
            self.is_init = True
            logger.info(f"Enable host monitoring with interval {self.interval} s")
        self.peak_memory_mb = 0
        self.peak_vir_memory_mb = 0
        self.pid_memory_history = []
        self.vir_memory_history = []
        self.last_known_processes = {}
        self.stop_monitor = threading.Event()
        self.monitor_thread = None

        # Track current training step
        self.current_step = None

    def _parse_host_monitor_steps(self, steps):
        """
        Parse step ranges into individual step numbers.

        Converts input like [1, 3, [7, 10]] into {1, 3, 7, 8, 9, 10}.

        Args:
            steps (list): List containing integers or [start,end] ranges.

        Returns:
            set: Set of parsed step numbers.
        """
        parsed_steps = set()
        if steps is None:
            return parsed_steps  # Monitor all steps if steps is None
        for item in steps:
            if isinstance(item, int):
                parsed_steps.add(item)
            elif isinstance(item, list) and len(item) == 2:
                start, end = item
                parsed_steps.update(range(start, end + 1))
        return parsed_steps

    def update_current_step(self, current_step):
        """
        Update the current training step from the training loop.

        Args:
            current_step (int): Current step in the training process.
        """
        self.current_step = current_step
        self.is_init = False  # Reset initialization flag after first update

    def get_active_processes(self, root_process):
        """
        Identify active processes in the training session (excluding idle/sleeping).

        Args:
            root_process (psutil.Process): Main training process.

        Returns:
            list: Active child processes (psutil.Process objects).
        """
        current_time = time.time()
        active_processes = []

        try:
            children = root_process.children(recursive=True)
            all_processes = [root_process] + children
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            all_processes = [root_process]

        for proc in all_processes:
            try:
                status = proc.status()
                if status in ['sleeping', 'idle']:
                    continue
                active_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        new_known = {}
        for proc in active_processes:
            pid = proc.pid
            new_known[pid] = current_time

        for pid, last_seen in list(self.last_known_processes.items()):
            if current_time - last_seen > 10:
                if pid in self.last_known_processes:
                    del self.last_known_processes[pid]

        self.last_known_processes.update(new_known)
        return active_processes

    @staticmethod
    def get_virtual_memory_usage():
        """
        Get system-wide virtual memory usage statistics.

        Returns:
            tuple: (used_memory_GB, total_memory_GB)
        """
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        return used_gb, total_gb

    def get_pid_memory_usage(self):
        """
        Calculate pid memory usage of training-related processes.

        Returns:
            float: pid memory usage in GB.
        """
        current_pid = os.getpid()
        try:
            root_process = psutil.Process(current_pid)
        except psutil.NoSuchProcess:
            return 0

        current_total_mem = 0
        active_count = 0
        active_processes = self.get_active_processes(root_process)
        for proc in active_processes:
            try:
                if not proc.is_running():
                    continue

                mem_info = proc.memory_info()
                current_total_mem += mem_info.rss
                active_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return current_total_mem / (1024 ** 3)  # Convert to GB

    def monitor_resource_usage(self):
        """
        Main monitoring loop for tracking resource consumption.

        Responsibilities:
        1. Checks if current step requires monitoring
        2. Enforces memory protection thresholds
        3. Records memory usage history
        4. Logs resource utilization
        """
        #pylint: disable=R1702
        try:
            while not self.stop_monitor.is_set():
                # Check if current step is in the monitored steps
                if (self.enable_host_monitor and
                        (not self.host_monitor_steps or
                         self.current_step in self.host_monitor_steps or
                         self.is_init)):
                    current_pid_mem_mb = self.get_pid_memory_usage()
                    cur_virtual_memory, total_memory = self.get_virtual_memory_usage()

                    # Host memory protection
                    if self.host_memory_protection:
                        if cur_virtual_memory / total_memory > self.host_max_memory_threshold:
                            logger.info(
                                f"Host memory exceeded {self.host_max_memory_threshold * 100:.2f}%, "
                                f"terminating process."
                            )
                            os.system("pkill -9 -f python")
                    if cur_virtual_memory > self.peak_vir_memory_mb:
                        self.peak_vir_memory_mb = cur_virtual_memory
                    # Update peak memory and history
                    if current_pid_mem_mb > self.peak_memory_mb:
                        self.peak_memory_mb = current_pid_mem_mb
                    self.pid_memory_history.append(
                        (time.strftime("%H:%M:%S"), current_pid_mem_mb))
                    self.vir_memory_history.append(
                        (time.strftime("%H:%M:%S"), cur_virtual_memory))
                    logger.info(
                        f"Virtual Memory: {cur_virtual_memory:.2f} GB | Process Memory: {current_pid_mem_mb:.2f} GB")
                time.sleep(self.interval)
        except (OSError, MemoryError, ValueError) as e:
            logger.exception("Monitoring error: %s", str(e))
        except Exception as e:
            logger.critical("Unexpected monitoring error: %s", str(e))
            raise
        finally:
            logger.info("Monitoring has stopped")

    def start(self):
        """
        Start the resource monitoring thread.

        Initializes monitoring state and launches a daemon thread to monitor resources.
        """
        if not self.enable_host_monitor:
            return
        self.peak_memory_mb = 0
        self.last_known_processes.clear()
        self.stop_monitor.clear()

        self.monitor_thread = threading.Thread(
            target=self.monitor_resource_usage,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Resource monitoring thread started")

    def stop(self):
        """
        Stop the resource monitoring thread.

        Signals the monitoring thread to terminate and waits for it to finish.
        """
        if not self.enable_host_monitor:
            return
        self.stop_monitor.set()
        logger.info(
            f"Stopping resource monitoring, peak pid memory: {self.peak_memory_mb:.2f} GB, "
            f"peak virtual memory: {self.peak_vir_memory_mb:.2f} GB")
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
