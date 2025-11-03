# Copyright 2025 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import List, Optional, Union

import torch
import torch.distributed
from torch.distributed import Backend

import mindspore as ms

def init_model_parallel_group(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    use_message_queue_broadcaster: bool = False,
    group_name: Optional[str] = None,
) -> "GroupCoordinator":
    from vllm.distributed.parallel_state import GroupCoordinator

    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=True,
        use_message_queue_broadcaster=False,
        group_name=group_name,
    )


def init_group_coordinator(
    self,
    group_ranks: List[List[int]],
    local_rank: int,
    torch_distributed_backend: Union[str, Backend],
    use_device_communicator: bool,
    use_message_queue_broadcaster: bool = False,
    group_name: Optional[str] = None,
):
    from vllm.distributed.parallel_state import _get_unique_name, _register_group
    from vllm.utils import resolve_obj_by_qualname

    group_name = group_name or "anonymous"
    self.unique_name = _get_unique_name(group_name)
    _register_group(self)

    self.rank = torch.distributed.get_rank()
    self.local_rank = local_rank
    self.device_group = None
    self.cpu_group = None

    for ranks in group_ranks:
        device_group = torch.distributed.new_group(ranks, backend=torch_distributed_backend)
        # CPU not ready now, use device to communication now.
        cpu_group = torch.distributed.new_group(ranks, backend="hccl")
        if self.rank in ranks:
            self.ranks = ranks
            self.world_size = len(ranks)
            self.rank_in_group = ranks.index(self.rank)
            self.device_group = device_group
            self.cpu_group = cpu_group

    if self.cpu_group is  None:
        raise AssertionError("cpu_group cannot be None")
    if self.device_group is None:
        raise AssertionError("device_group cannot be None")

    from vllm.platforms import current_platform

    # TODO: fix it for other platforms
    if current_platform.is_cuda_alike():
        self.device = torch.device(f"cuda:{local_rank}")
    else:
        self.device = torch.device("cpu")

    self.use_device_communicator = use_device_communicator

    self.device_communicator: DeviceCommunicatorBase = None  # type: ignore
    if use_device_communicator and self.world_size > 1:
        device_comm_cls = resolve_obj_by_qualname(current_platform.get_device_communicator_cls())
        self.device_communicator = device_comm_cls(
            cpu_group=self.cpu_group,
            device=self.device,
            device_group=self.device_group,
            unique_name=self.unique_name,
        )

    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    self.mq_broadcaster: Optional[MessageQueue] = None
    if use_message_queue_broadcaster and self.world_size > 1:
        self.mq_broadcaster = MessageQueue.create_from_process_group(self.cpu_group, 1 << 22, 6)

    from vllm.platforms import current_platform

    self.use_custom_op_call = current_platform.is_cuda_alike()


def initialize_parallel_state(
    distributed_init_method: str = "env://",
    backend: str = "nccl",
    tensor_model_parallel_size: int = 1,
    num_tp_per_train_tp: int = 1,
    pipeline_model_parallel_size: int = 1,
):
    # torch.distributed.all_reduce does not free the input tensor until
    # the synchronization point. This causes the memory usage to grow
    # as the number of all_reduce calls increases. This env var disables
    # this behavior.
    from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel

    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    num_rank = ms.device_context.ascend.device_count()

    # NOTE(sgm): Modify for verl, Env vars will be set by TORCHRUN.
    rank = int(os.getenv("RANK", "-1"))
    if os.getenv("LOCAL_RANK") is None:
        os.environ["LOCAL_RANK"] = str(rank % num_rank)
    local_rank = int(os.getenv("LOCAL_RANK")) % num_rank

    # Use the world_size set by TORCHRUN
    world_size = int(os.getenv("WORLD_SIZE", "-1"))
    if world_size == -1:
        raise AssertionError("The world_size is set to -1, not initialized by TORCHRUN")
    init_distributed_environment(world_size, rank, distributed_init_method, local_rank, backend)
    initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, backend)
