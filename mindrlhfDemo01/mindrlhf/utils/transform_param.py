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
"""Resharding Weight"""
import gc
import numpy as np

import mindspore.log as logger
from mindspore import ops, Tensor, Parameter, mint, context
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.communication import get_rank, get_group_size, create_group
from mindspore.parallel.shard import Layout, _DistributedTensorInfo
from mindspore.common.api import _pynative_executor
from mindspore.parallel._parallel_serialization import _build_searched_strategy, _convert_to_list
from mindspore.parallel.function.reshard_func import _redistribute
from mindspore.parallel._cell_wrapper import _insert_virtual_pp_dim

# pylint: disable=C0413
ENABLE_PYNATIVE_REDIST = True
try:
    from mindspore.parallel._tensor import _get_pipeline_operator_map, _get_resharding_operator_map
except ImportError:
    logger.warning("MindSpore version is not compatible, can only run redistribution in graph mode")
    ENABLE_PYNATIVE_REDIST = False

from mindrlhf.utils.utils import enable_pynative_async

BROADCAST_GROUP_CACHE = []
ALLGATHER_GROUP_CACHE = []


def _get_used_dev_mat(dev_mat, tensormap):
    used_dev_mat = []
    for i in range(len(dev_mat)):
        idx = len(dev_mat) - i - 1
        used_dev_mat.append(idx in tensormap)
    return used_dev_mat


def _tensor_map_flatten(tensor_map):
    tensor_map = np.array(tensor_map)
    if tensor_map.ndim == 1:
        return tensor_map.tolist()
    if tensor_map.ndim == 2 and tensor_map.shape[1] == 1:
        return tensor_map.flatten().tolist()
    raise ValueError(f"tensor_map shape: {tensor_map.shape} is not supported")


# pylint: disable=R1705
# pylint: disable=W0212
def _get_dev_mat_for_opt_shard(opt_shard, dev_mat, tensor_map):
    """generate device matrix for opt shard scenario"""
    if opt_shard == 0:
        return dev_mat, -1, tensor_map, True
    used_dev_num = int(np.prod(np.array([dev_mat[len(dev_mat) - i - 1] if i != -1 else 1 for i in tensor_map])))
    if opt_shard == -1 or used_dev_num * opt_shard == int(np.prod(np.array(dev_mat))):
        # 饱和切分不需要调整dev mat
        return dev_mat, -1, tensor_map, True
    # 非饱和根据场景调整dev mat
    remain_dev_num = int(np.prod(np.array(dev_mat))) // (used_dev_num * opt_shard)
    real_remain_dev_num = 1
    used_dev_mat_mask = _get_used_dev_mat(dev_mat, tensor_map)
    counter = -1
    for i, value in enumerate(dev_mat):
        if used_dev_mat_mask[i]:
            # 当前维度已经被使用，不被计算在有优化器并行中
            continue
        if real_remain_dev_num == remain_dev_num:
            # 现有dev mat已经满足需求，不需要调整
            # 通过counter值修改tensor map
            return dev_mat, counter, tensor_map, False
        elif real_remain_dev_num < remain_dev_num:
            real_remain_dev_num *= value
        else:
            # 对counter位置的维度进行拆分
            splitted_dev_value = dev_mat[counter]
            new_dev_mat_value_first = remain_dev_num // (real_remain_dev_num // splitted_dev_value)
            new_dev_mat_value_second = splitted_dev_value // new_dev_mat_value_first
            new_dev_mat = dev_mat[:counter] + \
                          [new_dev_mat_value_first, new_dev_mat_value_second] + dev_mat[counter + 1:]
            new_tensor_map = [value if value < len(dev_mat) - counter - 1 else value + 1 for value in tensor_map]
            return new_dev_mat, counter, new_tensor_map, False
        counter += 1
    if real_remain_dev_num == remain_dev_num:
        # 现有dev mat已经满足需求，不需要调整
        # 通过counter值修改tensor map
        return dev_mat, counter, tensor_map, False
    else:
        # 对counter位置的维度进行拆分
        splitted_dev_value = dev_mat[counter]
        new_dev_mat_value_first = remain_dev_num // (real_remain_dev_num // splitted_dev_value)
        new_dev_mat_value_second = splitted_dev_value // new_dev_mat_value_first
        new_dev_mat = dev_mat[:counter] + \
                      [new_dev_mat_value_first, new_dev_mat_value_second] + dev_mat[counter + 1:]
        new_tensor_map = [value if value < len(dev_mat) - counter - 1 else value + 1 for value in tensor_map]
        return new_dev_mat, counter, new_tensor_map, False


def _get_tensor_map_for_opt_shard(dev_mat, tensor_map, full_opt_shard, counter, alias_name):
    """generate tensor map for opt shard scenario"""
    used_dev_mat = _get_used_dev_mat(dev_mat, tensor_map)
    if full_opt_shard:
        unused_idx = [len(used_dev_mat) - i - 1 for i in range(len(used_dev_mat)) if not used_dev_mat[i]]
    else:
        unused_idx = []
        real_idx = 0
        for i, used_dev_mat_idx in enumerate(used_dev_mat):
            if used_dev_mat_idx:
                continue
            if real_idx > counter:
                unused_idx.append(len(used_dev_mat) - i - 1)
            if not used_dev_mat_idx:
                real_idx += 1

    opt_shard_dim = unused_idx
    if tensor_map[0] != -1:
        opt_shard_dim = [tensor_map[0]] + opt_shard_dim
    opt_shard_name = tuple([alias_name[len(alias_name) - i - 1] if i != -1 else "None" for i in opt_shard_dim])
    og_name = [alias_name[len(alias_name) - i - 1] if i != -1 else "None" for i in tensor_map[1:]]
    if len(opt_shard_name) > 1:
        new_tensor_map = tuple([opt_shard_name] + og_name)
    else:
        new_tensor_map = tuple(list(opt_shard_name) + og_name)
    return new_tensor_map


def _layout_process(stra_layout, pp_stages):
    """return the layout list."""
    rank_list = None
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    opt_shard = stra_layout[5]
    dev_mat = stra_layout[0]
    tensor_map = stra_layout[1]
    tensor_map = _tensor_map_flatten(tensor_map)
    new_dev_mat, counter, new_tensor_map, full_opt_shard = _get_dev_mat_for_opt_shard(opt_shard, dev_mat,
                                                                                      tensor_map)
    alias_name = [alphabet[i] for i in range(len(new_dev_mat))]
    if opt_shard != 0:
        # 开了优化器并行，需要修改new_tensor_map
        new_tensor_map = _get_tensor_map_for_opt_shard(new_dev_mat, new_tensor_map, full_opt_shard, counter,
                                                       alias_name)
    else:
        new_tensor_map = tuple([alias_name[len(alias_name) - i - 1] if i != -1 else "None" for i in new_tensor_map])
    all_dev_num = get_group_size()
    local_dev_num = int(np.prod(np.array(new_dev_mat)))
    if all_dev_num != local_dev_num:
        # 其中一个网络开启了pp，导致实际设备数和计算出的设备数不一致
        dev_num_in_stage = get_group_size() // pp_stages
        stage_id = stra_layout[6][0]
        rank_list = list(
            range(dev_num_in_stage * stage_id, dev_num_in_stage * (stage_id + 1)))
    new_dev_mat = tuple(new_dev_mat)
    alias_name = tuple(alias_name)
    layout = Layout(new_dev_mat, alias_name, rank_list)
    final_layout = layout(*new_tensor_map)
    logger.debug("The final layout is ", final_layout.to_dict())
    return final_layout


class TransformParametersD2D:
    """
    Transform parameter from source network's layout to destination network's layout. All the parameters will do
    transformation from device to device.

    Args:
         src_network (Cell): The network provides the source parameters.
         dst_network (Cell): The parameters in destination network will be assigned by source network
            after transformation.
        src_strategy_path (str): The path of source merged strategy end with ckpt. Default: None.
        dst_strategy_path (str): The path of destination merged strategy end with ckpt. Default: None.
        match_func (function): Check whether two input parameters are matched. It takes source param and dest param
            as input and return boolean. Default value is None, which means that the parameters name must equal.
            Default: None.
        reshard_mode (dict): reshard_mode can be set 0, 1, 2, they mean run in pynative mode, run in graph mode and run
            in hybrid mode respectively. default value is 0.

    Raises:
        TypeError: The output of `match_func` is not a boolean.
        ValueError: `src_network` and `dst_network` does not have matched parameter.
        ValueError: `src_strategy_path` or `dst_strategy_path` is None.
    """

    def __init__(self, src_network, dst_network, src_strategy_path=None, dst_strategy_path=None, match_func=None,
                 reshard_mode=0):
        if src_strategy_path is not None:
            src_strategy = _build_searched_strategy(src_strategy_path)
            src_stra_info = _convert_to_list(src_strategy, get_rank())
        else:
            raise ValueError("The src_strategy_path should not be None.")
        if dst_strategy_path is not None:
            dst_strategy = _build_searched_strategy(dst_strategy_path)
            dst_stra_info = _convert_to_list(dst_strategy, get_rank())
        else:
            raise ValueError("The dst_strategy_path should not be None.")

        # 获取src和dst的pp stage个数
        src_pp_stages = max(stra_info[6][0] for stra_info in src_stra_info.values()) + 1
        dst_pp_stages = max(stra_info[6][0] for stra_info in dst_stra_info.values()) + 1

        # 检查src和dst是否是同总卡数倒换
        self._check_total_dev_num(src_stra_info, src_pp_stages, dst_stra_info, dst_pp_stages)

        # 将存在strategy里面的param给存到dict中，不存在strategy.ckpt中的param不做倒换
        # 不能存在在策略文件中的权重也会做倒换，默认按照全1的切分策略并且默认不开启PP，数据放在stage0
        src_net_param_dict = {}
        for name, param in src_network.parameters_and_names():
            src_net_param_dict[name] = param
            if name not in src_stra_info.keys():
                logger.warning(f"for param {name}, it's not in strategy file, set default strategy.")
                src_stra_info[name] = [[get_group_size(),], [-1] * len(param.shape), [], 0, 0, 0, [0]]

        dst_net_param_dict = {}
        for name, param in dst_network.parameters_and_names():
            dst_net_param_dict[name] = param
            if name not in dst_stra_info.keys():
                logger.warning(f"for param {name}, it's not in strategy file, set default strategy.")
                dst_stra_info[name] = [[get_group_size(),], [-1] * len(param.shape), [], 0, 0, 0, [0]]

        # 获取src和dst的param的交集
        src_param_name_intersection = []
        dst_param_name_intersection = []
        for src_name, src_value in src_net_param_dict.items():
            for dst_name, dst_value in dst_net_param_dict.items():
                # 如果有match_func则通过match_func来判断两个网络中的param是否匹配
                if match_func is not None:
                    match_flag = match_func(src_name, dst_name)
                    if not isinstance(match_flag, bool):
                        raise TypeError("The return value of match_func should be bool.")
                    if match_flag:
                        src_layout = _layout_process(src_stra_info[src_name], src_pp_stages)
                        src_value._dtensor_info = _DistributedTensorInfo(src_layout)
                        dst_layout = _layout_process(dst_stra_info[dst_name], dst_pp_stages)
                        dst_value._dtensor_info = _DistributedTensorInfo(dst_layout)
                        src_param_name_intersection.append(src_value)
                        dst_param_name_intersection.append(dst_value)
                        break
                # 如果match_func为None，则需要src_name和dst_name完全匹配
                else:
                    if src_name == dst_name:
                        src_layout = _layout_process(src_stra_info[src_name], src_pp_stages)
                        src_value._dtensor_info = _DistributedTensorInfo(src_layout)
                        dst_layout = _layout_process(dst_stra_info[dst_name], dst_pp_stages)
                        dst_value._dtensor_info = _DistributedTensorInfo(dst_layout)
                        src_param_name_intersection.append(src_value)
                        dst_param_name_intersection.append(dst_value)
                        break
        self._log_not_in_intersection_param(src_param_name_intersection, src_net_param_dict, "src_network")
        self._log_not_in_intersection_param(dst_param_name_intersection, dst_net_param_dict, "dst_network")
        if not src_param_name_intersection or not dst_param_name_intersection:
            raise ValueError("The intersection of src_network and dst_network is empty.")
        if len(src_param_name_intersection) != len(dst_param_name_intersection):
            raise ValueError("The length of src_param_name_intersection and dst_param_name_intersection is not equal.")

        self._src_param_name_intersection = src_param_name_intersection
        self._dst_param_name_intersection = dst_param_name_intersection

        self.assign = P.Assign()

        self._auto_paralell_context_value_map = {}
        self._pipeline_config = {}
        self._reshard_mode = reshard_mode

    @enable_pynative_async
    def transform(self, input_on_device_flag=(True, True)):
        """
        transform the parameters from source network layout to dest network layout and assign the parameter to
        dest network
        """
        if not isinstance(input_on_device_flag, tuple) or len(input_on_device_flag) != 2:
            raise ValueError(f"The input_on_device_flag must be tuple, and its length must be 2."
                             f"But got {type(input_on_device_flag)}")
        for _, flag in enumerate(input_on_device_flag):
            if not isinstance(flag, bool):
                raise TypeError(f"elements in input_on_device_flag must be bool, but got ")
        mem_opt_level = self._check_input_flag(input_on_device_flag)
        logger.info(f"The input on device flag is {input_on_device_flag}, memory optimize level is {mem_opt_level}")
        src_param_name_intersection = self.preprocess_for_src_param(input_on_device_flag)
        if self._reshard_mode == 1 or not ENABLE_PYNATIVE_REDIST:
            logger.info("Force run in GRAPH redistribution")
            self.transform_graph(src_param_name_intersection, mem_opt_level)
            return True
        force_run_in_pynative = self._reshard_mode == 0
        logger.info(f"Run in HYBRID redistribution with force_run_in_pynative {force_run_in_pynative}")
        self.transform_hybrid(src_param_name_intersection, mem_opt_level, force_run_in_pynative)
        return True

    def transform_graph(self, src_param_name_intersection, mem_opt_level):
        """
        transform the parameters from source network layout to dest network layout and assign the parameter to
        dest network
        """
        for i, src_param in enumerate(src_param_name_intersection):
            if not isinstance(src_param, (Parameter, Tensor)):
                continue
            if mem_opt_level in [1]:
                src_param._load()
                self._dst_param_name_intersection[i]._load()
            redist_src_param = _redistribute(src_param, self._dst_param_name_intersection[i]._dtensor_info)
            redist_src_param = ops.cast(redist_src_param, self._dst_param_name_intersection[i].dtype)
            _pynative_executor.sync()
            if get_rank() in self._dst_param_name_intersection[i]._dtensor_info.layout.to_dict()["rank_list"]:
                _check_shape_match(redist_src_param, src_param._dtensor_info, self._dst_param_name_intersection[i])
                self.assign(self._dst_param_name_intersection[i], redist_src_param)
            if mem_opt_level in [1]:
                src_param._offload()
                self._dst_param_name_intersection[i]._offload()

    def transform_hybrid(self, src_param_name_intersection, mem_opt_level, force_run_in_pynative=False):
        """
        transform the parameters from source network layout to dest network layout and assign the parameter to
        dest network
        """
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        pp_stages = context.get_auto_parallel_context("pipeline_stages")
        context.set_auto_parallel_context(parallel_mode="stand_alone", device_num=get_group_size(), pipeline_stages=1)
        for i, src_param in enumerate(src_param_name_intersection):
            if not isinstance(src_param, (Parameter, Tensor)):
                continue
            if mem_opt_level in [1]:
                src_param._load()
                self._dst_param_name_intersection[i]._load()
            merge_pp_src_param = merge_params_in_pipeline_stages(src_param,
                                                                 self._dst_param_name_intersection[i]._dtensor_info)
            reshard(merge_pp_src_param, self._dst_param_name_intersection[i], force_run_in_pynative)
            if mem_opt_level in [1]:
                src_param._offload()
                self._dst_param_name_intersection[i]._offload()
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=get_group_size(),
                                          pipeline_stages=pp_stages)

    # pylint: disable=W0613
    def preprocess_for_src_param(self, input_on_device_flag):
        """preprocess src param"""
        return self._src_param_name_intersection

    @staticmethod
    def _log_not_in_intersection_param(param_intersection, all_param, debug_string):
        """find the param in all_param but not in param_intersection"""
        param_intersection_name = [
            param.name for param in param_intersection]
        for param_name in all_param.keys():
            if param_name not in param_intersection_name:
                logger.warning(f"The param {param_name} of {debug_string} is not in the intersection of src_network "
                               f"and dst_network.")

    @staticmethod
    def _check_total_dev_num(src_stra_info, src_pp_stages, dst_stra_info, dst_pp_stages):
        _, first_src_stra_info = next(iter(src_stra_info.items()))
        src_dev_num = np.prod(np.array(first_src_stra_info[0])) * src_pp_stages
        _, first_dst_stra_info = next(iter(dst_stra_info.items()))
        dst_dev_num = np.prod(np.array(first_dst_stra_info[0])) * dst_pp_stages
        if src_dev_num != dst_dev_num:
            raise ValueError(f"src network device number must equal to dest network device number,"
                             f"but got src {src_dev_num}, dst {dst_dev_num}")

    @staticmethod
    def _check_input_flag(input_on_device_flag):
        if input_on_device_flag[0] and input_on_device_flag[1]:
            return 0
        elif not (input_on_device_flag[0] or input_on_device_flag[1]):
            return 1
        else:
            raise ValueError(f"The input_on_device_flag must be in following case (True, True), "
                             f"(False, False), but got {input_on_device_flag}")


class TransformParametersD2DForDSv3(TransformParametersD2D):
    """
    Transform parameter from source network's layout to destination network's layout. All the parameters will do
    transformation from device to device.
    """

    def __init__(self, src_network, dst_network, transform_args, src_strategy_path=None, dst_strategy_path=None,
                 match_func=None, reshard_mode=0):
        super().__init__(src_network, dst_network, src_strategy_path, dst_strategy_path, match_func, reshard_mode)
        if not isinstance(transform_args, dict):
            raise TypeError("transform args must be dict")

        self._n_head = transform_args.get("n_head")
        self._qk_nope_head_dim = transform_args.get("qk_nope_head_dim")
        self._qk_rope_head_dim = transform_args.get("qk_rope_head_dim")
        self._tok_embedding_shape = transform_args.get("tok_embedding_shape")
        self.l2q_nope_proj = None
        self.l2q_pe_proj = None
        self.kv2l_k_pe = None
        self.kv2l_latent_kv = None
        if len(self._src_param_name_intersection) != len(self._dst_param_name_intersection):
            raise ValueError("The length of src_param_name_intersection and dst_param_name_intersection is not equal.")

    def preprocess_for_src_param(self, input_on_device_flag):
        """
        preprocess for train param
        """
        mem_opt_level = self._check_input_flag(input_on_device_flag)
        new_src_param_intersection = []
        dev_mat = Layout((get_group_size(),), ("all_dev",))

        for i, src_param in enumerate(self._src_param_name_intersection):
            tensor_map = ["None"] * len(src_param.shape)
            standalone_layout = dev_mat(*tensor_map)
            standalone_dtensor_info = _DistributedTensorInfo(standalone_layout)
            keywords = [
                "feed_forward.routed_experts.ffn.w1",
                "feed_forward.routed_experts.ffn.w2",
                "feed_forward.routed_experts.ffn.w3"
            ]
            if any(keyword in src_param.name for keyword in keywords):
                if mem_opt_level in [1]:
                    src_param._load()
                    self._dst_param_name_intersection[i]._load()
                merged_params = merge_params_in_pipeline_stages(src_param,
                                                                self._dst_param_name_intersection[i]._dtensor_info)
                transposed_params = F.transpose(merged_params, (0, 2, 1))
                transposed_params._dtensor_info = self._transpose_layout(src_param._dtensor_info, (0, 2, 1))
                reshard(transposed_params, self._dst_param_name_intersection[i], False)
                if mem_opt_level in [1]:
                    src_param._offload()
                    self._dst_param_name_intersection[i]._offload()
                new_src_param_intersection.append("skip")
                continue
            if "tok_embeddings" in src_param.name:
                if mem_opt_level in [1]:
                    src_param._load()
                    self._dst_param_name_intersection[i]._load()
                merged_params = merge_params_in_pipeline_stages(src_param,
                                                                self._dst_param_name_intersection[i]._dtensor_info,
                                                                self._tok_embedding_shape)
                reshard(merged_params, self._dst_param_name_intersection[i])
                if mem_opt_level in [1]:
                    src_param._offload()
                    self._dst_param_name_intersection[i]._offload()
                new_src_param_intersection.append("skip")
                continue
            if "attention.l2q_nope_proj.weight" in src_param.name:
                self._redist_func("l2q_nope_proj", mem_opt_level, src_param, standalone_dtensor_info)
            elif "attention.l2q_pe_proj.weight" in src_param.name:
                self._redist_func("l2q_pe_proj", mem_opt_level, src_param, standalone_dtensor_info)
            elif "attention.kv2l_k_pe.weight" in src_param.name:
                self._redist_func("kv2l_k_pe", mem_opt_level, src_param, standalone_dtensor_info)
            elif "attention.kv2l_latent_kv.weight" in src_param.name:
                self._redist_func("kv2l_latent_kv", mem_opt_level, src_param, standalone_dtensor_info)
            else:
                new_src_param_intersection.append(src_param)
                continue
            value_merged = self._transfer_train_to_infer(src_param, new_src_param_intersection)
            if value_merged is None:
                continue
            value_merged = Tensor(value_merged)
            value_merged._dtensor_info = standalone_dtensor_info
            param_obj = _redistribute(value_merged, self._dst_param_name_intersection[i]._dtensor_info)
            self._assign_value(param_obj, mem_opt_level, i, standalone_dtensor_info)
            new_src_param_intersection.append("skip")
        self._clean_cache()
        return new_src_param_intersection

    def _clean_cache(self):
        self.l2q_nope_proj = None
        self.l2q_pe_proj = None
        self.kv2l_k_pe = None
        self.kv2l_latent_kv = None
        gc.collect()

    def _check_value_is_none(self, input_var, value):
        if getattr(self, input_var) is None:
            setattr(self, input_var, value)
        else:
            raise ValueError(f"{input_var} has value")

    def _redist_func(self, input_var, mem_opt_level, param, dtensor_info):
        if mem_opt_level in [1]:
            param._load()
        param_obj = _redistribute(param, dtensor_info)
        self._check_value_is_none(input_var, param_obj.asnumpy())
        if mem_opt_level in [1]:
            param._offload()
        return param_obj

    def _transpose_layout(self, dtensor_info, transpose_in):
        """transpose tensor layout according to transpose argument"""
        layout = dtensor_info.layout
        layout_info = layout.to_dict()
        new_dev_mat = Layout(layout_info["device_matrix"], layout_info["alias_name"],
                             rank_list=layout_info["rank_list"])
        new_tensor_map = tuple([layout_info["tensor_map"][i] for i in transpose_in])
        new_alias_name = []
        for map_idx in new_tensor_map:
            if isinstance(map_idx, (tuple, list)):
                local_alias_name = []
                for idx in map_idx:
                    if idx != -1:
                        local_alias_name.append(layout_info["alias_name"][len(layout_info["alias_name"]) - idx - 1])
                    else:
                        local_alias_name.append("None")
                new_alias_name.append(tuple(local_alias_name))
            else:
                if map_idx != -1:
                    new_alias_name.append(layout_info["alias_name"][len(layout_info["alias_name"]) - map_idx - 1])
                else:
                    new_alias_name.append("None")
        new_layout = new_dev_mat(*new_alias_name)
        return _DistributedTensorInfo(new_layout)

    def _assign_value(self, input_value, mem_opt_level, i, standalone_dtensor_info):
        if get_rank() in self._dst_param_name_intersection[i]._dtensor_info.layout.to_dict()["rank_list"]:
            if mem_opt_level in [1]:
                self._dst_param_name_intersection[i]._load()
            _check_shape_match(input_value, standalone_dtensor_info, self._dst_param_name_intersection[i])
            ops.assign(self._dst_param_name_intersection[i],
                       input_value.astype(self._dst_param_name_intersection[i].dtype))
            if mem_opt_level in [1]:
                self._dst_param_name_intersection[i]._offload()

    def _transfer_train_to_infer(self, src_param, new_src_param_intersection):
        """Get two parts param in train, transfer them to infer"""
        value_merged = None
        is_l2q = ("attention.l2q_nope_proj.weight" in src_param.name or
                  "attention.l2q_pe_proj.weight" in src_param.name)
        is_kv2l = ("attention.kv2l_k_pe.weight" in src_param.name or
                   "attention.kv2l_latent_kv.weight" in src_param.name)
        if is_l2q and self.l2q_nope_proj is not None and self.l2q_pe_proj is not None:
            value_nope = self.l2q_nope_proj
            value_pe = self.l2q_pe_proj
            value_nope = value_nope.reshape(self._n_head, self._qk_nope_head_dim, -1)
            value_pe = value_pe.reshape(self._n_head, self._qk_rope_head_dim, -1)
            value_merged = np.concatenate([value_nope, value_pe], axis=1)
            value_merged = value_merged.reshape(-1, value_merged.shape[-1])
            self.l2q_nope_proj = None
            self.l2q_pe_proj = None
        elif is_kv2l and self.kv2l_k_pe is not None and self.kv2l_latent_kv is not None:
            value_k_pe = self.kv2l_k_pe
            value_latent_kv = self.kv2l_latent_kv
            value_k_pe = value_k_pe.reshape(-1, value_k_pe.shape[-1])
            value_merged = np.concatenate([value_latent_kv, value_k_pe], axis=0)
            value_merged = value_merged.reshape(-1, value_merged.shape[-1])
            self.kv2l_k_pe = None
            self.kv2l_latent_kv = None
        else:
            new_src_param_intersection.append("skip")
        return value_merged


def _transfer_layout_to_tuple(input_layout, input_shape):
    input_layout_dict = input_layout.to_dict()
    return (list(input_layout_dict["device_matrix"]), list(input_layout_dict["tensor_map"]),
            input_shape, input_layout_dict["rank_list"])


def merge_params_in_pipeline_stages(tensor, to_dtensor_info, broadcast_global_shape=None):
    """
    merge params between pp stages

    Args:
        tensor (Tensor)
        to_dtensor_info (Layout)
        broadcast_global_shape (tuple)
    """
    from_dtensor_layout = tensor._dtensor_info.layout
    from_shape = [tensor.shape[i] * tensor._dtensor_info.sharding_strategy[i] for i in range(len(tensor.shape))]
    to_dtensor_layout = to_dtensor_info.layout
    broadcast_op_map = _get_pipeline_operator_map(_transfer_layout_to_tuple(from_dtensor_layout, from_shape),
                                                  _transfer_layout_to_tuple(to_dtensor_layout, from_shape),
                                                  get_rank())
    if not broadcast_op_map or get_rank() not in broadcast_op_map:
        logger.info(f"broadcast_op_map is {broadcast_op_map}, current rank {get_rank()} no in it or it's empty.")
        return tensor
    broadcast_op_map = broadcast_op_map[get_rank()][0]
    root_idx = broadcast_op_map[1]
    broadcast_rank_list = broadcast_op_map[2]
    str_rank_list = '-'.join([str(rank) for rank in broadcast_rank_list])
    broadcast_group = f"pp_broadcast_group-{str_rank_list}"
    global BROADCAST_GROUP_CACHE
    if broadcast_group not in BROADCAST_GROUP_CACHE:
        logger.info(f"create broadcast group {broadcast_group} for rank list {broadcast_rank_list}")
        create_group(broadcast_group, broadcast_rank_list)
        BROADCAST_GROUP_CACHE.append(broadcast_group)
    if get_rank() not in from_dtensor_layout.to_dict()["rank_list"]:
        global_shape = tensor.shape if broadcast_global_shape is None else broadcast_global_shape
        new_tensor_shape = tuple([global_shape[i] // tensor._dtensor_info.sharding_strategy[i]
                                  for i in range(len(tensor.shape))])
        tensor = mint.empty(new_tensor_shape, dtype=tensor.dtype)
        tensor._dtensor_info = _DistributedTensorInfo(from_dtensor_layout)
    _ = mint.distributed.broadcast(tensor, root_idx, broadcast_group, async_op=False)
    return tensor


def _run_redist_pynative(tensor, ops_list):
    """run redistribution ops"""
    out_tensor = tensor
    for ops_info in ops_list:
        ops_name = ops_info[0]
        if ops_name == "StridedSlice":
            out_tensor = ops.strided_slice(out_tensor, ops_info[1], ops_info[2], ops_info[3])
        elif ops_name == "AllConcat":
            empty_tensor = [mint.empty(out_tensor.shape, dtype=tensor.dtype) for _ in range(ops_info[1])]
            out_tensor = out_tensor.contiguous()
            _ = mint.distributed.all_gather(empty_tensor, out_tensor, ops_info[2])
            out_tensor = mint.concat(empty_tensor, ops_info[3])
        elif ops_name == "Reshape":
            out_tensor = out_tensor.view(ops_info[1])
        else:
            raise ValueError(f"ops name {ops_name} is not valid, only support ['StridedSlice', 'AllConcat', 'Reshape']")
    return out_tensor


def _preprocess_op_map(operator_map):
    """preprocess op map, append all inputs of ops to list"""
    ops_list = []
    for ops_info in operator_map:
        ops_name = ops_info[0]
        if ops_name == "StridedSlice":
            tensor_dims = len(ops_info[1]) // 3
            ops_list.append((ops_name, (ops_info[1][:tensor_dims]), (ops_info[1][tensor_dims: 2 * tensor_dims]),
                             (ops_info[1][2 * tensor_dims:])))
        elif ops_name == "AllConcat":
            global ALLGATHER_GROUP_CACHE
            allgather_rank_list = ops_info[1][:-1]
            str_rank_list = '-'.join([str(rank) for rank in allgather_rank_list])
            allgather_group = f"reshard_allgather_group-{str_rank_list}"
            if allgather_group not in ALLGATHER_GROUP_CACHE:
                logger.info(f"create allgather group {allgather_group} for rank list {allgather_group}")
                create_group(allgather_group, allgather_rank_list)
                ALLGATHER_GROUP_CACHE.append(allgather_group)
            ops_list.append((ops_name, len(ops_info[1]) - 1, allgather_group, ops_info[1][-1]))
        elif ops_name == "Reshape":
            ops_list.append((ops_name, tuple(ops_info[1])))
        else:
            raise ValueError(f"ops name {ops_name} is not valid, only support ['StridedSlice', 'AllConcat', 'Reshape']")
    return tuple(ops_list)


def reshard(tensor, out_tensor, force_run_in_pynative=False):
    """
    reshard tensor in pp stages
    """
    to_dtensor_info = out_tensor._dtensor_info
    from_rank_list = tensor._dtensor_info.layout.to_dict()["rank_list"]
    to_rank_list = to_dtensor_info.layout.to_dict()["rank_list"]
    if get_rank() not in set(from_rank_list).union(set(to_rank_list)):
        return True
    if len(from_rank_list) < len(to_rank_list):
        logger.info(f"rank list of from_layout is not less than rank list of to_layout,"
                    f"which are {tensor._dtensor_info.layout.to_dict()['rank_list']} "
                    f"and {to_dtensor_info.layout.to_dict()['rank_list']}."
                    f"Please first merge pp stages by using _get_pipeline_operator_map")
    from_dtensor_layout = _insert_virtual_pp_dim(tensor._dtensor_info.layout)
    to_dtensor_layout = _insert_virtual_pp_dim(to_dtensor_info.layout)
    from_shape = [tensor.shape[i] * tensor._dtensor_info.sharding_strategy[i] for i in range(len(tensor.shape))]
    from_layout_tuple = _transfer_layout_to_tuple(from_dtensor_layout, from_shape)
    to_layout_tuple = _transfer_layout_to_tuple(to_dtensor_layout, from_shape)
    if from_layout_tuple == to_layout_tuple:
        logger.debug(f"from layout {from_layout_tuple} is equal to to layout {to_layout_tuple}")
        _check_shape_match(tensor, tensor._dtensor_info, out_tensor)
        ops.assign(out_tensor, tensor.astype(out_tensor.dtype))
        return True
    operator_map = _get_resharding_operator_map(from_layout_tuple, to_layout_tuple, get_rank())[get_rank()]
    tensor_name = tensor.name if isinstance(tensor, Parameter) else "tensor"
    logger.debug(f"for input {tensor_name}, from layout is {from_layout_tuple}, to layout is {to_layout_tuple} "
                 f"and the redistribute operator map is {operator_map}")
    is_last_all_gather = False
    last_allgather_rank_list = None
    if operator_map and operator_map[-1][0] == "AllConcat" and operator_map[-1][1][-1] == 0:
        # last op is allgather
        is_last_all_gather = True
        last_allgather_rank_list = operator_map[-1][1][:-1]
        operator_map = operator_map[:-1]

    # for debugging, force run redistribution in pynative mode
    force_run_in_pynative_redist = force_run_in_pynative
    run_in_graph = (len(to_rank_list) == get_group_size() and
                    not is_last_all_gather and not force_run_in_pynative_redist)
    redist_out = _run_redist(tensor, run_in_graph, from_dtensor_layout, to_dtensor_layout, operator_map)
    if is_last_all_gather:
        _run_redist_for_last_allgather(redist_out, out_tensor, last_allgather_rank_list)
        return True
    if get_rank() in to_dtensor_info.layout.to_dict()["rank_list"]:
        _check_shape_match(redist_out, tensor._dtensor_info, out_tensor)
        ops.assign(out_tensor, redist_out.astype(out_tensor.dtype))
    return True


def _run_redist(tensor, run_in_graph, from_dtensor_layout, to_dtensor_layout, operator_map):
    """run redistribution"""
    if run_in_graph:
        og_dtensor_info = tensor._dtensor_info
        tensor._dtensor_info = _DistributedTensorInfo(from_dtensor_layout)
        redist_out = _redistribute(tensor, _DistributedTensorInfo(to_dtensor_layout))
        tensor._dtensor_info = og_dtensor_info
    else:
        ops_list = _preprocess_op_map(operator_map)
        if ops_list:
            redist_out = _run_redist_pynative(tensor, ops_list)
        else:
            redist_out = tensor
    return redist_out


def _run_redist_for_last_allgather(redist_out, out_tensor, last_allgather_rank_list):
    """run redistribution for the scenario that last ops is allgather"""
    logger.info("The last op is all_gather, use all_gather_into_tensor to replace assign")
    global ALLGATHER_GROUP_CACHE
    str_rank_list = '-'.join([str(rank) for rank in last_allgather_rank_list])
    allgather_group = f"reshard_allgather_group-{str_rank_list}"
    if allgather_group not in ALLGATHER_GROUP_CACHE:
        logger.info(f"create last allgather group {allgather_group} for rank list {last_allgather_rank_list}")
        create_group(allgather_group, last_allgather_rank_list)
        ALLGATHER_GROUP_CACHE.append(allgather_group)
    redist_out = redist_out.contiguous()
    mint.distributed.all_gather_into_tensor(out_tensor, redist_out.astype(out_tensor.dtype),
                                            group=allgather_group, async_op=False)


# pylint: disable=W0702
def _check_shape_match(tensor, src_dtensor_info, out_tensor):
    """check whether shape of tensor is equal to out_tensor"""
    if out_tensor.shape != tensor.shape:
        try:
            src_layout = src_dtensor_info.layout.to_dict()
        except AttributeError:
            logger.error("Get src layout has error, src_layout will be set to None")
            src_layout = None
        try:
            dst_layout = out_tensor._dtensor_info.layout.to_dict()
        except AttributeError:
            logger.error("Get dst layout has error, dst_layout will be set to None")
            dst_layout = None
        raise ValueError(f"For param {out_tensor.name}, "
                         f"src shape must be equal to dst shape, but got "
                         f"src shape {tensor.shape}, "
                         f"dst shape {out_tensor.shape}. "
                         f"src layout is {src_layout}, "
                         f"dst layout is {dst_layout}")
