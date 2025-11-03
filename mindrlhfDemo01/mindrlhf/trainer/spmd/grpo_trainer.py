# Copyright 2025 Huawei Technologies Co., Ltd
#
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
""" GRPO Trainer """
import os  # 操作系统库，用于环境变量等
import time  # 时间库，用于计时
import numpy as np  # Numpy库，用于数值计算

import mindspore as ms  # MindSpore主库
from mindspore import Tensor  # MindSpore张量
from mindspore.common.api import _pynative_executor  # PyNative执行器
from mindspore.communication import get_rank, get_group_size  # MindSpore分布式通信，获取秩和组大小
from mindspore._c_expression import MSContext  # MindSpore上下文

from mindformers import logger  # MindFormers日志
from mindformers import MindFormerConfig  # MindFormers配置
from mindformers.utils.tensorboard import get_tensorboard_writer, _set_tensorboard_writer  # TensorBoard工具

from mindrlhf.utils import (
    transfer_from_str_to_bool,  # 字符串转布尔值
    set_perf_stats,  # 设置性能统计
    TimeConsumingCollector,  # 耗时收集器
    convert_index_json_total,  # 转换safetensors索引
    profiler_start,  # 性能分析器启动
    profiler_step,  # 性能分析器打点
    set_pack_level,  # 设置打包级别
    set_infer_dp_size,  # 设置推理的数据并行大小
)
from mindrlhf.tools.host_monitor import ResourceMonitor  # 主机资源监视器
from mindrlhf.worker.infer_worker import InferWorker  # 推理工作节点
from mindrlhf.worker.ref_worker import get_ref_worker  # 参考模型工作节点
from mindrlhf.worker.train_worker import TrainWorker  # 训练工作节点
from mindrlhf.worker.old_policy_worker import get_old_policy_worker, set_enable_old_policy  # 旧策略模型工作节点
from mindrlhf.worker.transform_worker import TransformWorker  # 模型参数转换工作节点
import mindrlhf.utils.reshard_optimizer as reshard_optimizer  # 重分片优化器
from mindrlhf.configs.grpo_configs import GRPOConfig, VllmMode  # GRPO配置
from mindrlhf.models import TokenizerFactory  # Tokenizer工厂
from mindrlhf.trainer.spmd.grpo_experience_maker import GRPOExperienceMaker  # GRPO经验生成器
from mindrlhf.utils.utils import MsProbe  # MsProbe工具


class GRPOTrainer:
    """GRPO 训练器"""

    def __init__(self, no_patch_tensor_shape, args=None):
        """初始化"""
        self.args = args  # 保存命令行参数
        self._set_hccl_op_expansion_mode()  # 设置HCCL通信算子的扩展模式

        self.grpo_config = self._init_grpo_configs(args)  # 初始化GRPO配置

        self._init_msprobe()  # 初始化MsProbe

        self.tokenizer = TokenizerFactory.init_tokenizer(self.grpo_config)  # 初始化Tokenizer

        if isinstance(self.grpo_config.rl_config.seed, int):
            ms.set_seed(self.grpo_config.rl_config.seed)  # 设置随机种子

        self._set_vllm_generation_config()  # 设置vLLM生成配置

        self.no_patch_tensor_shape = no_patch_tensor_shape  # 保存传入的 no_patch_tensor_shape

        self.tensor_writer = self._init_tensorboard()  # 初始化TensorBoard写入器

        setattr(self.args, "tensor_writer", self.tensor_writer)  # 将writer附加到args上，以便其他模块使用

        self.host_monitor = ResourceMonitor(  # 初始化主机资源监视器
            self.grpo_config.monitor_config.host_monitor_interval,
            self.grpo_config.monitor_config.host_monitor_steps,
            self.grpo_config.monitor_config.host_memory_protection,
            self.grpo_config.monitor_config.host_max_memory_threshold,
        )

        self.host_monitor.start()  # 启动资源监视

        self.reshard_optimizer = None  # 初始化重分片优化器为None

        if self.grpo_config.rl_config.enable_reshard_optimizer:  # 如果启用了重分片优化
            logger.info("GRPOTrainer: start init Reshard Optimizer")
            # 初始化重分片优化器，用于处理训练和推理之间不同的并行策略
            self.reshard_optimizer = reshard_optimizer.ReshardOptimizer(
                src_parallel=reshard_optimizer.Parallel(  # 源并行策略（训练）
                    dp=self.grpo_config.actor_config.parallel_config.data_parallel,
                    tp=self.grpo_config.actor_config.parallel_config.model_parallel,
                    pp=self.grpo_config.actor_config.parallel_config.pipeline_stage,
                ),
                dst_parallel=reshard_optimizer.Parallel(  # 目标并行策略（推理）
                    dp=self.grpo_config.generate_config.parallel_config.data_parallel,
                    tp=self.grpo_config.generate_config.parallel_config.model_parallel,
                    pp=self.grpo_config.generate_config.parallel_config.pipeline_stage,
                ),
            )
            # 设置通信组
            reshard_optimizer.OPT_COMMUNICATION_GROUPS = self.reshard_optimizer.opt_communication_groups
        logger.info("GRPOTrainer: start init workers")
        # 初始化各个工作节点
        self.infer = InferWorker(grpo_config=self.grpo_config, args=self.args, tokenizer=self.tokenizer)  # 推理（生成）
        self.ref = get_ref_worker(grpo_config=self.grpo_config, args=self.args)  # 参考模型
        self.train = TrainWorker(grpo_config=self.grpo_config, args=self.args)  # 训练
        self.old_policy = get_old_policy_worker(grpo_config=self.grpo_config, args=self.args)  # 旧策略模型（用于KL散度）
        logger.info(f"config of sft_model_config_train {self.train.sft_model_config_train}")

        # 处理数据打包（Packing）配置
        if self.grpo_config.rl_config.packing:
            if self.grpo_config.rl_config.pack_num < 1:
                raise ValueError("pack_num must >= 1!")
        else:
            # 默认启用Packing，数量为1
            self.grpo_config.rl_config.packing = True
            self.grpo_config.rl_config.pack_num = 1
        logger.info("GRPOTrainer: finish init workers")

        self.reshard_mem_opt_level = self.grpo_config.rl_config.reshard_mem_opt_level  # 获取重分片内存优化级别

        if self.reshard_mem_opt_level not in [0, 1]:
            raise ValueError(f"reshard_mem_opt_level can only be 0 or 1, but got {self.reshard_mem_opt_level}")

        if self.grpo_config.rl_config.load_ckpt_format == "hf_safetensors":
            self.rename_safetensors_weights()  # 如果加载safetensors，重命名权重

        # 动态打包级别配置
        if self.grpo_config.rl_config.dynamic_pack_level:
            self.max_pack_level = self.grpo_config.rl_config.max_pack_level
            self.min_pack_level = self.grpo_config.rl_config.min_pack_level
            self.max_pack_num = 2 ** (self.max_pack_level)
        else:
            self.max_pack_level = 0
            self.min_pack_level = 0
            self.max_pack_num = 1

        self._compile()  # 编译模型

        # 初始化TransformWorker，用于在不同模型（train, infer, ref, old_policy）之间同步参数
        self.transform = TransformWorker(
            self.grpo_config,
            self.train.sft_model_config_train,
            self.train.model(),
            self.infer.model(),
            self.ref.model(),
            self.old_policy.model(),
        )
        self.i_step = 0  # 当前步骤
        self.n_epoch = 0  # 当前轮次
        self.start_step, self.start_epoch = 0, 0  # 起始步骤和轮次（用于恢复训练）
        self.total_time = 0  # 总时间
        self._load_checkpoint()  # 加载检查点
        self._init_net_parameters()  # 初始化网络参数
        if not self.grpo_config.generate_config.load:
            # 如果没有加载推理模型的特定检查点，则执行一次参数重分片（例如，从训练模型同步）
            self.transform.reshard_params(0)

        if self.grpo_config.rl_config.save_ckpt_interval <= 0:
            raise ValueError(
                f"save_ckpt_interval should be lager than 0, but got "
                f"{self.grpo_config.rl_config.save_ckpt_interval}"
            )
        self.world_group_size = get_group_size()  # 获取分布式组大小

        # 初始化经验生成器
        self.experience_maker = GRPOExperienceMaker(
            self.train,
            self.infer,
            self.ref,
            self.old_policy,
            self.grpo_config,
            self.tokenizer,
            self.tensor_writer,
            self.i_step,
        )
        self.step_num = self.experience_maker.step_num  # 获取总步数
        # 设置推理模型为非训练模式
        if self.infer.use_vllm == VllmMode.ORIGIN:
            self.infer.grpo_model_infer.grpo_model.policy_model.model.set_train(False)
        else:
            self.infer.grpo_model_infer.grpo_model.policy_model.set_train(False)
        self.infer.refresh_policy_model_phase()  # 刷新推理模型阶段

    def _init_msprobe(self):
        """初始化 msprobe"""
        msprobe_config = self.grpo_config.msprobe_config
        MsProbe.config_init(msprobe_config)  # 初始化配置
        MsProbe.save_configs(  # 保存相关配置信息
            {
                "actor": self.grpo_config.actor_config.__dict__,
                "ref": self.grpo_config.ref_config.__dict__,
                "reward": self.grpo_config.reward_config.__dict__,
                "rl": self.grpo_config.rl_config.__dict__,
                "generate": self.grpo_config.generate_config.__dict__,
            }
        )

    def _init_net_parameters(self):
        """初始化网络参数"""
        # 为所有模型（推理、旧策略、参考、训练）执行参数初始化
        if self.infer.use_vllm != VllmMode.ORIGIN:
            self.infer.grpo_model_infer.grpo_model.policy_model.model.init_parameters_data()
        else:
            self.infer.grpo_model_infer.grpo_model.policy_model.model.init_parameters_data()
        if self.grpo_config.rl_config.enable_oldpolicy:
            self.old_policy.old_policy_model.model.init_parameters_data()
        if self.grpo_config.rl_config.enable_ref:
            self.ref.ref_model.model.init_parameters_data()
        self.train.grpo_model_train.grpo_model_train.policy_model.model.init_parameters_data()

    @staticmethod
    def _set_hccl_op_expansion_mode():
        """设置HCCL通信算子扩展模式，以避免AllReduce超时"""
        soc_version = MSContext.get_instance().get_ascend_soc_version()
        if soc_version == "ascend910b":
            os.environ["HCCL_OP_EXPANSION_MODE"] = "HOST"  # 910B使用HOST模式
        elif soc_version == "ascend910_93":
            os.environ["HCCL_OP_EXPANSION_MODE"] = "AI_CPU"  # 910(93)使用AI_CPU模式
        else:
            raise NotImplementedError(
                f"only support ascend910b and ascend910_93, "
                f"but get {soc_version}"
            )

    def _init_tensorboard(self):
        """初始化 tensorboard"""
        if self.grpo_config.rl_config.tensorboard and self.grpo_config.rl_config.tensorboard_dir:
            # 为每个rank创建独立的TensorBoard日志目录
            self.grpo_config.rl_config.tensorboard_dir = os.path.join(
                self.grpo_config.rl_config.tensorboard_dir, f"rank_{get_rank()}"
            )
            _set_tensorboard_writer(self.grpo_config.rl_config)
        tensor_writer = get_tensorboard_writer()
        return tensor_writer

    def _set_vllm_generation_config(self):
        """设置vLLM生成配置的环境变量"""
        os.environ["MINDFORMERS_MODEL_CONFIG"] = self.grpo_config.generate_config.model_config

    def __del__(self):
        """析构函数，清理环境变量"""
        if os.getenv("MINDFORMERS_MODEL_CONFIG"):
            del os.environ["MINDFORMERS_MODEL_CONFIG"]

    @staticmethod
    def _set_args_to_config(args, grpo_config: GRPOConfig):
        """将命令行参数 (args) 设置到 grpo_config 配置对象中"""
        # 逐一检查
        if args.dataset_file is not None:
            grpo_config.rl_config.dataset_file = args.dataset_file
        if args.tokenizer_dir is not None:
            grpo_config.rl_config.tokenizer_dir = args.tokenizer_dir
        if args.actor_checkpoint_path is not None:
            grpo_config.actor_config.load = args.actor_checkpoint_path
        if args.ref_checkpoint_path is not None:
            grpo_config.ref_config.load = args.ref_checkpoint_path
        if args.generate_checkpoint_path is not None:
            grpo_config.generate_config.load = args.generate_checkpoint_path
        if args.verifier_function is not None:
            if "," in args.verifier_function:
                verifier_function = args.verifier_function.split(",")
            else:
                verifier_function = [args.verifier_function]
            grpo_config.reward_config.verifier_function = verifier_function
        if args.verifier_weight is not None:
            if "," in args.verifier_weight:
                verifier_weight = args.verifier_weight.split(",")
                verifier_weight = [float(_) for _ in verifier_weight]
            else:
                verifier_weight = [float(args.verifier_weight)]
            grpo_config.reward_config.verifier_weight = verifier_weight
        if args.tensorboard is not None:
            tensorboard = transfer_from_str_to_bool(args.tensorboard)
            grpo_config.rl_config.tensorboard = tensorboard
        if args.save_checkpoint_dir is not None:
            grpo_config.actor_config.save = args.save_checkpoint_dir
        if args.model_name:
            grpo_config.rl_config.model_name = args.model_name
        if args.tokenizer_type:
            if args.model_name and args.model_name != args.tokenizer_type:
                logger.warning(
                    f"tokenizer_type [{args.tokenizer_type}] is different from "
                    f"model_name [{args.model_name}], tokenizer init will use "
                    f"tokenizer_type [{args.tokenizer_type}]"
                )
            grpo_config.rl_config.tokenizer_type = args.tokenizer_type
        else:
            logger.info(
                "tokenizer_type is unset, " f"set tokenizer_type as model_name [{grpo_config.rl_config.model_name}]"
            )
            grpo_config.rl_config.tokenizer_type = grpo_config.rl_config.model_name
        return grpo_config

    def _init_grpo_configs(self, args):
        """初始化 GRPO 配置"""
        logger.info(f"GRPOTrainer: _init_grpo_configs {args} in main task")
        # 1. 从配置文件加载基础配置
        grpo_config = GRPOConfig(args.config)
        # 2. 使用命令行参数覆盖配置
        grpo_config = self._set_args_to_config(args, grpo_config)
        # 3. 设置相关工具和标志位
        set_perf_stats(grpo_config)
        set_enable_old_policy(grpo_config)
        set_infer_dp_size(grpo_config)
        set_pack_level(grpo_config)
        # 如果 beta (KL散度系数) 不为0，则需要启用参考模型
        grpo_config.rl_config.enable_ref = grpo_config.rl_config.beta != 0
        if grpo_config.generate_config.use_vllm not in range(len(VllmMode)):
            logger.warning(f"use_vllm should be 0, 1 or 2, but got {grpo_config.generate_config.use_vllm}. Reset to 0.")
            grpo_config.generate_config.use_vllm = 0
        grpo_config.generate_config.use_vllm = VllmMode(grpo_config.generate_config.use_vllm)
        logger.info(
            f"vllm mode: {grpo_config.generate_config.use_vllm}, "
            f"hf_config_path: {grpo_config.generate_config.hf_config_path}"
        )
        if (
                grpo_config.rl_config.save_prompt_completions_data
                and grpo_config.rl_config.save_prompt_completions_interval <= 0
        ):
            logger.warning(
                f"save_prompt_completions_interval should be positive, "
                f"but got {grpo_config.rl_config.save_prompt_completions_interval}. "
                f"Set save_prompt_completions_data to False."
            )
            grpo_config.rl_config.save_prompt_completions_data = False
        return grpo_config

    @property
    def use_parallel(self):
        """是否使用并行"""
        return transfer_from_str_to_bool(self.grpo_config.rl_config.use_parallel)

    def _compile(self):
        """
        编译模型
        """
        with TimeConsumingCollector("GRPOTrainer compile"):
            self.infer.generate_strategy(self.reshard_optimizer)  # 生成推理策略
            origin_shape = Tensor.shape  # 备份原始的 Tensor.shape
            Tensor.shape = self.no_patch_tensor_shape  # 替换 Tensor.shape (可能是为了解决编译时的特定问题)

            # 编译各个模型
            self.ref.compile()
            self.old_policy.compile()
            # 训练模型需要针对不同的打包级别进行编译
            for i in range(self.max_pack_level + 1 - self.min_pack_level):
                self.train.compile(i)
                logger.info(f"train compile for pack level {i} done!")

            Tensor.shape = origin_shape  # 恢复原始的 Tensor.shape

    def _load_checkpoint(self):
        """
        加载检查点文件
        """
        if self.args.resume_training:  # 如果是恢复训练
            logger.info("Resuming training from checkpoint...")
            epoch_step_info = self.train.reload_ckpt()  # 加载训练检查点（包含模型和优化器状态）
            if epoch_step_info is None:
                raise ValueError("epoch/step info not read")

            if self.grpo_config.ref_config.sync_ref_model:
                self.ref.reload_ckpt()  # 如果参考模型需要同步，也加载其状态
            else:
                self.ref.load_checkpoint()  # 否则加载初始检查点

            # 从设备卸载模型到内存，为参数重分片做准备
            self.train.offload_model()
            self.infer.offload()
            self.old_policy.offload()

            # 定义参数重分片时，各个模型是否在设备上
            input_on_device_flag_dict = {
                "policy2infer": (False, False),
                "policy2ref": (True, True),
                "policy2old": (False, False),
            }
            # 执行重分片，将加载的训练模型参数同步到推理、参考和旧策略模型
            self.transform.reshard_params(0, input_on_device_flag_dict=input_on_device_flag_dict)
            self.infer.load(skip_kv_cache=False)  # 将推理模型加载回设备

            # 恢复训练的轮次和步数
            epoch_num = epoch_step_info["epoch_num"]
            data_skip_steps = epoch_step_info["step_num"]
            if epoch_num > 0:
                logger.info(f"epoch in resume training is: {epoch_num}.")
                self.n_epoch = epoch_num
                self.start_epoch = epoch_num
            if data_skip_steps > 0:
                logger.info(f"Skip step in resume training is: {data_skip_steps}.")
                self.i_step = data_skip_steps
                self.start_step = data_skip_steps
            return

        # 如果不是恢复训练，则正常加载各自的初始检查点
        with TimeConsumingCollector("GRPOTrainer load checkpoint"):
            self.infer.load_checkpoint()
            self.ref.load_checkpoint()
            self.old_policy.load_checkpoint()
            self.train.load_checkpoint()

    def _reshard_train_to_infer(self):
        """将训练模型 (TrainWorker) 的参数重分片（同步）到推理模型 (InferWorker) 和其他模型"""

        # 根据内存优化级别，决定模型在重分片时是加载在设备上还是在内存中
        if self.reshard_mem_opt_level == 1:  # 级别1：内存优化（模型在CPU）
            self.train.offload_model()  # 确保训练模型在CPU
            if self.train.model_on_device:
                raise RuntimeError(
                    "when reshard_mem_opt_level is equal to 1, train model must not on device before transform param"
                )
            if self.infer.on_device:
                raise RuntimeError(
                    "when reshard_mem_opt_level is equal to 1, infer model must not on device before transform param"
                )
            self.old_policy.check_not_on_device()
        else:  # 级别0：速度优化（模型在Device）
            self.infer.load(skip_kv_cache=True)  # 加载推理模型到Device（跳过KV Cache）
            self.old_policy.load()  # 加载旧策略模型到Device
            if not self.train.model_on_device:
                raise RuntimeError(
                    "when reshard_mem_opt_level is equal to 0, train model must on device before transform param"
                )
            if not self.infer.on_device:
                raise RuntimeError(
                    "when reshard_mem_opt_level is equal to 0, infer model must on device before transform param"
                )

        # 检查是否需要同步参考模型
        if self.transform.sync_ref_model and ((self.i_step + 1) % self.transform.ref_model_sync_steps == 0):
            if self.reshard_mem_opt_level == 0:
                self.ref.load()  # 级别0时，加载参考模型到Device
            input_on_device_flag_dict = {
                "policy2infer": (self.train.model_on_device, self.infer.on_device),
                "policy2ref": (self.train.model_on_device, self.ref.on_device),  # 同步到Ref
                "policy2old": (self.train.model_on_device, self.old_policy.on_device),
            }
            self.transform.reshard_params(self.i_step, input_on_device_flag_dict)  # 执行重分片
            if self.reshard_mem_opt_level == 0:
                self.ref.offload()  # 卸载参考模型
        else:
            # 仅同步到Infer和Old Policy
            input_on_device_flag_dict = {
                "policy2infer": (self.train.model_on_device, self.infer.on_device),
                "policy2ref": (self.train.model_on_device, self.ref.on_device),  # 即使不同步，也可能需要此标志
                "policy2old": (self.train.model_on_device, self.old_policy.on_device),
            }
            self.transform.reshard_params(self.i_step, input_on_device_flag_dict)  # 执行重分片

        # 重分片后的状态管理
        if self.reshard_mem_opt_level == 0:  # 级别0
            if not self.train.model_on_device:
                raise RuntimeError("...")
            if not self.infer.on_device:
                raise RuntimeError("...")
            self.train.offload_model()  # 卸载训练模型（已同步完毕）
            self.old_policy.check_on_device()
            self.old_policy.offload()  # 卸载旧策略模型（已同步完毕）
            self.infer.init_kvcache()  # 为推理模型初始化KV Cache，准备生成
        else:  # 级别1
            if self.train.model_on_device:
                raise RuntimeError("...")
            if self.infer.on_device:
                raise RuntimeError("...")
            self.old_policy.check_not_on_device()

    def run_grpo_train(self):
        """
        MindRLHF GRPO 训练的主入口。
        """
        logger.info(
            f"Start training epoch num:{self.grpo_config.rl_config.epochs}, step num:{self.step_num}, "
            f"generation num:{self.grpo_config.rl_config.num_generations}"
        )
        np.set_printoptions(threshold=1024)  # 设置Numpy打印选项

        # --- 轮次循环 (Epoch) ---
        while self.n_epoch < self.grpo_config.rl_config.epochs:
            grpo_profiler = profiler_start(self.grpo_config.profiler_config, role="grpo_all_stage")

            # --- 步骤循环 (Step) ---
            while self.i_step < self.step_num:
                # 如果同步参考模型，并在保存间隔时，保存参考模型
                if (
                        self.grpo_config.ref_config.sync_ref_model
                        and self.i_step % self.grpo_config.rl_config.save_ckpt_interval == 0
                ):
                    with TimeConsumingCollector("before save ref offload infer"):
                        self.infer.offload()  # 卸载推理模型以释放显存
                    with TimeConsumingCollector("save ref model"):
                        self.ref.save_checkpoints(  # 保存参考模型
                            epochs=self.n_epoch,
                            steps=self.i_step,
                            start_epoch=self.start_epoch,
                            start_step=self.start_step,
                        )
                self.host_monitor.update_current_step(self.i_step)  # 更新资源监视器
                logger.info(f"epoch: {self.n_epoch}, step: {self.i_step} start")

                with TimeConsumingCollector(f"whole epoch {self.n_epoch} train stage") as perf_collector:
                    # --- 阶段 1: 生成经验 (Rollout) ---
                    with TimeConsumingCollector("make_experience"):
                        self.experience_maker.make_experience(num_rollouts=self.grpo_config.rl_config.num_rollouts,
                                                              num_generations=self.grpo_config.rl_config.num_generations,)

                    # --- 阶段 2: 准备训练 (Load) ---
                    with TimeConsumingCollector("load train optimizer"):
                        self.train.load_optimizer()  # 加载优化器状态（例如，从CPU到Device）
                    with TimeConsumingCollector("load train model"):
                        self.train.load_model()  # 加载训练模型（例如，从CPU到Device）
                    with TimeConsumingCollector("load accu_grads"):
                        self.train.load_accu_grads()  # 加载累积梯度

                    # --- 阶段 3: 执行训练 (Train) ---
                    with TimeConsumingCollector("train model"):
                        update_profiler = profiler_start(
                            self.grpo_config.profiler_config, role="actor_update", profiler_iteration=self.n_epoch
                        )
                        self.train.train()  # 执行一步或多步训练
                        profiler_step(update_profiler)

                    # --- 阶段 4: 保存检查点 (Save) ---
                    if (self.i_step + 1) % self.grpo_config.rl_config.save_ckpt_interval == 0:
                        with TimeConsumingCollector("save train model and optimizer"):
                            self.train.save_checkpoints(  # 保存训练检查点（模型+优化器）
                                epochs=self.n_epoch,
                                steps=self.i_step + 1,
                                start_epoch=self.start_epoch,
                                start_step=self.start_step,
                            )

                    # --- 阶段 5: 训练后卸载 (Offload) ---
                    with TimeConsumingCollector("offload train optimizer"):
                        self.train.offload_optimizer()  # 卸载优化器状态（例如，从Device到CPU）
                    with TimeConsumingCollector("offload accu_grads"):
                        self.train.offload_accu_grads()  # 卸载累积梯度

                    # --- 阶段 6: 同步参数 (Reshard) ---
                    with TimeConsumingCollector("reshard train to infer"):
                        # 将刚训练好的策略权重同步到推理模型和旧策略模型
                        self._reshard_train_to_infer()

                self.total_time += perf_collector.duration  # 累积总时间
                # 打印性能日志
                logger.info(
                    "step processed tokens {}, tokens/s/p {}".format(
                        self.experience_maker.step_total_tokens,
                        self.experience_maker.step_total_tokens / perf_collector.duration / self.world_group_size,
                    )
                )
                logger.info(
                    "total processed tokens {}, total tokens/s/p {}".format(
                        self.experience_maker.total_processed_tokens,
                        self.experience_maker.total_processed_tokens / self.total_time / self.world_group_size,
                    )
                )
                self.i_step += 1  # 步骤+1
                MsProbe.step()
                profiler_step(grpo_profiler)
            # --- 步骤循环结束 ---
            self.i_step = 0  # 重置步骤计数器
            self.n_epoch += 1  # 轮次+1
        # --- 轮次循环结束 ---

        # 训练结束后，保存最终的检查点
        with TimeConsumingCollector("save checkpoint"):
            with TimeConsumingCollector("load train model"):
                self.train.load_model()
            self.train.save_checkpoints(epochs=self.grpo_config.rl_config.epochs, steps=self.step_num)
        self.host_monitor.stop()  # 停止资源监视器
        logger.info("run grpo train end")

    def rename_safetensors_weights(self):
        """重命名 safetensors 权重并写入 param_name_map.json"""
        # 假设所有模型加载相同的safetensors文件，使用actor_config处理
        config = MindFormerConfig(self.grpo_config.actor_config.model_config)
        config.load_checkpoint = self.grpo_config.actor_config.load

        if config.model.model_config.get("qkv_concat", False):
            raise ValueError("safetensors only support qkv_concat=False for now")

        if get_rank() == 0:  # 仅 rank 0 执行转换
            convert_func_lst = []
            # 收集所有需要加载此权重的模型的参数映射表
            convert_func_lst.append(self.infer.convert_map_dict)
            if self.grpo_config.rl_config.enable_ref:
                convert_func_lst.append(self.ref.convert_map_dict)
            if self.grpo_config.rl_config.enable_oldpolicy:
                convert_func_lst.append(self.old_policy.convert_map_dict)
            convert_func_lst.append(self.train.convert_map_dict)

            # 调用工具函数，根据所有模型的参数名映射，转换safetensors的index.json文件
            convert_index_json_total(config.load_checkpoint, config.load_checkpoint, convert_func_lst, False)
        else:
            # 其他 rank 等待 rank 0 完成
            time.sleep(10)
        ms.mint.distributed.barrier()  # 分布式屏障，等待所有进程
        _pynative_executor.sync()  # 同步执行器