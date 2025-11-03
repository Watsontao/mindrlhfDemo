<div align="center">

# MindRLHF

[![license](https://img.shields.io/github/license/mindspore-lab/mindrlhf.svg)](https://github.com/mindspore-lab/mindrlhf/blob/main/LICENSE.md)
[![open issues](https://img.shields.io/github/issues/mindspore-lab/mindrlhf)](https://github.com/mindspore-lab/mindrlhf/issues)
[![PRs](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/mindspore-lab/mindrlhf/pulls)
[![Code style: autopep8](https://img.shields.io/badge/code_style-autopep8-blue)](https://github.com/hhatto/autopep8)

[English](README.md) | 中文

[简介](#简介) |
[新闻](#新闻) |
[安装](#安装) |
[支持列表](#支持列表) |
[快速入门](#快速入门) |
[贡献](#贡献) |
[许可证](#许可证)

</div>

## 简介

OPENAI的[ChatGPT](https://openai.com/blog/chatgpt)在自然语言方面表现出了令人震惊的效果，开启了通用人工智能的序幕,它的优秀表现，与 RLHF（[Reinforcement Learning from Human Feedback](https://openai.com/research/learning-from-human-preferences)）算法密不可分。

`MindSpore RLHF`（简称 `MindRLHF`）以[MindSpore](https://gitee.com/mindspore/mindspore)作为基础框架，利用框架具备的大模型并行训练、推理、部署等能力，助力客户快速训练及部署带有百亿、千亿级别基础模型的RLHF算法流程。MindRLHF包含3个阶段的学习流程：

* 阶段1： 预训练模型训练
* 阶段2： 奖励模型训练
* 阶段3： 强化学习训练

MindRLHF集成了大模型套件[MindFormers](https://github.com/mindspore-lab/mindformers)中丰富的模型库， 提供了`Qwen2_5`等基础模型的微调流程。MindRLHF完全继承MindSpore的并行接口，可以一键将模型部署到训练集群上，开启大模型的训练和推理。

### 特性

* MindRLHF中集成了`增量推理`以提升推理性能，通过状态复用，相比于全量推理，推理性能可提升`30%`以上。
* MindRLHF组件化解耦训练流程与模型定义，支持用户自定义修改模型结构、奖励函数、训练超参等。
* 通过训练和推理权重在线快速自动重排，MindRLHF实现了训推共部署，避免权重文件落盘操作，节省离线转换保存权重文件的时间开销。
* 通过异构内存Swap技术，MindRLHF按需加载模型至显存，避免训练和推理的权重同时存在，支持更大规模模型的训练任务。

## 新闻

* [2025.3] 🚀🚀🚀 MindRLHF已实现[DeepSeek V3 + GRPO强化学习训练全流程支持](https://mp.weixin.qq.com/s?__biz=MzkxMTM2MjMzNg==&mid=2247630704&idx=1&sn=e340c4170b1ea13865fcf6325ef07d92&chksm=c0fbe4f3f595138cbf561f7f840a99a4d38b2237b67faadd3d79298365e0f42e589629bce868&scene=126&sessionid=1743684839#rd) 🚀🚀🚀
* [2025.2] 🚀🚀🚀 MindRLHF与鹏城实验室基于Qwen2.5-7B、32B打通[GRPO强化学习训练全流程](https://mp.weixin.qq.com/s/up7vYWn3NmNiW9KA_n4P4w) 🚀🚀🚀

## 安装

MindRLHF支持源码安装或通过Docker镜像安装。

### 源码安装

当前版本`0.5.0`支持源码安装：

```bash
git clone https://gitee.com/mindspore/mindrlhf.git
cd mindrlhf
pip install -e .
```

当前版本所依赖框架:

| 依赖     | 版本                                                         |
|--------|------------------------------------------------------------|
| 固件&驱动  | 24.1.RC3.3                                                 |
| CANN   | 8.1                                                        |
| Python | 3.10 or 3.11.9                                             |
| MindSpore | master, commit id：8f35b18d992cacea735567ab011e91f83a074731 |
| MindFormers | dev, commit id：6a52b43                                     |
| MindRLHF | master, commit id：0cd0874559d5658e5987ba11718a920384691c59 |
| vLLM   | master, commit id：8f35b18d992cacea735567ab011e91f83a074731 |
| vLLM_MindSpore | dev, commit id：6a52b43                                     |
| msadapter      | master, commit id：0cd0874559d5658e5987ba11718a920384691c59 |
| mindspore_gs      | master, commit id：0cd0874559d5658e5987ba11718a920384691c59 |

用户可直接执行bash install.sh安装相关依赖，如果安装失败，可参考[安装指导](./docs/install.md)。其中MindRLHF和MindFormers需要通过指定PYTHONPATH使用。

### 镜像安装

用户也可以通过Docker镜像一键安装MindRLHF和所有相关依赖：

1. 下载[Docker镜像](https://openi.pcl.ac.cn/PCL-Reasoner/GRPO-Training-Container.git)(推荐使用`git LFS`下载)。
2. 基于镜像创建容器

    ```bash
    docker run -itd  --privileged  --network=host \
           --shm-size 500g \
           --device=/dev/davinci0 \
           --device=/dev/davinci1 \
           --device=/dev/davinci2 \
           --device=/dev/davinci3 \
           --device=/dev/davinci4 \
           --device=/dev/davinci5 \
           --device=/dev/davinci6 \
           --device=/dev/davinci7 \
           --device=/dev/davinci_manager \
           --device=/dev/hisi_hdc \
           --device /dev/devmm_svm \
           -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
           -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
           -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
           -v /usr/local/sbin:/usr/local/sbin \
           -v /etc/hccn.conf:/etc/hccn.conf \
           CONTAINER_NAME:TAG \
           bash
    ```

3. 进入容器

    ```bash
    docker exec -it CONTAINER_ID bash
    ```

## 支持列表

当前版本集成了`Qwen2_5`、`DeepSeek V3`等模型，用户可以基于这些模型进行探索。未来，我们将提供更多模型如`Qwen3`等，帮助用户快速实现自己的应用。

MindSpore RLHF中不同模型对不同训练阶段的支持情况如下表所示：

表 1： MindSpore RLHF支持的模型和阶段

| 训练阶段                             | Qwen2_5  | DeepSeek V3 |
|----------------------------------|----------|-------------|
| [GRPO强化学习训练](examples/grpo)      | ✅        | ✅           |

## 快速入门

* GRPO强化学习训练: 在[`examples/grpo/qwen_grpo_tutorial/README.md`](examples/grpo/qwen_grpo_tutorial/README.md)中展示了如何利用MindRLHF中的`GRPOOTrainer`组件使用`Qwen2.5`模型进行GRPO强化学习训练的过程，包括模型与数据集获得、
  模型切分、数据集处理、配置设置以及拉起训练任务。

下面是`MindRLHF`中使用`GRPOTrainer`拉起训练任务的主要代码步骤。

```python
grpo_config = GRPOConfig()
sft_model_config_infer = LlamaConfig(**sft_config_infer.model.model_config)
sft_model_config_train = LlamaConfig(**sft_config_train.model.model_config)
ref_model_config = LlamaConfig(**ref_config.model.model_config)
tokenizer = Qwen2_5Tokenizer(args.vocab_path, args.merges_file_path, add_bos_token=False, add_eos_token=False)
trainer = GRPOTrainer(
    grpo_config=grpo_config,
    sft_model_config_infer=sft_model_config_infer,
    sft_model_config_train=sft_model_config_train,
    ref_model_config=ref_model_config,
    reward_funcs=[accuracy_reward, format_reward],
    tokenizer=tokenizer,
)
grpo_with_grad = init_grpo_network_and_optimizer(trainer)
for n in range(grpo_config.epochs):
    steps = trainer.prompt_dataset.get_dataset_size() // trainer.prompt_dataset.get_batch_size()
    for i in range(steps):
        trainer.make_experience(num_generations=grpo_config.num_generations, rank_id=rank_id)
        dataset = init_grpo_dataset(trainer)
        trainer.train(grpo_with_grad, dataset)
```

## 贡献

欢迎参与社区贡献，可参考MindSpore贡献要求Contributor Wiki。

## 许可证

Apache 2.0许可证
