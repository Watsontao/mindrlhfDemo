# MindRLHF 模型在线评测指南

本指南详细介绍如何将 MindRLHF 训练出的模型权重基于vllm-mindspore部署为在线服务，并基于aisbench进行自动化评测。

## 前置准备

1. **推理环境准备**：基于[mindspore日包](https://gitee.com/mindspore/mindrlhf/blob/master/.jenkins/test/config/dependent_packages.yaml)构建相互配套的推理服务环境：

```bash
# 基于日包安装核心组件（以MindSpore为例）
wget /path/to/whl
pip install mindspore-2.7.0rc1-cp311-cp311-linux_x86_64.whl
```

也可以直接通过[vllm-mindspore](https://gitee.com/mindspore/vllm-mindspore)的gitee仓构建环境：

```bash
# 卸载本身的相关环境
pip3 uninstall vllm mindspore msadapter mindspore_gs mindformers vllm-mindspore

# 安装vllm-mindspore核心组件
git clone -b r0.3.0.rc1 https://gitee.com/mindspore/vllm-mindspore.git
cd vllm-mindspore/

# 安装Python依赖
bash install_depend_pkgs.sh
pip3 install .
```

2. **评测环境准备**：由于[aisbench](https://gitee.com/aisbench/benchmark)对torch, transformers都有依赖，因此建议另起一个容器或者conda虚拟环境：

```bash
# 创建独立评测环境
conda create -n aisbench python=3.11 -y
conda activate aisbench

# 安装AISBench核心组件
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/

# 安装Python依赖
pip install nltk==3.8 pyext==0.5
pip install -e .
pip install -r requirements.txt
```

3. **模型权重准备**：MindRLHF强化学习过程保存的`.safetensor`格式格式模型权重文件，如果需要评测的权重为`.ckpt`格式，则需要先进行[权重转换](https://gitee.com/mindspore/mindrlhf/blob/master/docs/features/weight_convert.md)

## 推理服务

服务启动核心参数

| 参数 | 说明 |
|------|------|
| `model` | 模型权重路径（.safetensors格式），指定要加载的模型文件位置 |
| `host` | 服务绑定地址，指定API服务监听的网络接口，localhost：仅本地访问，0.0.0.0：开放所有网络接口 |
| `port` | 服务监听端口，指定API服务使用的TCP端口号，有效范围：1024-65535，默认值：8000 |
| `tensor-parallel-size` | 张量并行度，指定模型在多个NPU/GPU上的分片数量，必须等于物理设备数量，取值范围：1-8 |
| `max-model-len` | 最大上下文长度，指定模型能处理的最大token数量，必须≤模型预训练时的最大长度，典型值：4096（7B模型），32768（70B模型）|
| `gpu-memory-utilization` | GPU显存利用率，控制显存分配比例，取值范围：0.8-0.95，最大化吞吐量（可能增加OOM风险）|
| `trust-remote-code` |  远程代码信任标志，启用自定义模型架构加载，必须设置当使用以下情况：非标准HuggingFace模型, 自定义Attention机制, 修改过的Transformer结构 |

```bash
python -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server \
  --model /path/to/model/ \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code \
  --host localhost \
  --port 8000
```

验证方法：
启动后访问 http://localhost:8000/v1/models，应返回当前模型信息。

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "/path/to/model/",
        "messages": [{"role": "user", "content": "你好，介绍一下MindSpore"}]
      }'
```

## 评测任务

AISBench核心参数解析

| 参数 | 作用 | 示例值 |
|------|------|------|
| `models`| 评测配置模板| `vllm_api_general_chat`|
| `datasets` | 评测数据集 | `aime2025_gen_0_shot_chat_prompt`|

```bash
# 查询需要配置的数据集和模型文件
ais_bench \
  --models vllm_api_general_chat \
  --datasets aime2025_gen_0_shot_chat_prompt \
  --search

# 启动评测指令
ais_bench \
  --models vllm_api_general_chat \
  --datasets aime2025_gen_0_shot_chat_prompt \
```