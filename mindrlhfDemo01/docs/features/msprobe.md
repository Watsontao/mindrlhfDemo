# 精度分析（Msprobe）

## 概述

msprobe模块为强化学习训练流程提供了配置采集、关键过程数据采集比对的能力，帮助精度问题分析和调优。

### 前置条件

安装msprobe三方库，[安装指南](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/01.installation.md)。

### 配置选项

精度分析工具通过 YAML 配置文件中的 `msprobe_config` 部分进行配置：

```yaml
msprobe_config:
  msprobe: false
  dump_path: "./msprobe_dump"
  key_data_dump: false
  configurations_dump: false
```

### 配置参数说明

| 参数 | 说明 | 可选值 |
|------|------|--------|
| msprobe | 是否使能msprobe | true/false，开启后，下列的采集项才会生效 |
| dump_path | 存盘路径 | str，默认值"./msprobe_dump" |
| key_data_dump | 关键过程数据采集 | true/false，默认false，是否采集关键过程数据，包括prompt、response、ref_log_prob、advantage、log_prob、kl_loss、loss的真实数据 |
| configurations_dump | 训练配置采集 | true/false，默认false，是否采集训练配置 |

### 落盘数据说明

```txt
msprobe_dump/
├── data/  # 训练过程关键数据
│   └── advantages/  
│   └── kl_loss/  
│   └── log_prob/
│   └── old_log_prob/
│   └── loss/  
│   └── prompts/  
│   └── ref_log_prob/  
│   └── responses/
├── configurations.json  # 训练配置文件
```

## 核心功能

### 1. 配置管理

```python
@classmethod
def config_init(cls, msprobe_config):
    """初始化 MsProbe 调试工具包"""
    if not msprobe_config.msprobe:
        return
    cls.config = msprobe_config
    cls.enabled = True
    print("msprobe enabled")
```

### 2. 配置数据保存

```python
@classmethod
def save_configs(cls, data):
    """保存配置数据到 JSON 格式"""
    if not cls.enabled:
        return
    if not cls.config.configurations_dump:
        return
    save_json(data, os.path.join(cls.config.dump_path, "configurations.json"))
```

### 3. 训练步骤标记

```python
@classmethod
def step(cls):
    """标记执行步骤用于时序调试"""
    if not cls.enabled:
        return
    from msprobe.mindspore import step
    step()
```

### 4. 张量数据保存

```python
@classmethod
def save_data(cls, name, data):
    """保存张量数据到 NPY 格式"""
    if not cls.enabled:
        return
    if not cls.config.key_data_dump:
        return
    from msprobe.mindspore import save
    save(cls.config.dump_path + "/data/" + name, name, data)
```

### 5. 字符串列表保存

```python
@classmethod
def save_string_list(cls, name, string_list):
    """保存字符串列表为序列化张量"""
    if not cls.enabled:
        return
    if not cls.config.key_data_dump:
        return

    for text in string_list:
        encoded_bytes = np.array(list(text.encode("utf-8")), dtype=np.uint8)
        text_tensor = Tensor(encoded_bytes)
        cls.save_data(name, text_tensor)
```

## 使用流程

### 1. 配置初始化

在训练脚本中添加初始化代码：

```python
def _init_msprobe(self):
    """初始化 msprobe"""
    msprobe_config = self.grpo_config.msprobe_config
    MsProbe.config_init(msprobe_config)
    MsProbe.save_configs({
        "actor": self.grpo_config.actor_config.__dict__,
        "ref": self.grpo_config.ref_config.__dict__,
        "reward": self.grpo_config.reward_config.__dict__,
        "rl": self.grpo_config.rl_config.__dict__,
    })
```

### 2. 在训练循环中收集数据

```python
def train_step(self, data):
    # 前向计算
    outputs = self.model(data)

    # 收集关键数据
    MsProbe.save_data("layer_output", outputs)

    # 计算损失
    loss = self.loss_fn(outputs)

    # 收集损失数据
    MsProbe.save_data("loss", loss)

    # 标记步骤完成
    MsProbe.step()

    return loss
```

### 3. 收集文本数据

```python
def generate_text(self, prompt):
    # 生成文本
    generated_text = self.model.generate(prompt)

    # 保存生成的文本
    MsProbe.save_string_list("generated_text", [generated_text])

    return generated_text
```

## 常见问题排查

### 1. 数据未生成

• 检查配置文件中的 msprobe 是否为 true

• 确认 key_data_dump 或 configurations_dump 已启用

• 验证训练代码中调用了数据收集方法

### 2. 导入错误

Failed to import save from msprobe.mindspore

解决方案：安装 msprobe 包：pip install mindstudio-probe

### 3. 数据类型不匹配

• 张量数据：直接使用 save_data

• 文本数据：使用 save_string_list

• 其他类型：转换为张量后保存