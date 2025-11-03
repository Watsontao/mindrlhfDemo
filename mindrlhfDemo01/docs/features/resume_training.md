# 断点续训

断点续训功能允许在训练中断后从最近的检查点继续训练，保持训练进度的连续性。该功能通过智能管理检查点文件和训练状态实现无缝恢复。

## 核心实现机制

### 检查点元数据管理

使用 meta.json 文件记录最新检查点信息：

```python
record_last_ckpt_to_json(
    epoch=epochs,
    step=steps,
    ckpt_file=os.path.basename(ckpt_file),
    meta_json=os.path.join(rank_path, "meta.json")
)
```

### 智能检查点定位

自动查找最新有效检查点：

```python
resume_ckpt = get_resume_checkpoint_by_meta(self.sft_ckpt_path_train, formats)
ckpt_path = os.path.join(src_ckpt_file, resume_ckpt)
```

### 完整状态恢复

同时加载模型参数和优化器状态：

```python
# 加载模型参数
param_dict = ms.load_checkpoint(ckpt_path, format=formats)
load_param_to_net(self.grpo_model_train.grpo_model_train.policy_model, param_dict)

# 加载优化器状态
param_dict_opt = ms.load_checkpoint(ckpt_path_opt, format=formats)
load_param_to_net(self.grpo_with_grad.optimizer, param_dict_opt)
```

## 使用方式

### 启动恢复训练

在启动命令中添加 `--resume_training` 参数：

```bash
msrun ... \
--resume_training \
--save_strategy_dir $worker_dir/strategy \
--actor_checkpoint_path /path/to/train_checkpoints \
--ref_checkpoint_path /path/to/ref_checkpoints
```

### 关键参数说明

| 参数 | 必需 | 说明 | 示例值 |
|------|------|------|--------|
| `--resume_training` | 是 | 启用恢复训练模式 | N/A (标志参数) |
| `--save_strategy_dir` | 是 | 策略文件保存目录 | `$worker_dir/strategy` |
| `--actor_checkpoint_path` | 是 | 策略模型检查点路径 | `/path/to/train_checkpoints` |
| `--ref_checkpoint_path` | 是 | 参考模型检查点路径 | `/path/to/ref_checkpoints` |
| `--save_checkpoint_dir` | 是 | 新检查点保存目录 | `$worker_dir/save_ckpt` |

## 工作流程

### 初始化阶段：

   • 清理旧策略文件：`rm -rf $WORK_DIR/strategy`
   • 创建日志目录：`mkdir -p "${LOG_DIR}"`

### 检查点定位：

   • 读取 meta.json 确定最新检查点
   • 验证检查点完整性

### 状态恢复：

   • 加载模型参数
   • 加载优化器状态
   • 恢复训练进度（epoch/step）

### 继续训练：

   • 从上次中断处继续训练循环
   • 保存新的检查点

## 故障排查

### 常见问题解决

#### 找不到 meta.json

   ```bash
   # 检查路径是否存在
   ls -l /path/to/checkpoints/rank_0

   # 验证路径配置
   echo $LOCAL_DEFAULT_PATH
   ```

#### 参数加载失败

   ```python
   # 检查参数名称匹配
   logger.info(f"train model para names: {param.name}")

   # 手动验证检查点
   ms.load_checkpoint("/path/to/checkpoint.ckpt")
   ```

#### 分布式同步问题

   ```bash
   # 增加NCCL超时设置
   export HCCL_CONNECT_TIMEOUT=2400
   export HCCL_EXEC_TIMEOUT=7200
   ```

## 示例启动命令

```bash
# 环境初始化
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mindspore2.4.1_py310
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 恢复训练参数
WORK_DIR=/path/to/your/workdir
MSRUN_CMD="msrun ... \
    --resume_training \
    --save_strategy_dir $WORK_DIR/strategy \
    --actor_checkpoint_path $WORK_DIR/save_ckpt/train \
    --ref_checkpoint_path $WORK_DIR/save_ckpt/ref \
    --save_checkpoint_dir $WORK_DIR/new_save_ckpt"
```

### 执行训练

```bash
eval $EXECUTE_ORDER
```