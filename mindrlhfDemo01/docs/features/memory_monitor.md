# ResourceMonitor 资源监控工具

## 概述

ResourceMonitor 是一个用于深度学习训练过程中的系统资源监控工具，专门设计用于 MindRLHF 框架。该工具提供以下核心功能：

- **实时内存监控**：持续跟踪训练进程及其子进程的内存使用情况
- **虚拟内存监控**：监控整个系统的虚拟内存使用情况
- **内存保护机制**：在内存使用超过阈值时自动终止训练进程
- **步骤级监控**：针对特定训练步骤进行精细化的资源监控
- **可视化分析**：生成内存使用趋势图表，帮助分析资源消耗模式

## 功能特性

### 1. 灵活的监控配置

- **周期性监控**：按固定时间间隔（秒）进行资源采样
- **步骤级监控**：针对特定训练步骤进行监控（支持单个步骤和范围）
- **内存保护**：设置内存使用阈值，自动终止超限进程

### 2. 全面的资源跟踪

- 记录进程内存使用历史
- 记录系统虚拟内存使用历史
- 跟踪峰值内存使用量
- 区分活动进程和空闲进程

### 3. 可视化分析

- 自动生成内存使用趋势图
- 图表包含进程内存和虚拟内存对比
- 支持自定义输出目录

### 4. 无缝集成

- 与 MindRLHF 框架原生集成
- 通过简单 API 调用控制监控
- 后台线程监控，不影响主训练流程

## 使用指南

### 初始化监控器

```python
monitor = ResourceMonitor(
    host_mention_interval=1.0,                     # 监控间隔（秒）
    host_monitor_steps=[1, [5, 10]],       # 监控步骤：第1步，5-10步
    host_memory_protection=True,      # 启用内存保护
    host_max_memory_threshold=0.90           # 内存使用阈值（90%）
)
```

### 集成到训练循环

```python
# 在训练开始时启动监控
monitor.start()

for step, data in enumerate(dataset):
    # 更新当前训练步骤
    monitor.update_current_step(step)
    # 训练步骤代码
    # ...
# 训练结束时停止监控
monitor.stop()
```

### 参数详解

| 参数名                         | 类型 | 默认值 | 描述 |
|-----------------------------|------|--------|------|
| `host_mention_interval`     | float | -1.0 | 监控间隔（秒），负值表示禁用监控 |
| `host_monitor_steps`        | list | None | 需要监控的步骤列表（支持整数和范围） |
| `host_memory_protection`    | bool | False | 是否启用内存保护 |
| `host_max_memory_threshold` | float | 0.95 | 内存保护触发阈值（0.0-1.0） |

## 工作原理

### 资源监控流程

1. **初始化监控器**：配置监控参数
2. **启动监控线程**：后台线程开始资源采样
3. **更新训练步骤**：训练循环中更新当前步骤
4. **资源采样**：按间隔收集内存数据
5. **内存保护检查**：超过阈值则终止进程
6. **停止监控**：训练结束时停止线程并生成图表

### 进程监控机制

- 识别主训练进程及其所有子进程
- 排除休眠/空闲状态的进程
- 仅监控活动进程的内存使用
- 自动清理超过10秒未见的进程记录

### 内存保护机制

当系统虚拟内存使用率超过设定阈值时：

1. 记录当前内存状态
2. 生成内存使用趋势图
3. 终止所有相关的 Python 进程
4. 防止系统因内存耗尽而崩溃

## 输出示例

### 日志输出

```text
[INFO] Virtual Memory: 32.45GB | Process Memory: 28.76GB
[INFO] Peak memory usage - Process: 29.12GB, Virtual: 34.20GB
[INFO] Memory usage visualization saved to: ./monitor_logs/memory_usage_20250722_153045.png
```

### 绘图

如果需要绘制内存变化图，可通过一下命令：

```shell
python /mindrlhf/tools/plot_host_memory.py --log /path/to/worker_0.log --output_file /path/to/host_memory.png
```