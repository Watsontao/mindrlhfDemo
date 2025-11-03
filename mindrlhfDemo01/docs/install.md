# 环境安装

MindRLHF配套环境安装可以通过源码安装和whl包安装，但由于MindRLHF涉及多个模块，各个模块不同版本之间相互依赖，直接源码安装可能会造成版本冲突，因此更推荐使用whl包安装。

MindRLHF基于MindSpore框架，依赖MindFormers提供大模型套件，并采用vllm进行推理，因此主要需要安装的whl包包括：MindSpore，vLLM，vLLM_mindspore,msadapter,mindspore_gs与ray。其中，ray可以直接通过pip安装，无需whl包。安装链接与安装顺序如下：

步骤1：

vLLM([下载链接](https://repo.mindspore.cn/mirrors/vllm/version/202505/20250514/v0.8.4.dev0_newest/any/))：

```shell
pip install vllm-0.8.4.dev0+g296c657.d20250514.empty-py3-none-any.whl
```

步骤2：

在安装完vLLM后，需要执行一下命令卸载torch torch-npu torchvision：

```shell
pip3 uninstall torch torch-npu torchvision
```

步骤3：

vLLM_mindspore([下载链接](https://repo.mindspore.cn/mindspore/vllm-mindspore/version/202506/20250606/master_20250606160020_5943579d3a76de5147d07a86f7f8ccd14ea75b51_newest/ascend/aarch64/))：

```shell
pip install vllm_mindspore-0.3.0-cp310-cp310-linux_aarch64.whl
```

步骤4：

msadapter([下载链接](https://repo.mindspore.cn/mindspore/msadapter/version/202505/20250526/master_20250526120007_b76cb7804d1c9555e32a57439c1d412ff86293d1_newest/any/))：

```shell
pip install msadapter-0.0.1-py3-none-any.whl
```

步骤5：

mindspore_gs([下载链接](https://repo.mindspore.cn/mindspore/golden-stick/version/202506/20250604/master_20250604160014_35fcbec4406d3b18faf02ef99fcbe2741e80348e_newest/any/))：

```shell
pip install mindspore_gs-1.2.0.dev20250604-py3-none-any.whl
```

步骤6：
mindspore([下载链接](https://repo.mindspore.cn/mindspore/mindspore/version/202506/20250609/master_20250609160019_8f35b18d992cacea735567ab011e91f83a074731_newest/unified/aarch64/))：

```shell
pip install mindspore-2.7.0-cp310-cp310-linux_aarch64.whl
```

步骤7：

ray:

```shell
pip install ray
```

步骤8：

在通过whl包安装完上述依赖后，需要安装MindFormers与MindRLHF，这俩个环境需要通过指定PYTHONPATH的方式运行，安装命令如下：
MindFormers([下载链接](https://gitee.com/mindspore/mindformers)):

```shell
git clone -b dev https://gitee.com/mindspore/mindformers.git
git reset --hard 6a52b43
export PYTHONPATH=/path/to/mindformers:$PYTHONPATH
```

在下载完MindFormers后，执行一下命令：

```shell
cd mindformers
bash build.sh
```

步骤9：

MindRLHF([下载链接](https://gitee.com/mindspore/mindrlhf)):

```shell
git clone https://gitee.com/mindspore/mindrlhf.git
export PYTHONPATH=/path/to/mindrlhf:$PYTHONPATH
```