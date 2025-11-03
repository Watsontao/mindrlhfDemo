# Copyright 2021 Huawei Technologies Co., Ltd
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
"""GPT training wrapper"""

from mindformers.wrapper import MFPipelineWithLossScaleCell, MFTrainOneStepCell

class TrainOneStepWithLossScaleGRPO(MFTrainOneStepCell):
    """
    Encapsulation class of PanguAlpha network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_sense (Cell): Cell to do the loss scale. Default: None.
    """


class TrainPipelineWithLossScaleCellGRPO(MFPipelineWithLossScaleCell):
    """
    Encapsulation class of network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_sense (Cell): Cell to do the loss scale. Default: None.
    """
