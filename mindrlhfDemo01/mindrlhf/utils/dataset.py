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
"""Dataset"""


class GRPOIteratorStore:
    """iterator for storing data"""

    def __init__(self, store):
        self._index = 0
        self.length = len(store)
        self.store = store

    def __next__(self):
        if self._index >= self.length:
            raise StopIteration
        item = (
            self.store[self._index].prompt_completion_ids,
            self.store[self._index].responses_mask,
            self.store[self._index].ref_per_token_logps,
            self.store[self._index].advantages,
            self.store[self._index].actual_sequence_length,
            self.store[self._index].sample_index,
            self.store[self._index].sample_valid_length,
            self.store[self._index].old_per_token_logps,
        )
        self._index += 1
        return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return self.length
