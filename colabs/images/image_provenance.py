# Copyright 2025 DeepMind Technologies Limited.
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

"""Stores provenance information for images used in OneTwo Colab.

This module contains metadata about images, including their external URLs and
licensing information, primarily used for image provenance tracking.
"""

import dataclasses
from typing import Dict


@dataclasses.dataclass
class ImageSource:
  """Stores provenance and licensing information for an image asset."""
  # The external URL where the image was originally found.
  external_url: str = ''
  # The local path where the image is stored in the repository.
  local_path: str = ''
  # The license under which the image is distributed.
  license: str = ''


# A constant mapping image names (keys) to their ImageSource metadata (values).
image_sources: Dict[str, ImageSource] = {
    'paligemma_fox': ImageSource(
        external_url='https://big-vision-paligemma.hf.space/file=/tmp/gradio/4aa2d3fd01a6308961397f68e043b2015bc91493/image.png',
        local_path='paligemma_fox.png',
        license=(
            'CC0 by [XiaohuaZhai@](https://sites.google.com/corp/view/xzhai)'
        ),
    ),
    'paligemma_puffin': ImageSource(
        external_url='https://big-vision-paligemma.hf.space/file=/tmp/gradio/78f93b49088f8d72ee546d656387403d647b413f/image.png',
        local_path='paligemma_puffin.png',
        license=(
            'CC0 by [XiaohuaZhai@](https://sites.google.com/corp/view/xzhai)'
        ),
    ),
}
