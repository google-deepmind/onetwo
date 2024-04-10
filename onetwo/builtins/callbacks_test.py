# Copyright 2024 DeepMind Technologies Limited.
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

from collections.abc import Sequence
from typing import TypeVar

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.builtins import callbacks
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import templating


_T = TypeVar('_T')


Chunk = content_lib.Chunk
ChunkList = content_lib.ChunkList


# TODO: For now all the tests are in prompt_templating_test.py.
# We should move them here.
class CallbacksTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # This class tests various `llm` builtins. In case `import llm` is not
    # executed (this may happen when running `pytest` with multiple tests that
    # import `llm` module) various builtins from `llm` may be already configured
    # elsewhere in unexpected ways. We manually reset all the default builtin
    # implementations to make sure they are set properly.
    llm.reset_defaults()

    def generate(
        prompt: str | content_lib.ChunkList,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> str:
      del prompt, temperature, max_tokens, stop, top_k, top_p
      return ' done'

    def score(
        prompt: str | content_lib.ChunkList,
        targets: Sequence[str]
    ) -> Sequence[float]:
      del prompt
      # We score by the length of the target.
      return [float(len(target)) for target in targets]

    llm.generate_text.configure(generate)
    llm.score_text.configure(score)

  def test_generate_text(self):
    tpl = templating.JinjaTemplate(text='{{ generate_text() }}')
    tpl.register_callback(
        'generate_text', callbacks.generate_text, pass_context=True
    )
    res = executing.run(tpl.render())
    self.assertEqual(res['prefix'], ' done')


if __name__ == '__main__':
  absltest.main()
