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

"""Tests for Gemma backend."""

import collections
from collections.abc import Iterator
import contextlib
from typing import Final
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from gemma.deprecated import sampler as sampler_lib
from onetwo.backends import gemma_local
from onetwo.builtins import llm
from onetwo.core import core_test_utils
from onetwo.core import executing


_BATCH_SIZE: Final[int] = 1


@contextlib.contextmanager
def mock_backend(reply: str, **kwargs) -> Iterator[gemma_local.Gemma]:
  with mock.patch('gemma.deprecated.sampler.Sampler') as mock_sampler:
    with mock.patch(
        'onetwo.backends.gemma_local.Gemma._load_model'
    ) as mock_load_model:

      mock_load_model.return_value = (None, None, {'transformer': None})

      def sampler_call(
          input_strings: list[str], total_generation_steps: int
      ) -> sampler_lib.SamplerOutput:
        del input_strings, total_generation_steps
        return sampler_lib.SamplerOutput(
            text=[reply], logits=[[0.5]], tokens=[[1]]
        )

      mock_sampler.return_value.side_effect = sampler_call
      backend = gemma_local.Gemma(batch_size=_BATCH_SIZE, **kwargs)
      backend.register()
      yield backend


class GemmaTest(parameterized.TestCase, core_test_utils.CounterAssertions):

  def setUp(self):
    super().setUp()

    # This class tests various `llm` builtins. In case `import llm` is not
    # executed (this may happen when running `pytest` with multiple tests that
    # import `llm` module) various builtins from `llm` may be already configured
    # elsewhere in unexpected ways. We manually reset all the default builtin
    # implementations to make sure they are set properly.
    llm.reset_defaults()

  def test_generate_text(self):
    with mock_backend('a b') as backend:
      prompt = 'Something'
      result = executing.run(
          llm.generate_text(prompt=prompt, stop=[' '])  # pytype: disable=wrong-keyword-args
      )
      with self.subTest('returns_correct_reply'):
        self.assertEqual(result, 'a')

      expected_backend_counters = collections.Counter({
          'generate_text': 1,
          'generate_text_batches': 1,
      })

      with self.subTest('sends_correct_number_of_api_calls'):
        self.assertCounterEqual(backend._counters, expected_backend_counters)

  def test_generate_text_with_details(self):
    with mock_backend('a b') as backend:
      prompt = 'Something'
      result = executing.run(
          llm.generate_text(prompt=prompt, stop=[' '], include_details=True)  # pytype: disable=wrong-keyword-args
      )
      reply, details = result
      with self.subTest('returns_correct_reply_and_details'):
        self.assertEqual(reply, 'a')
        self.assertEqual(details[gemma_local.REPLY_TEXT], 'a b')
        self.assertLess(abs(details[gemma_local.REPLY_SCORE] - 0.5), 1e-5)

      expected_backend_counters = collections.Counter({
          'generate_text': 1,
          'generate_text_batches': 1,
      })

      with self.subTest('sends_correct_number_of_api_calls'):
        self.assertCounterEqual(backend._counters, expected_backend_counters)

if __name__ == '__main__':
  absltest.main()
