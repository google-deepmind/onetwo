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

import pprint

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import caching
from onetwo.core import executing
from onetwo.core import sampling


@executing.make_executable
async def process(request: str, suffix: str = '') -> tuple[str, str]:
  key = caching.context_sampling_key.get()
  return (request+suffix, key)


class SamplingTest(parameterized.TestCase):

  def test_repeat(self):
    executable = process('test')
    executables = sampling.repeat(executable, 3)

    res = executing.run(executing.par_iter(executables))

    with self.subTest('different_sampling_keys'):
      self.assertEqual(res, [('test', ''), ('test', '1'), ('test', '2')])

  def test_repeat_multiple_times(self):
    executable = process('test')
    executables = sampling.repeat(executable, 3)
    res = list(executing.run(executing.par_iter(executables)))
    executables = sampling.repeat(executable, 3, start_index=3)
    res += executing.run(executing.par_iter(executables))
    executables = sampling.repeat(executable, 3, start_index=6)
    res += executing.run(executing.par_iter(executables))
    res_keys = [key for _, key in res]

    with self.subTest('different_sampling_keys'):
      self.assertEqual(res_keys, [''] + [str(i) for i in range(1, 9)])

  def test_nested_repeat(self):
    """We test the nesting of repeats, together with the dynamic keys."""
    # Whether we create the repeated executable before executing them (i.e.
    # statically) or while executing them (i.e. dynamically), the effect on
    # sampling keys should be the same.

    # Static creation
    static_executable = executing.serial(
        sampling.repeat_and_execute(process('test1'), 3),
        sampling.repeat_and_execute(process('test2'), 3),
    )

    # Dynamic creation
    @executing.make_executable
    async def dynamic_executable():
      return await executing.serial(
          sampling.repeat_and_execute(process('test1'), 3),
          sampling.repeat_and_execute(process('test2'), 3),
      )

    executable1 = sampling.repeat_and_execute(static_executable, 2)
    executable2 = sampling.repeat_and_execute(dynamic_executable(), 2)
    res1 = executing.run(executable1)
    res2 = executing.run(executable2)

    expected_results = [
        [
            [('test1', ''), ('test1', '1'), ('test1', '2')],
            [('test2', ''), ('test2', '1'), ('test2', '2')],
        ],
        [
            [('test1', '1#0'), ('test1', '1#1'), ('test1', '1#2')],
            [('test2', '1#0'), ('test2', '1#1'), ('test2', '1#2')],
        ],
    ]

    with self.subTest('static'):
      self.assertEqual(
          res1,
          expected_results,
          msg=pprint.pformat(res1),
      )

    with self.subTest('dynamic'):
      self.assertEqual(
          res2,
          expected_results,
          msg=pprint.pformat(res2),
      )

  def test_update(self):
    sample_size = 2

    def update_result(result, sample_id):
      return result[0], result[1], sample_id, sample_size

    executable = sampling.repeat_and_execute(
        process('test'), sample_size, update_result_fn=update_result
    )
    res = executing.run(executable)
    with self.subTest('should_update_results'):
      for i in range(sample_size):
        self.assertEqual(res[i], ('test', str(i) if i else '', i, sample_size))

  def test_streaming(self):
    executable = executing.par_iter(sampling.repeat(process('test'), 3))

    with executing.safe_stream(executable) as iterator:
      results = sum(iterator, start=executing.Update()).to_result()

    self.assertEqual(
        results,
        [('test', ''), ('test', '1'), ('test', '2')],
    )

  def test_repeat_sampler(self):
    sampler = sampling.Repeated(process)
    # Note that we pass in both a positional and a keyword argument to verify
    # that the sampler is correctly passing them through.
    res = executing.run(sampler('test', suffix='-a', num_samples=3))
    expected = [('test-a', ''), ('test-a', '1'), ('test-a', '2')]
    self.assertEqual(expected, res, res)

  def test_round_robin_sampler(self):
    @executing.make_executable
    async def strategy1(request, **kwargs):
      return await process(f'{request}-1', **kwargs)

    @executing.make_executable
    async def strategy2(request, **kwargs):
      return await process(f'{request}-2', **kwargs)

    sampler = sampling.RoundRobin([
        sampling.Repeated(strategy1),
        sampling.Repeated(strategy2),
    ])

    # Note that we pass in both a positional and a keyword argument to verify
    # that the sampler is correctly passing them through.
    res = executing.run(sampler('test', suffix='-a', num_samples=5))
    expected = [
        ('test-1-a', ''),
        ('test-2-a', '1'),
        ('test-1-a', '2'),
        ('test-2-a', '3'),
        ('test-1-a', '4'),
    ]
    with self.subTest('default_start_index_starts_at_0_with_first_sampler'):
      self.assertEqual(expected, res, res)

    # This time we omit the suffix kwarg for simplicity.
    res = executing.run(sampler(request='test', num_samples=5, start_index=1))
    expected = [
        ('test-2', '1'),
        ('test-1', '2'),
        ('test-2', '3'),
        ('test-1', '4'),
        ('test-2', '5'),
    ]
    with self.subTest('start_index_1_starts_at_1_with_second_sampler'):
      self.assertEqual(expected, res, res)


if __name__ == '__main__':
  absltest.main()
