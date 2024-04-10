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

from collections.abc import AsyncIterator, Sequence
import pprint

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import batching
from onetwo.core import executing
from onetwo.core import routing
from onetwo.core import utils


make_executable = executing.make_executable


class ReplyForTest:

  def __init__(self, content):
    self.content = content

  def __str__(self):
    return self.content

  def __repr__(self):
    return self.__str__()

  def __eq__(self, other):
    return self.__str__() == other.__str__()


class RequestForTest:

  def __init__(self, engine, content):
    self.engine = engine
    self.content = content

  def __str__(self):
    return f'{self.engine} {self.content}'

  def __repr__(self):
    return f"RequestForTest('{self.engine}', '{self.content}')"

  def __eq__(self, other):
    return self.__str__() == other.__str__()


@batching.add_batching
class EngineForTest:

  def __init__(self, name, batch_size=3, out_of_order=False):
    self.name = name
    self.batch_size = batch_size
    self.batches = []
    self.out_of_order = out_of_order
    self.skip = False

  def register(self, name: str | None = None):
    del name
    registry = routing.function_registry
    registry[self.name] = self.execute

  def batching_function(
      self,
      batch: Sequence[RequestForTest],
      request: RequestForTest,
  ) -> tuple[bool, bool]:
    """See base class."""
    del request
    if not batch:
      # We always add the first request to the batch.
      return True, self.batch_size == 1

    if len(batch) >= self.batch_size:
      # Batch is full, we cannot add anything.
      return False, True

    if not self.out_of_order:
      return True, len(batch) >= self.batch_size - 1
    else:
      if self.skip:
        self.skip = False
        return False, len(batch) >= self.batch_size
      else:
        self.skip = True
        return True, len(batch) >= self.batch_size - 1

  async def execute(self, request: RequestForTest) -> ReplyForTest:
    replies = await self._process_requests([request])
    return replies[0]

  @batching.batch_method(
      batching_function=utils.FromInstance('batching_function')
  )
  def _process_requests(
      self,
      requests: Sequence[RequestForTest]
  ) -> list[ReplyForTest]:
    """See base class."""
    self.batches.append(requests)
    replies = [
        ReplyForTest(r.content.replace('req', 'rep'))  # pytype: disable=attribute-error
        for r in requests
    ]
    return replies


def plan(num_requests, engine_from_id):
  @make_executable
  async def send_request(engine, request):
    return await routing.function_registry[engine](request)

  requests = []
  for i in range(num_requests):
    llm_id = engine_from_id(i)
    requests.append(
        send_request(
            f'llm{llm_id}',
            RequestForTest(f'llm{llm_id}', f'req {i} for llm{llm_id}'),
        )
    )
  return executing.parallel(*requests)


class RoutingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # This class tests routing.function_registry. In case `import routing` is
    # not executed (this may happen when running `pytest` with multiple tests
    # that import `llm` module) the `function_registry` may be already filled
    # with various functions elsewhere in unexpected ways. We manually remove
    # all the keys to make sure it is empty.
    # TODO:` or such
    # for better control of reproducibility.
    routing.function_registry.clear()

  @parameterized.named_parameters(
      (
          'even_batches',
          10,
          lambda x: 1 + (x % 2),
          [
              [
                  [
                      RequestForTest('llm1', 'req 0 for llm1'),
                      RequestForTest('llm1', 'req 2 for llm1'),
                      RequestForTest('llm1', 'req 4 for llm1'),
                  ],
                  [
                      RequestForTest('llm1', 'req 6 for llm1'),
                      RequestForTest('llm1', 'req 8 for llm1'),
                  ],
              ],
              [
                  [
                      RequestForTest('llm2', 'req 1 for llm2'),
                      RequestForTest('llm2', 'req 3 for llm2'),
                      RequestForTest('llm2', 'req 5 for llm2'),
                  ],
                  [
                      RequestForTest('llm2', 'req 7 for llm2'),
                      RequestForTest('llm2', 'req 9 for llm2'),
                  ],
              ],
          ],
          [
              ReplyForTest('rep 0 for llm1'),
              ReplyForTest('rep 1 for llm2'),
              ReplyForTest('rep 2 for llm1'),
              ReplyForTest('rep 3 for llm2'),
              ReplyForTest('rep 4 for llm1'),
              ReplyForTest('rep 5 for llm2'),
              ReplyForTest('rep 6 for llm1'),
              ReplyForTest('rep 7 for llm2'),
              ReplyForTest('rep 8 for llm1'),
              ReplyForTest('rep 9 for llm2'),
          ],
      ),
      (
          'uneven_batches',
          10,
          lambda x: 1 + (x % 2),
          [
              [
                  [
                      RequestForTest('llm1', 'req 0 for llm1'),
                      RequestForTest('llm1', 'req 2 for llm1'),
                      RequestForTest('llm1', 'req 4 for llm1'),
                  ],
                  [
                      RequestForTest('llm1', 'req 6 for llm1'),
                      RequestForTest('llm1', 'req 8 for llm1'),
                  ],
              ],
              [
                  [
                      RequestForTest('llm2', 'req 1 for llm2'),
                      RequestForTest('llm2', 'req 3 for llm2'),
                      RequestForTest('llm2', 'req 5 for llm2'),
                  ],
                  [
                      RequestForTest('llm2', 'req 7 for llm2'),
                      RequestForTest('llm2', 'req 9 for llm2'),
                  ],
              ],
          ],
          [
              ReplyForTest('rep 0 for llm1'),
              ReplyForTest('rep 1 for llm2'),
              ReplyForTest('rep 2 for llm1'),
              ReplyForTest('rep 3 for llm2'),
              ReplyForTest('rep 4 for llm1'),
              ReplyForTest('rep 5 for llm2'),
              ReplyForTest('rep 6 for llm1'),
              ReplyForTest('rep 7 for llm2'),
              ReplyForTest('rep 8 for llm1'),
              ReplyForTest('rep 9 for llm2'),
          ],
      ),
      (
          'more_for_engine1',
          10,
          lambda x: 1 if x < 7 else 2,
          [
              [
                  [
                      RequestForTest('llm1', 'req 0 for llm1'),
                      RequestForTest('llm1', 'req 1 for llm1'),
                      RequestForTest('llm1', 'req 2 for llm1'),
                  ],
                  [
                      RequestForTest('llm1', 'req 3 for llm1'),
                      RequestForTest('llm1', 'req 4 for llm1'),
                      RequestForTest('llm1', 'req 5 for llm1'),
                  ],
                  [
                      RequestForTest('llm1', 'req 6 for llm1'),
                  ],
              ],
              [[
                  RequestForTest('llm2', 'req 7 for llm2'),
                  RequestForTest('llm2', 'req 8 for llm2'),
                  RequestForTest('llm2', 'req 9 for llm2'),
              ]],
          ],
          [
              ReplyForTest('rep 0 for llm1'),
              ReplyForTest('rep 1 for llm1'),
              ReplyForTest('rep 2 for llm1'),
              ReplyForTest('rep 3 for llm1'),
              ReplyForTest('rep 4 for llm1'),
              ReplyForTest('rep 5 for llm1'),
              ReplyForTest('rep 6 for llm1'),
              ReplyForTest('rep 7 for llm2'),
              ReplyForTest('rep 8 for llm2'),
              ReplyForTest('rep 9 for llm2'),
          ],
      ),
      (
          'missing_engine1',
          10,
          lambda x: 2,
          [
              [],
              [
                  [
                      RequestForTest('llm2', 'req 0 for llm2'),
                      RequestForTest('llm2', 'req 1 for llm2'),
                      RequestForTest('llm2', 'req 2 for llm2'),
                  ],
                  [
                      RequestForTest('llm2', 'req 3 for llm2'),
                      RequestForTest('llm2', 'req 4 for llm2'),
                      RequestForTest('llm2', 'req 5 for llm2'),
                  ],
                  [
                      RequestForTest('llm2', 'req 6 for llm2'),
                      RequestForTest('llm2', 'req 7 for llm2'),
                      RequestForTest('llm2', 'req 8 for llm2'),
                  ],
                  [
                      RequestForTest('llm2', 'req 9 for llm2'),
                  ],
              ],
          ],
          [
              ReplyForTest('rep 0 for llm2'),
              ReplyForTest('rep 1 for llm2'),
              ReplyForTest('rep 2 for llm2'),
              ReplyForTest('rep 3 for llm2'),
              ReplyForTest('rep 4 for llm2'),
              ReplyForTest('rep 5 for llm2'),
              ReplyForTest('rep 6 for llm2'),
              ReplyForTest('rep 7 for llm2'),
              ReplyForTest('rep 8 for llm2'),
              ReplyForTest('rep 9 for llm2'),
          ],
      ),
  )
  def test_running_with_registry(
      self,
      num_requests,
      engine_from_id,
      expected_batches,
      expected_results,
  ):
    engine1 = EngineForTest('llm1')
    engine2 = EngineForTest('llm2')
    engine1.register('llm1')
    engine2.register('llm2')
    results = executing.run(plan(num_requests, engine_from_id))

    with self.subTest('should_return_enough_results'):
      self.assertLen(results, num_requests)

    with self.subTest('should_have_correct_batches'):
      self.assertEqual(
          engine1.batches, expected_batches[0], pprint.pformat(engine1.batches)
      )
      self.assertEqual(
          engine2.batches, expected_batches[1], pprint.pformat(engine2.batches)
      )

    with self.subTest('should_have_correct_results'):
      self.assertEqual(results, expected_results)

  @parameterized.named_parameters(
      (
          'more_for_engine1',
          10,
          lambda x: 1 if x < 7 else 2,
          [
              [
                  [
                      RequestForTest('llm1', 'req 0 for llm1'),
                      RequestForTest('llm1', 'req 1 for llm1'),
                      RequestForTest('llm1', 'req 3 for llm1'),
                  ],
                  [
                      RequestForTest('llm1', 'req 2 for llm1'),
                      RequestForTest('llm1', 'req 5 for llm1'),
                  ],
                  [
                      RequestForTest('llm1', 'req 4 for llm1'),
                      RequestForTest('llm1', 'req 6 for llm1'),
                  ],
              ],
              [
                  [
                      RequestForTest('llm2', 'req 7 for llm2'),
                      RequestForTest('llm2', 'req 8 for llm2'),
                  ],
                  [
                      RequestForTest('llm2', 'req 9 for llm2'),
                  ],
              ],
          ],
          [
              ReplyForTest('rep 0 for llm1'),
              ReplyForTest('rep 1 for llm1'),
              ReplyForTest('rep 2 for llm1'),
              ReplyForTest('rep 3 for llm1'),
              ReplyForTest('rep 4 for llm1'),
              ReplyForTest('rep 5 for llm1'),
              ReplyForTest('rep 6 for llm1'),
              ReplyForTest('rep 7 for llm2'),
              ReplyForTest('rep 8 for llm2'),
              ReplyForTest('rep 9 for llm2'),
          ],
      ),
  )
  def test_out_of_order_running(
      self,
      num_requests,
      engine_from_id,
      expected_batches,
      expected_results,
  ):
    engine1 = EngineForTest('llm1', out_of_order=True)
    engine2 = EngineForTest('llm2', out_of_order=True)
    engine1.register('llm1')
    engine2.register('llm2')
    results = executing.run(plan(num_requests, engine_from_id))

    with self.subTest('should_return_enough_results'):
      self.assertLen(results, num_requests)

    with self.subTest('should_have_correct_batches'):
      self.assertEqual(
          engine1.batches, expected_batches[0], pprint.pformat(engine1.batches)
      )
      self.assertEqual(
          engine2.batches, expected_batches[1], pprint.pformat(engine2.batches)
      )

    with self.subTest('should_have_correct_results'):
      self.assertEqual(results, expected_results, pprint.pformat(results))

  def test_function_call(self):
    def simple_function(text: str, request: RequestForTest) -> ReplyForTest:
      return ReplyForTest(text + str(request))

    engine = EngineForTest('llm')
    engine.register()
    routing.function_registry['fn'] = simple_function

    with self.subTest('call_normal_function'):
      result = executing.run(
          routing.function_registry(
              'fn', text='test', request=RequestForTest('', 'req')
          )
      )
      self.assertEqual(result, 'test req')

    with self.subTest('call_async_function'):
      result = executing.run(
          routing.function_registry('llm', request=RequestForTest('', 'req'))
      )
      self.assertEqual(result, 'rep')

  def test_register_and_call(self):

    def f(i: int) -> str:
      return f'f:{i}'

    def f_with_varied_args(a: int, b: int, /, c: int, *, d: int) -> str:
      return f'f:{a},{b},{c},{d}'

    routing.function_registry['f'] = f
    routing.function_registry['f2'] = f_with_varied_args

    with self.subTest('should_return_correct_result'):
      result = executing.run(routing.function_registry('f', i=0))
      self.assertEqual(result, 'f:0')

      result = executing.run(routing.function_registry('f2', 0, 1, 2, d=3))
      self.assertEqual(result, 'f:0,1,2,3')
      result = executing.run(routing.function_registry('f2', 0, 1, c=2, d=3))
      self.assertEqual(result, 'f:0,1,2,3')

    with self.subTest('fails_to_get_unknown_key'):
      with self.assertRaises(KeyError):
        _ = executing.run(routing.function_registry('unregistered', i=0))

    with self.subTest('get_unknown_key'):
      self.assertIsNone(routing.function_registry.get('unregistered', None))

  def test_registry_update(self):

    def f1(i: int) -> str:
      return f'f1:{i}'

    def f2(i: int) -> str:
      return f'f2:{i}'

    @executing.make_executable
    async def f_call(i: int) -> str:
      return routing.function_registry('f', i=i)

    @executing.make_executable
    async def wrapper():
      reply1 = await f_call(0)
      with routing.RegistryContext():
        routing.function_registry['f'] = f2
        reply2 = await f_call(1)
      reply3 = await f_call(2)
      return [reply1, reply2, reply3]

    routing.function_registry['f'] = f1
    result = executing.run(wrapper())
    expected_result = ['f1:0', 'f2:1', 'f1:2']

    with self.subTest('should_return_correct_results'):
      self.assertListEqual(result, expected_result, pprint.pformat(result))

  def test_config_registry_update(self):
    with routing.RegistryContext():
      routing.config_registry.get()['key1'] = 'value1'
      self.assertEqual(routing.config_registry.get()['key1'], 'value1')

    with self.assertRaises(KeyError):
      _ = routing.config_registry.get()['key1']

  def test_registry_update_with_yield(self):
    def f1(i: int) -> str:
      return f'f1:{i}'

    def f2(i: int) -> str:
      return f'f2:{i}'

    @executing.make_executable
    async def call(i: int) -> AsyncIterator[str]:
      for _ in range(3):
        yield await routing.function_registry('f', i=i)

    @executing.make_executable
    async def register_and_call(fn: int, i: int) -> AsyncIterator[str]:
      with routing.RegistryContext():
        routing.function_registry['f'] = f1 if fn == 1 else f2
        for _ in range(3):
          yield await routing.function_registry('f', i=i)

    @executing.make_executable
    async def register_each_time(fn: int, i: int) -> AsyncIterator[str]:
      for _ in range(3):
        with routing.RegistryContext():
          routing.function_registry['f'] = f1 if fn == 1 else f2
          res = await routing.function_registry('f', i=i)
        yield res

    def process(executable):
      with executing.safe_stream(executable, iteration_depth=-1) as stream:
        result = []
        for update in stream:
          result.append(update.to_simplified_result())
      return result

    with self.subTest('should_raise_value_error'):
      executable = executing.parallel(
          register_and_call(1, 1),
          register_and_call(2, 2),
      )
      # The function `register_and_call`` does a yield within a
      # RegistryContext() context manager, this means the registry might be
      # changed during the alternate iterations and this will lead to the
      # context variable not being properly set.
      with self.assertRaises(ValueError):
        _ = process(executable)

    # When using `register_each_time`, we set the RegistryContext before calling
    # the executable which is the right way.
    executable = executing.parallel(
        register_each_time(1, 1), register_each_time(2, 2)
    )

    result = process(executable)
    expected_result = [
        ['f1:1'],
        ['f2:2'],
        ['f1:1'],
        ['f2:2'],
        ['f1:1'],
        ['f2:2'],
    ]

    results_with_wrong_registry = [
        ['f1:1'],
        ['f1:2'],
        ['f1:1'],
        ['f1:2'],
        ['f1:1'],
        ['f1:2'],
    ]

    with self.subTest('parallelize_and_alternate_registers'):
      self.assertListEqual(result, expected_result, pprint.pformat(result))

    routing.function_registry['f'] = f1
    executable = executing.parallel(call(1), call(2))
    result = process(executable)
    with self.subTest('register_once_does_not_use_the_right_registry'):
      self.assertListEqual(
          result, results_with_wrong_registry, pprint.pformat(result)
      )

    routing.function_registry['f'] = f1
    registry1 = routing.copy_registry()

    routing.function_registry['f'] = f2
    registry2 = routing.copy_registry()

    executable = executing.parallel(
        routing.with_registry(call(1), registry1),
        routing.with_registry(call(2), registry2),
    )
    result = process(executable)

    with self.subTest('using_with_registry_sets_registry_correctly'):
      self.assertListEqual(result, expected_result, pprint.pformat(result))

  def test_current_registry(self):
    def f1() -> str:
      return 'f1'

    def f2() -> str:
      return 'f2'

    routing.function_registry['f'] = f1
    # We create the executable, it does not store the registry.
    executable = routing.function_registry('f')
    routing.function_registry['f'] = f2
    # We now attach the current registry to the executable.
    executable = routing.with_current_registry(executable)

    with self.subTest('should_use_stored_registry'):
      self.assertEqual(executing.run(executable), 'f2')
    # We change the registry, the executable should keep using the stored
    # registry.
    routing.function_registry['f'] = f1

    with self.subTest('should_use_stored_registry_after_change'):
      self.assertEqual(executing.run(executable), 'f2')

    with self.subTest('should_use_stored_registry_if_wrapped'):
      # We wrap the executable in a changed registry, it still should use the
      # registry it stored.
      self.assertEqual(
          executing.run(
              routing.with_registry(executable, routing.copy_registry())
          ),
          'f2',
      )

  def test_set_registry(self):
    def f1() -> str:
      return 'f1'

    def f2() -> str:
      return 'f2'

    routing.function_registry['f'] = f1
    r = routing.copy_registry()
    routing.function_registry['f'] = f2
    executable = routing.function_registry('f')

    with self.subTest('should_call_latest_registered'):
      self.assertEqual(executing.run(executable), 'f2')

    with self.subTest('should_call_saved_version'):
      routing.set_registry(r)
      self.assertEqual(executing.run(executable), 'f1')

  def test_copy_registry(self):
    def f1() -> str:
      return 'f1'

    class F(routing.RegistryReference):
      def __call__(self) -> str:
        return 'f2'

    with self.subTest('copy_of_function_is_shallow'):
      routing.function_registry['f'] = f1
      r = routing.copy_registry()
      self.assertEqual(id(r[0]['f']), id(f1))

    with self.subTest('copy_of_function_reference_is_deep'):
      f2 = F()
      routing.function_registry['f'] = f2
      r = routing.copy_registry()
      self.assertNotEqual(id(r[0]['f']), id(f2))


if __name__ == '__main__':
  absltest.main()
