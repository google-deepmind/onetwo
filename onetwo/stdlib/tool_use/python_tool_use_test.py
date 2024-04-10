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

import dataclasses
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.builtins import tool_use
from onetwo.core import executing
from onetwo.core import routing
from onetwo.stdlib.code_execution import python_execution
from onetwo.stdlib.code_execution import python_execution_test_utils
from onetwo.stdlib.tool_use import llm_tool_use
from onetwo.stdlib.tool_use import python_tool_use

_ExecutionStatus = python_execution.ExecutionStatus
_SandboxResult = python_execution.SandboxResult


def _add_sync(arg1: Any, arg2: Any) -> Any:
  return arg1 + arg2


async def _add_async(arg1: Any, arg2: Any) -> Any:
  return arg1 + arg2


@executing.make_executable
def _add_executable(arg1: Any, arg2: Any) -> Any:
  return arg1 + arg2


class PythonToolUseEnvironmentTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # This class tests routing.function_registry. In case `import routing` is
    # not executed (this may happen when running `pytest` with multiple tests
    # that import `llm` module) the `function_registry` may be already filled
    # with various functions elsewhere in unexpected ways. We manually remove
    # all the keys to make sure it is empty.
    routing.function_registry.clear()
    # Unfortunately, we also removed all the builtins configured when importing
    # tool_use. Let's re-set them.
    # TODO:` or such
    # for better control of reproducibility.
    tool_use.reset_defaults()

  # pylint: disable=invalid-name
  def assertRunCodeResultEqualIgnoringTiming(
      self,
      expected_result: python_tool_use.RunCodeResult | str,
      actual_result: python_tool_use.RunCodeResult | str,
  ) -> None:
    # Remove timing-related content (similarly to what is done in
    # `python_execution_test_utils.SandboxResultAssertions`).
    if isinstance(expected_result, python_tool_use.RunCodeResult):
      expected_without_timing = dataclasses.replace(
          expected_result,
          sandbox_result=dataclasses.replace(
              expected_result.sandbox_result, timing=None
          ),
      )
    else:
      expected_without_timing = expected_result

    if isinstance(actual_result, python_tool_use.RunCodeResult):
      actual_without_timing = dataclasses.replace(
          actual_result,
          sandbox_result=dataclasses.replace(
              actual_result.sandbox_result, timing=None
          ),
      )
    else:
      actual_without_timing = actual_result

    return self.assertEqual(
        expected_without_timing,
        actual_without_timing,
        'RunCodeResult differed by more than just'
        f' timing.\nExpected:\n{expected_result!r}\nActual:\n{actual_result!r}',
    )

  def test_should_raise_error_if_running_code_without_starting(self):
    config = python_tool_use.PythonToolUseEnvironmentConfig()
    env = python_tool_use.PythonToolUseEnvironment(config=config)
    state = tuple()
    with self.assertRaisesRegex(ValueError, 'Environment not started'):
      executing.run(env.run_code(sandbox_state=state, code='x = 1 + 2'))

  def test_start_unsafe_and_stop_manually(self):
    config = python_tool_use.PythonToolUseEnvironmentConfig()
    env = python_tool_use.PythonToolUseEnvironment(config=config)

    executing.run(env.start_unsafe())

    state = tuple()
    result = executing.run(env.run_code(sandbox_state=state, code='x = 1 + 2'))

    expected_final_result = python_tool_use.RunCodeResult(
        code='x = 1 + 2',
        sandbox_result=python_execution.SandboxResult(final_expression_value=3),
    )

    with self.subTest('run_code_should_succeed_after_start_unsafe'):
      self.assertRunCodeResultEqualIgnoringTiming(
          expected_final_result, result
      )

    with self.subTest('sandbox_should_stay_cached_until_stop_is_called'):
      self.assertLen(env._sandbox_cache._objects, 1)

    env.stop()
    with self.subTest('sandbox_should_be_destroyed_when_stop_is_called'):
      self.assertEmpty(env._sandbox_cache._objects)

  def test_get_and_cache_sandbox_for_state_sequential(self):
    # This represents the most basic case of sandbox usage, where we execute
    # a sequence of code blocks, on each step requesting a sandbox equivalent
    # to the end of the previous step, and then recaching it again after based
    # on the new state. Under this scenario, we expect to keep reusing the same
    # sandbox.
    config = python_tool_use.PythonToolUseEnvironmentConfig()

    @executing.make_executable()
    async def wrapper() -> tuple[python_tool_use.RunCodeResult | None, int]:
      """Returns the final result and number of sandboxes."""
      with python_tool_use.PythonToolUseEnvironment(config=config) as env:
        result = None
        state = tuple()
        code_blocks = [
            'x = 1 + 2',
            'y = x + 3',
            'print(y + 4)\nexit()',
        ]
        for code in code_blocks:
          result = await env.run_code(sandbox_state=state, code=code)
          state += (code,)
        return result, len(env._sandbox_cache._objects)

    final_result, num_sandboxes = executing.run(wrapper())

    expected_final_result = python_tool_use.RunCodeResult(
        code='print(y + 4)\nexit()',
        sandbox_result=python_execution.SandboxResult(stdout='10\n'),
        exit_hook_called=True,
    )

    with self.subTest('should_return_correct_final_result'):
      self.assertRunCodeResultEqualIgnoringTiming(
          expected_final_result, final_result
      )

    with self.subTest('should_create_only_one_sandbox'):
      self.assertEqual(1, num_sandboxes)

  def test_get_and_cache_sandbox_for_state_non_sequential(self):

    # Simple class for configuring one step of the test (like in a parameterized
    # test, but all run within a single call to `executing.run`).
    @dataclasses.dataclass
    class StepForTest:
      sandbox_state: tuple[str, ...]
      code: str
      expected_result: python_tool_use.RunCodeResult | None
      expected_num_sandboxes: int

    # Simple class for storing results, which we will later assert on.
    @dataclasses.dataclass
    class StepResult:
      result: python_tool_use.RunCodeResult | None
      num_sandboxes: int

    # Steps to perform (like in a parameterized test).
    steps = [
        (
            # Get a sandbox, run code on it. The environment should
            # automatically create a sandbox in the desired state, including
            # running any past code blocks.
            StepForTest(
                sandbox_state=("x = 'a'",),
                code="x += 'b'\nx",
                expected_result=python_tool_use.RunCodeResult(
                    code="x += 'b'\nx",
                    sandbox_result=python_execution.SandboxResult(
                        final_expression_value='ab'
                    ),
                ),
                expected_num_sandboxes=1,
            )
        ),
        (
            # Get a sandbox in the same starting state as before, run different
            # code on it. Even thought we've been in requested state before,
            # the environment will need to create a new sandbox, since the
            # sandbox we created earlier has already moved onto a new state.
            StepForTest(
                sandbox_state=("x = 'a'",),
                code="x += 'c'\nx",
                expected_result=python_tool_use.RunCodeResult(
                    code="x += 'c'\nx",
                    sandbox_result=python_execution.SandboxResult(
                        final_expression_value='ac'
                    ),
                ),
                expected_num_sandboxes=2,
            )
        ),
        (
            # Get a sandbox in the state that would have resulted from the first
            # step, run more code on it. This should reuse the existing sandbox.
            StepForTest(
                sandbox_state=("x = 'a'", "x += 'b'\nx"),
                code="x += 'd'\nx",
                expected_result=python_tool_use.RunCodeResult(
                    code="x += 'd'\nx",
                    sandbox_result=python_execution.SandboxResult(
                        final_expression_value='abd'
                    ),
                ),
                expected_num_sandboxes=2,
            )
        ),
        (
            # Get a sandbox in the original starting state, and run exactly the
            # same code that we ran in the first step. This again requires
            # creating a new sandbox, since both of the sandboxes that were
            # previously in the requested starting state have moved onto new
            # states. In the future, however, we could potentially avoid the
            # need to create the new sandbox here, if we were to introduce a
            # request-reply cache for `run_code` requests (similar to the
            # caching that we do for LLM requests), in addition to caching the
            # sandboxes themselves.
            StepForTest(
                sandbox_state=("x = 'a'",),
                code="x += 'b'\nx",
                expected_result=python_tool_use.RunCodeResult(
                    code="x += 'b'\nx",
                    sandbox_result=python_execution.SandboxResult(
                        final_expression_value='ab'
                    ),
                ),
                expected_num_sandboxes=3,
            )
        ),
    ]

    # Run the steps and gather results.
    config = python_tool_use.PythonToolUseEnvironmentConfig()
    @executing.make_executable()
    async def wrapper() -> list[StepResult]:
      step_results = []
      with python_tool_use.PythonToolUseEnvironment(config=config) as env:
        for step in steps:
          result = await env.run_code(
              sandbox_state=step.sandbox_state, code=step.code
          )
          step_results.append(
              StepResult(
                  result=result, num_sandboxes=len(env._sandbox_cache._objects)
              )
          )
      return step_results

    step_results = executing.run(wrapper())

    # Compare the results to what was expected.
    for i, step_result in enumerate(step_results):
      with self.subTest(f'step_{i}_result'):
        self.assertRunCodeResultEqualIgnoringTiming(
            step_result.result, step_result.result
        )
      with self.subTest(f'step_{i}_num_sandboxes'):
        self.assertEqual(step_result.num_sandboxes, step_result.num_sandboxes)

  @parameterized.named_parameters(
      dict(
          testcase_name='without_retrying',
          max_retries_on_timeout=0,
          expected_final_result=python_tool_use.RunCodeResult(
              code='1 + 2',
              sandbox_result=_SandboxResult(
                  execution_status=_ExecutionStatus.SANDBOX_TIMEOUT,
                  sandbox_status=python_execution.SandboxStatus.INVALID,
              ),
          ),
          expected_calls_to_run=1,
      ),
      dict(
          testcase_name='with_retrying',
          max_retries_on_timeout=1,
          expected_final_result=python_tool_use.RunCodeResult(
              code='1 + 2',
              sandbox_result=python_execution.SandboxResult(
                  final_expression_value=3
              ),
          ),
          expected_calls_to_run=2,
      ),
  )
  def test_sandbox_timeout_retries(
      self,
      max_retries_on_timeout: int,
      expected_final_result: python_tool_use.RunCodeResult,
      expected_calls_to_run: int,
  ):
    # Here we configure the sandbox to time out the first time it is called, but
    # then succeed the 2nd time, so that we can check whether retry kicked in.
    sandbox = python_execution_test_utils.PythonSandboxForTest(
        reply_by_request={
            '1 + 2': [
                _SandboxResult(
                    execution_status=_ExecutionStatus.SANDBOX_TIMEOUT,
                    sandbox_status=python_execution.SandboxStatus.INVALID,
                ),
                _SandboxResult(final_expression_value=3),
            ]
        }
    )

    # Run the steps and gather results.
    config = python_tool_use.PythonToolUseEnvironmentConfig(
        sandbox_factory=python_execution_test_utils.PythonSandboxForTestFactory(
            default_sandbox=sandbox
        ),
        max_retries_on_timeout=max_retries_on_timeout,
    )
    with python_tool_use.PythonToolUseEnvironment(config=config) as env:
      result = executing.run(env.run_code(sandbox_state=tuple(), code='1 + 2'))

    with self.subTest('should_return_correct_final_result'):
      self.assertRunCodeResultEqualIgnoringTiming(expected_final_result, result)

    with self.subTest('should_return_correct_final_result'):
      self.assertLen(sandbox.requests, expected_calls_to_run)

    with self.subTest('should_generate_only_the_expected_requests'):
      self.assertEmpty(sandbox.unexpected_requests)

  @parameterized.named_parameters(
      ('sync', _add_sync),
      ('async', _add_async),
      ('executable', _add_executable),
  )
  def test_run_tool_directly(self, add_function):
    # No need to touch the function registry or builtins for calling
    # `env.run_tool` directly. Just configure the tool in the environment.
    config = python_tool_use.PythonToolUseEnvironmentConfig(
        tools=[llm_tool_use.Tool(name='add', function=add_function)]
    )
    with python_tool_use.PythonToolUseEnvironment(config=config) as env:
      result = executing.run(
          env.run_tool(tool_name='add', tool_args=['a', 'b'], tool_kwargs={})
      )

    self.assertEqual('ab', result)

  def test_run_tool_as_builtin(self):
    def f_in_registry(arg1: Any, arg2: Any) -> str:
      return f'f_in_registry: {arg1} {arg2}'

    def f_in_environment(arg1: Any, arg2: Any) -> Any:
      return f'f_in_environment: {arg1} {arg2}'

    routing.function_registry['f'] = f_in_registry

    config = python_tool_use.PythonToolUseEnvironmentConfig(
        tools=[llm_tool_use.Tool(name='f', function=f_in_environment)]
    )
    env = python_tool_use.PythonToolUseEnvironment(config=config)

    args = ['a', 'b']
    kwargs = {}

    # Before we call `env.register()`, `tool_use.run_tool` should invoke the
    # default implementation, which simply looks up 'f' in the function
    # registry.
    result_before = executing.run(tool_use.run_tool('f', args, kwargs))
    with self.subTest('before_registering_environment'):
      self.assertEqual('f_in_registry: a b', result_before)

    # After we call `env.register()`, `tool_use.run_tool()` should invoke
    # `env.run_tool()`, which will call the tool that is registered under the
    # name 'f' in the environment.
    with env, routing.RegistryContext():
      env.register()
      result_after = executing.run(tool_use.run_tool('f', args, kwargs))
    with self.subTest('after_registering_environment'):
      self.assertEqual('f_in_environment: a b', result_after)

  def test_run_tool_via_python(self):
    # No need to touch the function registry or builtins for using the tools as
    # hooks in a call to `env.run_code`directly. Just configure the tool in the
    # environment.
    config = python_tool_use.PythonToolUseEnvironmentConfig(
        tools=[llm_tool_use.Tool(name='add', function=lambda x, y: x + y)]
    )
    with python_tool_use.PythonToolUseEnvironment(config=config) as env:
      result = executing.run(
          env.run_code(sandbox_state=tuple(), code='add("a", "b")')
      )

    self.assertRunCodeResultEqualIgnoringTiming(
        python_tool_use.RunCodeResult(
            code='add("a", "b")',
            sandbox_result=python_execution.SandboxResult(
                final_expression_value='ab'
            ),
        ),
        result,
    )

  def test_run_tool_error_not_registered(self):
    # Note that we omit registering any tools in the environment config.
    config = python_tool_use.PythonToolUseEnvironmentConfig()
    with python_tool_use.PythonToolUseEnvironment(config=config) as env:
      result = executing.run(
          env.run_tool(tool_name='add', tool_args=['a', 'b'], tool_kwargs={})
      )

    self.assertStartsWith(
        str(result),
        '#ERROR#: Function add is not registered in the environment',
    )

  def test_run_tool_error_unexpected_argument(self):
    # Note that we register the function with arg `x`, but call it with `y`.
    config = python_tool_use.PythonToolUseEnvironmentConfig(
        tools=[llm_tool_use.Tool(name='double', function=lambda x: x *2)]
    )
    with python_tool_use.PythonToolUseEnvironment(config=config) as env:
      result = executing.run(
          env.run_tool(tool_name='double', tool_args=[], tool_kwargs={'y': 3})
      )

    self.assertRegex(
        str(result),
        "#ERROR#: .* got an unexpected keyword argument 'y'",
    )

if __name__ == '__main__':
  absltest.main()
