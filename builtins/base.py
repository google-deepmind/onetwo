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

"""Base definitions for built-in functions."""

import abc
from collections.abc import Callable
import functools
import inspect
import typing
from typing import Any, Generic, ParamSpec, TypeVar

from onetwo.core import executing
from onetwo.core import routing


_T = TypeVar('_T')
_Params = ParamSpec('_Params')


def _normalize_annotation(
    annotation: Any, typevar_dict: dict[str, int]
) -> Any:
  """Get a normalized version of types (generic or not) so they can be compared.

  Each typevar name is replaced with a number.

  Args:
    annotation: The type annotation
    typevar_dict: The dictionary collecting the

  Returns:
    The normalized version of the provided annotation.
  """
  if type(annotation) in [type(_T), type(_Params)]:
    # Normalize TypeVar
    typevar_name = annotation.__name__
    if typevar_name not in typevar_dict:
      typevar_dict[typevar_name] = len(typevar_dict) + 1
    return typevar_dict[typevar_name]

  if typing.get_origin(annotation):
    # Normalize generic types (like List[T], Tuple[T1, T2], etc.)
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    normalized_args = tuple(
        _normalize_annotation(arg, typevar_dict) for arg in args
    )
    return (origin, normalized_args)

  return annotation


def _compare_normalized_annotations(first: Any, second: Any) -> bool:
  """Compare two normalized annotations.

  Args:
    first: The first normalized annotation
    second: The second normalized annotation

  Returns:
    True if annotations are similar, False otherwise
  """
  first_type = type(first)
  second_type = type(second)
  if first_type != second_type:
    return False
  if isinstance(first, dict):
    if first.keys() != second.keys():
      return False
    return all(
        _compare_normalized_annotations(first[key], second[key])
        for key in first
    )
  if isinstance(first, (list, tuple)):
    return all(
        _compare_normalized_annotations(type1, type2)
        for type1, type2 in zip(first, second)
    )
  return first == second


class _BuiltinWrapper(Generic[_T], routing.RegistryReference):
  """Wrapper class that stores a particular implementation and its defaults.

  When configuring a Builtin with a particular implementation and default values
  we create an instance of this class and store it into the function_registry.
  Since it is callable, when calling the Builtin and executing it, the __call__
  function on the Builtin is executed which looks up the actual _BuiltinWrapper
  to use in the registry and calls its __call__ function which in tunrs calls
  the actual implementation.

  Note that we tag this class as an executing.RegistryReference so that it is
  deepcopied when we copy the registry.
  """

  def __init__(
      self,
      builtin_name: str,
      wrapped_signature: Callable[..., _T],
      implementation: Callable[..., _T],
      *args,
      **kwargs,
  ):
    # Using partial function when calling `configure`, i.e.,
    # `fn1.configure(functools.partial(fn2, arg_name=arg_value))`, will result
    # in error, because `executing.make_executable` does not support
    # functools.partial.
    if isinstance(implementation, functools.partial):
      raise ValueError(
          'Looks like `configure` method of a builtin was called with '
          'functools.partial object. Instead of '
          '`fn1.configure(functools.partial(fn2, arg_name=arg_value))`, '
          'please use `configure` directly with keyword arguments, e.g. '
          '`fn1.configure(fn2, arg_name=arg_value)`.'
      )

    # Make sure that implementation returns an Executable.
    if not getattr(implementation, 'decorated_with_make_executable', False):
      implementation = executing.make_executable(copy_self=False)(
          implementation
      )
    self._builtin_name = builtin_name
    self._implementation = implementation

    # We set `eval_str=True` to ensure that we can compare annotations defined
    # in modules with `from __future__ import annotations`.
    impl_signature = inspect.signature(implementation, eval_str=True)
    base_signature = inspect.signature(wrapped_signature, eval_str=True)
    # This is used to store typevars in the annotations. In order to be able to
    # compare generic types we replace, e.g. T by a serial number 1, S by 2, etc
    # with `_normalize_annotation` and compare using
    # `_compare_normalized_annotations`.
    impl_typevar_cache = {}
    base_typevar_cache = {}
    self._base_args_ignored_by_impl = []
    self._named_impl_args_not_in_base = []
    self._impl_has_kwargs = False
    for name, value in base_signature.parameters.items():
      if value.default == inspect.Parameter.empty:
        if name not in impl_signature.parameters:
          raise ValueError(
              f'Parameter {name} of {builtin_name} should be a parameter of the'
              ' implementation but it is missing.'
          )
      if name in impl_signature.parameters:
        # We cannot use the standard type comparisons since there could be
        # generic TypeVar-ed templates.
        normalized_base_annotation = _normalize_annotation(
            value.annotation, base_typevar_cache
        )
        normalized_impl_annotation = _normalize_annotation(
            impl_signature.parameters[name].annotation, impl_typevar_cache
        )
        if not _compare_normalized_annotations(
            normalized_impl_annotation, normalized_base_annotation
        ):
          raise ValueError(
              f'The implementation of {builtin_name} should have parameter'
              f' {name} with type "{value.annotation}" however it looks like'
              f' "{impl_signature.parameters[name].annotation}".'
          )
      else:
        self._base_args_ignored_by_impl.append(name)
    # We also ignore the extra arguments that are set at configuration time.
    for name, value in impl_signature.parameters.items():
      if value.kind in (
          inspect.Parameter.VAR_POSITIONAL,
          inspect.Parameter.VAR_KEYWORD,
      ):
        if value.kind == inspect.Parameter.VAR_KEYWORD:
          self._impl_has_kwargs = True
        continue
      if name not in base_signature.parameters.keys():
        if value.default == inspect.Parameter.empty:
          # Implementation has a non-optional arg that the base signature does
          # not have. We need to make sure its value is specified in kwargs:
          if name not in kwargs:
            raise ValueError(
                f'The implementation of {builtin_name} has non-optional '
                f'parameter {name} (with type "{value.annotation}") that '
                f'{builtin_name} does not have. The value of this non-optional '
                'parameter must be specified when configuring.'
            )
        self._named_impl_args_not_in_base.append(name)

    self._check_unknown_kwargs(implementation=self._implementation, **kwargs)

    # Finally we update the function with the positional argument and default
    # keyword arguments.
    self._registered_function = functools.partial(
        implementation,
        *args,
        **kwargs,
    )
    self._defaults = kwargs
    # We use functools.update_wrapper to attach all the properties of the
    # wrapped function to this object, so that it will behave as the wrapped
    # function, in particular it will have the same signature.
    functools.update_wrapper(self, wrapped_signature)

  def _check_unknown_kwargs(
      self, *, implementation: Callable[_Params, Any], **kwargs
  ) -> None:
    """Raise error if kwargs has arguments unknown to the implementation.

    Args:
      implementation: Function that implements this builtin.
      **kwargs: Keyword arguments with their default values that we are setting.

    Raises:
      ValueError: If implementations has no `VAR_KEYWORD` arguments and kwargs
        contain an argument unknown to implementations.
    """
    # If implementation takes `**kwargs` then we can pass any named arguments.
    if self._impl_has_kwargs:
      return
    impl_signature = inspect.signature(implementation)
    for name in kwargs:
      if name not in impl_signature.parameters:
        raise ValueError(
            f'Trying to set a default value for argument {name}, but '
            'provided implementation does not accept this argument.'
        )

  @property
  def defaults(self):
    """Default argument values specified when configuring the builtin."""
    return self._defaults

  def update(self, **new_defaults) -> None:
    self._check_unknown_kwargs(
        implementation=self._implementation, **new_defaults
    )
    self._registered_function = functools.partial(
        self._registered_function,
        **new_defaults,
    )
    self._defaults |= new_defaults

  # We do not decorate this function with make_executable since we want it
  # transparently call the registered function which is itself decorated.
  def __call__(self, *args, **kwargs):
    # Potentially replace the None values by their defaults.
    for key, value in self._defaults.items():
      if key in kwargs and kwargs[key] is None:
        kwargs[key] = value

    # Remove the arguments that are ignored by the implementation.
    for name in self._base_args_ignored_by_impl:
      kwargs.pop(name, None)

    # Make sure that all the passed named arguments appear in the builtin
    # signature.
    for name in kwargs:
      if name in self._named_impl_args_not_in_base:
        raise ValueError(
            f'Unknown argument {name} passed when calling the builtin '
            f'{self._builtin_name}.'
        )

    # Finally call the implementation ()
    return self._registered_function(*args, **kwargs)


class Builtin(Generic[_T], metaclass=abc.ABCMeta):
  """Decorator for built-in functions.

  This is meant to decorate a *signature*, i.e. a function without an
  implementation.
  Indeed, the implementation will be tied to the function at runtime by calling
  `configure`.
  A configured builtin function returns an Executable. Its returned value can
  be awaited or used with `async for`, `executing.run`, or
  `executing.safe_stream`.

  This performs the following operations:
  - it adds default parameters (similarly to functools.partial);
  - it possibly registers the function to the function registry so that the
    function can be called without passing a reference to it;
  - when the function is called, any None parameter value is replaced by its
    default (if provided at creation time);
  - when the function is called, the result of the wrapped function is returned.

  When decorating a function, the variable self._name takes the name of the
  decorated function (prepended with its module). This is the name that is
  used to register the _BuiltinWrapper into the function_registry.

  Usage:
  ```
  @Builtin[str]
  def builtin_function(a: int, b: str | None = None) -> str:
    raise NotImplementedError('Implementation provided at runtime')

  def implementation(a: int) -> str:
    # Since b has a default value we can ignore it in our implementation.
    return str(a)

  # We bind the builtin_function to its implementation, and this will add it
  # to the global function registry.
  builtin_function.configure(implementation)

  # We can now call the builtin_function (this will go through the function
  # registry to find the actual implementation) and get the result.
  result = await builtin_function(a=1, b='hello')

  # We can configure the function with default values.
  builtin_function.configure(implementation, b='hello')
  result = await builtin_function(a=1)

  # We can update the default value later on.
  builtin_function.update(b='world')
  result = await builtin_function(a=1)
  ```
  """

  def __init__(self, wrapped: Callable[_Params, _T]):
    self._wrapped = wrapped  # Builtin function.
    # We determine the name to use in the registry.
    module = getattr(wrapped, '__module__', None)
    self._name = getattr(wrapped, '__name__', None)
    if module is not None and self._name is not None:
      self._name = module + '.' + self._name
    if self._name is None:
      raise ValueError(
          f'Builtin is initialized with a function that has no name: {wrapped}.'
      )
    self._configured = False  # If the builtin was configured.
    functools.update_wrapper(self, wrapped)

  @property
  def name(self) -> str:
    """Return the name of the builtin function."""
    return self._name

  def configure(
      self,
      # We use Any because the implementation can be a function that is batched
      # or decorated with make_executable, etc.
      implementation: Callable[_Params, Any],
      *args,
      **kwargs,
  ) -> None:
    """Configures the implementation of the builtin and registers it.

    Implementation must satisfy the following contraints:
     1. Implementation signature has all the non-optional arguments of the base
      signature. If base function is `base_fn(a: int)` then implementation
      must have an argument called `a`.
     2. If implementation has any additional arguments compared to the base
      signature, i.e., `base_fn(a: int)` and `impl_fn(a: int, b: str)`, then
      values of these extra arguments (i.e., "b") must be specified when
      configuring. This can be achieved either by making these arguments
      optional `impl_fn(a: int, b: str = 'abc')`, or providing their value
      in keyword arguments `base_fn.configure(impl_fn, b='abc')`.

    Note: once configured, all the additional arguments of the implementation
    are frozen (see "2." above). In particular, specifying them when calling the
    builtin (e.g., base_fn(a=12, c='abc')) will raise ValueError. In those rare
    cases where we really want to pass some additional arguments to the builtin,
    implementation should do it using its "**kwargs". For example, if builtin
    signature is `base_fn(a: int)`, we can configure it with implementation that
    has a signature
    ```
      def impl_fn(a: int, b: str, **kwargs):
        ...
        if 'c' in kwargs:
          # do something.
        ...
    ```
    using
    ```
      base_fn.configure(impl_fn, b='abc')
    ```
    and then call `base_fn(a=1, c=12)`. A good example is `use_fewshot` kwarg in
    `_default_instruct` method of `onetwo.builtins.llm.py`.

    Note: we don't check that the return type of the implementation matches the
    one of the base. We often have decorators that change the return type and
    those decorators cannot properly adjust the signature (as returned by
    inspect.signature), nor can they reliably adjust the annotations, so there
    is no reliable method to assess a valid return type.

    Makes sure that whatever is registered as implementation of the builtin
    function is decorated with `executing.make_executable` and therefore returns
    an Executable.

    Args:
      implementation: The function that will be actually used when calling the
        builtin.
      *args: Default positional arguments to pass to the underlying function.
      **kwargs: Default keyword arguments to pass to the underlying function.
    """
    routing.function_registry[self._name] = _BuiltinWrapper[_T](
        self._name, self._wrapped, implementation, *args, **kwargs
    )
    self._configured = True

  def update(self, **kwargs) -> None:
    """Update defaults of an already configured function.

    Writing
    ```
    ...
    fn.configure(some_implementation, 1, 2, a=1, b=2, c=3, d=4, ...)
    fn.update(c=10)
    ```
    is equivalent to writing
    ```
    ...
    fn.configure(some_implementation, 1, 2, a=1, b=2, c=3, d=4, ...)
    fn.configure(some_implementation, 1, 2, a=1, b=2, c=10, d=4, ...)
    ```

    Args:
      **kwargs: Keyword arguments with their new default values.
    """
    if not self._configured:
      raise ValueError(
          'Attempting to amend a builtin function that has not been configured.'
      )
    routing.function_registry[self._name].update(**kwargs)

  @executing.make_executable(copy_self=False, execute_result=True)
  async def __call__(
      self, *args, **kwargs
  ) -> executing.FunctionExecWrapper[_T]:
    """Actually calls the wrapped function and handles default kwargs.

    Note that because this function is decorated with make_executable, calling
    it will not actually execute the function and in particular the actual
    implementation to use will not be determined until actual execution.

    Args:
      *args: Positional arguments.
      **kwargs: Keyword arguments.

    Returns:
      Returned value of the wrapped function or an executing.Executable wrapping
      the function.
    """
    if not self._configured:
      raise ValueError(
          f'Attempting to call a builtin function ({self._name}) that has not'
          ' been configured.'
      )
    # This corresponds to the case where the function has been explicitly
    # configured by calling `configure`, hence it has been registered
    # in the registry with its name.
    if self._name not in routing.function_registry:
      raise ValueError(
          f'The registry does not contain an entry for {self._name} even'
          ' though the function has been configured. This might be due to'
          ' the registry having been overridden or the entry deleted since'
          ' the registration of the function.'
      )
    # The registry contains the _BuiltinWrapper, and we want that when one does
    # await builtin(...) we get the same effect as doing
    # await implementation(...)
    # which is achieved by setting `execute_result=True`.
    # However, we also want that `await builtin(...).pre_execute()` has the same
    # effect as doing `await implementation(...).pre_execute()` which is why
    # we do `return await implementation(...).pre_execute()` since it will
    # just pass the result of the one-pass execution of the implementation.
    return await routing.function_registry[self._name](
        *args, **kwargs
    ).pre_execute()
