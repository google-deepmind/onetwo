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

"""OneTwo Model Server."""

import logging
import sys
import traceback
from typing import Any

import fastapi
from onetwo.builtins import llm
from onetwo.core import batching


_Body = fastapi.Body


def _get_http_exception(exception: Exception) -> Exception:
  error_message = 'An exception {} occurred. Arguments: {}.'.format(
      type(exception).__name__, exception.args
  )
  logging.info(
      '%s\\nTraceback: %s', error_message, traceback.format_exc()
  )
  # Converts all exceptions to HTTPException.
  return fastapi.HTTPException(status_code=500, detail=error_message)


class ModelServer:
  """Model server that wraps llm builtins and exposes them as API calls.

  See run_model_server.py for an example of how to use this class.
  """

  def __init__(self):
    """Initializes a fastapi application and sets the configs."""

    logging.basicConfig(
        format='%(asctime)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
        stream=sys.stdout,
    )

    # We disable batching on the server (it may still be enabled on the client).
    # This is specific to the way we run onetwo in an async application. In
    # general, in client applications it is best to call onetwo.run().
    batching._enable_batching.set(False)

    self._app = fastapi.FastAPI()

    @self._app.post('/tokenize')
    async def _tokenize(
        content: str = _Body(..., embed=True)  # Ellipsis: required parameter.
    ) -> list[int]:
      """Wraps llm.tokenize."""
      # We disable batching on the server (it may still be enabled on the
      # client). This is specific to the way we run onetwo in an async
      # application. In general, in client applications it is best to call
      # onetwo.run().
      batching._enable_batching.set(False)  # pylint: disable=protected-access

      try:
        res = await llm.tokenize(content)  # pytype: disable=wrong-arg-count
      except Exception as exception:  # pylint: disable=broad-exception-caught
        raise _get_http_exception(exception) from exception
      return res

    @self._app.post('/generate_text')
    async def _generate_text(
        prompt: str = _Body(...),  # Ellipsis: required parameter.
        temperature: float = _Body(default=None),
        max_tokens: int = _Body(default=None),
        include_details: bool = _Body(default=False),
    ) -> tuple[str, dict[str, Any]]:
      """Wraps llm.generate_text."""
      # We disable batching on the server (it may still be enabled on the
      # client). This is specific to the way we run onetwo in an async
      # application. In general, in client applications it is best to call
      # onetwo.run().
      batching._enable_batching.set(False)  # pylint: disable=protected-access
      try:
        res = await llm.generate_text(  # pytype: disable=wrong-keyword-args
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            include_details=include_details,
        )
      except Exception as exception:  # pylint: disable=broad-exception-caught
        raise _get_http_exception(exception) from exception
      if include_details:
        return res
      else:
        return res, {}

    @self._app.post('/count_tokens')
    async def _count_tokens(
        content: str = _Body(..., embed=True)  # Ellipsis: required parameter.
    ) -> int:
      """Wraps llm.count_tokens."""
      # We disable batching on the server (it may still be enabled on the
      # client). This is specific to the way we run onetwo in an async
      # application. In general, in client applications it is best to call
      # onetwo.run().
      batching._enable_batching.set(False)  # pylint: disable=protected-access
      try:
        res = await llm.count_tokens(content)  # pytype: disable=wrong-arg-count
      except Exception as exception:  # pylint: disable=broad-exception-caught
        raise _get_http_exception(exception) from exception
      return res

    self._app.add_api_route(
        path='/health',
        endpoint=self._health,
        methods=['GET'],
    )

  async def __call__(self, scope, receive, send):
    await self._app(scope, receive, send)

  def _health(self):
    """Executes a health check."""
    return {}
