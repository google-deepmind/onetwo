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

"""OneTwo Model Server."""

import logging
import sys
import traceback
from typing import Any

import fastapi
from onetwo.builtins import llm
from onetwo.core import batching
import pydantic


class GenerateTextRequest(pydantic.BaseModel):
  """Wrapper for llm.generate_text input parameters."""
  prompt: str
  temperature: float | None = None
  max_tokens: int | None = None
  include_details: bool = False


class GenerateTextResponse(pydantic.BaseModel):
  """Wrapper for llm.generate_text output."""
  result: str
  details: dict[str, Any]


class TokenizeRequest(pydantic.BaseModel):
  """Wrapper for llm.tokenize input parameters."""
  content: str


class TokenizeResponse(pydantic.BaseModel):
  """Wrapper for llm.tokenize output."""
  result: list[int]


class CountTokensRequest(pydantic.BaseModel):
  """Wrapper for llm.count_tokens input parameters."""
  content: str


class CountTokensResponse(pydantic.BaseModel):
  """Wrapper for llm.count_tokens output."""
  result: int


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
    self._app.add_api_route(
        path='/health',
        endpoint=self.health,
        methods=['GET'],
    )
    self._app.add_api_route(
        path='/generate_text',
        endpoint=self.generate_text,  # pytype: disable=wrong-arg-types
        methods=['POST'],
    )
    self._app.add_api_route(
        path='/count_tokens',
        endpoint=self.count_tokens,
        methods=['POST'],
    )

  async def __call__(self, scope, receive, send):
    await self._app(scope, receive, send)

  def health(self):
    """Executes a health check."""
    return {}

  async def tokenize(self, request: TokenizeRequest) -> TokenizeResponse:
    """Wraps llm.tokenize."""
    # We disable batching on the server (it may still be enabled on the client).
    # This is specific to the way we run onetwo in an async application. In
    # general, in client applications it is best to call onetwo.run().
    batching._enable_batching.set(False)  # pylint: disable=protected-access

    try:
      res = await llm.tokenize(request.content)
      return TokenizeResponse(result=res)
    except Exception as exception:
      error_message = 'An exception {} occurred. Arguments: {}.'.format(
          type(exception).__name__, exception.args
      )
      logging.info(
          '%s\\nTraceback: %s', error_message, traceback.format_exc()
      )
      # Converts all exceptions to HTTPException.
      raise fastapi.HTTPException(status_code=500, detail=error_message)

  async def generate_text(
      self, request: GenerateTextRequest
  ) -> GenerateTextResponse:
    """Wraps llm.generate_text."""
    # We disable batching on the server (it may still be enabled on the client).
    # This is specific to the way we run onetwo in an async application. In
    # general, in client applications it is best to call onetwo.run().
    batching._enable_batching.set(False)  # pylint: disable=protected-access
    try:
      res = await llm.generate_text(
          prompt=request.prompt,
          temperature=request.temperature,
          max_tokens=request.max_tokens,
          include_details=request.include_details,
      )
      if request.include_details:
        return GenerateTextResponse(result=res[0], details=res[1])
      else:
        return GenerateTextResponse(result=res, details={})
    except Exception as exception:
      error_message = 'An exception {} occurred. Arguments: {}.'.format(
          type(exception).__name__, exception.args
      )
      logging.info('%s\\nTraceback: %s', error_message, traceback.format_exc())
      # Converts all exceptions to HTTPException.
      raise fastapi.HTTPException(status_code=500, detail=error_message)

  async def count_tokens(
      self, request: CountTokensRequest
  ) -> CountTokensResponse:
    """Wraps llm.count_tokens."""
    # We disable batching on the server (it may still be enabled on the client).
    # This is specific to the way we run onetwo in an async application. In
    # general, in client applications it is best to call onetwo.run().
    batching._enable_batching.set(False)  # pylint: disable=protected-access
    try:
      res = await llm.count_tokens(request.content)
      return CountTokensResponse(result=res)
    except Exception as exception:
      error_message = 'An exception {} occurred. Arguments: {}.'.format(
          type(exception).__name__, exception.args
      )
      logging.info(
          '%s\\nTraceback: %s', error_message, traceback.format_exc()
      )
      # Converts all exceptions to HTTPException.
      raise fastapi.HTTPException(status_code=500, detail=error_message)
