# Copyright 2026 DeepMind Technologies Limited.
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

"""MCP-related QA strategies."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
import contextlib
import dataclasses
import logging
import re
from typing import Any

import anyio
import fastmcp
from fastmcp.client import transports as fastmcp_transports
from google.genai import types as genai_types
import httpx
from mcp import types as mcp_types
from mcp.client import streamable_http as mcp_streamable_http
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.core import executing
from onetwo.core import routing
from onetwo.core import tracing
from onetwo.stdlib.qa import qa
from onetwo.stdlib.qa import strategies_lib

_PromptContent = str | content_lib.ChunkList
StrategyResult = qa.StrategyResult

# TODO: Think of a standard way of speficying a
# template and coupling it to how its output is parsed. This way, we would not
# need to worry about coupling the prompt with the output parsing.

# Minimally-modified version of the original prompt used in the BrowseComp-Plus
# paper (https://arxiv.org/abs/2508.06600). We modified the first part of the
# prompt to mention "an MCP server" instead of "a search engine" to make it
# more general purpose. Also added additional_instructions to allow passing
# MCP-server-specific instructions if needed.
default_qa_template_mcp = (
    'You are a deep research agent. You need to answer the given question by'
    ' interacting with an MCP server, using the tools provided. Please perform'
    ' reasoning and use the tools in a step by step, interleaved manner. You'
    ' may use the tools multiple times. {additional_instructions}'
    '\n\n'
    'Question: {question}'
    '\n\n'
    'Your response should be in the following format:\n'
    'Explanation: {{your explanation for your final answer. For this'
    ' explanation section only, you should cite your evidence documents inline'
    ' by enclosing their docids in square brackets [] at the end of sentences.'
    ' For example, [20].}}\n'
    'Exact Answer: {{your succinct, final answer}}\n'
    'Confidence: {{your confidence score between 0% and 100% for your answer}}'
)


def _extract_cited_doc_ids(answer: str) -> list[str]:
  """Extracts unique cited document IDs from the answer.

  Parses bracket-enclosed citations from the model's response text and returns
  them as a deduplicated list of strings in the order they first appear.
  Supports both single citations like ``[16935]`` and comma-separated
  citations like ``[73758, 81636]``.

  Args:
    answer: The raw answer string from the model.

  Returns:
    A list of unique document ID strings.
  """
  seen = set()
  doc_ids = []
  for match in re.finditer(r'\[([^\[\]]+)\]', answer):
    for doc_id in match.group(1).split(','):
      doc_id = doc_id.strip()
      if doc_id and doc_id not in seen:
        seen.add(doc_id)
        doc_ids.append(doc_id)
  return doc_ids


# TODO: Create a helper library for these MCP transport wrappers.
class _ResilientStreamableHTTPTransport(
    mcp_streamable_http.StreamableHTTPTransport
):
  """StreamableHTTPTransport converting HTTP 4xx/5xx into JSON-RPC errors.

  Standard StreamableHTTPTransport calls response.raise_for_status() inside its
  background reader task when the MCP server returns HTTP 400/422 (e.g. on
  invalid tool arguments). This fatal unhandled exception crashes anyio's
  TaskGroup and permanently kills the transport, causing all subsequent tool
  calls in the multi-turn reasoning loop to hang indefinitely.

  This resilient transport intercepts HTTP error responses and sends a JSON-RPC
  error message to the read stream, keeping the session alive and allowing the
  LLM to receive the error and self-correct.
  """

  async def _handle_post_request(
      self, ctx: mcp_streamable_http.RequestContext
  ) -> None:
    headers = self._prepare_headers()
    message = ctx.session_message.message
    is_initialization = self._is_initialization_request(message)

    async with ctx.client.stream(
        'POST',
        self.url,
        json=message.model_dump(by_alias=True, mode='json', exclude_none=True),
        headers=headers,
    ) as response:
      if response.status_code == 202:
        return

      if response.status_code == 404:
        if isinstance(message.root, mcp_types.JSONRPCRequest):
          await self._send_session_terminated_error(
              ctx.read_stream_writer,
              message.root.id,
          )
        return

      if response.status_code >= 400:
        if isinstance(message.root, mcp_types.JSONRPCRequest):
          content = await response.aread()
          error_text = content.decode('utf-8', errors='replace')
          jsonrpc_error = mcp_types.JSONRPCError(
              jsonrpc='2.0',
              id=message.root.id,
              error=mcp_types.ErrorData(
                  code=(
                      mcp_types.INVALID_PARAMS
                      if response.status_code in (400, 422)
                      else mcp_types.INTERNAL_ERROR
                  ),
                  message=f'HTTP {response.status_code}: {error_text}',
              ),
          )
          session_message = mcp_streamable_http.SessionMessage(
              mcp_types.JSONRPCMessage(jsonrpc_error)
          )
          await ctx.read_stream_writer.send(session_message)
        return

      if is_initialization:
        self._maybe_extract_session_id_from_response(response)

      if isinstance(message.root, mcp_types.JSONRPCRequest):
        content_type = response.headers.get(
            mcp_streamable_http.CONTENT_TYPE, ''
        ).lower()
        if content_type.startswith(mcp_streamable_http.JSON):
          await self._handle_json_response(
              response, ctx.read_stream_writer, is_initialization
          )
        elif content_type.startswith(mcp_streamable_http.SSE):
          await self._handle_sse_response(response, ctx, is_initialization)
        else:
          await self._handle_unexpected_content_type(
              content_type,
              ctx.read_stream_writer,
          )


@contextlib.asynccontextmanager
async def _resilient_streamable_http_client(
    url: str,
    headers: dict[str, str] | None = None,
    auth: Any = None,
    http_read_timeout: float = 300.0,
    http_connect_timeout: float = 60.0,
    http_write_timeout: float = 60.0,
) -> AsyncGenerator[
    tuple[
        anyio.streams.memory.MemoryObjectReceiveStream[Any],
        anyio.streams.memory.MemoryObjectSendStream[Any],
        Any,
    ],
    None,
]:
  """Context manager providing streams via _ResilientStreamableHTTPTransport.

  Args:
    url: The URL of the MCP server to connect to.
    headers: Optional HTTP headers to send with requests.
    auth: Optional authentication handler for httpx.
    http_read_timeout: Maximum seconds to wait for an HTTP response chunk or SSE
      stream frame.
    http_connect_timeout: Maximum seconds to wait to establish an HTTP
      connection.
    http_write_timeout: Maximum seconds to wait when writing HTTP request data.

  Yields:
    A tuple of (read_stream, write_stream, get_session_id_fn).
  """
  read_stream_writer, read_stream = anyio.create_memory_object_stream[
      mcp_streamable_http.SessionMessage | Exception
  ](0)
  write_stream, write_stream_reader = anyio.create_memory_object_stream[
      mcp_streamable_http.SessionMessage
  ](0)

  client = mcp_streamable_http.create_mcp_http_client(
      headers=headers,
      auth=auth,
      timeout=httpx.Timeout(
          http_read_timeout,
          connect=http_connect_timeout,
          read=http_read_timeout,
          write=http_write_timeout,
      ),
  )
  transport = _ResilientStreamableHTTPTransport(url)

  async with anyio.create_task_group() as tg:
    try:
      async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(client)

        def start_get_stream() -> None:
          tg.start_soon(transport.handle_get_stream, client, read_stream_writer)

        tg.start_soon(
            transport.post_writer,
            client,
            write_stream_reader,
            read_stream_writer,
            write_stream,
            start_get_stream,
            tg,
        )

        try:
          yield (read_stream, write_stream, transport.get_session_id)
        finally:
          if transport.session_id:
            try:
              await transport.terminate_session(client)
            except Exception:  # pylint: disable=broad-except
              pass
          tg.cancel_scope.cancel()
    finally:
      await read_stream_writer.aclose()
      await write_stream.aclose()


class _ResilientStreamableHttpTransport(
    fastmcp_transports.StreamableHttpTransport
):
  """FastMCP transport that wraps _ResilientStreamableHTTPTransport."""

  def __init__(
      self,
      url: str,
      *args: Any,
      http_read_timeout: float = 300.0,
      http_connect_timeout: float = 60.0,
      http_write_timeout: float = 60.0,
      **kwargs: Any,
  ) -> None:
    """Initializes _ResilientStreamableHttpTransport.

    Args:
      url: The URL of the MCP server to connect to.
      *args: Positional arguments passed to base StreamableHttpTransport.
      http_read_timeout: Maximum seconds to wait for an HTTP response chunk or
        SSE stream frame.
      http_connect_timeout: Maximum seconds to wait to establish an HTTP
        connection.
      http_write_timeout: Maximum seconds to wait when writing HTTP request
        data.
      **kwargs: Keyword arguments passed to base StreamableHttpTransport.
    """
    super().__init__(url, *args, **kwargs)
    self.http_read_timeout = http_read_timeout
    self.http_connect_timeout = http_connect_timeout
    self.http_write_timeout = http_write_timeout

  @contextlib.asynccontextmanager
  async def connect_session(
      self, **session_kwargs
  ) -> AsyncIterator[fastmcp_transports.ClientSession]:
    headers = fastmcp_transports.get_http_headers() | self.headers
    async with _resilient_streamable_http_client(
        self.url,
        headers=headers,
        auth=self.auth,
        http_read_timeout=self.http_read_timeout,
        http_connect_timeout=self.http_connect_timeout,
        http_write_timeout=self.http_write_timeout,
    ) as (read_stream, write_stream, _):
      async with fastmcp_transports.ClientSession(
          read_stream,
          write_stream,
          **session_kwargs,
      ) as session:
        yield session


def _build_thinking_config(
    include_thoughts: bool,
) -> genai_types.ThinkingConfig:
  """Builds a ThinkingConfig preserving any defaults from backend registry."""
  chat_defaults = routing.function_registry[llm.chat.name].defaults
  existing_thinking_config = chat_defaults.get('thinking_config')
  if existing_thinking_config is not None:
    return existing_thinking_config.model_copy(
        update={'include_thoughts': include_thoughts}
    )
  return genai_types.ThinkingConfig(include_thoughts=include_thoughts)


def _setup_resilient_mcp_session(
    client: fastmcp.Client,
    tool_timeout: float,
    tool_call_history: list[tuple[str, dict[str, Any] | None, str]],
) -> None:
  """Configures an MCP client session with resilient error handling.

  Args:
    client: The fastmcp.Client instance whose active session will be wrapped.
    tool_timeout: Maximum seconds to wait for a tool execution.
    tool_call_history: Mutable list tracking (name, arguments, result_text) for
      every tool call across all sessions.
  """

  # Monkey-patch __deepcopy__ to prevent copy.deepcopy from failing on
  # live session objects (like Futures) during GenAI SDK's internal
  # config copying. Returning self preserves the shared live connection.
  # The try/except guards against the session being dead (e.g. after an
  # HTTP 400 from the MCP server kills the connection).
  def _safe_deepcopy(memo, client_ref=client):
    del memo  # Unused; required by __deepcopy__ protocol.
    try:
      return client_ref.session
    except RuntimeError:
      return None

  setattr(client.session, '__deepcopy__', _safe_deepcopy)

  original_call_tool = client.session.call_tool

  async def _safe_call_tool(
      name: str, arguments: dict[str, Any] | None = None, **call_kwargs
  ):
    try:
      result = await asyncio.wait_for(
          original_call_tool(name, arguments=arguments, **call_kwargs),
          timeout=tool_timeout,
      )
      result_text = ''
      if hasattr(result, 'content') and result.content:
        result_text = '\n'.join(
            getattr(c, 'text', str(c)) for c in result.content
        )
      tool_call_history.append((name, arguments, result_text))
      return result
    except Exception as exc:  # pylint: disable=broad-except
      logging.warning(
          'MCP call_tool(%s, %s) raised exception: %s',
          name,
          arguments,
          exc,
      )
      err_text = (
          f'Error executing tool {name}: {exc}. Please adjust arguments'
          ' or try another tool.'
      )
      tool_call_history.append((name, arguments, err_text))
      return mcp_types.CallToolResult(
          content=[
              mcp_types.TextContent(
                  type='text',
                  text=err_text,
              )
          ],
          isError=True,
      )
    finally:
      # Update the active LLM child stage in the trace in real time as each tool
      # executes. If llm.chat fails later (e.g. context limit), the trace will
      # already contain all executed tool calls and responses up to that point.
      tool_details = [
          {
              'type': 'function_call',
              'name': n,
              'args': a or {},
              'response': {
                  'result': {
                      'content': [{'type': 'text', 'text': r}],
                  }
              },
          }
          for n, a, r in tool_call_history
      ]
      tracing.update_info({
          'llm.tools.calls': len(tool_call_history),
          'llm.tools.details': tool_details,
      })

  setattr(client.session, 'call_tool', _safe_call_tool)


@dataclasses.dataclass
# TODO: b/535122834 - Rename to ClientSideMCPStrategy or something more generic
# once we standardize MCP tool passing and configuration across backends.
class GoogleGenAIMCPStrategy(qa.QAStrategy[_PromptContent, StrategyResult]):
  """A strategy that runs an MCP server client-side and passes it to the model.

  This strategy initiates a connection to the MCP server client-side and uses a
  manual async execution loop to call tools locally on the developer's machine.
  This allows testing with HTTP (localhost) MCP servers without needing HTTPS.

  Error Handling & Empty Returns:
    - Returns ``StrategyResult(answer='', docs=[])`` when:
        (1) The LLM chat encounters an unrecoverable error during automatic
            function calling (e.g. input token count exceeds the model's
            context window limit).
        (2) The model completes automatic function calling without outputting
            a final text answer (e.g. ends on a tool call or returns an empty
            response).
      In all these cases, all intermediate tool calls and responses executed
      before the failure are preserved in the execution trace
      (``llm.tools.calls`` and ``llm.tools.details``).
    - Transient tool execution errors (e.g. timeouts or exceptions from the MCP
      server) are caught and returned to the model as
      ``CallToolResult(isError=True)`` so it can adjust arguments and recover.
     - Raises an exception when:
        (1) Initial configuration is invalid (e.g. missing ``mcp_server_urls``
            or missing ``{question}`` placeholder in ``query_template``).
        (2) FastMCP transport connection fails before LLM execution begins.

  Attributes:
    mcp_server_urls: The URL of the MCP server to connect to (or comma-separated
      list of URLs).
    query_template: A format string used to wrap the question before sending it
      to the LLM. Must contain both ``{question}`` and
      ``{additional_instructions}`` placeholders.
    include_thoughts: Whether to include thoughts in the backend response.
      Useful to analyze the thought process of the model in between tool calls.
    tool_timeout: Maximum seconds to wait for an individual tool call before
      returning a timeout error to the model.
    http_read_timeout: Maximum seconds to wait for an HTTP response chunk or SSE
      stream frame from the MCP server.
    http_connect_timeout: Maximum seconds to wait to establish an HTTP
      connection with the MCP server.
    http_write_timeout: Maximum seconds to wait when writing HTTP request data
      to the MCP server.
    maximum_remote_calls: Maximum number of remote tool calls allowed during
      automatic function calling. Defaults to 100.
  """

  mcp_server_urls: str = ''
  query_template: str = default_qa_template_mcp
  include_thoughts: bool = True
  tool_timeout: float = 60.0
  http_read_timeout: float = 300.0
  http_connect_timeout: float = 60.0
  http_write_timeout: float = 60.0
  maximum_remote_calls: int = 100

  def __post_init__(self):
    if not self.mcp_server_urls:
      raise ValueError('GoogleGenAIMCPStrategy requires mcp_server_urls.')
    if '{question}' not in self.query_template:
      raise ValueError(
          'query_template must contain a {question} placeholder, got:'
          f' {self.query_template!r}'
      )
    if '{additional_instructions}' not in self.query_template:
      raise ValueError(
          'query_template must contain an {additional_instructions}'
          f' placeholder, got: {self.query_template!r}'
      )

  @property
  def _mcp_server_urls(self) -> list[str]:
    """Returns the list of MCP server URLs, split from mcp_server_urls."""
    return [u.strip() for u in self.mcp_server_urls.split(',') if u.strip()]

  def _format_question(
      self,
      question: _PromptContent,
      additional_instructions: str,
  ) -> _PromptContent:
    """Formats the question by applying the query template.

    Args:
      question: The raw question, either a string or a ChunkList.
      additional_instructions: Extra instructions to inject into the template.

    Returns:
      The formatted question with the template applied.
    """
    instructions_block = (
        f'\n{additional_instructions}' if additional_instructions else ''
    )

    if isinstance(question, str):
      return self.query_template.format(
          question=question,
          additional_instructions=instructions_block,
      )
    elif isinstance(question, content_lib.ChunkList):
      template_with_instructions = self.query_template.replace(
          '{additional_instructions}', instructions_block
      )
      parts = template_with_instructions.split('{question}')
      if len(parts) == 2:
        head, tail = parts
        return content_lib.ChunkList(
            [content_lib.Chunk(head)]
            + list(question)
            + [content_lib.Chunk(tail)]
        )
    return question

  @executing.make_executable(copy_self=False)
  async def __call__(
      self,
      question: _PromptContent,
      **query_time_kwargs: Any,
  ) -> StrategyResult:

    additional_instructions = query_time_kwargs.pop(
        'additional_instructions', ''
    )
    del query_time_kwargs

    question = self._format_question(question, additional_instructions)

    # Create fresh MCP client(s) per call.  This avoids two classes of bugs:
    #   1. Concurrency races when multiple coroutines (from ot.evaluate /
    #      par_iter) share a single session — one coroutine's error could
    #      tear down or poison another's session.
    #   2. Silent session death — when the MCP server returns HTTP 400 during
    #      AFC, the GenAI SDK handles it internally (no exception reaches us),
    #      leaving the session dead for all subsequent calls.
    # The overhead of a per-call session is negligible compared to the
    # multi-turn LLM interaction (milliseconds vs. tens of seconds).
    urls = self._mcp_server_urls
    logging.info(
        'Connecting to MCP server(s) client-side: %s',
        ', '.join(urls),
    )
    clients = [
        fastmcp.Client(
            _ResilientStreamableHttpTransport(
                url=url,
                http_read_timeout=self.http_read_timeout,
                http_connect_timeout=self.http_connect_timeout,
                http_write_timeout=self.http_write_timeout,
            )
        )
        for url in urls
    ]

    answer_str = ''
    tool_call_history: list[tuple[str, dict[str, Any] | None, str]] = []
    try:
      async with contextlib.AsyncExitStack() as stack:
        for client in clients:
          await stack.enter_async_context(client)
          _setup_resilient_mcp_session(
              client=client,
              tool_timeout=self.tool_timeout,
              tool_call_history=tool_call_history,
          )
        thinking_config = _build_thinking_config(self.include_thoughts)
        kwargs = {
            'tools': [client.session for client in clients],
            'automatic_function_calling': (
                genai_types.AutomaticFunctionCallingConfig(
                    maximum_remote_calls=self.maximum_remote_calls
                )
            ),
            'thinking_config': thinking_config,
        }

        messages = [
            content_lib.Message(
                role=content_lib.PredefinedRole.USER,
                content=question,
            )
        ]
        try:
          answer = await llm.chat(messages, **kwargs)
          answer_str = str(answer) if answer is not None else ''
        except Exception as chat_exc:  # pylint: disable=broad-except
          logging.warning(
              'llm.chat encountered error during AFC: %s. Preserving %d tool'
              ' calls in trace and returning no answer.',
              chat_exc,
              len(tool_call_history),
          )
          return StrategyResult(answer='', docs=[])

        # If AFC completed without a final text answer (e.g. ended on a tool
        # call), return a no-answer result.
        if not answer_str or answer_str.startswith('[Function Call:'):
          logging.info(
              'Model did not provide a final text answer (tool calls: %d).',
              len(tool_call_history),
          )
          return StrategyResult(answer='', docs=[])
    except Exception as e:  # pylint: disable=broad-except
      # If fastmcp.__aexit__ raised a background transport error (e.g.
      # HTTPStatusError from a previously caught tool call error) during
      # client teardown, but the answer was already produced by the model,
      # return the generated answer.
      if answer_str:
        logging.warning(
            'FastMCP client shutdown raised exception after answer was'
            ' generated: %s',
            e,
        )
        return StrategyResult(
            answer=answer_str,
            docs=[
                qa._Document(doc_id=d)
                for d in _extract_cited_doc_ids(answer_str)
            ],
        )
      logging.exception('Error during GoogleGenAIMCPStrategy execution')
      return StrategyResult(answer='', docs=[])

    return StrategyResult(
        answer=answer_str,
        docs=[
            qa._Document(doc_id=d) for d in _extract_cited_doc_ids(answer_str)
        ],
    )


# Self-register the 'mcp' strategy when this module is imported.
strategies_lib.register_strategy('mcp', GoogleGenAIMCPStrategy)
