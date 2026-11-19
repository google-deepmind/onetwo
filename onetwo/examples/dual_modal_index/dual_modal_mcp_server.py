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

r"""Dual-Modal Index MCP Server library.

Exposes a pre-built `DualModalDocumentIndex` (see
`onetwo.examples.dual_modal_index.indexing`), i.e. a text-only and an image-only
retrieval index alongside the preserved unchunked source documents, as MCP tools
(`search_text_chunks`, `search_images`, and `get_document`) for agentic
multimodal retrieval.

Unlike `onetwo.stdlib.retrieval.mcp.mcp_server`, which wraps a single document
index behind one search tool, this server has a search tool per modality and
resolves `get_document` against the preserved original documents. It has the
same API: serve an index (a saved index directory or an index object) with
`start_mcp`, or build the server with `create_server` to run it yourself or to
test it. See ``dual_modal_mcp_server_main.py`` in this directory for a launch
script.
"""

import base64
from collections.abc import Mapping
import dataclasses
import json
import re
from typing import Any, Literal

from absl import logging
from mcp import types as mcp_types
from mcp.server.fastmcp import server
from onetwo.agents import agent as agent_lib
from onetwo.core import content as content_lib
from onetwo.examples.dual_modal_index import indexing
from onetwo.stdlib.retrieval import index_building
from onetwo.stdlib.retrieval import indexing as indexing_lib

# Passed as additional_instructions to AgenticRetrievalStrategy when querying
# a dual-modal MCP server exposing search_text_chunks, search_images, and
# get_document. Empirically tuned on SpanishGov QA, though the guidelines are
# corpus-agnostic; customize as needed for your dataset.
AGENTIC_ADDITIONAL_INSTRUCTIONS = (
    'Retrieval and verification guidelines:\n'
    '- Route by modality: `search_images` indexes visual content only (no'
    ' surrounding text, titles, or metadata), so pass purely visual'
    ' descriptions of positive observable features (exclude entity names,'
    ' non-visual facts, negated exclusions, or unstated visual assumptions).'
    ' Use `search_text_chunks` with short, focused natural-language queries'
    ' (avoid quotes or `AND`/`OR`/`-` operators; include corpus-language terms'
    ' when applicable) for dense semantic retrieval. Query both modalities'
    ' when a question combines positive textual and visual constraints; when'
    ' one constraint is already highly selective or a visual condition is'
    ' negated, start from the selective modality and verify the remaining'
    ' conditions via `get_document`.\n'
    '- Verify intersections and compare finalists via `get_document`: Search'
    ' results show only potentially truncated snippets or isolated images.'
    ' Always inspect full documents with `get_document` in parallel batches'
    ' (e.g., 5-8 per turn), prioritizing `docid`s present in both text and'
    ' image results. Do not stop at the first plausible match—read remaining'
    ' intersection candidates (and close top hits when details are subtle) to'
    ' compare finalists side-by-side and calibrate ambiguous visual or textual'
    " attributes using only each document's own text and images, not outside"
    ' knowledge. If none satisfy every condition, rephrase or broaden your'
    " retrieval queries without relaxing the question's conditions.\n"
    '- Check numbers, negations, tense, and cardinality: Read numbers exactly'
    ' and negations literally. Unless a condition explicitly refers to the'
    ' past, judge it by what the document presents as true now rather than'
    ' former or past facts (even if visual similarity is high). Prioritize'
    ' candidates that satisfy all explicit factual conditions, and commit to'
    ' the single best-supported answer for single-target questions.\n'
    '- Cite exact `docid` strings verbatim: Cite only documents verified via'
    ' `get_document`, copying the full `docid` into square brackets without'
    ' modifying or stripping prefixes (e.g., `[DOC_20]`, never `[20]`), and'
    ' write `Exact Answer:` in plain text without Markdown formatting (e.g.,'
    ' no `**`).'
)

# Final-turn prompt passed to AgenticRetrievalStrategy when max_steps is hit.
AGENTIC_FINAL_INSTRUCTION = (
    f'{agent_lib.DEFAULT_FINAL_INSTRUCTION} Base your answer strictly on'
    ' documents you have already verified via `get_document`; never guess or'
    ' cite unread `docid`s from search snippets. Cite verified evidence'
    ' documents in square brackets using the exact `docid` string verbatim,'
    ' and write `Exact Answer:` in plain text without Markdown formatting.'
)

_CHUNK_SUFFIX_RE = re.compile(r'(_chunk\d+)+$')
# Round similarity scores so tool JSON avoids noisy full-precision floats.
_SCORE_DECIMAL_PLACES = 4
# Appended to text snippets that exceed snippet_max_chars.
_TRUNCATION_SUFFIX = '... [truncated; call get_document(docid) for full text]'


def strip_chunk_suffix(docid: str) -> str:
  """Strips trailing `_chunkN` suffixes from a document or chunk ID."""
  return _CHUNK_SUFFIX_RE.sub('', docid)


def _truncate_to_max_chars(text: str, max_chars: int) -> str:
  """Truncates `text` to `max_chars` characters (`< 0` disables truncation)."""
  if max_chars < 0 or len(text) <= max_chars:
    return text
  return f'{text[:max_chars]}{_TRUNCATION_SUFFIX}'


def _lookup_original_doc(
    dual_index: indexing.DualModalDocumentIndex,
    docid: str,
) -> indexing.Document | None:
  """Looks up an unchunked original document in dual_index."""
  base_id = strip_chunk_suffix(docid)
  try:
    return dual_index.get_doc(base_id)
  except KeyError:
    logging.warning('Original document %r not found.', base_id)
    return None


def _resolve_title(
    doc: indexing.Document,
    base_id: str,
    dual_index: indexing.DualModalDocumentIndex,
) -> str:
  """Returns doc.title, falling back to the original document's title."""
  if doc.title:
    return doc.title
  orig_doc = _lookup_original_doc(dual_index, base_id)
  return orig_doc.title if orig_doc is not None else ''


def _content_to_blocks(
    content: str | content_lib.ChunkList,
) -> list[mcp_types.ContentBlock]:
  """Converts document `content` into ordered MCP text and image blocks."""
  if isinstance(content, str):
    return [mcp_types.TextContent(type='text', text=content)] if content else []
  blocks: list[mcp_types.ContentBlock] = []
  for chunk in content:
    if chunk.content_type.startswith('image/'):
      if not isinstance(chunk.content, bytes):
        raise TypeError(
            f'Expected bytes for {chunk.content_type!r} chunk, got'
            f' {type(chunk.content).__name__}.'
        )
      blocks.append(
          mcp_types.ImageContent(
              type='image',
              data=base64.b64encode(chunk.content).decode('ascii'),
              mimeType=chunk.content_type,
          )
      )
    elif text_piece := str(chunk):
      blocks.append(mcp_types.TextContent(type='text', text=text_piece))
  return blocks


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ExtraMcpConfig:
  """The parsed `extra_mcp_config` of `create_server` (see there).

  Private on purpose: the extra settings are experimental, so they are not
  exposed as a public type. Callers pass a plain mapping, which is validated
  and parsed with `from_mapping`.

  Attributes:
    max_results: Default number of hits returned by each search tool.
    snippet_max_chars: Maximum number of characters of each text snippet, or
      None to return the full text.
  """

  max_results: int = 5
  snippet_max_chars: int | None = 1200

  @classmethod
  def from_mapping(
      cls, extra_mcp_config: Mapping[str, Any] | None
  ) -> '_ExtraMcpConfig':
    """Returns the config with the settings of `extra_mcp_config`.

    Args:
      extra_mcp_config: The `extra_mcp_config` of `create_server`, or None for
        the defaults.

    Raises:
      ValueError: If `extra_mcp_config` has an unsupported key.
    """
    settings = dict(extra_mcp_config or {})
    supported = {field.name for field in dataclasses.fields(cls)}
    unknown = settings.keys() - supported
    if unknown:
      raise ValueError(
          f'Unknown extra_mcp_config keys: {sorted(unknown)}. Supported keys:'
          f' {sorted(supported)}.'
      )
    return cls(**settings)


def _check_dual_modal_index(
    index: indexing_lib.DocumentIndex,
) -> indexing.DualModalDocumentIndex:
  """Returns `index` if the server can serve it.

  Args:
    index: The index to serve, e.g. as loaded by `index_building.load_index`.

  Raises:
    TypeError: If `index` is not a `DualModalDocumentIndex`.
    ValueError: If its text index does not have `preserve_original_docs=True`,
      which `get_document` needs.
  """
  if not isinstance(index, indexing.DualModalDocumentIndex):
    raise TypeError(
        'The Dual-Modal Index MCP server requires a DualModalDocumentIndex, but'
        f' got a {type(index).__name__}.'
    )
  if not index.text_index.preserve_original_docs:
    raise ValueError(
        'The Dual-Modal Index MCP server requires the text_index of the'
        ' DualModalDocumentIndex to have preserve_original_docs=True.'
    )
  return index


def create_server(
    index: str | indexing_lib.DocumentIndex,
    *,
    host: str = 'localhost',
    port: int = 8775,
    extra_backend_kwargs: Mapping[str, Any] | None = None,
    extra_mcp_config: Mapping[str, Any] | None = None,
) -> server.FastMCP:
  """Creates the Dual-Modal Index MCP server for a `DualModalDocumentIndex`.

  The server has the `search_text_chunks`, `search_images` and `get_document`
  tools. To serve it, call its `run` method with a transport, or use
  `start_mcp` instead.

  Args:
    index: The index to serve: the directory of a saved index, loaded with
      ``index_building.load_index``, or an index object. Either must be a
      `DualModalDocumentIndex` whose text index has
      ``preserve_original_docs=True``. Its text index backs `search_text_chunks`
      and `get_document`, and its image index `search_images`.
    host: Host/interface to bind the server on. Only used with
      ``transport='streamable-http'``.
    port: Port to listen on. Only used with ``transport='streamable-http'``.
    extra_backend_kwargs: Extra keyword arguments for the constructors of the
      backends that this call creates (e.g. an API key) when ``index`` is a
      directory, see ``index_building.load_index``. Must be None for an index
      object, which already has its backends. Usually not needed: credentials
      are read from the environment (e.g. `GOOGLE_API_KEY`). Not a stable part
      of the API.
    extra_mcp_config: Extra, advanced settings of the MCP tools, or None to use
      the defaults. Unlike the other arguments, these settings are NOT part of
      the stable API. They are still evolving, so keys may be added, renamed or
      removed, and their meanings and defaults may change, in any release and
      without notice. Most users should not need to set them. Every key is
      optional (missing keys use their defaults), and unknown keys raise a
      ``ValueError`` (rather than being ignored) so that typos and removed keys
      fail loudly. Currently supported keys: - ``'max_results'`` (``int``,
      default 5): default number of hits returned by ``search_text_chunks`` and
      ``search_images`` (the agent can pass another ``max_results`` per call). -
      ``'snippet_max_chars'`` (``int | None``, default 1200, about 300 tokens):
      maximum number of characters of each ``search_text_chunks`` snippet, or
      None to return the full text. Images are never truncated, nor are
      ``get_document`` results.

  Returns:
    A configured FastMCP server instance.

  Raises:
    ValueError: If ``extra_mcp_config`` has an unsupported key,
      ``extra_backend_kwargs`` is set for an index object, or the text index
      does not have ``preserve_original_docs=True``.
    TypeError: If the index is not a ``DualModalDocumentIndex``.
  """
  config = _ExtraMcpConfig.from_mapping(extra_mcp_config)
  if isinstance(index, str):
    logging.info('Loading dual-modal index from: %s', index)
    dual_index = _check_dual_modal_index(
        index_building.load_index(
            index, extra_backend_kwargs=extra_backend_kwargs
        )
    )
  elif extra_backend_kwargs is not None:
    raise ValueError(
        'extra_backend_kwargs are only used to load an index from a directory,'
        f' not with an index object ({type(index).__name__}).'
    )
  else:
    dual_index = _check_dual_modal_index(index)
  logging.info(
      'Serving a dual-modal index with %d text documents and %d image'
      ' documents.',
      dual_index.text_index.num_docs,
      dual_index.image_index.num_docs,
  )

  # `host` and `port` are only used with the streamable-http transport.
  mcp = server.FastMCP(
      'OneTwo Dual-Modal Index MCP Server', host=host, port=port
  )

  # `_truncate_to_max_chars` does not truncate with `snippet_max_chars=-1`.
  snippet_max_chars = (
      -1 if config.snippet_max_chars is None else config.snippet_max_chars
  )
  default_max_results = config.max_results

  @mcp.tool()
  async def search_text_chunks(
      query: str,
      max_results: int = default_max_results,
  ) -> list[dict[str, Any]]:
    """Searches document text chunks by dense semantic similarity.

    Use this tool as the primary search for conceptual or factual text
    queries. Snippets ending with `... [truncated; ...]` are incomplete; call
    `get_document(docid)` to read the full unchunked source document. Treat
    every `docid` as an opaque string: pass it unchanged to `get_document`
    and cite the exact `docid` verbatim without modifying or stripping any
    prefix or characters.

    Args:
      query: Natural-language semantic search query (do not use quotes, AND,
        OR, or -exclusion operators).
      max_results: Maximum number of text chunks to return.

    Returns:
      Top-k hit dicts with keys `docid` (exact parent document ID to pass to
      `get_document` and cite verbatim), `chunk_id`, `title`, `score`, and
      `snippet`.
    """
    results_with_scores = await dual_index.text_index.retrieve_with_scores(
        query, max_results=max_results
    )
    hits: list[dict[str, Any]] = []
    for doc, score in results_with_scores:
      base_id = strip_chunk_suffix(doc.doc_id)
      hits.append({
          'docid': base_id,
          'chunk_id': doc.doc_id,
          'title': _resolve_title(doc, base_id, dual_index),
          'score': round(score, _SCORE_DECIMAL_PLACES),
          'snippet': _truncate_to_max_chars(doc.text, snippet_max_chars),
      })
    return hits

  del search_text_chunks

  @mcp.tool()
  async def search_images(
      query: str,
      max_results: int = default_max_results,
  ) -> mcp_types.CallToolResult:
    """Searches the image-only index using a purely visual description.

    Images are indexed by visual content only (surrounding document text,
    titles, and metadata are not in the image embeddings), so describe only
    positive observable visual features (exclude non-visual text, entity
    names, negated exclusions, or unstated visual assumptions).
    Cross-reference matched `docid` values via `get_document(docid)` or
    `search_text_chunks`. Treat every `docid` as an opaque string and cite it
    verbatim without modifying or stripping any prefix or characters.

    Args:
      query: Purely visual description of positive observable image features
        (exclude entity names, titles, non-visual facts, or negated
        exclusions).
      max_results: Maximum number of image hits to return.

    Returns:
      A `CallToolResult` interleaving JSON metadata `TextContent` blocks
      (`docid`, `title`, `score`) with `ImageContent` blocks for the top-k
      hits.
    """
    results_with_scores = await dual_index.image_index.retrieve_with_scores(
        query, max_results=max_results
    )
    blocks: list[mcp_types.ContentBlock] = []
    for doc, score in results_with_scores:
      base_id = strip_chunk_suffix(doc.doc_id)
      hit_metadata = {
          'docid': base_id,
          'title': _resolve_title(doc, base_id, dual_index),
          'score': round(score, _SCORE_DECIMAL_PLACES),
      }
      blocks.append(
          mcp_types.TextContent(
              type='text',
              text=json.dumps(hit_metadata, ensure_ascii=False),
          )
      )
      blocks.extend(_content_to_blocks(doc.content))
    if not blocks:
      blocks.append(mcp_types.TextContent(type='text', text='[]'))
    return mcp_types.CallToolResult(content=blocks)

  del search_images

  @mcp.tool()
  def get_document(docid: str) -> mcp_types.CallToolResult:
    """Retrieves the full unchunked source document for `docid`.

    Trailing `_chunkN` suffixes are stripped automatically so either a base
    `docid` or a `chunk_id` from `search_text_chunks` may be passed. When
    citing a document in your answer, use the returned header `docid` string
    verbatim without modifying or stripping any prefix or characters.

    Args:
      docid: Exact document or chunk ID string returned by a search tool.

    Returns:
      A `CallToolResult` whose `content` starts with a JSON header
      `TextContent` block (`docid`, `title`, `url`, `metadata`) followed by
      the preserved document's `TextContent` and `ImageContent` blocks in
      order, or `'null'` if `docid` is not found.
    """
    doc = _lookup_original_doc(dual_index, docid)
    if doc is None:
      return mcp_types.CallToolResult(
          content=[mcp_types.TextContent(type='text', text='null')]
      )
    header = {
        'docid': doc.doc_id,
        'title': doc.title,
        'url': doc.url,
        'metadata': doc.metadata,
    }
    blocks: list[mcp_types.ContentBlock] = [
        mcp_types.TextContent(
            type='text', text=json.dumps(header, ensure_ascii=False)
        ),
        *_content_to_blocks(doc.content),
    ]
    return mcp_types.CallToolResult(content=blocks)

  del get_document

  return mcp


def start_mcp(
    index: str | indexing_lib.DocumentIndex,
    *,
    transport: Literal['stdio', 'streamable-http'] = 'stdio',
    host: str = 'localhost',
    port: int = 8775,
    extra_backend_kwargs: Mapping[str, Any] | None = None,
    extra_mcp_config: Mapping[str, Any] | None = None,
) -> None:
  """Serves the Dual-Modal Index MCP server until it is stopped.

  Builds the server with `create_server` (see there for the arguments other
  than `transport`) and runs it. This blocks: e.g. to evaluate a strategy with
  the server (`rag.evaluate` with `mcp_urls=['http://...']` and
  `strategy='agentic_retrieval'`), start it in another process.

  Args:
    index: The index to serve: the directory of a saved `DualModalDocumentIndex`
      or an index object, see `create_server`.
    transport: MCP transport: 'stdio' (e.g. when an MCP client launches the
      server as a subprocess) or 'streamable-http' (serves
      http://<host>:<port>/mcp).
    host: Host/interface to bind the server on. Only used with
      ``transport='streamable-http'``.
    port: Port to listen on. Only used with ``transport='streamable-http'``.
    extra_backend_kwargs: See `create_server`.
    extra_mcp_config: EXPERIMENTAL extra settings of the tools, see
      `create_server`.
  """
  mcp = create_server(
      index,
      host=host,
      port=port,
      extra_backend_kwargs=extra_backend_kwargs,
      extra_mcp_config=extra_mcp_config,
  )
  mcp.run(transport=transport)
