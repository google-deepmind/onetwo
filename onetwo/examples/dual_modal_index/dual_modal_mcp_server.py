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

"""Dual-Modal Index MCP Server library.

Exposes pre-built OneTwo text-only and image-only retrieval indices alongside
their preserved unchunked source documents as MCP tools (search_text_chunks,
search_images, and get_document) for agentic multimodal retrieval.

Unlike onetwo.stdlib.retrieval.mcp.mcp_server, which wraps a single
EmbeddingBasedDocumentIndex behind one search tool, this server loads separate
text and image rewriting indices with modality-specific search tools and
resolves get_document against the preserved original documents.
"""

import base64
from collections.abc import Mapping
import json
import re
from typing import Any

from absl import logging
from mcp import types as mcp_types
from mcp.server.fastmcp import server
from onetwo.agents import agent as agent_lib
from onetwo.core import content as content_lib
from onetwo.examples.dual_modal_index import indexing
from onetwo.stdlib.retrieval import index_building

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
  if not dual_index.text_index.preserve_original_docs:
    return None
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


# TODO: Update the signature to match the one in
# onetwo.stdlib.retrieval.mcp.mcp_server.
def create_dual_modal_server(
    *,
    index_path: str,
    max_results: int = 5,
    snippet_max_chars: int = 1200,
    port: int = 8775,
    transport: str = 'streamable-http',
    host: str = 'localhost',
    include_get_document: bool = True,
    index_runtime_kwargs: Mapping[str, Any] | None = None,
) -> server.FastMCP:
  """Creates and configures the Dual-Modal Index MCP server.

  Args:
    index_path: Directory of a saved DualModalDocumentIndex (e.g. saved by
      rag.build_index). Its text index backs search_text_chunks, and its image
      index search_images.
    max_results: Default number of top hits returned by each search tool.
    snippet_max_chars: Maximum characters per text snippet (`-1` disables
      truncation). Defaults to `1200` (~300 tokens).
    port: Port to bind for HTTP/SSE transports.
    transport: MCP transport (`'stdio'`, `'sse'`, or `'streamable-http'`).
    host: Host/interface to bind for HTTP/SSE transports.
    include_get_document: Whether to register the `get_document` tool.
    index_runtime_kwargs: Optional runtime kwargs forwarded to
      `index_building.load_index`. `None` uses default backend settings.

  Returns:
    A configured `FastMCP` server instance with dual-modal retrieval tools.

  Raises:
    TypeError: If index_path is not a saved DualModalDocumentIndex.
    ValueError: If include_get_document=True and dual_index.text_index does not
      have preserve_original_docs=True.
  """
  if transport == 'stdio':
    mcp = server.FastMCP('OneTwo Dual-Modal Index MCP Server')
  else:
    mcp = server.FastMCP(
        'OneTwo Dual-Modal Index MCP Server', host=host, port=port
    )

  logging.info('Loading dual-modal index from: %s', index_path)
  dual_index = index_building.load_index(
      index_path, extra_backend_kwargs=index_runtime_kwargs
  )
  if not isinstance(dual_index, indexing.DualModalDocumentIndex):
    raise TypeError(f'{index_path!r} is not a saved DualModalDocumentIndex.')
  logging.info(
      'Dual-modal index loaded (%d text documents, %d image documents).',
      dual_index.text_index.num_docs,
      dual_index.image_index.num_docs,
  )

  if include_get_document and not dual_index.text_index.preserve_original_docs:
    raise ValueError(
        'include_get_document=True requires text_index to be saved with'
        ' preserve_original_docs=True.'
    )

  default_max_results = max_results

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

  if include_get_document:

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
