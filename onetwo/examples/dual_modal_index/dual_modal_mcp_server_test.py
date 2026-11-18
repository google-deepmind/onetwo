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

"""Unit tests for `dual_modal_mcp_server`."""

import asyncio
import base64
import io
import json
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from mcp import types as mcp_types
from mcp.server.fastmcp import server as fastmcp_server
from onetwo import ot
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.examples.dual_modal_index import dual_modal_mcp_server
from onetwo.examples.dual_modal_index import indexing as dual_modal_indexing
from onetwo.stdlib.retrieval import chunking
from onetwo.stdlib.retrieval import config_serialization
from onetwo.stdlib.retrieval import corpus_rewriting
from onetwo.stdlib.retrieval import document_formatting
from onetwo.stdlib.retrieval import indexing
from onetwo.stdlib.retrieval import retrieval_data_structures
from PIL import Image as PILImage

_Document = retrieval_data_structures.Document


def _make_png_bytes(color: tuple[int, int, int]) -> bytes:
  image = PILImage.new('RGB', (4, 4), color=color)
  buffer = io.BytesIO()
  image.save(buffer, format='PNG')
  return buffer.getvalue()


def _call_tool(
    mcp: fastmcp_server.FastMCP,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
  """Invokes `mcp.call_tool` and unwraps FastMCP's structured result payload."""
  res: Any = asyncio.run(mcp.call_tool(tool_name, arguments))
  if isinstance(res, tuple):
    _, structured = res
    return structured.get('result', structured)
  return res


class DualModalMcpServerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._solar_png = _make_png_bytes((220, 20, 20))
    self._marine_png = _make_png_bytes((20, 20, 220))
    solar_png = self._solar_png

    @ot.make_executable
    async def fake_embed(
        content: str | content_lib.ChunkList, **kwargs
    ) -> list[float]:
      del kwargs
      if (
          isinstance(content, content_lib.ChunkList)
          and len(content.chunks) == 1
          and content.chunks[0].content == solar_png
      ):
        return [1.0, 0.0]
      text = str(content)
      if 'orange ring' in text or 'Observatory' in text:
        return [1.0, 0.0]
      return [0.0, 1.0]

    llm.embed.configure(fake_embed)

    self._docs = [
        _Document(
            doc_id='doc_solar',
            title='Solar Observatory Report',
            content=content_lib.ChunkList([
                content_lib.Chunk(
                    content=self._solar_png, content_type='image/png'
                ),
                content_lib.Chunk(
                    'The Solar Observatory tracks coronal mass ejections and'
                    ' solar flares.'
                ),
            ]),
            url='https://example.org/solar',
            metadata={'topic': 'astronomy'},
        ),
        _Document(
            doc_id='doc_marine',
            title='Marine Lab Report',
            content=content_lib.ChunkList([
                content_lib.Chunk(
                    content=self._marine_png, content_type='image/png'
                ),
                content_lib.Chunk(
                    'The Marine Lab monitors deep-sea hydrothermal vents and'
                    ' ocean currents.'
                ),
            ]),
            url='https://example.org/marine',
            metadata={'topic': 'oceanography'},
        ),
    ]

    self._index_dir = self.create_tempdir('dual_modal_index').full_path

    text_index = indexing.RewritingEmbeddingBasedDocumentIndex(
        rewriter=corpus_rewriting.ChunkingCorpusRewriter(
            chunker=chunking.ChunkByMaxWords(
                max_words_per_chunk=6, overlap_window=0
            )
        ),
        preserve_original_docs=True,
    )
    image_index = indexing.RewritingEmbeddingBasedDocumentIndex(
        rewriter=corpus_rewriting.FormattingCorpusRewriter(
            formatter=document_formatting.KeepImageFormatter()
        ),
        inner_index=indexing.EmbeddingBasedDocumentIndex(
            document_task_type=None
        ),
    )
    self._dual_index = dual_modal_indexing.DualModalDocumentIndex(
        text_index=text_index, image_index=image_index
    )
    ot.run(self._dual_index.add_docs(self._docs))
    self._dual_index.save(self._index_dir)
    config_serialization.save_index_config(self._dual_index, self._index_dir)
    self._mcp = dual_modal_mcp_server.create_server(
        self._index_dir,
        extra_mcp_config={'max_results': 2, 'snippet_max_chars': None},
    )

  @parameterized.named_parameters(
      ('base_id', 'doc_123', 'doc_123'),
      ('single_chunk', 'doc_123_chunk0', 'doc_123'),
      ('multi_digit_chunk', 'doc_123_chunk14', 'doc_123'),
      ('nested_chunk_suffix', 'doc_123_chunk1_chunk2', 'doc_123'),
      ('non_trailing_chunk_word', 'chunk0_doc_123', 'chunk0_doc_123'),
  )
  def test_strip_chunk_suffix(self, raw_id: str, expected: str):
    self.assertEqual(dual_modal_mcp_server.strip_chunk_suffix(raw_id), expected)

  def test_list_tools_registers_three_core_tools(self):
    tools = asyncio.run(self._mcp.list_tools())
    self.assertSequenceEqual(
        [tool.name for tool in tools],
        ['search_text_chunks', 'search_images', 'get_document'],
    )
    tools_by_name = {tool.name: tool for tool in tools}
    for tool in tools:
      self.assertIn('verbatim', tool.description or '')
    for search_tool_name in ('search_text_chunks', 'search_images'):
      max_results_schema = tools_by_name[search_tool_name].inputSchema[
          'properties'
      ]['max_results']
      self.assertEqual(max_results_schema['default'], 2)

  def test_search_text_chunks(self):
    hits = _call_tool(
        self._mcp, 'search_text_chunks', {'query': 'Solar Observatory'}
    )
    self.assertLen(hits, 2)
    self.assertEqual(hits[0]['docid'], 'doc_solar')
    self.assertStartsWith(hits[0]['chunk_id'], 'doc_solar_chunk')
    self.assertEqual(hits[0]['title'], 'Solar Observatory Report')
    self.assertAlmostEqual(hits[0]['score'], 1.0, places=4)
    self.assertIn('Observatory', hits[0]['snippet'])
    self.assertNotIn('truncated', hits[0]['snippet'])

    with self.subTest('per_call_max_results_overrides_server_default'):
      top_one_hit = _call_tool(
          self._mcp,
          'search_text_chunks',
          {'query': 'Solar Observatory', 'max_results': 1},
      )
      self.assertLen(top_one_hit, 1)
      self.assertEqual(top_one_hit[0]['docid'], 'doc_solar')

    with self.subTest('truncates_to_snippet_max_chars'):
      truncated_mcp = dual_modal_mcp_server.create_server(
          self._index_dir,
          extra_mcp_config={'max_results': 1, 'snippet_max_chars': 10},
      )
      short_hits = _call_tool(
          truncated_mcp, 'search_text_chunks', {'query': 'Solar Observatory'}
      )
      expected_prefix = hits[0]['snippet'][:10]
      self.assertEqual(
          short_hits[0]['snippet'],
          f'{expected_prefix}{dual_modal_mcp_server._TRUNCATION_SUFFIX}',
      )
      self.assertEqual(
          dual_modal_mcp_server._truncate_to_max_chars('hello', 5), 'hello'
      )

  def test_search_images_returns_original_title_and_raw_image_bytes(self):
    res = _call_tool(
        self._mcp, 'search_images', {'query': 'bright orange ring'}
    )
    self.assertFalse(res.isError)
    first_meta, first_img, _, second_img = res.content
    self.assertEqual(
        json.loads(first_meta.text),
        {
            'docid': 'doc_solar',
            'title': 'Solar Observatory Report',
            'score': 1.0,
        },
    )
    self.assertEqual(first_img.mimeType, 'image/png')
    self.assertEqual(base64.b64decode(first_img.data), self._solar_png)
    self.assertEqual(base64.b64decode(second_img.data), self._marine_png)

    with self.subTest('per_call_max_results_overrides_server_default'):
      top_one_result = _call_tool(
          self._mcp,
          'search_images',
          {'query': 'bright orange ring', 'max_results': 1},
      )
      self.assertLen(top_one_result.content, 2)

  def test_concurrent_search_calls_succeed_in_shared_event_loop(self):
    async def run_concurrent_searches() -> tuple[Any, Any]:
      return await asyncio.gather(
          self._mcp.call_tool(
              'search_text_chunks',
              {'query': 'Solar Observatory', 'max_results': 1},
          ),
          self._mcp.call_tool(
              'search_images',
              {'query': 'bright orange ring', 'max_results': 1},
          ),
      )

    text_result, image_result = asyncio.run(run_concurrent_searches())
    _, text_structured = text_result
    self.assertLen(text_structured['result'], 1)
    self.assertEqual(text_structured['result'][0]['docid'], 'doc_solar')
    self.assertFalse(image_result.isError)
    self.assertLen(image_result.content, 2)

  @parameterized.named_parameters(
      ('by_base_doc_id', 'doc_solar'),
      ('by_chunk_id', 'doc_solar_chunk2'),
  )
  def test_get_document(self, lookup_id: str):
    res = _call_tool(self._mcp, 'get_document', {'docid': lookup_id})
    header_block, image_block, text_block = res.content
    self.assertEqual(
        json.loads(header_block.text),
        {
            'docid': 'doc_solar',
            'title': 'Solar Observatory Report',
            'url': 'https://example.org/solar',
            'metadata': {'topic': 'astronomy'},
        },
    )
    self.assertIsInstance(image_block, mcp_types.ImageContent)
    self.assertEqual(image_block.mimeType, 'image/png')
    self.assertEqual(base64.b64decode(image_block.data), self._solar_png)
    self.assertIn('solar flares', text_block.text)

    with self.assertLogs(level='WARNING') as log_output:
      missing = _call_tool(self._mcp, 'get_document', {'docid': 'unknown_doc'})
    self.assertEqual(missing.content[0].text, 'null')
    self.assertLen(log_output.output, 1)
    self.assertIn(
        "Original document 'unknown_doc' not found.",
        log_output.output[0],
    )

  def test_create_server_serves_an_index_object(self):
    mcp = dual_modal_mcp_server.create_server(
        self._dual_index, extra_mcp_config={'max_results': 1}
    )
    hits = _call_tool(mcp, 'search_text_chunks', {'query': 'Solar Observatory'})
    self.assertLen(hits, 1)
    self.assertEqual(hits[0]['docid'], 'doc_solar')
    res = _call_tool(mcp, 'get_document', {'docid': 'doc_solar'})
    self.assertEqual(json.loads(res.content[0].text)['docid'], 'doc_solar')

  def test_create_server_host_and_port(self):
    self.assertEqual(self._mcp.settings.host, 'localhost')
    self.assertEqual(self._mcp.settings.port, 8775)
    http_mcp = dual_modal_mcp_server.create_server(
        self._index_dir, host='0.0.0.0', port=9123
    )
    self.assertEqual(http_mcp.settings.host, '0.0.0.0')
    self.assertEqual(http_mcp.settings.port, 9123)

  def test_create_server_rejects_unknown_extra_mcp_config_keys(self):
    with self.assertRaisesRegex(
        ValueError, r"Unknown extra_mcp_config keys: \['max_result'\]"
    ):
      dual_modal_mcp_server.create_server(
          self._index_dir, extra_mcp_config={'max_result': 3}
      )

  def test_create_server_rejects_extra_backend_kwargs_with_index_object(self):
    with self.assertRaisesRegex(ValueError, 'extra_backend_kwargs'):
      dual_modal_mcp_server.create_server(
          self._dual_index, extra_backend_kwargs={'api_key': 'key'}
      )

  def test_create_server_requires_preserved_original_docs(self):
    index_without_orig = dual_modal_indexing.DualModalDocumentIndex(
        text_index=indexing.RewritingEmbeddingBasedDocumentIndex(
            preserve_original_docs=False
        ),
        image_index=indexing.RewritingEmbeddingBasedDocumentIndex(),
    )
    ot.run(index_without_orig.add_docs(self._docs))
    with self.subTest('index_object'):
      with self.assertRaisesRegex(ValueError, 'preserve_original_docs=True'):
        dual_modal_mcp_server.create_server(index_without_orig)

    with self.subTest('saved_index'):
      no_orig_dir = self.create_tempdir('no_orig').full_path
      index_without_orig.save(no_orig_dir)
      config_serialization.save_index_config(index_without_orig, no_orig_dir)
      with self.assertRaisesRegex(ValueError, 'preserve_original_docs=True'):
        dual_modal_mcp_server.create_server(no_orig_dir)

  def test_create_server_requires_a_dual_modal_index(self):
    plain_index = indexing.RewritingEmbeddingBasedDocumentIndex()
    ot.run(plain_index.add_docs(self._docs))
    with self.subTest('index_object'):
      with self.assertRaisesRegex(
          TypeError,
          'requires a DualModalDocumentIndex, but got a'
          ' RewritingEmbeddingBasedDocumentIndex',
      ):
        dual_modal_mcp_server.create_server(plain_index)

    with self.subTest('saved_index'):
      plain_dir = self.create_tempdir('plain').full_path
      plain_index.save(plain_dir)
      config_serialization.save_index_config(plain_index, plain_dir)
      with self.assertRaisesRegex(
          TypeError, 'requires a DualModalDocumentIndex'
      ):
        dual_modal_mcp_server.create_server(plain_dir)

  def test_start_mcp_runs_the_server_with_the_transport(self):
    with mock.patch.object(
        fastmcp_server.FastMCP, 'run', autospec=True
    ) as mock_run:
      dual_modal_mcp_server.start_mcp(
          self._dual_index,
          transport='streamable-http',
          port=9124,
          extra_mcp_config={'max_results': 1},
      )
    mock_run.assert_called_once()
    mcp = mock_run.call_args.args[0]
    self.assertEqual(mcp.settings.port, 9124)
    self.assertEqual(
        mock_run.call_args.kwargs, {'transport': 'streamable-http'}
    )

  def test_content_to_blocks_rejects_non_bytes_image_chunk(self):
    pil_chunk = content_lib.Chunk(PILImage.new('RGB', (2, 2)))
    with self.assertRaisesRegex(TypeError, 'Expected bytes'):
      dual_modal_mcp_server._content_to_blocks(
          content_lib.ChunkList([pil_chunk])
      )


if __name__ == '__main__':
  absltest.main()
