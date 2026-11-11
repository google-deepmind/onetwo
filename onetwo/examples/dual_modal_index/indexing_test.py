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

import copy
import shutil
import tempfile
from unittest import mock

from absl.testing import absltest
from etils import epath
import numpy as np
from onetwo import ot
from onetwo.backends import backend_registry
from onetwo.backends import backends_test_utils
from onetwo.core import content as content_lib
from onetwo.examples.dual_modal_index import indexing
from onetwo.stdlib.retrieval import config_serialization
from onetwo.stdlib.retrieval import retrieval_data_structures

Document = retrieval_data_structures.Document


class DualModalDocumentIndexTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.mkdtemp()

    @ot.make_executable
    async def fake_embed(
        content: str | content_lib.ChunkList, **kwargs
    ) -> np.ndarray:
      del kwargs
      if not content:
        raise ValueError('Cannot embed empty content.')
      if isinstance(content, content_lib.ChunkList):
        for chunk in content.chunks:
          if chunk.content == b'alice_img':
            return np.array([1.0, 0.0])
      text = str(content)
      if 'Alice' in text or 'visual_q' in text:
        return np.array([1.0, 0.0])
      return np.array([0.0, 1.0])

    @ot.make_executable
    async def fake_generate_content(
        prompt,
        **kwargs,
    ) -> content_lib.ChunkList:
      del prompt, kwargs
      return content_lib.ChunkList(
          [content_lib.Chunk(content='Visual: visual_q\nText: Alice')]
      )

    self.enter_context(mock.patch('onetwo.builtins.llm.embed', new=fake_embed))
    self.enter_context(
        mock.patch(
            'onetwo.builtins.llm.generate_content', new=fake_generate_content
        )
    )
    backend = backends_test_utils.LLMForTest()
    backend.register(register_tokenize=True)
    # The default sub-indexes create real tokenizer and embedder backends from
    # their configs: stub them out, so that the fakes registered above are used.
    self.enter_context(
        mock.patch.object(
            backend_registry, 'get_backend', return_value=mock.MagicMock()
        )
    )

  def tearDown(self):
    shutil.rmtree(self.temp_dir)
    super().tearDown()

  def _sample_docs(self) -> list[Document]:
    return [
        Document(
            doc_id='doc_alice',
            title='Alice',
            content=content_lib.ChunkList([
                content_lib.Chunk(
                    content=b'alice_img', content_type='image/png'
                ),
                content_lib.Chunk('Alice document text'),
            ]),
        ),
        Document(
            doc_id='doc_bob',
            title='Bob',
            content=content_lib.ChunkList([
                content_lib.Chunk(content=b'bob_img', content_type='image/png'),
                content_lib.Chunk('Bob document text'),
            ]),
        ),
    ]

  def test_add_retrieve_save_and_load(self):
    index = indexing.DualModalDocumentIndex(rrf_k=60, retrieval_depth=10)
    docs = self._sample_docs()
    ot.run(copy.deepcopy(index.add_docs(docs)))

    self.assertEqual(index.num_docs, 2)
    self.assertEqual(index.get_doc('doc_alice').doc_id, 'doc_alice')

    retrieved_with_scores = list(
        ot.run(index.retrieve_with_scores('Who is Alice?', max_results=2))
    )
    self.assertLen(retrieved_with_scores, 2)
    self.assertEqual(retrieved_with_scores[0][0].doc_id, 'doc_alice')
    self.assertGreater(retrieved_with_scores[0][1], retrieved_with_scores[1][1])

    retrieved = list(ot.run(index.retrieve('Who is Alice?', max_results=1)))
    self.assertLen(retrieved, 1)
    self.assertEqual(retrieved[0].doc_id, 'doc_alice')

    index.save(self.temp_dir)
    loaded = indexing.DualModalDocumentIndex(rrf_k=60, retrieval_depth=10)
    ot.run(copy.deepcopy(loaded.load(self.temp_dir)))
    self.assertEqual(loaded.num_docs, 2)
    self.assertEqual(loaded.get_doc('doc_alice').doc_id, 'doc_alice')
    loaded_retrieved = list(
        ot.run(loaded.retrieve('Who is Alice?', max_results=1))
    )
    self.assertLen(loaded_retrieved, 1)
    self.assertEqual(loaded_retrieved[0].doc_id, 'doc_alice')

  def test_deduplicates_chunked_text_sub_index_by_original_doc_id(self):
    docs = [
        Document(
            doc_id='doc_alice',
            title='Alice',
            content=content_lib.ChunkList([
                content_lib.Chunk(
                    content=b'alice_img', content_type='image/png'
                ),
                content_lib.Chunk('Alice ' * 100),
            ]),
        ),
        Document(
            doc_id='doc_bob',
            title='Bob',
            content=content_lib.ChunkList([
                content_lib.Chunk(content=b'bob_img', content_type='image/png'),
                content_lib.Chunk('Bob document text'),
            ]),
        ),
    ]
    index = indexing.DualModalDocumentIndex(rrf_k=60, retrieval_depth=10)
    ot.run(index.add_docs(docs))

    self.assertEqual(index.text_index.num_docs, 3)
    retrieved = list(ot.run(index.retrieve('Who is Alice?', max_results=2)))
    self.assertEqual([d.doc_id for d in retrieved], ['doc_alice', 'doc_bob'])

    index.save(self.temp_dir)
    loaded = indexing.DualModalDocumentIndex(rrf_k=60, retrieval_depth=10)
    ot.run(loaded.load(self.temp_dir))
    self.assertEqual(loaded.num_docs, index.num_docs)
    self.assertEqual(loaded.get_doc('doc_alice').doc_id, 'doc_alice')

  def test_skips_empty_modality_views_for_unimodal_documents(self):
    docs = [
        Document(
            doc_id='doc_both',
            title='Alice',
            content=content_lib.ChunkList([
                content_lib.Chunk(
                    content=b'alice_img', content_type='image/png'
                ),
                content_lib.Chunk('Alice document text'),
            ]),
        ),
        Document(
            doc_id='doc_text_only',
            title='Bob',
            content='Bob text without images',
        ),
        Document(
            doc_id='doc_img_only',
            title='Charlie',
            content=content_lib.ChunkList([
                content_lib.Chunk(
                    content=b'alice_img', content_type='image/png'
                ),
            ]),
        ),
    ]
    index = indexing.DualModalDocumentIndex(rrf_k=60, retrieval_depth=10)
    ot.run(index.add_docs(docs))

    self.assertEqual(index.text_index.num_docs, 2)
    self.assertEqual(index.image_index.num_docs, 2)
    self.assertEqual(index.get_doc('doc_both').doc_id, 'doc_both')
    self.assertEqual(index.get_doc('doc_text_only').doc_id, 'doc_text_only')
    self.assertEqual(index.get_doc('doc_img_only').title, 'Charlie')

    retrieved = list(ot.run(index.retrieve('Who is Alice?', max_results=3)))
    self.assertCountEqual(
        [d.doc_id for d in retrieved],
        ['doc_both', 'doc_text_only', 'doc_img_only'],
    )

  def test_filter_empty_documents_drops_empty_content(self):
    docs = [
        Document(doc_id='non_empty', content='Alice text'),
        Document(doc_id='empty_str', content=''),
        Document(doc_id='empty_chunks', content=content_lib.ChunkList([])),
    ]
    filtered = list(ot.run(indexing.FilterEmptyDocuments()(docs)))
    self.assertSequenceEqual([d.doc_id for d in filtered], ['non_empty'])

  def test_config_serialization_round_trip(self):
    index_dir = self.create_tempdir().full_path
    index = indexing.DualModalDocumentIndex(rrf_k=60, retrieval_depth=10)
    ot.run(index.add_docs(self._sample_docs()))

    index.save(index_dir)
    config_serialization.save_index_config(index, index_dir)
    loaded = config_serialization.make_index_from_config_file(
        index_dir, runtime_kwargs={'api_key': 'secret'}
    )
    ot.run(loaded.load(index_dir))

    assert isinstance(loaded, indexing.DualModalDocumentIndex)
    self.assertEqual(
        config_serialization.get_config(loaded),
        config_serialization.get_config(index),
    )
    self.assertEqual(
        loaded.text_index.inner_index.runtime_kwargs, {'api_key': 'secret'}
    )
    self.assertEqual(
        loaded.image_index.inner_index.runtime_kwargs, {'api_key': 'secret'}
    )
    saved_json = (
        epath.Path(index_dir) / config_serialization.INDEX_CONFIG_FILENAME
    ).read_text()
    self.assertNotIn('secret', saved_json)
    retrieved = list(ot.run(loaded.retrieve('Who is Alice?', max_results=1)))
    self.assertSequenceEqual([d.doc_id for d in retrieved], ['doc_alice'])

  def test_use_full_query_for_text_queries_text_index_with_raw_query(self):
    docs = [
        Document(doc_id='doc_alice', title='Alice', content='Alice text'),
        Document(doc_id='doc_bob', title='Bob', content='Bob text'),
    ]
    decomposed_index = indexing.DualModalDocumentIndex(
        rrf_k=60, retrieval_depth=10, use_full_query_for_text=False
    )
    ot.run(decomposed_index.add_docs(docs))
    decomposed_retrieved = list(
        ot.run(decomposed_index.retrieve('Who is Bob?', max_results=1))
    )
    self.assertSequenceEqual(
        [d.doc_id for d in decomposed_retrieved], ['doc_alice']
    )

    full_query_index = indexing.DualModalDocumentIndex(
        rrf_k=60, retrieval_depth=10, use_full_query_for_text=True
    )
    ot.run(full_query_index.add_docs(docs))
    full_query_retrieved = list(
        ot.run(full_query_index.retrieve('Who is Bob?', max_results=1))
    )
    self.assertSequenceEqual(
        [d.doc_id for d in full_query_retrieved], ['doc_bob']
    )


if __name__ == '__main__':
  absltest.main()
