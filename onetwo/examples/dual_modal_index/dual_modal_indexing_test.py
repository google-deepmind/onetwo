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
import dataclasses
import json
import shutil
import tempfile
from unittest import mock

from absl.testing import absltest
from etils import epath
import numpy as np
from onetwo import ot
from onetwo.backends import backend_registry
from onetwo.backends import backends_test_utils
from onetwo.builtins import llm
from onetwo.core import content as content_lib
from onetwo.core import routing
from onetwo.examples.dual_modal_index import indexing
from onetwo.stdlib.retrieval import config_serialization
from onetwo.stdlib.retrieval import rag_defaults
from onetwo.stdlib.retrieval import retrieval_data_structures

Document = retrieval_data_structures.Document


@dataclasses.dataclass
class _DecomposerForTest:
  """Generation backend that gives the same reply to every prompt."""

  reply: str
  prompts: list[str] = dataclasses.field(default_factory=list)

  @ot.make_executable
  async def generate_content(
      self, prompt: llm.Prompt, **kwargs
  ) -> content_lib.ChunkList:
    del kwargs
    self.prompts.append(str(prompt))
    return content_lib.ChunkList([content_lib.Chunk(self.reply)])

  def register(self) -> None:
    llm.generate_content.configure(self.generate_content)


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

    self.enter_context(mock.patch('onetwo.builtins.llm.embed', new=fake_embed))
    backend = backends_test_utils.LLMForTest()
    backend.register(register_tokenize=True)
    # Indexes create their backends from their configs: generation configs get
    # the fake decomposer, other configs (tokenizer, embedder) a stub, so that
    # the fakes registered above are used.
    self.decomposer = _DecomposerForTest(reply='Visual: visual_q\nText: Alice')

    def fake_get_backend(config, **runtime_kwargs):
      del runtime_kwargs
      if isinstance(config, backend_registry.GenerateBackendConfig):
        return self.decomposer
      return mock.MagicMock()

    self.enter_context(
        mock.patch.object(
            backend_registry, 'get_backend', side_effect=fake_get_backend
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
    backend_config = backend_registry.GenerateBackendConfig(
        backend_class_name='GoogleGenAIAPI', model_name='decomposer'
    )
    index = indexing.DualModalDocumentIndex(
        rrf_k=60, retrieval_depth=10, backend_config=backend_config
    )
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
    self.assertEqual(loaded.runtime_kwargs, {'api_key': 'secret'})
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

  def test_loads_config_without_backend_config_with_default(self):
    index_dir = self.create_tempdir().full_path
    index = indexing.DualModalDocumentIndex(rrf_k=60, retrieval_depth=10)
    config_serialization.save_index_config(index, index_dir)
    config_path = (
        epath.Path(index_dir) / config_serialization.INDEX_CONFIG_FILENAME
    )
    saved = json.loads(config_path.read_text())
    del saved['index']['config']['backend_config']
    config_path.write_text(json.dumps(saved))

    loaded = config_serialization.make_index_from_config_file(index_dir)

    assert isinstance(loaded, indexing.DualModalDocumentIndex)
    self.assertEqual(
        loaded.backend_config,
        rag_defaults.default_index_generate_backend_config(),
    )

  def test_decomposes_query_with_own_backend(self):
    backend_config = backend_registry.GenerateBackendConfig(
        backend_class_name='GoogleGenAIAPI', model_name='decomposer'
    )
    index = indexing.DualModalDocumentIndex(
        rrf_k=60, retrieval_depth=10, backend_config=backend_config
    )
    ot.run(index.add_docs(self._sample_docs()))

    retrieved = list(ot.run(index.retrieve('Who is Bob?', max_results=1)))

    # The decomposed queries (visual_q, Alice) both rank Alice first.
    self.assertSequenceEqual([d.doc_id for d in retrieved], ['doc_alice'])
    self.assertLen(self.decomposer.prompts, 1)
    self.assertIn('Who is Bob?', self.decomposer.prompts[0])

  def test_without_backend_config_decomposes_with_registered_model(self):
    index = indexing.DualModalDocumentIndex(
        rrf_k=60, retrieval_depth=10, backend_config=None
    )
    ot.run(index.add_docs(self._sample_docs()))

    with routing.RegistryContext():
      self.decomposer.register()
      retrieved = list(ot.run(index.retrieve('Who is Bob?', max_results=1)))

    self.assertSequenceEqual([d.doc_id for d in retrieved], ['doc_alice'])
    self.assertLen(self.decomposer.prompts, 1)

  def test_without_any_model_searches_with_full_query(self):
    index = indexing.DualModalDocumentIndex(
        rrf_k=60, retrieval_depth=10, backend_config=None
    )
    ot.run(index.add_docs(self._sample_docs()))

    with self.assertLogs(level='WARNING') as logs:
      retrieved = list(ot.run(index.retrieve('Who is Bob?', max_results=1)))

    # The full query ranks Bob first in both indexes.
    self.assertSequenceEqual([d.doc_id for d in retrieved], ['doc_bob'])
    self.assertEmpty(self.decomposer.prompts)
    self.assertIn('no generation model', logs.output[0])

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
