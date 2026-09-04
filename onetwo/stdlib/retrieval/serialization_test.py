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

import json
import os
import shutil
import tempfile
from typing import Any, cast

from absl.testing import absltest
import numpy as np
from onetwo import ot
from onetwo.backends import backend_registry
from onetwo.core import content as content_lib
from onetwo.stdlib.retrieval import index_state
from onetwo.stdlib.retrieval import retrieval_data_structures
from onetwo.stdlib.retrieval import serialization

_Document = retrieval_data_structures.Document


class SerializationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_simple_serialization_roundtrip(self):
    serializer = serialization.SimpleConstrainedEmbeddingBasedIndexSerializer(
        _Document
    )

    doc1 = _Document(
        doc_id='id1', content='content1', metadata={'key1': 'val1'}
    )
    doc2 = _Document(
        doc_id='id2', content='content2', metadata={'key2': 'val2'}
    )
    state = index_state.ConstrainedEmbeddingBasedIndexState(
        docs=[doc1, doc2],
        doc_embeddings=[
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ],
        doc_indices_by_discrete_value={
            'external_keys': {'key1': [0], 'key2': [1]}
        },
    )
    serializer.save(state, self.test_dir)

    self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'docs.jsonl')))
    self.assertTrue(
        os.path.exists(os.path.join(self.test_dir, 'embeddings.npy'))
    )
    self.assertTrue(
        os.path.exists(os.path.join(self.test_dir, 'discrete_indices.json'))
    )

    new_state = ot.run(serializer.load(self.test_dir))
    self.assertLen(new_state.docs, 2)
    self.assertEqual(new_state.docs[0].doc_id, 'id1')
    self.assertEqual(new_state.docs[1].doc_id, 'id2')
    self.assertEqual(new_state.docs[0].metadata, {'key1': 'val1'})

    self.assertLen(new_state.doc_embeddings, 2)
    np.testing.assert_array_equal(
        new_state.doc_embeddings[0], state.doc_embeddings[0]
    )
    np.testing.assert_array_equal(
        new_state.doc_embeddings[1], state.doc_embeddings[1]
    )

    self.assertEqual(
        new_state.doc_indices_by_discrete_value,
        state.doc_indices_by_discrete_value,
    )
    self.assertIsNone(new_state.embed_backend_config)

  def test_serialization_with_manifest_and_kwargs(self):
    serializer = serialization.SimpleEmbeddingBasedIndexSerializer(_Document)
    state = index_state.EmbeddingBasedIndexState(
        docs=[_Document(doc_id='id1', content='content1')],
        doc_embeddings=[np.array([1.0], dtype=np.float32)],
        embed_backend_config=backend_registry.EmbedBackendConfig(
            backend_class_name='MyBackend',
            model_name='MyModel',
            kwargs={'api_key': 'secret', 'retries': 3, 'timeout': 10.5},
        ),
    )
    serializer.save(state, self.test_dir)
    self.assertTrue(
        os.path.exists(os.path.join(self.test_dir, 'embed_backend_config.json'))
    )

    new_state = ot.run(serializer.load(self.test_dir))
    self.assertIsNotNone(new_state.embed_backend_config)
    self.assertEqual(
        new_state.embed_backend_config.backend_class_name, 'MyBackend'
    )
    self.assertEqual(new_state.embed_backend_config.model_name, 'MyModel')
    self.assertEqual(
        new_state.embed_backend_config.kwargs,
        {'api_key': 'secret', 'retries': 3, 'timeout': 10.5},
    )

  def test_serialization_unserializable_kwargs(self):
    serializer = serialization.SimpleEmbeddingBasedIndexSerializer(_Document)
    state = index_state.EmbeddingBasedIndexState(
        docs=[_Document(doc_id='id1', content='content1')],
        doc_embeddings=[np.array([1.0], dtype=np.float32)],
        embed_backend_config=backend_registry.EmbedBackendConfig(
            backend_class_name='MyBackend',
            kwargs={'non_serializable': object()},
        ),
    )
    with self.assertRaisesRegex(
        TypeError, 'Failed to serialize EmbedBackendConfig to JSON'
    ):
      serializer.save(state, self.test_dir)

  def test_simple_serializer_init_raises_error(self):
    class InvalidDoc:
      """Missing to_json and from_json."""

      pass

    with self.assertRaisesRegex(TypeError, 'InvalidDoc must implement to_json'):
      serialization.SimpleEmbeddingBasedIndexSerializer(
          doc_class=cast(Any, InvalidDoc)
      )

  def test_embedding_index_serializer_init_raises_error(self):
    class IncompleteDoc:
      """Has one but not both required methods."""

      def to_json(self):
        return '{}'

    with self.assertRaisesRegex(
        TypeError, 'IncompleteDoc must implement from_json'
    ):
      serialization.SimpleEmbeddingBasedIndexSerializer(
          doc_class=cast(Any, IncompleteDoc)
      )

  def test_sharded_serialization_roundtrip(self):
    serializer = serialization.SimpleEmbeddingBasedIndexSerializer(
        _Document, num_shards=2
    )

    doc1 = _Document(doc_id='id1', content='content1')
    doc2 = _Document(doc_id='id2', content='content2')
    state = index_state.EmbeddingBasedIndexState(
        docs=[doc1, doc2],
        doc_embeddings=[
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ],
    )
    serializer.save(state, self.test_dir)

    self.assertTrue(
        os.path.exists(os.path.join(self.test_dir, 'docs-00000-of-00002.jsonl'))
    )
    self.assertTrue(
        os.path.exists(os.path.join(self.test_dir, 'docs-00001-of-00002.jsonl'))
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(self.test_dir, 'embeddings-00000-of-00002.npy')
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(self.test_dir, 'embeddings-00001-of-00002.npy')
        )
    )

    new_state = ot.run(serializer.load(self.test_dir))
    self.assertLen(new_state.docs, 2)
    self.assertEqual(new_state.docs[0].doc_id, 'id1')
    self.assertEqual(new_state.docs[1].doc_id, 'id2')
    self.assertLen(new_state.doc_embeddings, 2)
    np.testing.assert_array_equal(
        new_state.doc_embeddings[0], state.doc_embeddings[0]
    )
    np.testing.assert_array_equal(
        new_state.doc_embeddings[1], state.doc_embeddings[1]
    )

  def test_serialization_with_manifest(self):
    serializer = serialization.SimpleEmbeddingBasedIndexSerializer(_Document)

    doc1 = _Document(doc_id='id1', content='content1')
    doc2 = _Document(doc_id='id2', content='content2')
    state = index_state.EmbeddingBasedIndexState(
        docs=[doc1, doc2],
        doc_embeddings=[
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ],
    )
    serializer.save(state, self.test_dir)

    manifest_path = os.path.join(self.test_dir, 'manifest.json')
    self.assertTrue(os.path.exists(manifest_path))
    with open(manifest_path, 'r') as f:
      manifest_data = json.load(f)
    self.assertEqual(manifest_data['serializer_type'], 'simple')
    self.assertEqual(manifest_data['format_version'], '1.0')
    self.assertEqual(manifest_data['docs_count'], 2)
    self.assertEqual(manifest_data['embeds_count'], 2)

    new_state = ot.run(serializer.load(self.test_dir))
    self.assertLen(new_state.docs, 2)
    self.assertEqual(new_state.docs[0].doc_id, 'id1')
    self.assertEqual(new_state.docs[1].doc_id, 'id2')
    self.assertLen(new_state.doc_embeddings, 2)
    np.testing.assert_array_equal(
        new_state.doc_embeddings[0], state.doc_embeddings[0]
    )
    np.testing.assert_array_equal(
        new_state.doc_embeddings[1], state.doc_embeddings[1]
    )

  def test_serializer_registry(self):
    self.assertIn('simple', serialization.SERIALIZER_REGISTRY)
    self.assertIs(
        serialization.SERIALIZER_REGISTRY['simple'],
        serialization.SimpleEmbeddingBasedIndexSerializer,
    )

  def test_get_serializer(self):
    serializer = serialization.get_serializer('simple', _Document)
    self.assertIsInstance(
        serializer, serialization.SimpleEmbeddingBasedIndexSerializer
    )
    with self.assertRaisesRegex(
        ValueError, 'Unsupported serializer type: unknown'
    ):
      serialization.get_serializer('unknown', _Document)

  def test_multimodal_serialization_roundtrip(self):
    """Verifies that multimodal documents with ChunkList content survive save/load."""
    serializer = serialization.SimpleEmbeddingBasedIndexSerializer(_Document)

    text_doc = _Document(doc_id='text1', content='Plain text document.')
    multimodal_doc = _Document(
        doc_id='multi1',
        title='Multimodal Doc',
        content=content_lib.ChunkList([
            content_lib.Chunk(
                content=b'\x89PNG\r\n\x1a\nfake_image_bytes',
                content_type='image/png',
            ),
            content_lib.Chunk(
                content='A caption for the image.',
                content_type='text/plain',
            ),
            content_lib.Chunk(
                content=b'\x00\x01\x02audio_bytes',
                content_type='audio/wav',
            ),
        ]),
        metadata={'source': 'test'},
    )

    state = index_state.EmbeddingBasedIndexState(
        docs=[text_doc, multimodal_doc],
        doc_embeddings=[
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ],
    )
    serializer.save(state, self.test_dir)

    new_state = ot.run(serializer.load(self.test_dir))
    self.assertLen(new_state.docs, 2)

    # Verify plain text document is unchanged.
    self.assertEqual(new_state.docs[0].doc_id, 'text1')
    self.assertEqual(new_state.docs[0].content, 'Plain text document.')

    # Verify multimodal document content is a ChunkList, not a raw dict.
    loaded_doc = new_state.docs[1]
    self.assertEqual(loaded_doc.doc_id, 'multi1')
    self.assertEqual(loaded_doc.title, 'Multimodal Doc')
    self.assertIsInstance(loaded_doc.content, content_lib.ChunkList)

    chunks = loaded_doc.content.chunks
    self.assertLen(chunks, 3)

    # Verify each chunk is a Chunk instance with correct content and type.
    self.assertIsInstance(chunks[0], content_lib.Chunk)
    self.assertEqual(chunks[0].content, b'\x89PNG\r\n\x1a\nfake_image_bytes')
    self.assertEqual(chunks[0].content_type, 'image/png')

    self.assertIsInstance(chunks[1], content_lib.Chunk)
    self.assertEqual(chunks[1].content, 'A caption for the image.')
    self.assertEqual(chunks[1].content_type, 'text/plain')

    self.assertIsInstance(chunks[2], content_lib.Chunk)
    self.assertEqual(chunks[2].content, b'\x00\x01\x02audio_bytes')
    self.assertEqual(chunks[2].content_type, 'audio/wav')

    # Verify metadata is preserved.
    self.assertEqual(loaded_doc.metadata, {'source': 'test'})

  def test_multimodal_sharded_serialization_roundtrip(self):
    """Verifies that sharded multimodal documents also survive save/load."""
    serializer = serialization.SimpleEmbeddingBasedIndexSerializer(
        _Document, num_shards=2
    )

    docs = [
        _Document(
            doc_id=f'doc{i}',
            content=content_lib.ChunkList([
                content_lib.Chunk(
                    content=f'text_{i}'.encode(),
                    content_type='audio/wav',
                ),
            ]),
        )
        for i in range(4)
    ]

    state = index_state.EmbeddingBasedIndexState(
        docs=docs,
        doc_embeddings=[
            np.array([float(i)], dtype=np.float32) for i in range(4)
        ],
    )
    serializer.save(state, self.test_dir)

    new_state = ot.run(serializer.load(self.test_dir))
    self.assertLen(new_state.docs, 4)

    for i, doc in enumerate(new_state.docs):
      self.assertIsInstance(doc.content, content_lib.ChunkList)
      self.assertLen(doc.content.chunks, 1)
      self.assertIsInstance(doc.content.chunks[0], content_lib.Chunk)
      self.assertEqual(doc.content.chunks[0].content, f'text_{i}'.encode())


if __name__ == '__main__':
  absltest.main()

