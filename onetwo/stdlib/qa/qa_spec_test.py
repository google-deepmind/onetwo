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

from absl.testing import absltest
from onetwo.backends import backend_registry
from onetwo.stdlib.qa import qa_spec


class QaSpecTest(absltest.TestCase):

  def test_rag_strategy_spec_serialization(self):
    backend_config = backend_registry.GenerateBackendConfig(
        backend_class_name='GoogleGenAIAPI',
        model_name='gemini-3-flash-preview',
        kwargs={'temperature': 0.5},
    )
    spec = qa_spec.QAStrategySpec(
        index_path='/path/to/index',
        generate_backend_config=backend_config,
        query_time_kwargs={'max_results': 10, 'min_score': 0.8},
    )

    json_str = spec.to_json()
    reconstructed = qa_spec.QAStrategySpec.from_json(json_str)

    self.assertEqual(spec, reconstructed)
    self.assertEqual(reconstructed.qa_strategy_name, 'rag')
    self.assertEqual(reconstructed.index_path, '/path/to/index')
    self.assertEqual(
        reconstructed.generate_backend_config.model_name,
        'gemini-3-flash-preview',
    )
    self.assertEqual(
        reconstructed.query_time_kwargs, {'max_results': 10, 'min_score': 0.8}
    )

  def test_rag_strategy_spec_str(self):
    backend_config = backend_registry.GenerateBackendConfig(
        backend_class_name='GoogleGenAIAPI',
        model_name='gemini-3-flash-preview',
    )
    spec = qa_spec.QAStrategySpec(
        index_path='/path/to/index',
        generate_backend_config=backend_config,
        query_time_kwargs={'max_results': 10},
    )
    self.assertEqual(
        str(spec), 'rag-max_results=10-GoogleGenAIAPI-gemini-3-flash-preview'
    )

  def test_rag_strategy_spec_str_no_model(self):
    spec = qa_spec.QAStrategySpec(
        index_path='/path/to/index',
        query_time_kwargs={'max_results': 10},
    )
    self.assertEqual(str(spec), 'rag-max_results=10')


if __name__ == '__main__':
  absltest.main()
