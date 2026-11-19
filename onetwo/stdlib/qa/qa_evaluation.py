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

# Copyright 2026 Google LLC
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

"""QA evaluation library."""

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import itertools
import json
from typing import Any

from etils import epath
from onetwo.backends import backend_registry
from onetwo.core import content as content_lib
from onetwo.core import results
from onetwo.stdlib.qa import qa
from onetwo.stdlib.retrieval import indexing

_PromptContent = str | content_lib.ChunkList
_StrategyResult = qa.StrategyResult
EVAL_CONFIG_FILENAME = 'eval_config.json'
# The index and MCP servers are saved separately, as 'index' and 'mcp_urls'.
# TODO: Also save prompt_builder and doc_ids_from_answer (they
# hold functions, which have no stable JSON form).
_NOT_SAVED_FIELDS = (
    'runtime_kwargs',
    'retriever',
    'tools',
    'prompt_builder',
    'doc_ids_from_answer',
)


def save_html_report(
    summary: results.EvaluationSummary,
    output_dir_path: str | epath.Path,
    filename: str = 'results.html',
    levels_to_expand: int = 2,
) -> epath.Path:
  """Saves an EvaluationSummary as an HTML report.

  Args:
    summary: The evaluation summary to save.
    output_dir_path: Directory where the report should be saved.
    filename: Filename for the report.
    levels_to_expand: Depth of results to expand in the HTML view.

  Returns:
    The path to the saved HTML file.
  """
  output_dir = epath.Path(output_dir_path)
  output_dir.mkdir(parents=True, exist_ok=True)
  html_report_path = output_dir / filename
  html_report_path.write_text(
      results.HTMLRenderer(levels_to_expand=levels_to_expand).render(summary)
  )
  return html_report_path


def _strategy_config(
    strategy: qa.QAStrategy[_PromptContent, _StrategyResult],
) -> dict[str, Any]:
  """Returns init fields of `strategy` excluding runtime kwargs and index."""
  if not dataclasses.is_dataclass(strategy):
    return {}
  return {
      f.name: getattr(strategy, f.name)
      for f in dataclasses.fields(strategy)
      if f.init and f.name not in _NOT_SAVED_FIELDS
  }


def _run_dir_prefix(
    strategy_name: str,
    config: Mapping[str, Any],
    eval_config_json: str,
) -> str:
  """Returns `<readable>__hash_<8>` for a run directory name."""
  parts = [strategy_name]
  backend_cfg = config.get('backend_config')
  if isinstance(backend_cfg, backend_registry.BackendConfig):
    model = backend_cfg.model_name or backend_cfg.backend_class_name
    if model:
      parts.append(model)
  for name, value in config.items():
    if isinstance(value, (bool, int, float)):
      parts.append(f'{name}={value}')
  config_hash = hashlib.sha256(eval_config_json.encode()).hexdigest()[:8]
  # Model names may contain '/' (e.g. "models/...").
  return f"{'_'.join(parts)}__hash_{config_hash}".replace('/', '-')


def _create_run_directory(dataset_dir: epath.Path, prefix: str) -> epath.Path:
  """Creates and returns the first free `<dataset_dir>/<prefix>_run<i>`."""
  dataset_dir.mkdir(parents=True, exist_ok=True)
  for i in itertools.count(1):
    run_dir = dataset_dir / f'{prefix}_run{i}'
    try:
      # Without parents=True, mkdir is atomic, also on CNS.
      run_dir.mkdir()
      return run_dir
    except FileExistsError:
      pass
  raise AssertionError('Unreachable')


def save_run_outputs(
    summary: results.EvaluationSummary,
    *,
    output: str,
    examples_name: str,
    index: str | indexing.DocumentIndex | None,
    mcp_urls: Sequence[str] = (),
    preset_name: str | None,
    strategy: qa.QAStrategy[_PromptContent, _StrategyResult],
    metric_functions: Mapping[str, Any] | None,
) -> epath.Path:
  """Saves `results.html` and `eval_config.json` in a new run folder."""
  strategy_cls = type(strategy).__name__
  resolved_config = _strategy_config(strategy)
  eval_config = {
      'examples': examples_name,
      'index': (
          index
          if index is None or isinstance(index, str)
          else type(index).__name__
      ),
      'mcp_urls': list(mcp_urls),
      'strategy': {
          'preset': preset_name,
          'class': strategy_cls,
          'config': resolved_config,
      },
      'metric_functions': list(metric_functions or ()),
  }
  eval_config_json = json.dumps(
      eval_config, sort_keys=True, default=dataclasses.asdict
  )
  run_dir = _create_run_directory(
      epath.Path(output) / examples_name,
      _run_dir_prefix(
          preset_name or strategy_cls, resolved_config, eval_config_json
      ),
  )
  html_report_path = save_html_report(
      summary=summary,
      output_dir_path=run_dir,
  )
  (run_dir / EVAL_CONFIG_FILENAME).write_text(
      json.dumps(eval_config, indent=2, default=dataclasses.asdict) + '\n'
  )
  print(f'Results saved to {run_dir}')
  print(f'HTML report saved to {html_report_path}')
  return run_dir
