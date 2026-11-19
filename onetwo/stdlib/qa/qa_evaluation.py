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

from etils import epath
from onetwo.core import results


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
