# Copyright 2024 DeepMind Technologies Limited.
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

"""Agent implementations for use in tests."""

import dataclasses
from typing import TypeAlias

from onetwo.agents import agents_base
from onetwo.agents import distribution
from onetwo.core import executing

_SU: TypeAlias = agents_base.ScoredUpdate[str]
_ULS: TypeAlias = agents_base.UpdateListState[str, _SU]


@dataclasses.dataclass
class DistributionAgentForTest(
    distribution.DistributionAgent[
        str, str, str, _SU, None
    ]
):
  """Given a distribution over strings, form a distribution over next chars."""
  distribution: dict[str, float] = dataclasses.field(
      default_factory=dict
  )

  @executing.make_executable(copy_self=False)
  async def initialize_state(self, inputs: str) -> str:
    """Overridden from base class (Agent)."""
    return inputs

  def extract_output(self, state: str) -> str:
    """Overridden from base class (Agent)."""
    return state

  def is_finished(self, state: str) -> bool:
    """Overridden from base class (Agent)."""
    # We reached a final state if we have the end-of-sequence token `$`.
    if state.endswith('$'):
      return True
    # But we also consider the state to be final if it cannot be reached.
    found = False
    for word in self.distribution:
      if word.startswith(state):
        found = True
    return not found

  def score_state(self, state: str) -> float:
    """Returns the probability of reaching this state from an empty state."""
    sum_probabilities = 0.0
    for word, prob in self.distribution.items():
      if (word + '$').startswith(state):
        sum_probabilities += prob
    return sum_probabilities

  @executing.make_executable(copy_self=False)
  async def get_next_step_distribution(
      self, state: str, environment: None = None
  ) -> list[_SU]:
    """Overridden from base class (DistributionAgent)."""
    if self.is_finished(state):
      # If we are in a final state, we return a distribution with score 1 for
      # the empty update (which leaves the state unchanged).
      return [agents_base.ScoredUpdate(update='', score=1.0)]
    next_letter_probs = {}
    for word, score in self.distribution.items():
      if word.startswith(state):
        if len(word) > len(state):
          next_letter = word[len(state)]
        else:
          next_letter = '$'  # End of sequence token.
        if next_letter not in next_letter_probs:
          next_letter_probs[next_letter] = score
        else:
          next_letter_probs[next_letter] += score
    # Normalize the distribution.
    total_prob = sum(next_letter_probs.values())
    if total_prob > 0.0:
      for k in next_letter_probs:
        next_letter_probs[k] /= total_prob
    else:
      # If the total probability is 0, we assume we cannot continue hence the
      # next update is necessarily empty.
      return [agents_base.ScoredUpdate(update='', score=1.0)]
    # Convert the disrtibution to ScoredUpdates.
    return [
        agents_base.ScoredUpdate(update=k, score=v)
        for k, v in next_letter_probs.items()
    ]


_StringAgentState: TypeAlias = agents_base.UpdateListState[str, str]


@dataclasses.dataclass
class StringAgent(
    agents_base.SingleSampleAgent[str, str, _StringAgentState, str, None]
):
  """Simple test agent, whose input / updates are strings.

  Its output is a concatenation of the update strings, separate by space.

  Attributes:
    max_length: Maximum length of the agent's state (i.e., of its update list).
      If specified, then will finish when this length is reached. If None, then
      will by default run forever.
    sequence: A sequence of strings to be used by the agent to produce samples.
  """

  max_length: int = 5
  sequence: list[str] = dataclasses.field(default_factory=list)

  @executing.make_executable(copy_self=False)
  async def initialize_state(
      self, inputs: str
  ) -> _StringAgentState:
    return _StringAgentState(inputs=inputs)

  def extract_output(self, state: _StringAgentState) -> str:
    """Overridden from base class (Agent)."""
    return ' '.join(state.updates)

  def is_finished(self, state: _StringAgentState) -> bool:
    """Overridden from base class (Agent)."""
    return len(state.updates) >= self.max_length

  @executing.make_executable(copy_self=False)
  async def _sample_single_next_step(
      self, state: _StringAgentState, environment=None
  ) -> str:
    """Overridden from base class (SingleSampleAgent)."""
    return self.sequence.pop(0)
