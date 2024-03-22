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

"""Data config for applying IterativeThought to the Game of 24 problem."""

import textwrap

from onetwo.agents import iterative_thought


GAME_OF_24_DESCRIPTION = textwrap.dedent("""\
    Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
    Each number can be only used once, but results can be used in subsequent
    steps.
""")


# Exemplars for use with IterativeThoughtAgent.
GAME_OF_24_EXAMPLES = [
    iterative_thought.IterativeThoughtState(
        inputs='10 5 2 11',
        updates=[
            '10 5 2 11, we try 2 * 11 = 22 (remaining: 10 5 22)',
            '10 5 22, we try 10 / 5 = 2 (remaining: 22 2)',
            '22 2, we try 22 + 2 = 24 (remaining: 24, success: we got 24)',
        ],
    ),
    iterative_thought.IterativeThoughtState(
        inputs='10 5 2 11',
        updates=[
            '10 5 2 11, we try 2 * 11 = 22 (remaining: 10 5 22)',
            '10 5 22, we try 22 - 5 = 17 (remaining: 10 17)',
            '10 17, we try 10 + 17 = 27 '
            + '(remaining: 27, failure: we did not get 24)',
        ],
    ),
    iterative_thought.IterativeThoughtState(
        inputs='10 5 2 11',
        updates=[
            '10 5 2 11, we try 11 - 10 = 1 (remaining: 5 2 1)',
            '5 2 1, we try 2 + 1 = 3 (remaining: 5 3)',
            '5 3, we try 5 * 3 = 15 '
            + '(remaining: 15, failure: we did not get 24)',
        ],
    ),
    iterative_thought.IterativeThoughtState(
        inputs='2 6 5 3',
        updates=[
            '2 6 5 3, we try 5 - 3 = 2 (remaining: 2 6 2)',
            '2 6 2, we try 2 * 6 = 12 (remaining: 2 12)',
            '2 12, we try 2 * 12 = 24 (remaining: 24, success: we got 24)',
        ],
    ),
    iterative_thought.IterativeThoughtState(
        inputs='2 6 5 3',
        updates=[
            '2 6 5 3, we try 3 * 2 = 6 (remaining: 5 6 6)',
            '5 6 6, we try 6 + 6 = 12 (remaining: 5 12)',
            '5 12, we try 12 + 5 = 17 '
            + '(remaining: 17, failure: we did not get 24)',
        ],
    ),
    iterative_thought.IterativeThoughtState(
        inputs='2 6 5 3',
        updates=[
            '2 6 5 3, we try 5 - 3 = 2 (remaining: 2 6 2)',
            '2 6 2, we try 6 / 2 = 3 (remaining: 2 3)',
            '2 3, we try 2 * 3 = 6 '
            + '(remaining: 6, failure: we did not get 24)',
        ],
    ),
    iterative_thought.IterativeThoughtState(
        inputs='2 6 5 3',
        updates=[
            '2 6 5 3, we try 2 * 5 = 10 (remaining: 6 3 10)',
            '6 3 10, we try 6 * 10 = 60 (remaining: 3 60)',
            '3 60, we try 60 / 3 = 20 '
            + '(remaining: 20, failure: we did not get 24)',
        ],
    ),
]


# Exemplars for use with IterativeThoughtProposerAgent.
GAME_OF_24_PROPOSER_EXAMPLES = [
    iterative_thought.IterativeThoughtProposerExemplar(
        iterative_thought.IterativeThoughtState(
            inputs='10 5 2 11',
            updates=[],
        ),
        next_steps=[
            '10 5 2 11, we try 2 * 11 = 22 (remaining: 10 5 22)',
            '10 5 2 11, we try 11 - 10 = 1 (remaining: 5 2 1)',
        ],
    ),
    iterative_thought.IterativeThoughtProposerExemplar(
        iterative_thought.IterativeThoughtState(
            inputs='10 5 2 11',
            updates=[
                '10 5 2 11, we try 2 * 11 = 22 (remaining: 10 5 22)',
            ],
        ),
        next_steps=[
            '10 5 22, we try 10 / 5 = 2 (remaining: 22 2)',
            '10 5 22, we try 22 - 5 = 17 (remaining: 10 17)',
        ],
    ),
    iterative_thought.IterativeThoughtProposerExemplar(
        iterative_thought.IterativeThoughtState(
            inputs='10 5 2 11',
            updates=[
                '10 5 2 11, we try 2 * 11 = 22 (remaining: 10 5 22)',
                '10 5 22, we try 10 / 5 = 2 (remaining: 22 2)',
            ],
        ),
        next_steps=[
            '22 2, we try 22 + 2 = 24 (remaining: 24, success: we got 24)',
        ],
    ),
    iterative_thought.IterativeThoughtProposerExemplar(
        iterative_thought.IterativeThoughtState(
            inputs='10 5 2 11',
            updates=[
                '10 5 2 11, we try 2 * 11 = 22 (remaining: 10 5 22)',
                '10 5 22, we try 22 - 5 = 17 (remaining: 10 17)',
            ],
        ),
        next_steps=[
            '10 17, we try 10 + 17 = 27 '
            + '(remaining: 27, failure: we did not get 24)',
        ],
    ),
    iterative_thought.IterativeThoughtProposerExemplar(
        iterative_thought.IterativeThoughtState(
            inputs='2 6 5 3',
            updates=[],
        ),
        next_steps=[
            '2 6 5 3, we try 5 - 3 = 2 (remaining: 2 6 2)',
            '2 6 5 3, we try 3 * 2 = 6 (remaining: 5 6 6)',
            '2 6 5 3, we try 2 * 5 = 10 (remaining: 6 3 10)',
        ],
    ),
    iterative_thought.IterativeThoughtProposerExemplar(
        iterative_thought.IterativeThoughtState(
            inputs='2 6 5 3',
            updates=[
                '2 6 5 3, we try 5 - 3 = 2 (remaining: 2 6 2)',
            ],
        ),
        next_steps=[
            '2 6 2, we try 6 / 2 = 3 (remaining: 2 3)',
            '2 6 2, we try 2 * 6 = 12 (remaining: 2 12)',
        ],
    ),
    iterative_thought.IterativeThoughtProposerExemplar(
        iterative_thought.IterativeThoughtState(
            inputs='2 6 5 3',
            updates=[
                '2 6 5 3, we try 5 - 3 = 2 (remaining: 2 6 2)',
                '2 6 2, we try 2 * 6 = 12 (remaining: 2 12)',
            ],
        ),
        next_steps=[
            '2 12, we try 2 * 12 = 24 (remaining: 24, success: we got 24)',
        ],
    ),
    iterative_thought.IterativeThoughtProposerExemplar(
        iterative_thought.IterativeThoughtState(
            inputs='2 6 5 3',
            updates=[
                '2 6 5 3, we try 5 - 3 = 2 (remaining: 2 6 2)',
                '2 6 2, we try 6 / 2 = 3 (remaining: 2 3)',
            ],
        ),
        next_steps=[
            '2 3, we try 2 * 3 = 6 (remaining: 6, failure: we did not get 24)',
        ],
    ),
]
