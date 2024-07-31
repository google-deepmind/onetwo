# OneTwo Change Log

## v0.2.0

* **Backends**
  * **VertexAI**: Add VertexAI chat support.
  * **Space healing:** Add token/space healing options to builtin functions,
    including proper support for space healing in `llm.generate_text` and
    `llm.chat` of `GeminiAPI`.
* **Core**
  * **Caching:** Enable loading from multiple cache files, while merging the
    contents. This is useful, for example, when collaborating in a group, where
    each person can save to a personal cache file, while loading from both their
    own and ones from teammates.
* **Standard library**
  * **Chain-of-thought:** Define a library of helper functions and data
    structures for implementing chain-of-thought
    [[Wei, et al., 2023]](https://arxiv.org/pdf/2201.11903) strategies,
    including off-the-shelf implementations of several commonly-used approaches,
    and add a corresponding section to the tutorial colab. Variants illustrated
    include:
    * Chain-of-thought implemented using a prompt template alone (w/2 calls).
    * Chain-of-thought implemented using a prompt template (1 call) + answer
      parser.
    * Few-shot chain-of-thought.
    * Few-shot exemplars represented as data, so as to be reusable across
      different styles of prompt template.
    * Few-shot chain-of-thought with different exemplars specified for each
      question (e.g., for dynamic exemplar selection).
  * **Self-consistency:** Define a generic implementation of self-consistency
    [[Wang, et al., 2023]](https://arxiv.org/pdf/2203.11171) and add a
    corresponding section to the tutorial colab. In this implementation, we
    reformulate self-consistency as a meta-strategy that wraps some underlying
    strategy that outputs a single answer (typically via some kind of reasoning
    path or other intermediate steps) and converts it into a strategy that
    outputs a marginal distribution over possible answers (marginalizing over
    the intermediate steps). The marginal distribution is estimated via repeated
    sampling from the underlying strategy. Supported variations include:
    * Self-consistency over chain-of-thought (like in the original paper).
    * Self-consistency over a multi-step prompting strategy (e.g., ReAct).
    * Self-consistency over a multi-arg strategy (e.g., Retrieval QA).
    * Self-consistency over diverse parameterizations of the underlying strategy
      (e.g., with samples taken using different choices of few-shot exemplars).
    * Self-consistency over diverse underlying strategies.
    * Self-consistency with answer normalization applied during bucketization.
    * Self-consistency with weighted voting.
    * Evaluation based on the consensus answer alone.
    * Evaluation based on the full answer distribution (e.g., accuracy@k).
    * Evaluation taking into account a representative reasoning path.
* **Evaluation**
  * Add a new `agent_evaluation` library, which is similar to the existing
    `evaluation` library, but automatically packages the results of the
    evaluation run in a standardized `EvaluationSummary` object, with options to
    include detailed debugging information for each example. This can be used
    for evaluating arbitrary prompting strategies, but contains particular
    optimizations for agents.
  * Add library for writing an `EvaluationSummary` to disk.
* **Visualization**
  * Update `HTMLRenderer` to support rendering of `EvaluationSummary` objects,
    to render structured Python objects in an expandable/collapsible form, and
    to allow specification of custom renderers for other data types.
* **Documentation**
  * Add sections to the tutorial colab on chain-of-thought, self-consistency,
    and swapping backends.
* **Other**
  * Various other bug fixes and incremental improvements to `VertexAIAPI`
    backend, `ReActAgent`, caching, composables, and handling of multimodal
    content chunks.

## v0.1.1

* Add support for Vertex AI models.
* Add an `HTMLRenderer` library for rendering an `ExecutionResult`,
  `EvaluationResult`, or `EvaluationSummary` as a block of HTML suitable for
  interactive display in colab.
* Various bug fixes and incremental improvements to `ReActAgent`,
  `PythonPlanningAgent`, tracing, and handling of multimodal content chunks.

## v0.1.0

* Initial release including support for two kinds of models:
  * `GeminiAPI`: remote connection to any of the models supported by the
    [Gemini API](https://ai.google.dev/).
  * `Gemma`: possibility to load and use open-weights
    [Gemma](https://github.com/google-deepmind/gemma) models.
  * `OpenAIAPI`: remote connection to any of the models supported by the
    [OpenAI API](https://platform.openai.com/docs/overview).
* OneTwo core includes support for asynchronous execution, batching, caching,
  tracing, sequential and parallel flows, prompting templating via Jinja2 or
  composables, and multimodal support.
* OneTwo standard library includes off-the-shelf implementations of two popular
  tool use strategies in the form of a `ReActAgent` and a `PythonPlanningAgent`,
  along with a `PythonSandbox` API, simple autorater critics, and a generic
  `BeamSearch` implementation that can be composed with arbitrary underlying
  agents for producing multi-trajectory strategies such as Tree-of-Thoughts.
