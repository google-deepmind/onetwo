# OneTwo Change Log

## v0.3.0

*   **Backends**
    *   **GoogleGenAIAPI:** Implement a `GoogleGenAIAPI` backend that can be
        used to connect to Gemini on both the Gemini Developer API and on Vertex
        AI via the new unified
        [Google Gen AI SDK](https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview).
        This new backend subsumes OneTwo's existing `GeminiAPI` and
        `VertexAIAPI` backends, while providing additional functionality such as
        control over the model's thinking config, multimodal output, support for
        the tokenize operation, configurable retry delays, smarter retry based
        on the specific error code, more efficient parallelization, more
        flexible control over the system instruction, and better control over
        the operations for which the backend is registered and the default
        parameters to be applied.
*   **Core**
    *   **Structured Output:** Introduce enhanced support for structured output
        generation across various backends with the `generate_object` method.
        This feature enables you to provide a Python type or `dataclass` or
        `Pydantic` model, and have the backend return a response that conforms
        to the specified structure.
    *   **Multimodal Output:** Introduce a new operation `llm.generate_content`
        that subsumes the functionality of `llm.chat`, but returns a `ChunkList`
        rather than a string, so as to allow for multimodal output. Ensure that
        return values of type `ChunkList` are handled properly through the core
        code, including for example the caching layer. Provide support for
        returning details of thoughts, etc., via the `include_details=True`
        option. Implement concrete support for this operation in the
        `GoogleGenAIAPI` backend.
    *   **Parallelism:** Provide a `max_parallel_executions` context manager
        that allows overriding the maximum number of parallel execution
        branches, and improve the efficiency of the parallel execution code so
        as to ensure that the worker pool is constantly kept full up to the
        specified `max_parallel_executions`. While previously it had only been
        possible to maintain up to 100 simultaneous LLM requests in parallel, by
        adjusting the `max_parallel_executions` value, it is now possible in
        some cases to achieve as many as 10,000 LLM requests in parallel within
        a single process.
*   **Agents**
    *   **Search Tool (Agent as a tool):** Using `GoogleGenAIAPI` backend, we
        provide an improved search tool by wrapping an LLM call that leverages
        the Google Generative AI SDK's built-in search functionality. This
        demonstrates how easily custom tools can be created in OneTwo, even by
        encapsulating complex operations like LLM calls, effectively creating an
        "Agent as a tool".
*   **Standard library**
    *   **Sandbox:** Enhanced sandboxing capabilities with the introduction of
        `PythonSandboxSafeSubsetMultiProcess`, that wraps the existing
        `PythonSandboxSafeSubset` but executes the code in a separate,
        persistent process using Python's standard `multiprocessing` library.
        This new sandbox offers true parallelism and greater robustness against
        crashes or hangs, especially when executing LLM-generated code in
        parallel.
*   **Evaluation**
    *   **Fuzzy LLM critic:** Implement a `naive_fuzzy_evaluation_critic` that
        is similar to `naive_evaluation_critic`, except returns a continuous
        score between 0 and 1 rather than a binary rating.
*   **Visualization**
    *   Make `HTMLRenderer` more robust to rendering of objects with cyclic
        references.
*   **Documentation**
    *   Update the tutorial colab to support the latest Gemini and OpenAI models
        and `GoogleGenAIAPI` backend and to illustrate best practices for
        structured output generation. Includes, among other things, new sections
        illustrating using LLMs as a `Tool` in `ReActAgent` and
        `PythonPlanningAgent`, and sandbox selection.
*   **Other**
    *   **Fix to the Multimodal section of Colab tutorial:** Added images
        directly to the codebase, with licensing and source details tracked in
        `image_provenance.py`. Also resolved issues with image data handling in
        the `GoogleGenAIAPI` backend.
    *   Various other bug fixes and incremental improvements to caching,
        exception handling, tracing , `ReActAgent`, `PythonPlanningAgent`,
        evaluation summary output, conversion and formatting of chunks and chunk
        lists, test utilities, type annotations, and the `GeminiAPI` and
        `VertexAIAPI` backends.

## v0.2.1

*   **Backends**
    *   **Multimodal:** Enable support for sending arbitrary content chunk types
        to Gemini models, including video, audio, PDFs, vision embeddings, more
        image formats, and other chunk types.
    *   **Formatting:** Define a `ConcatFormatter` that simply concatenates the
        contents of a list of chat messages, while ignoring the roles. When
        applied to certain styles of prompt, this provides a way to reuse the
        same prompt across both chat-style LLMs (using
        `formatter=formatting.FormatterName.API`) and plain text LLMs (using
        `formatter=formatting.FormatterName.CONCAT`).
    *   **Automatic retry:** Implement a generic retry mechanism for use with
        arbitrary LLMs, which can be used, for example, to automatically retry
        upon receipt of a rate-limiting error.
*   **Core**
    *   **Tracing:** Apply comprehensive tracing to all calls to built-in
        methods (`generate_text`, `chat`, `instruct`, `score_text`, `select`,
        etc.) of the standard LLM backends, so that we no longer depend on the
        use of Jinja templates for tracing.
    *   **Chat:** Improve support for chat operations throughout the OneTwo
        codebase, including adding chat support to composables and ensuring that
        caching works robustly for chat messages that contain multimodal
        content.
*   **Agents**
    *   **Error handling:** Improve error handling in `PythonPlanningAgent`,
        including providing a way to configure inside of each `Tool` definition
        which types of errors are recoverable or not, so as to surface any
        tool-generated error messages to the LLM for potentially recoverable
        errors to allow the LLM to retry with adjusted syntax, while
        automatically terminating the agent quickly if an irrecoverable error
        occurs.
*   **Standard library**
    *   **Chat:** Re-implement the standard components such as `ReActAgent`,
        `PythonPlanningAgent`, and chain-of-thought components to use chat
        operations and to improve performance on the latest generations of
        chat-tuned models.
    *   **Multimodal:** Add support for multimodal inputs in `ReActAgent` and
        `PythonPlanningAgent`.
*   **Evaluation**
    *   **LLM critic:** Re-implement naive_evaluation_critic using chat
        operations and with a parser that is robust to more diverse reply
        formats, including reply formats commonly output by Gemini 1.5 models.
*   **Visualization**
    *   Improve `HTMLRenderer` to ensure that strings are properly escaped
        before rendering and to robustly handle a broader range of data types,
        including graceful fallbacks for images and other large byte objects.
*   **Documentation**
    *   Update the tutorial colab to support the latest Gemini and OpenAI models
        and to illustrate best practices for chat semantics and multimodal
        support. Includes, among other things, new sections illustrating
        multimodal `ReAct` and `PythonPlanning` agents.
*   **Other**
    *   Move commonly used utility functions (e.g., for cache management, etc.)
        from the tutorial colab into a `colab_utils` library to facilitate reuse
        in other colabs.
    *   Various bug fixes and incremental improvements to the `GeminiAPI` and
        `OpenAIAPI` backends, the `VertexAIAPI` backend, multi-threading
        support, and Jinja templates.

## v0.2.0

*   **Backends**
    *   **VertexAI**: Add VertexAI chat support.
    *   **Space healing:** Add token/space healing options to built-in
        functions, including proper support for space healing in
        `llm.generate_text` and `llm.chat` of `GeminiAPI`.
*   **Core**
    *   **Caching:** Enable loading from multiple cache files, while merging the
        contents. This is useful, for example, when collaborating in a group,
        where each person can save to a personal cache file, while loading from
        both their own and ones from teammates.
    *   **Retries:** Implement a generic `with_retry` decorator that
        automatically retries a given function with exponential backoff when an
        exception occurs, and enable this for the `GeminiAPI` and `OpenAIAPI`
        backends.
*   **Standard library**
    *   **Chain-of-thought:** Define a library of helper functions and data
        structures for implementing chain-of-thought
        [[Wei, et al., 2023]](https://arxiv.org/pdf/2201.11903) strategies,
        including off-the-shelf implementations of several commonly-used
        approaches, and add a corresponding section to the tutorial colab.
        Variants illustrated include:
    *   Chain-of-thought implemented using a prompt template alone (w/2 calls).
    *   Chain-of-thought implemented using a prompt template (1 call) + answer
        parser.
    *   Few-shot chain-of-thought.
    *   Few-shot exemplars represented as data, so as to be reusable across
        different styles of prompt template.
    *   Few-shot chain-of-thought with different exemplars specified for each
        question (e.g., for dynamic exemplar selection).
    *   **Self-consistency:** Define a generic implementation of
        self-consistency
        [[Wang, et al., 2023]](https://arxiv.org/pdf/2203.11171) and add a
        corresponding section to the tutorial colab. In this implementation, we
        reformulate self-consistency as a meta-strategy that wraps some
        underlying strategy that outputs a single answer (typically via some
        kind of reasoning path or other intermediate steps) and converts it into
        a strategy that outputs a marginal distribution over possible answers
        (marginalizing over the intermediate steps). The marginal distribution
        is estimated via repeated sampling from the underlying strategy.
        Supported variations include:
    *   Self-consistency over chain-of-thought (like in the original paper).
    *   Self-consistency over a multi-step prompting strategy (e.g., ReAct).
    *   Self-consistency over a multi-arg strategy (e.g., Retrieval QA).
    *   Self-consistency over diverse parameterizations of the underlying
        strategy (e.g., with samples taken using different choices of few-shot
        exemplars).
    *   Self-consistency over diverse underlying strategies.
    *   Self-consistency with answer normalization applied during bucketization.
    *   Self-consistency with weighted voting.
    *   Evaluation based on the consensus answer alone.
    *   Evaluation based on the full answer distribution (e.g., accuracy@k).
    *   Evaluation taking into account a representative reasoning path.
*   **Evaluation**
    *   Add a new `agent_evaluation` library, which is similar to the existing
        `evaluation` library, but automatically packages the results of the
        evaluation run in a standardized `EvaluationSummary` object, with
        options to include detailed debugging information for each example. This
        can be used for evaluating arbitrary prompting strategies, but contains
        particular optimizations for agents.
    *   Add library for writing an `EvaluationSummary` to disk.
*   **Visualization**
    *   Update `HTMLRenderer` to support rendering of `EvaluationSummary`
        objects, to render structured Python objects in an
        expandable/collapsible form, and to allow specification of custom
        renderers for other data types.
*   **Documentation**
    *   Add sections to the tutorial colab on chain-of-thought,
        self-consistency, and swapping backends.
*   **Other**
    *   Various other bug fixes and incremental improvements to `VertexAIAPI`
        backend, `ReActAgent`, caching, composables, and handling of multimodal
        content chunks.

## v0.1.1

*   Add support for Vertex AI models.
*   Add an `HTMLRenderer` library for rendering an `ExecutionResult`,
    `EvaluationResult`, or `EvaluationSummary` as a block of HTML suitable for
    interactive display in colab.
*   Various bug fixes and incremental improvements to `ReActAgent`,
    `PythonPlanningAgent`, tracing, and handling of multimodal content chunks.

## v0.1.0

*   Initial release including support for two kinds of models:
    *   `GeminiAPI`: remote connection to any of the models supported by the
        [Gemini API](https://ai.google.dev/).
    *   `Gemma`: possibility to load and use open-weights
        [Gemma](https://github.com/google-deepmind/gemma) models.
    *   `OpenAIAPI`: remote connection to any of the models supported by the
        [OpenAI API](https://platform.openai.com/docs/overview).
*   OneTwo core includes support for asynchronous execution, batching, caching,
    tracing, sequential and parallel flows, prompting templating via Jinja2 or
    composables, and multimodal support.
*   OneTwo standard library includes off-the-shelf implementations of two
    popular tool use strategies in the form of a `ReActAgent` and a
    `PythonPlanningAgent`, along with a `PythonSandbox` API, simple autorater
    critics, and a generic `BeamSearch` implementation that can be composed with
    arbitrary underlying agents for producing multi-trajectory strategies such
    as Tree-of-Thoughts.
