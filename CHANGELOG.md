# OneTwo Change Log

## v0.6.0

*   **⚠️ Breaking changes**
    *   **OneTwo is now statically typed.** The core decorators were rewritten
        so that type checkers (like Pyrefly) see the real signatures and return
        types of decorated functions instead of `Any`. Code that type-checked
        against v0.5.0 may now report errors, and some decorators changed their
        calling convention:
        *   **`ot.make_executable`:** Note that runtime behaviour is unchanged.
            What breaks is type checking:
            *   Decorated functions are now typed as returning
                `FunctionExecWrapper[T]` instead of `Any`, where `T` is the
                result type of the function, unwrapped from coroutines, async
                generators and returned `Executable`s. E.g.
                `async def f(...) -> str` is typed as returning
                `FunctionExecWrapper[str]`, so using `f(...)` where a `str` is
                expected (instead of `await f(...)` or `ot.run(f(...))`) now
                fails type checking.
            *   Without arguments (`@ot.make_executable`), the decorated
                function keeps its parameters (via `ParamSpec`), so calls with
                wrong argument types or counts are now flagged.
        *   **`tracing.trace`:** Must now always be called with parentheses:
            `@tracing.trace()` (or `@tracing.trace(name=...)`). A bare
            `@tracing.trace` no longer works.
        *   **`caching.cache_method`:** Now only supports `async def` methods,
            and preserves the signature of the decorated method.
        *   **`utils.with_retry` / `utils.rate_limit_method`:** Preserve the
            signature of the decorated method instead of collapsing it to
            `Callable[..., Any]`.
        *   **`Composable.execute`** is now an executable.
        *   **Legacy built-ins:** `llm.generate_text` and `llm.generate_texts`
            are deprecated and moved to `builtins/legacy_llm_builtins.py`.
    *   **Python >= 3.13 is now required**.
*   **OneTwo Agent with MCP support**
    *   **`Agent`:** Added a generic agent that runs model-native function
        calling to completion. Tools can be plain Python callables (their
        declarations are inferred from signatures and docstrings), `Tool`s, or
        **MCP servers**, mixed freely, e.g. `Agent(tools=[get_weather,
        mcp_tools.server('http://localhost:8777/mcp')])`. Supports multimodal
        questions, parallel tool calls, a final answer without tools when steps
        run out, and recovery from empty model turns. Added
        `agent.format_tool_calls` and `agent.format_agent_steps` to inspect
        runs.
    *   **MCP tools:** Added `mcp_tools.server` (streamable HTTP) and
        `mcp_tools.stdio_server` to use any MCP server as a source of agent
        tools, with sessions that stay alive across failing tool calls and
        support for image tool results.
    *   **`llm.generate_with_tools`:** New built-in for model-native function
        calling, implemented in the `GoogleGenAIAPI`, `ClaudeAPI` and
        `OpenAIAPI` backends.
    *   **`core.function_calling`:** New core types `ToolSpec`,
        `ConversationTurn`, `FunctionCall` and `FunctionResponse` (with
        correlation `id`s). Function-calling runs, including thoughts and tool
        calls, are recorded in execution traces.
    *   **Tool schemas:** Added `tool_schema.tool_from_callable` to infer a tool
        declaration from a Python callable. `llm_tool_use.Tool` gained
        `parameters` and `to_spec()`.
*   **RAG**
    *   **New top-level `rag` module** (`from onetwo import rag`) to build,
        serve and evaluate retrieval pipelines in a few calls:
        *   **`rag.build_index(corpus, index, *, output, ...)`:** Builds and
            saves a document index from a registered `DatasetLoader` or a folder
            of files of any type, and returns the built index. `index` is either
            a ready-made `rag.IndexType` (`EMBEDDING` or `GRAPHRAG`) or any
            `DocumentIndex` object. Also available as a command-line tool
            (`build_index`).
        *   **`rag.evaluate(examples, index, strategy, *, ...)`:** Evaluates a
            QA strategy on a QA dataset. `strategy` is either a ready-made
            `rag.StrategyType` (`CLOSED_BOOK`, `RAG`, `RETRIEVAL_ONLY`,
            `AGENTIC_RETRIEVAL`) or any `QAStrategy` object; `mcp_urls` points
            agentic strategies at MCP servers. Results are saved in descriptive
            folders under `output`. Also available as a command-line tool
            (`evaluate_rag`).
        *   **`rag.start_mcp` / `rag.start_mcp_in_background`:** Serve a saved
            index or an index object as an MCP server (stdio or streamable HTTP)
            with `search` and `get_document` tools.
        *   **`rag.StorageFormat`:** Chooses how an index is saved, at save
            time.
    *   **Indexing:**
        *   `use_task_type`, the `prepare_*` hooks, `default_prepare_query` and
            `default_prepare_document` were removed. Embedding indexes are
            configured with an `EmbedBackendConfig` and a `searcher: Searcher`.
        *   `EmbeddingBasedDocumentIndex` is now coupled with its embedding
            backend: it takes an `embed_backend_config` and instantiates its own
            embedder backend from it (with credentials passed separately and
            never saved), instead of relying on the globally registered
            `llm.embed`. The config is saved with the index, so a loaded index
            embeds queries with the same backend and model it was built with.
        *   Chunk `doc_id`s now use a `_chunk` suffix format, and chunkers
            record the source span of each chunk in its metadata
            (`span_within_original_doc`).
        *   Added ID-based document lookup (`get_doc`, `get_docs_by_ids`,
            `get_doc_embedding`) to indexes, and preservation of the original
            (unchunked) documents in rewriting indexes across save and load.
    *   **Index configuration and persistence:** Indexes are their own configs:
        they are saved as `index_config.json` and rebuilt from it by
        `index_building.load_index`, whatever their type. Custom components
        (indexes, chunkers, formatters, searchers, ...) are registered by stable
        name with `registry.register`. Serializers are pluggable, support
        sharding, show progress bars and load faster.
    *   **QA strategy results:** `QAStrategy` implementations now return a
        `StrategyResult`, holding the answer together with the retrieved
        `Document`s. Strategies always receive `query_time_kwargs`.
    *   **Datasets:** `onetwo.evaluation.datasets` is now a package. The
        `max_documents` property of dataset loaders was renamed to
        `max_number_of_documents`.
    *   **GraphRAG:** Added `HippoRAGDocumentIndex`, a knowledge-graph index
        based on HippoRAG (with multimodal entity extraction), along with its
        own MCP server and serialization.
    *   **Chunking:**
        *   Added `RoutingChunker`, which chunks each document according to its
            content type, and is the default chunker of the ready-made indexes.
        *   Added `ChunkByMaxWords`, `ChunkByPdfPages`, `ChunkByAudioSeconds`
            and `ChunkByVideoSeconds`.
        *   The tokenizer of `ChunkByMaxTokens` is configurable through a
            `TokenizeBackendConfig`.
    *   **Formatting:** Added `KeepTextFormatter` and `KeepImageFormatter`.
    *   **QA strategies:**
        *   Added a strategies registry (`strategies_lib`) with
            `ClosedBookStrategy`, `LongContextStrategy`, `RAGStrategy` and
            `RetrievalOnlyStrategy`. `RAGContextBuilder` was extracted from
            `RAGStrategy` to ease writing custom RAG strategies.
        *   Added `AgenticRetrievalStrategy`, in which the model searches MCP
            servers step by step (with a configurable step budget and a forced
            final answer) until it can answer, and cites the documents it used.
        *   QA strategies own their generation backend and expose `max_results`,
            `min_score` and `additional_instructions`.
    *   **Examples:** Added a dual-modal (text and image) index with its MCP
        server under `examples/dual_modal_index`.
*   **Backends**
    *   **OllamaAPI:** Added new backend for local models served by Ollama,
        including `.embed`.
    *   **`llm.embed`:** Now takes `task_type` (`llm.EmbedTaskType` or `str`)
        and `title` arguments shared by all backends.
    *   **Backend registry:** Added `BackendConfig` (`GenerateBackendConfig`,
        `EmbedBackendConfig`, `TokenizeBackendConfig`) and `get_backend` to
        instantiate backends from serializable configs.
    *   **GoogleGenAIAPI:**
        *   Accepts a single `model_name`, like other backends.
        *   `CachedChunkList.refresh_cache=False` uses an existing prefix cache
            as is, without refreshing it on every call.
        *   Records tool call counts and token metrics in traces.
*   **Core**
    *   **Execution & Parallelism:**
        *   `par_iter` can show a progress bar (`show_progress`,
            `progress_desc`, `total`).
        *   `merge_iter` uses a sliding-window worker pool, so a slow iterator
            no longer blocks the others.
        *   `@to_thread_pool_method` can wrap async methods.
    *   **Caching:** Faster binary serialization of containers.
    *   **Content:** Added JSON-compatible serialization of `Document`, `Chunk`
        and `ChunkList`.
*   **Evaluation**
    *   **ot.evaluate:** Metric functions receive the `execution_result` and run
        in parallel per example.
    *   Added `EvaluationConfig` and `evaluate_with_config`.
    *   **RAG metrics:** Added `rag_metrics` with `recall@k`, a configurable
        AI-rater accuracy critic, runtime, token-count and tool-call metrics.
    *   **Datasets:** Added loaders for BrowseComp-Plus, Spanish Government
        Multimodal QA, JSON Lines files (`JsonlDatasetLoader`) and folders of
        multimodal files (`MultimodalDatasetLoader`). Added `example_id` to
        `QAExample`.
*   **Colabs & Docs**
    *   **RAG Tutorial:** Rewritten around the `rag` module, including an index
        MCP server used by an `Agent`, and Ollama examples.
    *   **Tutorials:** Reorganized the OneTwo and Getting Started tutorials and
        updated default models.
    *   Added a
        [guide](https://github.com/google-deepmind/onetwo/blob/main/docs/browsecomp_plus_qa.md)
        to run QA on BrowseComp-Plus with RAG and GraphRAG.
*   **Fixes**
    *   **Security:**
        *   `HTMLRenderer` now escapes LLM requests and replies (and dict keys
            and stage names).
        *   The `code_execution` multiprocess sandbox now uses a random
            per-sandbox authkey for its hook channel, and exchanges JSON instead
            of pickled messages.
    *   **Execution:**
        *   `par_iter` (with more executables than `chunk_size`) no longer hangs
            forever when an executable raises; the error is now reported.
        *   Fixed CPU-burning spin-locks in the queue processing of `iterating`,
            which starved the event loop threads (~30% faster evaluation).
        *   `ot.run` no longer creates and closes a new event loop on every
            call, which invalidated the connections of long-lived backends in
            multi-threaded hosts (`OSError: [Errno 9] Bad file descriptor`).
    *   **Backends:**
        *   Fixed rate-limiting deadlocks and 429 retry handling in
            `GoogleGenAIAPI` and `GeminiAPI`.
        *   `GoogleGenAIAPI.embed`: fixed double formatting of titles, and
            failures on empty content and multi-chunk text, for v1 embedding
            models such as `gemini-embedding-001`. Very large texts are now
            cropped client-side (without changing the embedding) instead of
            making the request fail.
    *   **Retrieval:**
        *   Creating an `EmbeddingBasedIndex` without documents no longer raises
            `ValueError: need at least one array to stack`.
        *   `EmbeddingBasedIndex.retrieve_doc_score` now scores like the
            searcher (cosine similarity when `normalize_embeds` is set) and
            returns a `float`.
        *   `ChunkByMaxTokens` with `overlap_window > 0` no longer creates
            redundant tail chunks that only repeat the end of the previous
            chunk.
    *   **Evaluation:**
        *   `ot.evaluate` now keeps the `extra_info` of all metrics.
        *   A single example failing with a `BaseException` (e.g.
            `asyncio.CancelledError`) no longer cancels the whole evaluation
            batch.
        *   `structured_evaluation_critic` returns a score of 0.0 with
            diagnostics instead of crashing when the rater returns invalid
            output.

## v0.5.0

*   **Backends**
    *   **ClaudeAPI:** Added new backend for Anthropic Claude models, including
        support for tool use and thinking control.
    *   **OpenAI API:** Migrated to the latest Responses API, adding support for
        reasoning, tools, `.embed`, and `.count_tokens`.
    *   **GoogleGenAIAPI:**
        *   Updated default Gemini API model to `gemini-3.1-flash-lite`.
        *   Enabled server-side prefix caching.
        *   Added `CachedChunkList` for robust prefix cache handling.
        *   Removed token healing functionality.
        *   Added `normalize` kwarg to `.embed`.
        *   Improved token usage reporting in tracers.
        *   Added support for Gemini Function Calling (FC2.0) metadata.
        *   Bug fixes and model updates.
    *   **SentencePieceTokenizer:** Added a new tokenizer backend using the
        standard sentencepiece library.
*   **Core**
    *   **Execution & Parallelism:** Fixed event loop leak in
        `asyncio_run_wrapper`.
    *   **Tracing:**
        *   Improved exception handling and reporting within the tracing system.
        *   Reduced tracing overhead by skipping large inputs/outputs in traces
            for several backends and modules.
*   **Standard library**
    *   **Retrieval & QA:**
        *   Architectural split for constrained retrieval, introducing
            `ConstrainedEmbeddingBasedIndex`.
        *   Removed use of lambdas in `EmbeddingBasedIndex` for better pickle
            support.
        *   Refined chunking logic in `ChunkByMaxTokens`.
        *   Added `QAPromptBuilder` for flexible QA prompt generation.
        *   Added implementation of RAG-Fusion.
    *   **code_execution:** Fixed test for Python 3.14.
*   **Evaluation**
    *   **ot.evaluate:** Optimized to write large results and `results_debug`
        JSON files incrementally to avoid OOM errors.
*   **Colabs & Docs**
    *   **RAG Tutorial:** Significantly revamped with improved explanations and
        diagrams.
    *   **Tutorial:** Updated with `ClaudeAPI` examples.

## v0.4.0

*   **Backends**
    *   **GoogleGenAIAPI:** Set `httpx` connection limits according to
        `threadpool_size` to allow higher throughput. Handled safety filter
        blocks more explicitly, raising a specific error. Removed redundant
        `@caching.cache_method` decorators from `generate_text` and `chat`
        methods to avoid double caching. Automatically retry on empty responses
        caused by safety filters. Refactored `generate_text` and `chat` to use
        `generate_content` as the base implementation. Added support for
        `output_dimensionality` and `task_type` in `llm.embed`.
    *   **Gemini API:** Updated default embedding model name, as
        `models/embedding-001` is deprecated.
*   **Core**
    *   **Caching:** Made `SimpleCache.cache_value` asynchronous for
        non-blocking I/O. Improved printing of cache summaries in `colab_utils`,
        including support for `TwoLayerCaches` and sorting keys. Made cache read
        resilient to empty file reads caused by race conditions. Added a generic
        `TwoLayerCache` implementation.
    *   **Execution & Parallelism:** Optimized `executing.parallel` to reduce
        overhead for nested and fixed-size sequences using `asyncio.gather`
        directly in a fast path. Allowed changing the default value of
        `max_parallel_executions` globally. Propagated context variables in
        `run_method_in_threadpool`. Used `ContextualExecutor` instead of
        `ThreadPoolExecutor` to ensure context propagation for debugging tools
        like Sherlog. Propagated OneTwo tracers across async boundaries in
        `batching.py`. Improved exception handling and reporting within the
        tracing system.
*   **Agents**
    *   **ReAct:** Added option `retry_on_parsing_error` to provide more
        detailed error messages on action parsing failures, including the
        thought and failed action string to help the LLM self-correct.
*   **Standard library**
    *   **Retrieval & QA:** Added a comprehensive suite of modules under
        `third_party/py/onetwo/stdlib/retrieval` and
        `third_party/py/onetwo/stdlib/qa` to support building advanced
        Retrieval-Augmented Generation (RAG) and Question Answering systems. Key
        features include:
        *   **Core Interfaces:** Introduced fundamental interfaces for RAG like
            `Retriever`, `Index`, `Searcher`, `CorpusRewriter`, `Chunker` and
            `DocumentFormatter`.
        *   **Data Structures:** Added the `Document` dataclass for representing
            content to be indexed and retrieved.
        *   **Indexing:** Introduced various index types: in particular
            `EmbeddingBasedIndex`, `RewritingIndex`.
        *   **CorpusRewriter:** `CorpusRewriter` interface and implementations
            to enable flexible document processing pipelines before indexing.
            Indexes leverage the `CorpusRewriter` abstraction for more general
            transformations.
        *   **Chunking:** Provided implementations like `TextChunker`,
            `NoChunking`, and `ChunkByMaxTokens` for splitting documents.
        *   **Formatter:** Provided implementations for formatting documents.
        *   **Searcher:** Added `BruteForceSearcher` for nearest neighbor search
            between embeddings.
        *   **Constrained Retrieval:** Introduced `RetrievalConstraint` classes
            and the `ConstrainedRetriever` and `ConstrainedSearcher` protocols
            to enable efficient pre-filtering of documents based on metadata
            fields before vector search.
        *   **Index Serialization:** Added `EmbeddingBasedIndexState` to store
            the state of `EmbeddingBasedIndex` and serializers for persisting
            and loading index data.
        *   **QA Strategies:** Defined interfaces for question answering,
            including `QAStrategy`, `ContextualQAStrategy`, and
            `RetrievalQAStrategy`.
*   **Colabs**
    *   Introduced a Getting Started tutorial.
    *   **OneTwo tutorial:** Demonstrated use of `llm.embed()` for multimodal
        contents.
    *   Introduced a RAG Tutorial.
*   **Testing**
    *   Modified `LLMForTest` tokenizer and added detokenizer for better
        testing.
*   **Evaluation**
    *   **Tracing:** Improved tracing for metrics in `metrics.py` and
        `evaluation.py`.
    *   **ot.evaluate:** Unified `evaluation` and `agent_evaluation` modules
        into `onetwo.evaluation.evaluation`. Made `ot.evaluate` more generic to
        support arbitrary example types and strategy signatures. Added
        `dataset_name` and `dataset_description` parameters to `evaluate`.

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
