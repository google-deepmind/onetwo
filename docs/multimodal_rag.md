# Multimodal RAG: the SpanishGov case study

OneTwo comes with ready-made indexes, `rag.IndexType.EMBEDDING` and
`rag.IndexType.GRAPHRAG` (see the
[RAG tutorial](https://colab.research.google.com/github/google-deepmind/onetwo/blob/main/colabs/rag_tutorial.ipynb)).
While these indexes are a strong starting point, you may want to build a custom
index for advanced use cases.

This tutorial illustrates how easy it is to create a custom OneTwo index, and
how much such an index can help when questions require information from both
text and images.

## Setup

Install OneTwo and
[set up your Gemini API key](https://aistudio.google.com/docs/api-key):

```shell
pip install git+https://github.com/google-deepmind/onetwo

export GOOGLE_API_KEY="YOUR_API_KEY"
```

## The dataset

![A question that needs two facts: only the CV text says Ministerio del
Interior, and only the portrait shows the orange
shirt](../assets/multimodal_example.png)

In this case study, we consider a dataset created from the public Spanish
Government website, `spanish_gov_multimodal_qa`. Each question pairs a fact from
a CV's **text** with a visual fact from its **portrait**. Like the other OneTwo
RAG datasets, it has two parts:

*   The **corpus**: 683 documents, the CVs of Spanish officials, most of them
    with a portrait.
*   The **examples**: 323 questions, each targeting both the text and the
    portrait of one CV, as shown in the image above.

```python
from onetwo import rag
from onetwo.evaluation import datasets

corpus = datasets.get_dataset_loader('spanish_gov_multimodal_qa')
```

## Baseline: the standard MCP server

![An agent calls search on a single vector index of whole CVs, where each CV's
text and portrait share one vector, and get_document on the
corpus](../assets/multimodal_whole_cv_agent.png)

As a baseline, we use OneTwo's ready-made building blocks. The default vector
index (`rag.IndexType.EMBEDDING`) is multimodal with the default
`gemini-embedding-2` embedder, so it embeds each CV, text and portrait together,
into a single vector. An agent searches it through the standard MCP server, with
agentic retrieval (`strategy='agentic_retrieval'`).

**1. Build the index:**

```python
index = rag.build_index(
    corpus,
    rag.IndexType.EMBEDDING,  # The default.
    output='/tmp/spanishgov_index'
)
```

**2. Serve it** with the standard MCP server. Its tools are `search` and
`get_document`, as illustrated in the figure above. The figure draws the corpus
separately for clarity, but `get_document` reads the full CVs from the index
itself.

```python
index_server = rag.start_mcp_in_background(index)
```

**3. Let an agent search it:**

```python
examples = datasets.get_dataset_loader('spanish_gov_multimodal_qa')
summary = rag.evaluate(
    examples,
    strategy='agentic_retrieval',
    mcp_urls=[index_server.url],  # http://localhost:8775/mcp
    output='/tmp/spanishgov_eval_baseline'
)

index_server.stop()
```

The ground-truth answer to each question is the full name of the official it
describes. An LLM rater compares the agent's answer with the ground-truth answer
for all 323 questions. With Gemini 3.8 Flash, the agent answers 59% of the
questions correctly, and with Gemini 4 Argon it increases to 70%. Argon works
hard for it, making about **60 tool calls per question**, three times as many as
Flash.

Can we do even better? Each question has a textual part and a visual part, so an
index that searches the text and the portraits separately is a natural next
step. The next section shows how easily you can build one with OneTwo.

## A custom dual-modal index and its MCP server

![An agent calls search_text_chunks, search_images and get_document on the
text index, the image index and the corpus, until it can
answer](../assets/multimodal_agentic_retrieval.png)

`DualModalDocumentIndex`, from the OneTwo examples, is a custom index made of a
separate text index and image index over the same CVs:

*   The **text index** keeps only the text of each CV, and splits it into chunks
    of up to 512 tokens with the default tokenizer. It also keeps the full CVs,
    for `get_document`.
*   The **image index** keeps only the portrait of each CV, and embeds it whole.
*   Both indexes use the default `gemini-embedding-2` embedder.

```python
# Simplified from examples/dual_modal_index/indexing.py, embedder omitted.
class DualModalDocumentIndex(indexing.DocumentIndex):
  text_index = RewritingEmbeddingBasedDocumentIndex(
      rewriter=SequentialCorpusRewriter(rewriters=[
          FormattingCorpusRewriter(KeepTextFormatter()),  # Drops the images.
          FilterEmptyDocuments(),
          ChunkingCorpusRewriter(RoutingChunker()),       # Chunks the text.
      ]),
      preserve_original_docs=True,
  )
  image_index = RewritingEmbeddingBasedDocumentIndex(
      rewriter=SequentialCorpusRewriter(rewriters=[
          FormattingCorpusRewriter(KeepImageFormatter()),  # Drops the text.
          FilterEmptyDocuments(),
      ]),
  )
```

**1. Build it.** And that is it! Pass the custom index to `rag.build_index`,
and OneTwo builds and saves it like any other index:

```python
from onetwo.examples.dual_modal_index import indexing as dual_modal_indexing

dual_index = rag.build_index(
    corpus,
    dual_modal_indexing.DualModalDocumentIndex(),
    output='/tmp/spanishgov_index'
)
```

**2. Serve it** just like before, but with the dual-modal MCP server. As shown
in the figure above, this server introduces a new `search_images` tool that
searches the image index for portraits with a purely visual description.
Additionally, `search_text_chunks` works like `search` from the standard MCP
server above, but only over the text chunks, whereas `get_document` is
unchanged.

```python
from onetwo.examples.dual_modal_index import dual_modal_mcp_server as server

dual_server = server.start_mcp_in_background(dual_index)
```

**3. Let an agent search it:**

```python
from onetwo.stdlib.qa import agentic_retrieval

examples = datasets.get_dataset_loader('spanish_gov_multimodal_qa')
summary = rag.evaluate(
    examples,
    strategy=agentic_retrieval.AgenticRetrievalStrategy(
        max_steps=30,
        final_instruction=server.AGENTIC_FINAL_INSTRUCTION,
        additional_instructions=server.AGENTIC_ADDITIONAL_INSTRUCTIONS
    ),
    mcp_urls=[dual_server.url],  # http://localhost:8775/mcp
    output='/tmp/spanishgov_eval_dual_modal'
)

dual_server.stop()
```

The Argon-based agent now answers up to **90.1%** of the questions correctly!
The additional instructions are optional, but they route each query to the right
tool: `search_images` only sees the portraits, so it gets purely visual
descriptions.

## Results

The following tables show the accuracy and the LLM and tool calls per question
of both setups with each model, on all 323 questions. The best number of each
model is in bold.

**Gemini 3.8 Flash**

| Setup                                   | Accuracy (%) | LLM calls | Tool calls |
| :-------------------------------------- | :----------: | :-------: | :--------: |
| Baseline: standard MCP server           | 58.5         | 20.2      | **20.4**   |
| Dual-modal index, dual-modal MCP server | **87.5**     | **15.4**  | 25.4       |

**Gemini 4 Argon**

| Setup                                   | Accuracy (%) | LLM calls | Tool calls |
| :-------------------------------------- | :----------: | :-------: | :--------: |
| Baseline: standard MCP server           | 70.1         | 12.2      | 59.7       |
| Dual-modal index, dual-modal MCP server | **90.1**     | **6.6**   | **27.4**   |

Splitting the index into text and images has multiple benefits:

*   **More accurate answers:** using this simple modification to the index, the
    agent's accuracy increases by 29% (Flash) and 20% (Argon).
*   **Less work:** by adding a specialized tool for images, both agents make
    fewer LLM calls. Flash makes 24% fewer LLM calls, and Argon does far less
    work: it requires around 50% fewer LLM calls, and its tool calls drop from
    **60** to **27** per question!
