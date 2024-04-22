# OneTwo FAQ

## Is there a way to see actual formatted prompts that are sent to the backends?

While some of the builtin functions (e.g., `llm.generate_text`) apply little to
no modification to the prompt before it is sent to the model, others (e.g.,
`llm.instruct` or `llm.chat`) may apply a lot of formatting. For example,
one natural way of implementing `llm.instruct` for a model that is only
pre-trained (PT) but not instruction-tuned (IT) is to first format the
user-provided task, e.g. `'Write me a short poem'`, into a longer prompt similar
to `'Here is the task: Write me a short poem. Here is the answer:'` and then
send it to pure completion with `llm.generate_text`. Indeed, this is precisely
how the default implementation of `llm.instruct`
(`onetwo.builtins.default_instruct`) works.

In cases like this, where a non-trivial formatting of prompts takes place, the
user may naturally want to see the actual fully formatted prompt that is sent to
a model.  A simple way to do it, which often comes handy when debugging, is to
configure (mock) `llm.generate_text` with a fake implementation that simply
returns the prompt (for convenience we provide such an implementation in
`onetwo.builtins.echo_generate_text`):

  ```python
  import ot
  from onetwo.builtins import llm
  backend = ...
  # Assume this backend only configures `llm.generate_text`, i.e. `llm.instruct`
  # is configured to use the default OneTwo implementation.
  backend.register()
  print(ot.run(llm.generate_text('Once upon a')))
  print(ot.run(llm.instruct('Name three cities in France.')))

  def fake_generate_text(prompt: str | content_lib.ChunkList, **kwargs):
    return prompt

  # Alternatively, use `onetwo.builtins.echo_generate_text`.
  llm.generate_text.configure(fake_generate_text)
  # Now `llm.generate_text` simply returns its input.

  assert ot.run(llm.generate_text('Once upon a')) == 'Once upon a'
  # `llm.instruct` formats the prompt and sends it to `llm.generate_text`,
  # which returns the formatted prompt.
  print(ot.run(llm.instruct('Name three cities in France.')))
  # We should get something like 'Task: Name three cities in France.\n Answer:'.

  backend.register()
  # Now `llm.generate_text` points again to the backend implementation.
  ```

This approach assumes that you know exactly where the first call to the
*external model API* happens (e.g., `llm.generate_text` in the example above) so
that you can mock it.

In the future we plan to introduce a more principled and unified way of doing
this.
Likely we will base it on the `onetwo.core.tracing.trace` decorator that is
already available in OneTwo (refer to the "Agents and Tool Use" section of our
[Colab](https://colab.research.google.com/github/google-deepmind/onetwo/blob/main/colabs/tutorial.ipynb)
for more details on tracing with OneTwo).
