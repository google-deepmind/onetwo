# OneTwo Basics

One of the key principles behind the OneTwo library is to enable the creation
of complex flows involving several calls to foundation models and possibly other
tools.
For ease of experimentation, it is important to easily change the backends or
their configuration and run the same flow on two backends/configurations, e.g.
when doing comparisons.

The bottleneck is often the multiple RPC requests that need to happen. This
makes fast iterations or experimenting on many examples slow and tedious. In
order to reduce this bottleneck, there are two strategies that are implemented
in the OneTwo library:

1. **Caching**: The result of the calls to the models are cached, which enables
one to very quickly replay a flow or an experiment which may have partially
executed (e.g. failed in the middle of execution). For example, if you have a
complex flow and want to add just one extra step, rerunning the whole thing
amounts to reading everything from cache and only executing for real that one
last step.
1. **Asynchronous Execution**: While some of the model calls might need to be
chained serially, there are many situations when you may want to execute some
calls in parallel (e.g. talking to different backends, running an experiment on
many examples, or having a step in your flow where several independent tasks are
performed). A natural way to do that is to use asynchronous programming, or
multi-threading.

## Builtins

In order to use a uniform language, we define a number of "built-in" functions
representing the basic operations one may want to perform using a model.

- `llm.generate_text()` - Generate raw text.
- `llm.generate_object()` - Generate and parse text into a Python object.
- `llm.select()` - Choose among alternatives.
- `llm.instruct()` - Generate answer to instructions.
- `llm.chat()` - Generate text in a multi-turn dialogue.

We also decouple these functions from their implementations.
So you can use them to define your **prompting strategy**, without specifying
which **model** or which **model parameters** you want to use, and only specify
those later.

## Executables

Many of the basic functions provided by the library actually return what we call
*Executables*. For example:

```python
from onetwo.builtins import llm

e = llm.generate_text(
    'Q: What are three not so well known cities in france?\nA:',
    stop=['Q:'],
    max_tokens=20,
)
# > Troyes, Dijon, Annecy.
```

Now this `e` variable has type `Executable[str]` and needs to be *executed* to
produce the final result. This happens by calling `ot.run()`:

```python
from onetwo import ot

result = ot.run(e)
```
The benefit of this two-step process is that one can define possibly complex
execution flows in a natural pythonic way, and decouple the definition of the
flow from the actual backends that are used to execute it.

## Function Registry

Specifying which backend is used to actually perform a built-in function like
`llm.generate_text` is done when calling the `register()` method on a backend.
This method registers the various function calls that the backend supports into
a global function registry.

You can temporarily override this registry if you want the calls to
`llm.generate_text` to be routed elsewhere.

For example,

```python
backend = ....
backend.register()

def fake_generate_text(prompt: str | ChunkList):
  return prompt

with ot.RegistryContext():
  llm.generate_text.configure(fake_generate_text)
  print(ot.run(llm.generate_text('test')))
```

As another example, assume you have two different backends, then it is possible
to create two distinct registries and pick one of those at execution time:

```python
backend1 = ...
backend2 = ...

with ot.RegistryContext():
  backend1.register()
  registry1 = ot.copy_registry()

with ot.RegistryContext():
  backend2.register()
  registry2 = ot.copy_registry()
```

```python
ot.run(ot.with_registry(e, registry1))
ot.run(ot.with_registry(e, registry2))
```

## Asynchronous execution

While it may take a bit of time to get used to `asyncio` if you never used it,
we tried to make it as simple as possible.

So if you need to perform two sequential calls to an LLM, you can of course run
one after the other:

```python
result = ot.run(llm.generate_text('Q: What is the southernmost city in France? A:'))
result2 = ot.run(llm.generate_text(f'Q: Who is the mayor of {result}? A:'))
```

But a better way is to create a single Executable by combining them into a
function decorated with `@ot.make_executable`:

```python
@ot.make_executable
async def f(*any_arguments):
  del any_arguments  # This example does not use arguments.
  result = await llm.generate_text('Q: What is the southernmost city in France? A:')
  result2 = await llm.generate_text(f'Q: Who is the mayor of {result}? A:')
  return result2

result = ot.run(f())
```

Indeed, the `ot.run()` function will actually block on execution and will
only return when the LLM has produced the output, while when an async function
is created with multiple await calls in its body, the execution of this function
can be interleaved with the execution of other async functions. This will be
beneficial when creating complex workflows with multiple calls as they can be
scheduled automatically in an optimal way. For example, if we were to repeatedly
call the `f` function on different inputs the inner generate_text calls could be
interleaved (see next section).

Functions decorated with `@ot.make_executable` return `Executable` objects
when called. I.e., `f()` is of type `Executable` and can be executed with
`ot.run()`.

## Combining executables

We provide in particular a way to combine executables in parallel:

```python
e1 = llm.generate_text('Q: What is the southernmost city in France? A:')
e2 = llm.generate_text('Q: What is the southernmost city in Spain? A:')
e = ot.parallel(e1, e2)
results = ot.run(e)
```

The creation of the `Executable` `e` as a parallel composition of two
executables indicates that one does not depend on the output of the other and
the calls to the LLM can thus be performed in parallel, assuming that the
backend supports it. Typically a backend will be a proxy to a remote server that
may support multiple simultaneous calls from different threads, or that may
support sending requests in batches. In this case, the execution will
automatically take advantage of this functionality to speed things up and not
having to wait for the first `generate_text` call to return before performing
the second one.

## Composing multi-step prompts

While using `async`, `await`, and `ot.parallel` lets you create arbitrary
complex flows using standard Python, there are cases where one might want a
simpler way to specify basic or typical combinations. We thus also provide ways
of composing prompts that can execute in multiple steps (i.e. involving multiple
calls to a model).

We support two different syntaxes for that:

-   Prompt templates in jinja2 templating language;
-   Composition via the `+` operator.

```python
from onetwo.builtins import composables as c

template = c.j("""\
What is the southernmost city in France? {{ generate_text() }}
Who is its mayor? {{ generate_text() }}
""")
result = ot.run(template)
```

```python
e = 'What is the southernmost city in France?' + c.generate_text() + \
    'Who is its mayor?' + c.generate_text()
result = ot.run(e)
```