# OneTwo

The OneTwo library provides tools to support research in *prompting*.
This includes making a *single call* to external large foundational models
as well as implementing arbitrarily complex scenarios with *multiple such calls* (some of the calls possibly depending on the outcomes of the others).
The OneTwo library focuses on experiments that **don't change weights** of the
models.

### Features
* OneTwo is designed with **concurrency and asynchronous execution** in mind.
Some of the calls to external models need to happen serially one after the
other (e.g., when the second request depends on the outcome of the first),
while others can be executed independently in parallel (executing the same
scenario on all examples in the test set). In either case you don't
want to be blocked on the model calls and may want to continue with your
scenario. OneTwo makes it easy to implement arbitrary scenarios in a familiar
pythonic way so that they can be later executed asynchronously.
* All the calls to external large models are **cached**. Perhaps your experiment
failed in the middle. Or you modified a little step in the scenario. In both
cases you will be able to quickly replay the experiment and reuse all the cached
replies.
* Scenarios can be easily executed against various models with various
parameters. OneTwo supports several public and open-sourced large foundational models out of the box. It can be easily extended to support other custom models
as well.

## Quick start

### Installation

...

### Running the unit tests

...

## Tutorial

This [Colab](https://colab.research.google.com/github/deepmind/onetwo/blob/master/colabs/tutorial.ipynb) is a good starting point and demonstrates most of the features available in onetwo.

## License

Copyright 2024 DeepMind Technologies Limited

This code is licensed under the Apache License, Version 2.0 (the \"License\");
you may not use this file except in compliance with the License. You may obtain
a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an AS IS BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

## Disclaimer

This is not an official Google product.