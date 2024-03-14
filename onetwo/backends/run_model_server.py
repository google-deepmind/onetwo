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

"""Binary that starts a OneTwo Model Server (serves as an example/test)."""

import argparse
import importlib
import json
from typing import Final

import uvicorn


_DEFAULT_PORT = 8000
_DEFAULT_BACKEND_MODULE: Final[str] = 'onetwo.backends.test_utils'
_DEFAULT_BACKEND_CLASS: Final[str] = 'LLMForTest'
_DEFAULT_BACKEND_ARGS: Final[str] = '{"default_reply": "Test reply"}'


if __name__ == '__main__':
  parser = argparse.ArgumentParser('OneTwo Model Server.')
  parser.add_argument(
      '--port', type=int, default=_DEFAULT_PORT, help='Port to listen on.'
  )
  parser.add_argument(
      '--backend_module',
      type=str,
      default=_DEFAULT_BACKEND_MODULE,
      help='Backend module to load.',
  )
  parser.add_argument(
      '--backend_class',
      type=str,
      default=_DEFAULT_BACKEND_CLASS,
      help='Backend class to instantiate.',
  )
  parser.add_argument(
      '--backend_args',
      type=str,
      default=_DEFAULT_BACKEND_ARGS,
      help='Arguments for the backend class constructor (in JSON format).',
  )
  args = parser.parse_args()
  backend_module = importlib.import_module(args.backend_module)
  backend_class = getattr(backend_module, args.backend_class)
  backend_args = json.loads(args.backend_args)
  backend = backend_class(**backend_args)
  backend.register()

  uvicorn.run(
      'onetwo.backends.model_server:ModelServer',
      host='0.0.0.0',
      port=args.port,
      factory=True,
  )
