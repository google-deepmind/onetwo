# Copyright 2026 DeepMind Technologies Limited.
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

"""Thread-safe RPC tracker for OneTwo `with_retry` instrumentation.

Purpose
-------
RpcTracker provides lightweight, opt-in observability for all functions
decorated with `@with_retry` (see `onetwo.core.utils`). When a tracker is
active, every retry-decorated call automatically records:

  * request start / success / failure / retry events
  * per-function in-flight counts, success counts and error counts
  * error classification (type + optional status code)
  * per-request elapsed-time tracking and completed-duration histograms

Usage
-----
1. Create a tracker and install it globally:

      from onetwo.core import rpc_tracker as tracker_lib

      tracker = tracker_lib.RpcTracker()
      tracker_lib.set_rpc_tracker(tracker)

2. Any function decorated with `@with_retry` will now automatically report
   events to the global tracker.

3. Inspect stats at any time:

      stats = tracker.get_stats()
      tracker.print_summary()      # logs a formatted table

4. For periodic monitoring, enable the background logger:

      tracker.start_periodic_logging(interval_seconds=30)
      # ... run workload ...
      tracker.stop_periodic_logging()

5. When done, reset to discard all accumulated state:

      tracker.reset()

Stats dictionary
----------------
`get_stats()` returns a dict with the following keys:

  total_invocations      int    total on_request_start calls
  in_flight_count        int    currently executing requests
  in_flight_by_func      dict   {func_name: count}
  in_flight_requests     dict   {request_id: "Ns" elapsed}
  successful             dict   {func_name: count}
  errors                 dict   {func_name: count}
  error_types            dict   {"ErrorType[:code]": count}
  retries                dict   {func_name: count}
  total_retries          int
  completed_durations    list   [float seconds per completed request]
"""

import collections
import re
import statistics
import threading
import time
from typing import Any

from absl import logging


def _error_key(error: Exception) -> str:
  """Returns a classification key for an error.

  E.g. 'ValueError' or 'RpcError:499 CANCELLED'.

  Args:
    error: The exception to classify.
  """
  name = type(error).__name__
  code = getattr(error, 'code', None)
  if code is not None:
    m = re.match(r'\d+\s+\S+', str(error))
    reason = m.group(0) if m else str(code)
    return f'{name}:{reason}'
  return name


class RpcTracker:
  """Tracks with_retry call statistics in a thread-safe manner.

  Provides unique request IDs, logs send/receive/retry events, and maintains
  global stats on in-flight, successful, errored, and retried calls.
  """

  def __init__(self):
    self._lock = threading.Lock()
    self._counter = 0
    self._total_invocations = 0
    self._in_flight: dict[str, float] = {}
    self._in_flight_counts: dict[str, int] = collections.defaultdict(int)
    self._successful: dict[str, int] = collections.defaultdict(int)
    self._errors: dict[str, int] = collections.defaultdict(int)
    self._error_types: dict[str, int] = collections.defaultdict(int)
    self._retries: dict[str, int] = collections.defaultdict(int)
    self._total_retries = 0
    self._completed_durations: list[float] = []

    self._periodic_timer: threading.Timer | None = None
    self._periodic_interval = 30
    self._periodic_logging_active = False

  def on_request_start(self, func_name: str) -> str:
    """Records the start of a request and returns a unique request ID."""
    with self._lock:
      request_id = f'rpc-{self._counter:06d}'
      self._counter += 1
      self._total_invocations += 1
      self._in_flight[request_id] = time.time()
      self._in_flight_counts[func_name] += 1
      if not self._periodic_logging_active:
        self.start_periodic_logging()
      total_in_flight = sum(self._in_flight_counts.values())
    logging.vlog(
        1,
        'RPC_REQUEST_START request_id=%s func=%s | in_flight=%d',
        request_id,
        func_name,
        total_in_flight,
    )
    return request_id

  def on_request_success(self, request_id: str, func_name: str) -> None:
    """Records a successful completion of the request."""
    with self._lock:
      start_time = self._in_flight.pop(request_id, None)
      if start_time is not None:
        self._completed_durations.append(time.time() - start_time)
      self._in_flight_counts[func_name] = max(
          0, self._in_flight_counts[func_name] - 1
      )
      self._successful[func_name] += 1
    logging.vlog(
        1,
        'RPC_REQUEST_SUCCESS request_id=%s func=%s',
        request_id,
        func_name,
    )

  def on_request_retry(
      self,
      request_id: str,
      func_name: str,
      retry_num: int,
      error: Exception,
      delay: float,
  ) -> None:
    """Records a retry attempt for the given request."""
    error_type = _error_key(error)
    with self._lock:
      self._retries[func_name] += 1
      self._total_retries += 1
      self._error_types[error_type] += 1
    logging.warning(
        'RPC_REQUEST_RETRY request_id=%s func=%s retry=%d'
        ' error_type=%s delay=%.1fs error=%s',
        request_id,
        func_name,
        retry_num,
        error_type,
        delay,
        str(error)[:200],
    )

  def on_request_failure(
      self, request_id: str, func_name: str, error: Exception
  ) -> None:
    """Records the final failure of a request.

    Called after all retries are exhausted.

    Args:
      request_id: The unique request identifier from on_request_start.
      func_name: The name of the function that failed.
      error: The final exception raised.
    """
    error_type = _error_key(error)
    with self._lock:
      start_time = self._in_flight.pop(request_id, None)
      if start_time is not None:
        self._completed_durations.append(time.time() - start_time)
      self._in_flight_counts[func_name] = max(
          0, self._in_flight_counts[func_name] - 1
      )
      self._errors[func_name] += 1
      self._error_types[error_type] += 1
    logging.warning(
        'RPC_REQUEST_FAILURE request_id=%s func=%s error_type=%s error=%s',
        request_id,
        func_name,
        error_type,
        str(error)[:200],
    )

  def get_stats(self) -> dict[str, Any]:
    """Returns a snapshot of all tracked RPC statistics."""
    with self._lock:
      now = time.time()
      in_flight_details = {
          rid: f'{now - start:.0f}s' for rid, start in self._in_flight.items()
      }
      in_flight_count = sum(self._in_flight_counts.values())
      stats = {
          'total_invocations': self._total_invocations,
          'in_flight_count': in_flight_count,
          'in_flight_by_func': dict(self._in_flight_counts),
          'in_flight_requests': in_flight_details,
          'successful': dict(self._successful),
          'errors': dict(self._errors),
          'error_types': dict(self._error_types),
          'retries': dict(self._retries),
          'total_retries': self._total_retries,
          'completed_durations': list(self._completed_durations),
      }
      return stats

  def print_summary(self) -> None:
    """Logs a human-readable summary of the current RPC stats."""
    stats = self.get_stats()
    logging.info('=' * 60)
    logging.info('RPC Tracker Summary')
    logging.info('=' * 60)
    logging.info('  Invocations : %d', stats['total_invocations'])
    logging.info(
        '  In-flight   : %d %s',
        stats['in_flight_count'],
        stats['in_flight_by_func'],
    )
    if stats['in_flight_requests']:
      durations = [
          float(v.rstrip('s')) for v in stats['in_flight_requests'].values()
      ]
      logging.info(
          '    elapsed: min=%.0fs avg=%.0fs median=%.0fs max=%.0fs',
          min(durations),
          sum(durations) / len(durations),
          statistics.median(durations),
          max(durations),
      )
    logging.info('  Successful  : %s', stats['successful'])
    if stats['completed_durations']:
      d = stats['completed_durations']
      logging.info(
          '    latency: min=%.0fs avg=%.0fs median=%.0fs max=%.0fs (n=%d)',
          min(d),
          sum(d) / len(d),
          statistics.median(d),
          max(d),
          len(d),
      )
    logging.info('  Errors      : %s', stats['errors'])
    logging.info('  Error types : %s', stats['error_types'])
    logging.info(
        '  Retries     : %s (total: %d)',
        stats['retries'],
        stats['total_retries'],
    )
    logging.info('=' * 60)

  def start_periodic_logging(self, interval_seconds: float = 30) -> None:
    self._periodic_interval = interval_seconds
    self._periodic_logging_active = True
    self._schedule_next_log()

  def stop_periodic_logging(self) -> None:
    self._periodic_logging_active = False
    if self._periodic_timer is not None:
      self._periodic_timer.cancel()
      self._periodic_timer = None

  def _schedule_next_log(self) -> None:
    if not self._periodic_logging_active:
      return
    self._periodic_timer = threading.Timer(
        self._periodic_interval, self._periodic_log_tick
    )
    self._periodic_timer.daemon = True
    self._periodic_timer.start()

  def _periodic_log_tick(self) -> None:
    if not self._periodic_logging_active:
      return
    self.print_summary()
    self._schedule_next_log()

  def reset(self) -> None:
    """Resets all tracked state and stops periodic logging."""
    self.stop_periodic_logging()
    with self._lock:
      self._counter = 0
      self._total_invocations = 0
      self._in_flight.clear()
      self._in_flight_counts.clear()
      self._successful.clear()
      self._errors.clear()
      self._error_types.clear()
      self._retries.clear()
      self._total_retries = 0
      self._completed_durations.clear()


_tracker: RpcTracker | None = None


def set_rpc_tracker(tracker: RpcTracker) -> None:
  global _tracker
  _tracker = tracker


def get_rpc_tracker() -> RpcTracker | None:
  return _tracker
