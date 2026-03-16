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

import threading
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from onetwo.core import rpc_tracker as tracker_lib


class ErrorKeyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('simple_value_error', ValueError('boom'), 'ValueError'),
      ('simple_runtime_error', RuntimeError('oops'), 'RuntimeError'),
      ('type_error', TypeError('bad type'), 'TypeError'),
  )
  def test_error_key_without_code(self, error, expected):
    self.assertEqual(tracker_lib._error_key(error), expected)

  def test_error_key_with_numeric_code_attribute(self):

    class CodedError(Exception):

      def __init__(self, msg, code):
        super().__init__(msg)
        self.code = code

    error = CodedError('499 CANCELLED something', 499)
    result = tracker_lib._error_key(error)
    self.assertEqual(result, 'CodedError:499 CANCELLED')

  def test_error_key_with_non_matching_code(self):

    class CodedError(Exception):

      def __init__(self, msg, code):
        super().__init__(msg)
        self.code = code

    error = CodedError('no numeric prefix', 'UNKNOWN')
    result = tracker_lib._error_key(error)
    self.assertEqual(result, 'CodedError:UNKNOWN')


class RpcTrackerRequestLifecycleTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tracker = tracker_lib.RpcTracker()
    self.tracker.stop_periodic_logging()

  def tearDown(self):
    super().tearDown()
    self.tracker.stop_periodic_logging()

  def test_single_successful_request(self):
    rid = self.tracker.on_request_start('my_func')
    self.tracker.on_request_success(rid, 'my_func')

    stats = self.tracker.get_stats()
    self.assertEqual(stats['total_invocations'], 1)
    self.assertEqual(stats['successful'], {'my_func': 1})
    self.assertEqual(stats['in_flight_count'], 0)
    self.assertEqual(stats['errors'], {})
    self.assertEqual(stats['total_retries'], 0)
    self.assertLen(stats['completed_durations'], 1)

  def test_single_failed_request(self):
    rid = self.tracker.on_request_start('my_func')
    self.tracker.on_request_failure(rid, 'my_func', ValueError('bad'))

    stats = self.tracker.get_stats()
    self.assertEqual(stats['errors'], {'my_func': 1})
    self.assertEqual(stats['error_types'], {'ValueError': 1})
    self.assertEqual(stats['successful'], {})
    self.assertEqual(stats['in_flight_count'], 0)
    self.assertLen(stats['completed_durations'], 1)

  def test_request_with_retries_then_success(self):
    rid = self.tracker.on_request_start('my_func')
    for i in range(3):
      self.tracker.on_request_retry(
          rid, 'my_func', i, ValueError('transient'), 0.1
      )
    self.tracker.on_request_success(rid, 'my_func')

    stats = self.tracker.get_stats()
    self.assertEqual(stats['retries'], {'my_func': 3})
    self.assertEqual(stats['total_retries'], 3)
    self.assertEqual(stats['successful'], {'my_func': 1})
    self.assertEqual(stats['errors'], {})
    self.assertEqual(stats['error_types'], {'ValueError': 3})

  def test_request_with_retries_then_failure(self):
    rid = self.tracker.on_request_start('my_func')
    self.tracker.on_request_retry(rid, 'my_func', 0, RuntimeError('err'), 0.5)
    self.tracker.on_request_failure(rid, 'my_func', RuntimeError('err'))

    stats = self.tracker.get_stats()
    self.assertEqual(stats['retries'], {'my_func': 1})
    self.assertEqual(stats['total_retries'], 1)
    self.assertEqual(stats['errors'], {'my_func': 1})
    self.assertEqual(stats['error_types'], {'RuntimeError': 2})

  def test_in_flight_tracking(self):
    rid1 = self.tracker.on_request_start('func_a')
    rid2 = self.tracker.on_request_start('func_b')

    stats = self.tracker.get_stats()
    self.assertEqual(stats['in_flight_count'], 2)
    self.assertEqual(stats['in_flight_by_func'], {'func_a': 1, 'func_b': 1})
    self.assertIn(rid1, stats['in_flight_requests'])
    self.assertIn(rid2, stats['in_flight_requests'])

    self.tracker.on_request_success(rid1, 'func_a')
    stats = self.tracker.get_stats()
    self.assertEqual(stats['in_flight_count'], 1)
    self.assertEqual(stats['in_flight_by_func'], {'func_a': 0, 'func_b': 1})

  def test_multiple_functions_tracked_independently(self):
    cases = [
        ('func_a', 3, True),
        ('func_b', 1, False),
    ]
    for func_name, count, succeeds in cases:
      for _ in range(count):
        rid = self.tracker.on_request_start(func_name)
        if succeeds:
          self.tracker.on_request_success(rid, func_name)
        else:
          self.tracker.on_request_failure(rid, func_name, ValueError('x'))

    stats = self.tracker.get_stats()
    self.assertEqual(stats['total_invocations'], 4)
    self.assertEqual(stats['successful'], {'func_a': 3})
    self.assertEqual(stats['errors'], {'func_b': 1})


class RpcTrackerRequestIdTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tracker = tracker_lib.RpcTracker()
    self.tracker.stop_periodic_logging()

  def tearDown(self):
    super().tearDown()
    self.tracker.stop_periodic_logging()

  def test_request_ids_are_unique(self):
    ids = set()
    for _ in range(100):
      rid = self.tracker.on_request_start('f')
      ids.add(rid)
      self.tracker.on_request_success(rid, 'f')
    self.assertLen(ids, 100)

  def test_request_id_format(self):
    rid = self.tracker.on_request_start('f')
    self.assertRegex(rid, r'^rpc-\d{6}$')
    self.tracker.on_request_success(rid, 'f')


class RpcTrackerDurationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tracker = tracker_lib.RpcTracker()
    self.tracker.stop_periodic_logging()

  def tearDown(self):
    super().tearDown()
    self.tracker.stop_periodic_logging()

  @mock.patch('time.time')
  def test_completed_durations_recorded_on_success(self, mock_time):
    mock_time.return_value = 1000.0
    rid = self.tracker.on_request_start('f')
    mock_time.return_value = 1002.5
    self.tracker.on_request_success(rid, 'f')

    stats = self.tracker.get_stats()
    self.assertLen(stats['completed_durations'], 1)
    self.assertAlmostEqual(stats['completed_durations'][0], 2.5)

  @mock.patch('time.time')
  def test_completed_durations_recorded_on_failure(self, mock_time):
    mock_time.return_value = 1000.0
    rid = self.tracker.on_request_start('f')
    mock_time.return_value = 1001.0
    self.tracker.on_request_failure(rid, 'f', ValueError('x'))

    stats = self.tracker.get_stats()
    self.assertLen(stats['completed_durations'], 1)
    self.assertAlmostEqual(stats['completed_durations'][0], 1.0)


class RpcTrackerResetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tracker = tracker_lib.RpcTracker()
    self.tracker.stop_periodic_logging()

  def tearDown(self):
    super().tearDown()
    self.tracker.stop_periodic_logging()

  def test_reset_clears_all_state(self):
    rid = self.tracker.on_request_start('f')
    self.tracker.on_request_retry(rid, 'f', 0, ValueError('x'), 0.1)
    self.tracker.on_request_success(rid, 'f')

    self.tracker.reset()
    stats = self.tracker.get_stats()

    empty_checks = {
        'total_invocations': 0,
        'in_flight_count': 0,
        'total_retries': 0,
    }
    for key, expected in empty_checks.items():
      with self.subTest(stat=key):
        self.assertEqual(stats[key], expected)

    for key in (
        'in_flight_by_func',
        'in_flight_requests',
        'successful',
        'errors',
        'error_types',
        'retries',
    ):
      with self.subTest(stat=key):
        self.assertEmpty(stats[key])

    self.assertEmpty(stats['completed_durations'])

  def test_reset_counter_restarts_at_zero(self):
    rid1 = self.tracker.on_request_start('f')
    self.tracker.on_request_success(rid1, 'f')
    self.tracker.reset()

    rid2 = self.tracker.on_request_start('f')
    self.assertEqual(rid2, 'rpc-000000')
    self.tracker.on_request_success(rid2, 'f')


class RpcTrackerGlobalAccessorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._original = tracker_lib.get_rpc_tracker()

  def tearDown(self):
    super().tearDown()
    if self._original is not None:
      tracker_lib.set_rpc_tracker(self._original)
    else:
      tracker_lib._tracker = None

  def test_get_returns_none_by_default(self):
    tracker_lib._tracker = None
    self.assertIsNone(tracker_lib.get_rpc_tracker())

  def test_set_and_get_round_trip(self):
    tracker = tracker_lib.RpcTracker()
    tracker_lib.set_rpc_tracker(tracker)
    self.assertIs(tracker_lib.get_rpc_tracker(), tracker)


class RpcTrackerPeriodicLoggingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tracker = tracker_lib.RpcTracker()

  def tearDown(self):
    super().tearDown()
    self.tracker.stop_periodic_logging()

  def test_periodic_logging_starts_and_stops(self):
    self.tracker.start_periodic_logging(interval_seconds=100)
    self.assertTrue(self.tracker._periodic_logging_active)
    self.assertIsNotNone(self.tracker._periodic_timer)

    self.tracker.stop_periodic_logging()
    self.assertFalse(self.tracker._periodic_logging_active)
    self.assertIsNone(self.tracker._periodic_timer)

  def test_periodic_logging_calls_print_summary(self):
    with mock.patch.object(self.tracker, 'print_summary') as mock_print:
      self.tracker.start_periodic_logging(interval_seconds=0.05)
      time.sleep(0.15)
      self.tracker.stop_periodic_logging()
      self.assertGreater(mock_print.call_count, 0)

  def test_auto_starts_on_first_request(self):
    self.assertFalse(self.tracker._periodic_logging_active)
    rid = self.tracker.on_request_start('f')
    self.assertTrue(self.tracker._periodic_logging_active)
    self.tracker.on_request_success(rid, 'f')
    self.tracker.stop_periodic_logging()


class RpcTrackerPrintSummaryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tracker = tracker_lib.RpcTracker()
    self.tracker.stop_periodic_logging()

  def tearDown(self):
    super().tearDown()
    self.tracker.stop_periodic_logging()

  def test_print_summary_does_not_crash_on_empty_tracker(self):
    self.tracker.print_summary()

  def test_print_summary_does_not_crash_with_data(self):
    rid = self.tracker.on_request_start('f')
    self.tracker.on_request_retry(rid, 'f', 0, ValueError('err'), 0.1)
    self.tracker.on_request_success(rid, 'f')
    self.tracker.print_summary()


class RpcTrackerThreadSafetyTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tracker = tracker_lib.RpcTracker()
    self.tracker.stop_periodic_logging()

  def tearDown(self):
    super().tearDown()
    self.tracker.stop_periodic_logging()

  def test_concurrent_requests(self):
    num_threads = 10
    requests_per_thread = 50
    barrier = threading.Barrier(num_threads)

    def worker():
      barrier.wait()
      for _ in range(requests_per_thread):
        rid = self.tracker.on_request_start('concurrent_func')
        self.tracker.on_request_success(rid, 'concurrent_func')

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    stats = self.tracker.get_stats()
    total = num_threads * requests_per_thread
    self.assertEqual(stats['successful']['concurrent_func'], total)
    self.assertEqual(stats['in_flight_count'], 0)
    self.assertLen(stats['completed_durations'], total)


if __name__ == '__main__':
  absltest.main()
