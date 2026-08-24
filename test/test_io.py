# Copyright 2026 Google LLC.
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

"""Tests for io operations in the doubleml_pipeline utils module."""

import unittest
from unittest import mock

import pandas as pd

from doubleml_pipeline.utils import io


class TestIO(unittest.TestCase):
  """Test suite for io functions."""

  @mock.patch(
      "doubleml_pipeline.utils.io.os.makedirs"
  )
  @mock.patch("pandas.DataFrame.to_csv")
  def test_save_step1_outputs_default_processlog(self, mock_to_csv, mock_makedirs):
    """Tests save_step1_outputs with default process_log='lightweight."""
    df = pd.DataFrame({"a": [1, 2]})
    io.save_step1_outputs(df, df, df, "test_model", "/tmp/out")
    mock_makedirs.assert_called_with("/tmp/out", exist_ok=True)
    call_args_list = mock_to_csv.call_args_list
    expected_calls = [
        mock.call("/tmp/out/1b_geo_item_test_model.csv", index=False),
        mock.call("/tmp/out/1c_metrics_test_model.csv", index=False)
    ]
    self.assertEqual(call_args_list, expected_calls)

  @mock.patch(
      "doubleml_pipeline.utils.io.os.makedirs"
  )
  @mock.patch("pandas.DataFrame.to_csv")
  def test_save_step1_outputs_custom_processlog(self, mock_to_csv, mock_makedirs):
    """Tests save_step1_outputs with a custom process_log='full'."""
    df = pd.DataFrame({"a": [1, 2]})
    custom_process_log = "full"
    io.save_step1_outputs(df, df, df, "test_model", "/tmp/out", process_log=custom_process_log)
    mock_makedirs.assert_called_with("/tmp/out", exist_ok=True)
    call_args_list = mock_to_csv.call_args_list
    expected_calls = [
        mock.call("/tmp/out/1a_full_df_test_model.csv", index=False),
        mock.call("/tmp/out/1b_geo_item_test_model.csv", index=False),
        mock.call("/tmp/out/1c_metrics_test_model.csv", index=False)
    ]
    self.assertEqual(call_args_list, expected_calls)

  @mock.patch(
      "doubleml_pipeline.utils.io.os.makedirs"
  )
  @mock.patch("pandas.DataFrame.to_csv")
  def test_save_step2_outputs(self, mock_to_csv, mock_makedirs):
    """Tests save_step2_outputs."""
    df = pd.DataFrame({"a": [1, 2]})
    io.save_step2_outputs(df, "/tmp/out")
    mock_makedirs.assert_called_with("/tmp/out", exist_ok=True)
    self.assertEqual(mock_to_csv.call_count, 1)


if __name__ == "__main__":
  unittest.main()
