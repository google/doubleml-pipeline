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

from doubleml_pipeline.utils.io import save_step1_outputs
from doubleml_pipeline.utils.io import save_step2_outputs
import pandas as pd


class TestIO(unittest.TestCase):
  """Test suite for io functions."""

  @mock.patch("doubleml_pipeline.utils.io.os.makedirs")
  @mock.patch("pandas.DataFrame.to_csv")
  def test_save_step1_outputs(self, mock_to_csv, mock_makedirs):
    """Tests save_step1_outputs."""
    df = pd.DataFrame({"a": [1, 2]})
    save_step1_outputs(df, df, df, "test_model", "/tmp/out")
    mock_makedirs.assert_called_with("/tmp/out", exist_ok=True)
    self.assertEqual(mock_to_csv.call_count, 3)

  @mock.patch("doubleml_pipeline.utils.io.os.makedirs")
  @mock.patch("pandas.DataFrame.to_csv")
  def test_save_step2_outputs(self, mock_to_csv, mock_makedirs):
    """Tests save_step2_outputs."""
    df = pd.DataFrame({"a": [1, 2]})
    save_step2_outputs(df, "/tmp/out")
    mock_makedirs.assert_called_with("/tmp/out", exist_ok=True)
    self.assertEqual(mock_to_csv.call_count, 1)


if __name__ == "__main__":
  unittest.main()
