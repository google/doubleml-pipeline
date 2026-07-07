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

"""Tests for plots in the doubleml_pipeline visualization module."""

import unittest
from unittest import mock

from doubleml_pipeline.visualization.plots import plot_incremental_kpi_line
import pandas as pd


class TestPlots(unittest.TestCase):
  """Test suite for plotting functions."""

  @mock.patch("matplotlib.pyplot.savefig")
  def test_plot_incremental_kpi_line(self, mock_savefig):
    """Tests plot_incremental_kpi_line."""
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02"],
        "kpi": [10, 20],
        "lower": [5, 10],
        "upper": [15, 30],
    })

    plot_incremental_kpi_line(
        df=df,
        date_col="date",
        kpi_col="kpi",
        lower_col="lower",
        upper_col="upper",
        out_path="/tmp/plot.png",
        title="Test Plot",
    )

    mock_savefig.assert_called_once()


if __name__ == "__main__":
  unittest.main()
