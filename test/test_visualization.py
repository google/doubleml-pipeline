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

"""Tests for the ``visualization`` package: ``plots`` and ``html_extraction``.

* ``plot_incremental_kpi_line``, the alpha profile chart (its 1x2 default
  layout, the grid-construction / curve / stats panels and the FINAL curve
  colour scheme) and the value formatting of the stats table.
* The CATE scatter matrix and the CATE response curves: square cells, large
  fonts, wrapped long names that stay inside the figure, the aligned right
  column of the scatter matrix, the single-row layout of the response curves
  (one cell per treatment plus the total, legend under the row) and the 95 %
  prediction band.
* The report file name of ``html_extraction`` and the Phase 0 rows of its
  header (``Selected family`` / ``Decision``).

``plot_sensitivity_contour`` is covered with the sensitivity analysis in
test_sensitivity.py.
"""

import json
import os
import re
import tempfile
import unittest
from unittest import mock

from doubleml_pipeline.visualization import html_extraction
from doubleml_pipeline.visualization import plots
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PlotsTest(unittest.TestCase):
  """The plotting functions."""

  @mock.patch("matplotlib.pyplot.savefig")
  def test_plot_incremental_kpi_line(self, mock_savefig):
    """Tests plot_incremental_kpi_line."""
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02"],
        "kpi": [10, 20],
        "lower": [5, 10],
        "upper": [15, 30],
    })

    plots.plot_incremental_kpi_line(
        df=df,
        date_col="date",
        kpi_col="kpi",
        lower_col="lower",
        upper_col="upper",
        out_path="/tmp/plot.png",
        title="Test Plot",
    )

    mock_savefig.assert_called_once()

  @mock.patch("matplotlib.pyplot.close")
  @mock.patch("matplotlib.pyplot.savefig")
  def test_plot_alpha_profile_two_panels_by_default(
      self, mock_savefig, mock_close
  ):
    """Without the optional args the chart keeps its original 1x2 layout."""
    del mock_close
    plots.plot_alpha_profile(
        alpha_grid=[0.2, 0.5, 0.8],
        mean_loss=[0.9, 0.7, 0.8],
        per_seed_loss=[[0.9, 0.7, 0.8]],
        per_seed_alpha=[0.5],
        alpha_hat=0.5,
        one_se_set=[False, True, False],
        out_path="/tmp/alpha_profile.png",
        title="Test",
    )

    mock_savefig.assert_called_once()
    self.assertEqual(len(plt.gcf().axes), 2)

  @mock.patch("matplotlib.pyplot.close")
  @mock.patch("matplotlib.pyplot.savefig")
  def test_plot_alpha_profile_adds_curve_and_stats_panels(
      self, mock_savefig, mock_close
  ):
    """`family`/`t_normalized`/`stats_dict` add the two extra panels."""
    del mock_close
    plots.plot_alpha_profile(
        alpha_grid=[0.2, 0.5, 0.8],
        mean_loss=[0.9, 0.7, 0.8],
        per_seed_loss=[[0.9, 0.7, 0.8]],
        per_seed_alpha=[0.5],
        alpha_hat=0.5,
        one_se_set=[False, True, False],
        out_path="/tmp/alpha_profile.png",
        title="Test",
        family="power",
        t_normalized=np.linspace(0.0, 4.0, 50),
        stats_dict={
            "alpha_hat": 0.5,
            "weak identification": False,
            "dispersion ratio": float("nan"),
            "ML model Y": "rf",
        },
        stats_tags={"dispersion ratio": "diag2"},
    )

    mock_savefig.assert_called_once()
    fig = plt.gcf()
    # 2 + 2 + 1 panels (grid construction, seed-wise distribution, R-loss
    # curve, saturation curve, stats) + the twin axis carrying the treatment
    # histogram.
    self.assertEqual(len(fig.axes), 6)

    titles = [ax.get_title() for ax in fig.axes]
    self.assertIn("Grid construction", titles)
    self.assertIn("Seed-wise alpha distribution", titles)
    self.assertIn("Profiled R-loss curve", titles)
    self.assertIn("Fitted saturation curve", titles)
    self.assertIn("Key stats & process metrics", titles)
    # Row order of the 2 + 2 + 1 layout: grid construction and seed-wise
    # distribution on top, R-loss and the saturation curve below, stats last.
    grid_ax = fig.axes[titles.index("Grid construction")]
    loss_ax = fig.axes[titles.index("Profiled R-loss curve")]
    stats_ax = fig.axes[titles.index("Key stats & process metrics")]
    self.assertGreater(grid_ax.get_position().y0, loss_ax.get_position().y0)
    self.assertGreater(loss_ax.get_position().y0, stats_ax.get_position().y0)
    # The stats cell spans the full width (both columns of the grid above).
    self.assertGreater(
        stats_ax.get_position().width, 1.8 * grid_ax.get_position().width
    )

    curve_ax = fig.axes[titles.index("Fitted saturation curve")]
    # The saturation curve itself is the thick red (FINAL) line.
    curve = next(
        ln for ln in curve_ax.get_lines() if ln.get_linewidth() == 3
    )
    self.assertEqual(curve.get_color(), "red")
    x_vals, y_vals = curve.get_xdata(), curve.get_ydata()
    # f(T) = T ** 0.5 must pass through the origin and stay monotone.
    self.assertAlmostEqual(y_vals[0], 0.0)
    self.assertTrue(np.all(np.diff(y_vals) >= 0))
    np.testing.assert_allclose(y_vals, np.sqrt(x_vals), atol=1e-9)

    # `Cell.get_text()` hands back a Text artist, not the string itself.  The
    # full-width cell holds two side-by-side tables.
    self.assertEqual(len(stats_ax.tables), 2)
    rendered = set()
    for table in stats_ax.tables:
      rendered.update(
          cell.get_text().get_text() for cell in table.get_celld().values()
      )
    self.assertIn("alpha_hat", rendered)
    self.assertIn("0.5", rendered)
    # Booleans render as yes/no and non-finite values as n/a.
    self.assertIn("no", rendered)
    self.assertIn("n/a", rendered)
    # The third column names the diagnostic (Diag1-Diag6) of a row (blank
    # otherwise).
    self.assertIn("diagnostics", rendered)
    self.assertIn("diag2", rendered)
    # Every chart ends with an explanatory footnote.
    footnotes = [t.get_text() for t in fig.texts if "Notes" in t.get_text()]
    self.assertTrue(footnotes)
    self.assertIn("summary_alpha_grid.csv", footnotes[0])
    self.assertIn(
        "summary_alpha_seed-wise_alpha_distribution.csv", footnotes[0]
    )

  def test_format_stat_value(self):
    """Booleans, NaN and None get human-readable renderings."""
    # pylint: disable=protected-access
    # `_format_stat_value` is deliberately exercised directly: its NaN / bool /
    # None handling is the part most likely to regress, and going through a
    # rendered figure to assert on it would be far more brittle.
    self.assertEqual(plots._format_stat_value(True), "yes")
    self.assertEqual(plots._format_stat_value(False), "no")
    self.assertEqual(plots._format_stat_value(None), "n/a")
    self.assertEqual(plots._format_stat_value(float("nan")), "n/a")
    self.assertEqual(plots._format_stat_value(1234), "1,234")
    self.assertEqual(plots._format_stat_value(0.16666666666), "0.166667")
    self.assertEqual(plots._format_stat_value("rf"), "rf")

  def test_wrap_axis_label_breaks_on_underscores(self):
    """Labels break after underscores / spaces; unbreakable tokens are cut."""
    # pylint: disable=protected-access
    self.assertEqual(
        plots._wrap_axis_label("Inc. Effect: estimated_incremental_KPI_tv"),
        "Inc. Effect: estimated_\nincremental_KPI_tv",
    )
    self.assertEqual(plots._wrap_axis_label("short"), "short")
    # Spaces are break opportunities too, and a token without any break
    # opportunity is cut at the width instead of overflowing.
    self.assertEqual(
        plots._wrap_axis_label("Grouped by: average_shelf_price", 20),
        "Grouped by: average_\nshelf_price",
    )
    self.assertEqual(
        plots._wrap_axis_label("abcdefghijklmnopqrstuvwxyz", 10),
        "abcdefghij\nklmnopqrst\nuvwxyz",
    )
    # pylint: enable=protected-access

  @staticmethod
  def _unwrap(text):
    """Reverses the label wrapping (breaks happen after `_` or a space)."""
    out = ""
    for line in text.split("\n"):
      if out and not out.endswith("_"):
        out += " "
      out += line
    return out

  def test_fit_text_wraps_on_measured_width_and_shrinks_last(self):
    """Labels wrap to the physical length; the font only shrinks beyond the line budget."""
    # pylint: disable=protected-access
    short, fs = plots._fit_text("Grouped by: price", 5.0, 26, max_lines=2)
    self.assertEqual(short, "Grouped by: price")
    self.assertEqual(fs, 26.0)
    # A long name wraps (after underscores) and keeps the full font size as
    # long as it fits the line budget ...
    name = "Grouped by: average_shelf_price_index"
    wrapped, fs = plots._fit_text(name, 5.0, 26, max_lines=3)
    self.assertEqual(fs, 26.0)
    self.assertIn("\n", wrapped)
    self.assertLessEqual(wrapped.count("\n") + 1, 3)
    self.assertEqual(self._unwrap(wrapped), name)
    for line in wrapped.split("\n"):
      self.assertLessEqual(plots._text_width_in(line, fs), 5.0)
    # ... and shrinks (never below the floor) when it would need more lines.
    squeezed, fs_small = plots._fit_text(name, 5.0, 26, max_lines=1)
    self.assertLess(fs_small, 26.0)
    self.assertGreaterEqual(fs_small, plots._CATE_MIN_CELL_FS)
    for line in squeezed.split("\n"):
      self.assertLessEqual(plots._text_width_in(line, fs_small), 5.0)
    # Measured widths agree with the renderer to within a few percent.
    fig = plt.figure()
    txt = fig.text(0, 0, name, fontsize=26)
    rendered_in = (
        txt.get_window_extent(renderer=fig.canvas.get_renderer()).width
        / fig.dpi
    )
    self.assertAlmostEqual(
        plots._text_width_in(name, 26) / rendered_in, 1.03, delta=0.03
    )
    plt.close(fig)
    # pylint: enable=protected-access

  def _long_name_frame(self):
    """A frame whose column names are long enough to overflow a cell."""
    frame = self._scatter_frame()
    renames = {
        "tv_spend": "national_tv_campaign_spend_total_jpy",
        "digital_spend": "digital_display_and_search_spend_total_jpy",
        "price": "average_shelf_price_index_normalised_by_category",
        "holiday": "is_national_holiday_or_weekend_flag_for_region",
    }
    renames.update({
        f"estimated_incremental_KPI_{k}": f"estimated_incremental_KPI_{v}"
        for k, v in list(renames.items())[:2]
    })
    return frame.rename(columns=renames), renames

  @staticmethod
  def _all_axes(fig):
    """The figure's axes plus their inset (child) axes, e.g. colorbars."""
    out = []
    for ax in fig.axes:
      out.append(ax)
      out.extend(ax.child_axes)
    return out

  def _assert_texts_inside_figure(self, fig):
    """Every visible text of ``fig`` lies within the figure bounds."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_box = fig.bbox
    tol = 2.0  # pixels
    for ax in self._all_axes(fig):
      texts = [ax.title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
      for txt in texts:
        if not txt.get_text():
          continue
        box = txt.get_window_extent(renderer=renderer)
        self.assertGreaterEqual(box.x0, fig_box.x0 - tol, txt.get_text())
        self.assertLessEqual(box.x1, fig_box.x1 + tol, txt.get_text())
        self.assertGreaterEqual(box.y0, fig_box.y0 - tol, txt.get_text())
        self.assertLessEqual(box.y1, fig_box.y1 + tol, txt.get_text())
      leg = ax.get_legend()
      if leg is not None:
        box = leg.get_window_extent(renderer=renderer)
        self.assertLessEqual(box.x1, fig_box.x1 + tol)
        self.assertGreaterEqual(box.x0, fig_box.x0 - tol)
    for leg in fig.legends:
      box = leg.get_window_extent(renderer=renderer)
      self.assertLessEqual(box.x1, fig_box.x1 + tol)
      self.assertGreaterEqual(box.x0, fig_box.x0 - tol)
      self.assertGreaterEqual(box.y0, fig_box.y0 - tol)
      self.assertLessEqual(box.y1, fig_box.y1 + tol)

  def _assert_single_row_with_legend_below(self, fig, n_cells):
    """The response chart: ``n_cells`` square cells in one row, legend under.

    Args:
      fig: The response-curve figure.
      n_cells: Expected number of cells (treatments + 1).

    Returns:
      The cells, left to right.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    cells = [ax for ax in fig.axes if ax.get_box_aspect() == 1]
    self.assertEqual(len(cells), n_cells)
    # No right-hand text column: the cells are the only axes.
    self.assertEqual(len(fig.axes), n_cells)
    x_lefts = [ax.get_position().x0 for ax in cells]
    self.assertEqual(x_lefts, sorted(x_lefts))
    self.assertEqual(len({round(ax.get_position().y0, 6) for ax in cells}), 1)
    # The figure legend lies entirely below every cell's x label.
    self.assertEqual(len(fig.legends), 1)
    leg_box = fig.legends[0].get_window_extent(renderer=renderer)
    for ax in cells:
      label_box = ax.xaxis.label.get_window_extent(renderer=renderer)
      self.assertLess(leg_box.y1, label_box.y0)
    return cells

  @mock.patch("matplotlib.pyplot.close")
  @mock.patch("matplotlib.pyplot.savefig")
  def test_cate_charts_keep_long_names_inside_the_figure(
      self, mock_savefig, mock_close
  ):
    """Long names wrap: nothing runs out of the figure, cells stay square."""
    del mock_savefig, mock_close
    frame, renames = self._long_name_frame()
    d_cols = [renames["tv_spend"], renames["digital_spend"]]
    cont_cols = [renames["price"]]
    bin_cols = [renames["holiday"]]
    eq = {d: f"CATE_{d} = 123456.789 + 25.123 * {cont_cols[0]}" for d in d_cols}
    stats = {
        d: {
            "coef": {cont_cols[0]: 123456789.123, bin_cols[0]: -0.000012},
            "pvalue": {cont_cols[0]: 0.0001, bin_cols[0]: 0.25},
            "stderr": {cont_cols[0]: 98765432.1, bin_cols[0]: 0.5},
        }
        for d in d_cols
    }
    limits = plots.plot_cate_scatter_matrix(
        frame, d_cols, cont_cols, bin_cols, eq, "unused.png", stats
    )
    scatter_fig = plt.gcf()
    self._assert_texts_inside_figure(scatter_fig)
    cells = [ax for ax in scatter_fig.axes if ax.get_box_aspect() == 1]
    self.assertEqual(len(cells), 3)
    # The returned scales are those of the cells.
    self.assertEqual(set(limits), {(0, 0), (1, 0), (2, 0)})
    self.assertEqual(limits[(0, 0)], (cells[0].get_xlim(), cells[0].get_ylim()))
    # Titles and x labels are wrapped (several lines) rather than cut, and
    # never drop below the readable floor.
    # pylint: disable=protected-access
    title = cells[0].title
    self.assertIn("\n", title.get_text())
    self.assertEqual(
        self._unwrap(title.get_text()), f"Grouped by: {cont_cols[0]}"
    )
    self.assertGreaterEqual(title.get_fontsize(), plots._CATE_MIN_CELL_FS)
    x_label = cells[0].xaxis.label
    self.assertIn("\n", x_label.get_text())
    self.assertEqual(
        self._unwrap(x_label.get_text()), f"Treatment: {d_cols[0]}"
    )
    self.assertGreaterEqual(x_label.get_fontsize(), plots._CATE_MIN_CELL_FS)
    # The colorbar label of the long covariate is wrapped too.
    cbar_labels = [
        ax.yaxis.label.get_text()
        for ax in self._all_axes(scatter_fig)
        if ax.yaxis.label.get_text().startswith("Covariate:")
    ]
    self.assertTrue(cbar_labels)
    self.assertIn("\n", cbar_labels[0])
    self.assertEqual(
        self._unwrap(cbar_labels[0]), f"Covariate: {cont_cols[0]}"
    )
    # The stats table formats the huge coefficient in scientific notation so
    # the row stays within the box width.
    stats_text = scatter_fig.axes[-1].texts[1].get_text()
    self.assertIn("1.235e+08", stats_text)
    self.assertNotIn("123456789.123", stats_text)
    for line in stats_text.split("\n"):
      self.assertLessEqual(len(line), plots._CATE_MAX_CHARS)

    plots.plot_cate_response_curves(frame, d_cols, bin_cols, "unused.png")
    fig = plt.gcf()
    self._assert_texts_inside_figure(fig)
    # One cell per treatment plus the total, side by side.
    cells = self._assert_single_row_with_legend_below(fig, 3)
    # The long treatment name is wrapped in the cell title as well (and the
    # x label, as in the scatter matrix).
    title = cells[0].title
    self.assertIn("\n", title.get_text())
    self.assertEqual(self._unwrap(title.get_text()), d_cols[0])
    self.assertGreaterEqual(title.get_fontsize(), plots._CATE_MIN_CELL_FS)
    self.assertEqual(
        self._unwrap(cells[0].xaxis.label.get_text()),
        f"Treatment: {d_cols[0]}",
    )
    self.assertEqual(cells[-1].get_title(), "Total")
    # pylint: enable=protected-access
    # The response note sits in the lower-right corner of every cell, the
    # area the curves rising from the origin leave clear.
    for ax in cells:
      notes = [t for t in ax.texts if t.get_text().startswith("response:")]
      self.assertEqual(len(notes), 1)
      self.assertEqual(notes[0].get_position(), (0.98, 0.03))
      self.assertEqual(notes[0].get_ha(), "right")
      self.assertEqual(notes[0].get_va(), "bottom")

  def _scatter_frame(self):
    """Builds a small synthetic frame for the CATE scatter-matrix tests.

    Returns:
      A 40-row DataFrame with a binary `holiday` column, a continuous `price`
      column, two treatment columns (`tv_spend`, `digital_spend`), their
      `estimated_incremental_KPI_<treatment>` counterparts (1.5x the spend)
      and the summed `total_estimated_incremental_KPI` column.
    """
    rng = np.random.default_rng(0)
    n = 40
    data = {
        "holiday": rng.integers(0, 2, n),
        "price": rng.normal(10, 2, n),
    }
    total = np.zeros(n)
    for d in ["tv_spend", "digital_spend"]:
      data[d] = rng.gamma(2.0, 50.0, n)
      kpi = data[d] * 1.5
      data[f"estimated_incremental_KPI_{d}"] = kpi
      total = total + kpi
    data["total_estimated_incremental_KPI"] = total
    return pd.DataFrame(data)

  @mock.patch("matplotlib.pyplot.close")
  @mock.patch("matplotlib.pyplot.savefig")
  def test_cate_scatter_matrix_cells_are_square(self, mock_savefig, mock_close):
    """Every scatter cell is square rather than stretched vertically."""
    del mock_savefig, mock_close
    d_cols = ["tv_spend", "digital_spend"]
    cont_cols = ["price"]
    plots.plot_cate_scatter_matrix(
        self._scatter_frame(),
        d_cols,
        cont_cols,
        ["holiday"],
        {d: f"CATE_{d} = 1.0" for d in d_cols},
        "unused.png",
    )
    fig = plt.gcf()
    squares = [ax for ax in fig.axes if ax.get_box_aspect() == 1]
    # One square cell per treatment row plus the trailing "total" row.
    self.assertEqual(len(squares), (len(d_cols) + 1) * len(cont_cols))

  @mock.patch("matplotlib.pyplot.close")
  @mock.patch("matplotlib.pyplot.savefig")
  def test_cate_scatter_matrix_uses_large_fonts(self, mock_savefig, mock_close):
    """Labels are large enough to stay legible once the report scales the png."""
    del mock_savefig, mock_close
    d_cols = ["tv_spend"]
    plots.plot_cate_scatter_matrix(
        self._scatter_frame(),
        d_cols,
        ["price"],
        ["holiday"],
        {d: f"CATE_{d} = 1.0" for d in d_cols},
        "unused.png",
    )
    fig = plt.gcf()
    cell = [ax for ax in fig.axes if ax.get_box_aspect() == 1][0]
    self.assertGreaterEqual(cell.xaxis.label.get_fontsize(), 20)
    self.assertGreaterEqual(cell.yaxis.label.get_fontsize(), 20)
    self.assertGreaterEqual(cell.title.get_fontsize(), 20)

  def _right_column_boxes(self, fig):
    """Returns ``(legend_bbox, [text_bboxes])`` of the right column in pixels."""
    renderer = fig.canvas.get_renderer()
    ax_right = fig.axes[-1]
    leg = ax_right.get_legend()
    leg_box = leg.get_window_extent(renderer=renderer)
    text_boxes = []
    for txt in ax_right.texts:
      patch = txt.get_bbox_patch()
      extent = txt.get_window_extent(renderer=renderer)
      pad = patch.get_boxstyle().pad * txt.get_fontsize() * fig.dpi / 72.0
      text_boxes.append((extent.x0 - pad, extent.x1 + pad))
    return (leg_box.x0, leg_box.x1), text_boxes

  @mock.patch("matplotlib.pyplot.close")
  @mock.patch("matplotlib.pyplot.savefig")
  def test_cate_scatter_matrix_right_column_is_aligned_and_filled(
      self, mock_savefig, mock_close
  ):
    """Legend and the two text boxes share their edges and fill the column."""
    del mock_savefig, mock_close
    d_cols = ["tv_spend", "digital_spend"]
    frame = self._scatter_frame()
    stats = {
        d: {
            "coef": {"holiday": 0.5, "price": -0.1},
            "pvalue": {"holiday": 0.01, "price": 0.2},
            "stderr": {"holiday": 0.1, "price": 0.05},
        }
        for d in d_cols
    }
    plots.plot_cate_scatter_matrix(
        frame,
        d_cols,
        ["price"],
        ["holiday"],
        {d: f"CATE_{d} = 1.0 + 0.5 * holiday" for d in d_cols},
        "unused.png",
        stats,
    )
    fig = plt.gcf()
    fig.canvas.draw()
    (leg_x0, leg_x1), text_boxes = self._right_column_boxes(fig)
    self.assertEqual(len(text_boxes), 2)
    width_px = fig.get_figwidth() * fig.dpi
    tol = 0.004 * width_px
    # The two text boxes are equally wide and the legend spans the same
    # horizontal extent (mode='expand' on the measured text-box width).
    for x0, x1 in text_boxes:
      self.assertAlmostEqual(x0, text_boxes[0][0], delta=tol)
      self.assertAlmostEqual(x1, text_boxes[0][1], delta=tol)
      self.assertAlmostEqual(leg_x0, x0, delta=tol)
      self.assertAlmostEqual(leg_x1, x1, delta=tol)
    # No blank column: the boxes end close to the right edge of the figure.
    self.assertGreater(text_boxes[0][1], 0.96 * width_px)
    # The monospace text is larger than before (19 pt unless it had to shrink
    # to fit the height; with two treatments it must not go below 12 pt).
    ax_right = fig.axes[-1]
    self.assertGreaterEqual(ax_right.texts[0].get_fontsize(), 12)

  @mock.patch("matplotlib.pyplot.close")
  @mock.patch("matplotlib.pyplot.savefig")
  def test_cate_response_curves_one_cell_per_treatment_in_a_row(
      self, mock_savefig, mock_close
  ):
    """1 x (treatments + 1) cells on one scale from the origin; lines, no dots."""
    del mock_savefig, mock_close
    d_cols = ["tv_spend", "digital_spend"]
    frame = self._scatter_frame()

    # tv_spend follows a tanh saturation; digital_spend stays linear.
    alpha, t_ref = 0.02, 150.0
    t_norm = frame["tv_spend"].to_numpy(dtype=float)
    f_row = np.tanh(alpha * t_norm) / np.tanh(alpha * t_ref)
    specs = {
        "tv_spend": {
            "f_row": f_row,
            "divisor_row": np.ones(len(frame)),
            "curve_fn": lambda t: np.tanh(alpha * t) / np.tanh(alpha * t_ref),
            "label": "tanh saturation",
        }
    }
    fitted = plots.plot_cate_response_curves(
        frame, d_cols, ["holiday"], "unused.png", response_specs=specs
    )
    fig = plt.gcf()
    # One cell per treatment plus the total, side by side in a single row
    # (no column per continuous covariate: the curves do not depend on them).
    cells = self._assert_single_row_with_legend_below(fig, len(d_cols) + 1)
    self.assertEqual(
        [ax.get_title() for ax in cells], ["tv_spend", "digital_spend", "Total"]
    )
    self.assertEqual(
        [ax.get_xlabel() for ax in cells],
        [
            "Treatment: tv_spend",
            "Treatment: digital_spend",
            "Total Treatment (Excl. Lags)",
        ],
    )
    for ax in cells:
      # One translucent line per binary group, no scatter; the only
      # collections are the translucent 95% prediction bands, one per group,
      # drawn beneath the lines.
      self.assertEqual(len(ax.get_lines()), 2)
      for line in ax.get_lines():
        self.assertLess(line.get_alpha(), 1.0)
      self.assertEqual(len(ax.collections), 2)
      for band in ax.collections:
        self.assertIsInstance(band, matplotlib.collections.PolyCollection)
        self.assertLess(band.get_alpha(), 0.5)
        self.assertLess(band.get_zorder(), ax.get_lines()[0].get_zorder())
    # One scale from the origin for every cell: identical limits starting at
    # zero, wide enough for the largest treatment (the total) and the highest
    # curve or band drawn anywhere.
    self.assertEqual(len({ax.get_xlim() for ax in cells}), 1)
    self.assertEqual(len({ax.get_ylim() for ax in cells}), 1)
    (x0, x1), (y0, y1) = cells[0].get_xlim(), cells[0].get_ylim()
    self.assertEqual((x0, y0), (0.0, 0.0))
    inputs = [frame[d].to_numpy() for d in d_cols]
    inputs.append(frame[d_cols].sum(axis=1).to_numpy())
    self.assertGreater(x1, max(x.max() for x in inputs))
    y_drawn = max(
        max(ln.get_ydata().max() for ln in ax.get_lines()) for ax in cells
    )
    bands = []
    for ax in cells:
      bands.extend(ax.collections)
    band_top = max(band.get_paths()[0].vertices[:, 1].max() for band in bands)
    self.assertGreater(y1, max(y_drawn, band_top))
    self.assertLess(y1, 1.2 * max(y_drawn, band_top))
    # Every curve covers its own observed treatment range, from zero.
    for ax, x_data in zip(cells, inputs):
      for line in ax.get_lines():
        self.assertEqual(line.get_xdata().min(), 0.0)
        self.assertAlmostEqual(line.get_xdata().max(), x_data.max())
    # Group colours match the dots of the scatter matrix: both charts assign
    # tab10 colours in the order the binary patterns first appear.
    cmap = plt.get_cmap("tab10")
    for i, ln in enumerate(cells[0].get_lines()):
      np.testing.assert_allclose(
          matplotlib.colors.to_rgba(ln.get_color())[:3], cmap(i)[:3]
      )
    # KPI = 1.5 * spend in the fixture, so the linear treatment recovers 1.5
    # per group and the saturated one a positive slope.
    for theta in fitted["digital_spend"].values():
      self.assertAlmostEqual(theta, 1.5, places=6)
    for theta in fitted["tv_spend"].values():
      self.assertGreater(theta, 0.0)
    # The legend under the row names every group (same labels and
    # first-appearance order as the scatter matrix) and explains the band; the
    # equations / stats boxes of the scatter matrix are not repeated.
    legend = fig.legends[0]
    legend_texts = [t.get_text().strip() for t in legend.get_texts()]
    groups = [f"holiday={v}" for v in frame["holiday"].drop_duplicates()]
    self.assertEqual(
        legend_texts,
        groups + ["95% prediction interval (rows of the group)"],
    )
    self.assertEqual(legend.get_title().get_text(), "Binary Covariates")
    self.assertEqual(sum(len(ax.texts) for ax in fig.axes), len(cells))
    self.assertIn("Response Curves", fig._suptitle.get_text())  # pylint: disable=protected-access

  @mock.patch("matplotlib.pyplot.close")
  @mock.patch("matplotlib.pyplot.savefig")
  def test_cate_response_curves_legend_wraps_to_the_row_width(
      self, mock_savefig, mock_close
  ):
    """Many binary groups: the legend wraps into several rows under the cells."""
    del mock_savefig, mock_close
    rng = np.random.default_rng(3)
    frame = self._scatter_frame()
    frame["promo_flag"] = rng.integers(0, 2, len(frame))
    bin_cols = ["holiday", "promo_flag"]
    d_cols = ["tv_spend", "digital_spend"]
    plots.plot_cate_response_curves(frame, d_cols, bin_cols, "unused.png")
    fig = plt.gcf()
    self._assert_texts_inside_figure(fig)
    cells = self._assert_single_row_with_legend_below(fig, len(d_cols) + 1)
    legend = fig.legends[0]
    n_groups = len(frame[bin_cols].drop_duplicates())
    self.assertEqual(n_groups, 4)
    self.assertEqual(len(legend.get_texts()), n_groups + 1)
    # Every cell (each treatment and the total) carries one response curve
    # and one 95% band per binary-covariate combination, coloured like the
    # dots of the scatter matrix (tab10 in first-appearance order).
    cmap = plt.get_cmap("tab10")
    for ax in cells:
      self.assertEqual(len(ax.get_lines()), n_groups)
      self.assertEqual(len(ax.collections), n_groups)
      for i, ln in enumerate(ax.get_lines()):
        np.testing.assert_allclose(
            matplotlib.colors.to_rgba(ln.get_color())[:3], cmap(i)[:3]
        )
    # The five entries do not fit one row of the figure, so the legend is laid
    # out in several columns (entries of one column share their left edge)
    # and several rows -- but not one entry per column.
    renderer = fig.canvas.get_renderer()
    column_edges = {
        round(t.get_window_extent(renderer=renderer).x0)
        for t in legend.get_texts()
    }
    row_edges = {
        round(t.get_window_extent(renderer=renderer).y0)
        for t in legend.get_texts()
    }
    self.assertGreater(len(column_edges), 1)
    self.assertLess(len(column_edges), n_groups + 1)
    self.assertGreater(len(row_edges), 1)
    # The legend spans no more than the cells' row.
    leg_box = legend.get_window_extent(renderer=renderer)
    row_x0 = cells[0].get_window_extent(renderer=renderer).x0
    row_x1 = cells[-1].get_window_extent(renderer=renderer).x1
    tol = 2.0  # pixels
    self.assertGreaterEqual(leg_box.x0, row_x0 - tol)
    self.assertLessEqual(leg_box.x1, row_x1 + tol)

  @mock.patch("matplotlib.pyplot.close")
  @mock.patch("matplotlib.pyplot.savefig")
  def test_cate_charts_share_the_cell_size_and_fonts(
      self, mock_savefig, mock_close
  ):
    """Scatter-matrix and response-curve cells are equal squares, same fonts.

    Both charts draw their cells at ``plots._CATE_CELL_IN`` inches whatever
    their number of columns, so that the report, which scales the two pngs by
    one common factor, shows equal squares with equal font sizes.
    """
    del mock_savefig, mock_close
    # pylint: disable=protected-access
    d_cols = ["tv_spend", "digital_spend"]
    frame = self._scatter_frame()
    plots.plot_cate_scatter_matrix(
        frame,
        d_cols,
        ["price"],
        ["holiday"],
        {d: f"CATE_{d} = 1.0" for d in d_cols},
        "unused.png",
    )
    matrix_fig = plt.gcf()
    plots.plot_cate_response_curves(frame, d_cols, ["holiday"], "unused.png")
    row_fig = plt.gcf()
    self.assertIsNot(matrix_fig, row_fig)
    matrix_fig.canvas.draw()
    row_fig.canvas.draw()

    def cell_edges_in(fig):
      """(width, height) in inches of every square cell of ``fig``."""
      return [
          (
              ax.get_position().width * fig.get_figwidth(),
              ax.get_position().height * fig.get_figheight(),
          )
          for ax in fig.axes
          if ax.get_box_aspect() == 1
      ]

    matrix_cells = cell_edges_in(matrix_fig)
    row_cells = cell_edges_in(row_fig)
    # 1 covariate column x 3 rows, and 3 cells in the row.
    self.assertEqual(len(matrix_cells), 3)
    self.assertEqual(len(row_cells), 3)
    for width, height in matrix_cells + row_cells:
      self.assertAlmostEqual(width, plots._CATE_CELL_IN, places=2)
      self.assertAlmostEqual(height, plots._CATE_CELL_IN, places=2)
    # Same fonts: the x labels fit at the full label font in both charts and
    # the tick labels share one size.
    matrix_cell = [ax for ax in matrix_fig.axes if ax.get_box_aspect() == 1][0]
    row_cell = [ax for ax in row_fig.axes if ax.get_box_aspect() == 1][0]
    self.assertEqual(
        matrix_cell.xaxis.label.get_fontsize(),
        row_cell.xaxis.label.get_fontsize(),
    )
    self.assertEqual(
        matrix_cell.get_xticklabels()[0].get_fontsize(),
        row_cell.get_xticklabels()[0].get_fontsize(),
    )
    # The figure widths differ with the number of columns; the report
    # compensates by scaling the pngs proportionally (html_extraction).
    self.assertNotEqual(matrix_fig.get_figwidth(), row_fig.get_figwidth())

  def test_cate_cell_geometry_margins_grow_with_the_wrapped_label_lines(self):
    """Every extra line of a wrapped label adds one line height to its margin.

    The top margin holds the cell titles, the bottom margin the x labels:
    each grows by ``_CATE_EXTRA_LINE_IN`` per extra line its labels wrap to,
    and the row gap by both.  The left margin never grows (it holds the full
    y-label budget already).  The base margins themselves are layout
    constants and are deliberately not pinned here.
    """
    # pylint: disable=protected-access
    short = plots._cate_cell_geometry(
        2, plots._cate_cell_texts(["tv_spend", "digital_spend"], ["price"])
    )
    self.assertEqual(short["cell_in"], plots._CATE_CELL_IN)
    # Long names: the cell title (`Grouped by: <covariate>`) and the x label
    # (`Treatment: <treatment>`) wrap to two lines each.
    _, renames = self._long_name_frame()
    texts = plots._cate_cell_texts([renames["tv_spend"]], [renames["price"]])
    cell_in = short["cell_in"]
    title, _ = plots._fit_cell_label("title", texts["title"][0], cell_in)
    x_label, _ = plots._fit_cell_label("x", texts["x"][0], cell_in)
    self.assertEqual(title.count("\n"), 1)
    self.assertEqual(x_label.count("\n"), 1)
    wrapped = plots._cate_cell_geometry(1, texts)
    extra = plots._CATE_EXTRA_LINE_IN
    self.assertAlmostEqual(wrapped["top_in"] - short["top_in"], extra)
    self.assertAlmostEqual(wrapped["bottom_in"] - short["bottom_in"], extra)
    self.assertAlmostEqual(
        wrapped["hspace"] - short["hspace"], 2 * extra / plots._CATE_CELL_IN
    )
    self.assertEqual(wrapped["left_in"], short["left_in"])
    # The response-curve chart has no colorbar gutter: only the room for the
    # last tick label is kept on the right; the other margins are the same.
    row = plots._cate_cell_geometry(
        3, plots._cate_response_cell_texts(["tv_spend", "digital_spend"]),
        colorbar=False,
    )
    self.assertEqual(row["right_in"], plots._CATE_ROW_RIGHT_IN)
    self.assertLess(row["right_in"], short["right_in"])
    self.assertEqual(row["top_in"], short["top_in"])
    self.assertEqual(row["bottom_in"], short["bottom_in"])
    # pylint: enable=protected-access

  def test_response_band_is_the_95pct_range_of_row_slopes(self):
    """The band is the 2.5-97.5% range of theta_i = KPI_i / f_i times the curve."""
    # pylint: disable=protected-access
    rng = np.random.default_rng(1)
    f_row = rng.gamma(2.0, 50.0, 400)
    # Row-level CATE driven by a continuous covariate: theta_i in [1, 3].
    theta_rows = 1.0 + 2.0 * rng.uniform(size=400)
    kpi = theta_rows * f_row
    x_grid = np.linspace(0.0, 200.0, 50)
    band = plots._response_band(f_row, kpi, x_grid)
    self.assertIsNotNone(band)
    lower, upper = band
    q_lo, q_hi = np.quantile(theta_rows, [0.025, 0.975])
    np.testing.assert_allclose(lower, q_lo * x_grid, rtol=1e-9)
    np.testing.assert_allclose(upper, q_hi * x_grid, rtol=1e-9)
    # The linear fit through the origin lies inside the band.
    y_grid, theta, band_fit = plots._fit_group_response(
        f_row, kpi, None, x_grid
    )
    self.assertTrue(np.all(band_fit[0][1:] <= y_grid[1:]))
    self.assertTrue(np.all(y_grid[1:] <= band_fit[1][1:]))
    self.assertGreater(theta, q_lo)
    self.assertLess(theta, q_hi)
    # Rows without treatment (f_i = 0) carry no slope and are ignored; fewer
    # than two usable rows give no band at all.
    with_zero = plots._response_band(
        np.concatenate([f_row, [0.0]]), np.concatenate([kpi, [5.0]]), x_grid
    )
    np.testing.assert_allclose(with_zero[1], upper)
    self.assertIsNone(plots._response_band([10.0, 0.0], [15.0, 0.0], x_grid))
    # Identical slopes collapse the band onto the curve.
    flat = plots._response_band(f_row, 1.5 * f_row, x_grid)
    np.testing.assert_allclose(flat[0], flat[1])
    np.testing.assert_allclose(flat[0], 1.5 * x_grid)
    # pylint: enable=protected-access

  @mock.patch("matplotlib.pyplot.close")
  @mock.patch("matplotlib.pyplot.savefig")
  def test_plot_alpha_profile_grid_panel_and_final_curve(
      self, mock_savefig, mock_close
  ):
    """`grid_result` fills the grid-construction panel and names FINAL."""
    del mock_savefig, mock_close
    rng = np.random.default_rng(0)
    t_values = np.concatenate([np.zeros(10), rng.gamma(2.0, 20.0, 190)])
    alphas = np.array([0.01, 0.02, 0.05])
    grid = mock.Mock()
    grid.alphas = alphas
    grid.t_half = np.arctanh(0.5) / alphas
    grid.q50 = float(np.median(t_values[t_values > 0]))
    grid.q95 = float(np.quantile(t_values[t_values > 0], 0.95))
    grid.c_min, grid.c_max = 0.25, 4.0
    grid.mode, grid.extensions, grid.coord_kind = "auto", 0, "t_half"
    result = mock.Mock()
    result.family = "tanh"
    result.grid = grid
    result.t_values = t_values
    result.t_ref = grid.q95
    result.t_ref_quantile = 0.95
    result.grid_edge_hit = "none"
    result.alpha_hat = 0.02
    result.alpha_hat_estimated = 0.02
    result.selected_family = None  # linear decision -> identity is FINAL
    result.decision_reason = "1se_fail"
    result.fallback_half_saturation_used = False
    result.grid_variable = "T_1/2"

    plots.plot_alpha_profile(
        alpha_grid=alphas,
        mean_loss=[0.9, 0.7, 0.8],
        per_seed_loss=[[0.9, 0.7, 0.8]],
        per_seed_alpha=[0.02],
        alpha_hat=0.02,
        one_se_set=[False, True, False],
        out_path="/tmp/alpha_profile.png",
        title="Test",
        family="tanh",
        t_normalized=t_values,
        stats_dict={"alpha_hat": 0.02},
        t_ref=grid.q95,
        grid_result=result,
    )
    fig = plt.gcf()
    titles = [ax.get_title() for ax in fig.axes]
    grid_ax = fig.axes[titles.index("Grid construction")]
    # Histogram bars and the q50 / q95 / grid-point lines were drawn.
    self.assertTrue(grid_ax.patches)
    self.assertEqual(grid_ax.get_xscale(), "log")
    legend_texts = [t.get_text() for t in grid_ax.get_legend().get_texts()]
    self.assertTrue(any("= T_ref" in t for t in legend_texts), legend_texts)
    self.assertTrue(any("grid points" in t for t in legend_texts), legend_texts)
    curve_ax = fig.axes[
        titles.index("Fitted saturation curve (normalized through (T_ref, 1))")
    ]
    lines = {ln.get_label(): ln for ln in curve_ax.get_lines()}
    labels = list(lines)
    self.assertTrue(any(lbl.startswith("FINAL: identity") for lbl in labels))
    self.assertTrue(any("NOT adopted" in lbl for lbl in labels))
    self.assertTrue(any(lbl.startswith("grid min alpha") for lbl in labels))
    # Colour scheme: red is reserved for FINAL and for the (T_ref, 1) dot, the
    # estimate that was not adopted stays dark green dashed, and the grid
    # shapes form a light-to-dark purple gradation (three distinct colours).
    final = next(ln for lbl, ln in lines.items() if lbl.startswith("FINAL"))
    self.assertEqual(final.get_color(), "red")
    self.assertEqual(final.get_linewidth(), 3)
    marker = next(ln for lbl, ln in lines.items() if lbl.startswith("(T_ref"))
    self.assertEqual(marker.get_color(), "red")
    self.assertEqual(marker.get_marker(), "o")
    rejected = next(ln for lbl, ln in lines.items() if "NOT adopted" in lbl)
    self.assertEqual(rejected.get_color(), "darkgreen")
    self.assertEqual(rejected.get_linestyle(), "--")
    shapes = [
        ln
        for lbl, ln in lines.items()
        if lbl.startswith(("grid min", "grid median", "grid max"))
    ]
    self.assertEqual(len(shapes), 3)
    self.assertEqual(len({ln.get_color() for ln in shapes}), 3)
    for ln in shapes:
      self.assertNotEqual(ln.get_color(), "red")
    footnotes = [t.get_text() for t in fig.texts if "Notes" in t.get_text()]
    # The block is wrapped to the figure width: compare without line breaks.
    footnote = " ".join(footnotes[0].split())
    self.assertIn("Grid construction", footnote)
    self.assertIn("FINAL curve used by Phases 1-4", footnote)
    self.assertIn("thick red = the FINAL curve", footnote)
    self.assertIn("red dot = (T_ref, 1)", footnote)


class ReportFilenameTest(unittest.TestCase):
  """The report is named ``<treatment>_<outcome>_<family>.html``."""

  def _name(self, treatments, outcome, family):
    # pylint: disable=protected-access
    return html_extraction._report_filename(treatments, outcome, family)
    # pylint: enable=protected-access

  def test_single_treatment_with_family(self):
    self.assertEqual(
        self._name(["tv_spend"], "sales", "power"),
        "tv_spend_sales_power.html",
    )

  def test_multiple_treatments_joined_with_hyphen(self):
    self.assertEqual(
        self._name(["tv_spend", "digital_spend"], "sales", "exp"),
        "tv_spend-digital_spend_sales_exp.html",
    )

  def test_family_segment_dropped_when_unset(self):
    self.assertEqual(
        self._name(["tv_spend"], "sales", None),
        "tv_spend_sales.html",
    )

  def test_unsafe_characters_are_sanitized(self):
    self.assertEqual(
        self._name(["tv spend/eu"], "net sales", "log"),
        "tv_spend_eu_net_sales_log.html",
    )

  def test_falls_back_when_nothing_usable(self):
    self.assertEqual(self._name([], "", None), "report.html")


class GenerateHtmlReportTest(unittest.TestCase):
  """``generate_html_report`` builds a self-contained HTML report."""

  def _write_run(self, tmp, metadata):
    """Writes ``metadata`` and the two consolidated CSVs the report needs."""
    with open(os.path.join(tmp, "metadata.json"), "w") as f:
      json.dump(metadata, f)
    step4 = os.path.join(tmp, "step4_consolidated_results")
    os.makedirs(step4, exist_ok=True)
    pd.DataFrame({
        "date": ["2023-01-01", "2023-01-08"],
        "sales": [100.0, 120.0],
        "tv_spend": [10.0, 15.0],
        "estimated_incremental_KPI_tv_spend": [20.0, 30.0],
    }).to_csv(os.path.join(step4, "consolidated_1a_full_df.csv"), index=False)
    pd.DataFrame({
        "treatment_col": ["tv_spend", "Total"],
        "model_group": ["lgbm_lgbm_5_2_5_0", ""],
        "total_input": [25.0, 25.0],
        "total_estimated_incremental_KPI": [50.0, 50.0],
        "total_estimated_roi": [2.0, 2.0],
    }).to_csv(os.path.join(step4, "consolidated_3b_summary.csv"), index=False)

  @staticmethod
  def _header_rows(html):
    """The ``<strong>label:</strong> value`` rows of the header box, in order."""
    start = html.index('class="header-bg"')
    end = html.index('class="tab-buttons"', start)
    return re.findall(
        r"<strong>([^<]+):</strong> ([^<]*?) ?(?:<br>|\n)", html[start:end]
    )

  def test_generates_report_without_ai_assistant_tab(self):
    """Generates the HTML report without any AI Assistant elements."""
    with tempfile.TemporaryDirectory() as tmp:
      self._write_run(
          tmp,
          {
              "treatment_cols": ["tv_spend"],
              "outcome_col": "sales",
              "treatment_types": {"tv_spend": "spend"},
              "x_cols_list": [["price"]],
              "date_col": "date",
          },
      )

      report_path = html_extraction.generate_html_report(tmp)
      self.assertEqual(
          os.path.basename(report_path), "tv_spend_sales.html"
      )
      with open(report_path, encoding="utf-8") as f:
        html = f.read()
      self.assertIn("tab_001_total", html)
      self.assertNotIn("tab_ai_assistant", html)
      self.assertNotIn("AI Assistant", html)
      # Header without a saturation family: no Phase 0, the treatment is
      # linear.  The two rows sit between CATE Intercept and Simple
      # Optimization.
      labels = [label for label, _ in self._header_rows(html)]
      self.assertEqual(
          labels[labels.index("CATE Intercept") :],
          ["CATE Intercept", "Selected family", "Decision",
           "Simple Optimization"],
      )
      rows = dict(self._header_rows(html))
      self.assertEqual(
          rows["Selected family"], "None (no saturation family configured)"
      )
      self.assertEqual(rows["Decision"], "linear")
      self.assertEqual(rows["Simple Optimization"], "n/a")

  def _header_rows_of_run(self, metadata):
    """Renders a run with ``metadata`` and returns its header rows as a dict."""
    with tempfile.TemporaryDirectory() as tmp:
      self._write_run(tmp, metadata)
      report_path = html_extraction.generate_html_report(tmp)
      with open(report_path, encoding="utf-8") as f:
        return dict(self._header_rows(f.read()))

  def test_header_reports_phase0_family_and_decision(self):
    """Saturation run: the configured family, the 1-SE decision per treatment."""
    base = {
        "treatment_cols": ["tv_spend"],
        "outcome_col": "sales",
        "treatment_types": {"tv_spend": "spend"},
        "x_cols_list": [["price"]],
        "date_col": "date",
        "saturation_family": "tanh",
        "simple_optimization": False,
    }
    # One treatment: the bare values.  `Selected family` is the family the
    # user configured; the family Phase 0 adopted is part of the decision.
    rows = self._header_rows_of_run({
        **base,
        "saturation_selected_family": {"tv_spend": "tanh"},
        "saturation_phase0": {
            "tv_spend": {
                "decision": "saturation",
                "selected_family": "tanh",
                "alpha_hat": 0.02,
            }
        },
    })
    self.assertEqual(rows["Selected family"], "tanh")
    self.assertEqual(rows["Decision"], "saturation (tanh)")
    # Older metadata without the user's selection: the final decision only.
    self.assertEqual(
        rows["Simple Optimization"],
        "user selection: n/a (older metadata) | final decision: False",
    )
    # Several treatments: one entry per treatment, in pipeline order; a
    # treatment kept linear by the 1-SE test shows `linear`.
    rows = self._header_rows_of_run({
        **base,
        "treatment_cols": ["tv_spend", "digital_spend"],
        "treatment_types": {"tv_spend": "spend", "digital_spend": "spend"},
        "saturation_selected_family": {
            "tv_spend": "tanh",
            "digital_spend": None,
        },
        "saturation_phase0": {
            "tv_spend": {"decision": "saturation", "selected_family": "tanh"},
            "digital_spend": {"decision": "linear", "selected_family": None},
        },
    })
    self.assertEqual(rows["Selected family"], "tanh")
    self.assertEqual(
        rows["Decision"], "tv_spend: saturation (tanh), digital_spend: linear"
    )
    # A family configured but no Phase 0 record (older metadata): the
    # configured family is still known, the decision is not.
    rows = self._header_rows_of_run(base)
    self.assertEqual(rows["Selected family"], "tanh")
    self.assertEqual(rows["Decision"], "n/a")

  def test_header_simple_optimization_shows_selection_and_fallback(self):
    """User selection and final decision: derived, forced, linear fallback."""
    base = {
        "treatment_cols": ["tv_spend"],
        "outcome_col": "sales",
        "treatment_types": {"tv_spend": "spend"},
        "x_cols_list": [["price"]],
        "date_col": "date",
        "saturation_family": "tanh",
    }
    # Not set (derived False) and no fallback: the simple optimizer did not
    # run.
    rows = self._header_rows_of_run({
        **base,
        "simple_optimization": False,
        "simple_optimization_requested": None,
        "simple_optimization_derived": False,
        "simple_optimization_forced": False,
    })
    self.assertEqual(
        rows["Simple Optimization"],
        "user selection: not set (default False: a saturation family is"
        " configured) | final decision: False",
    )
    # Explicit False, Phase 0 selected linear: the simple optimizer ran, so
    # the final decision is True and the reason is named.
    rows = self._header_rows_of_run({
        **base,
        "simple_optimization": False,
        "simple_optimization_requested": False,
        "simple_optimization_derived": False,
        "simple_optimization_forced": False,
        "saturation_phase0": {
            "tv_spend": {"decision": "linear", "selected_family": None}
        },
        "phase4_optimization": {
            "tv_spend": {"phase4_fallback_to_linear": True}
        },
    })
    self.assertEqual(
        rows["Simple Optimization"],
        "user selection: False | final decision: True (linear fallback:"
        " Phase 0 selected linear for tv_spend)",
    )
    # Two treatments, one fell back: the decision is listed per treatment.
    rows = self._header_rows_of_run({
        **base,
        "treatment_cols": ["tv_spend", "radio_spend"],
        "treatment_types": {"tv_spend": "spend", "radio_spend": "spend"},
        "simple_optimization": False,
        "simple_optimization_requested": False,
        "simple_optimization_derived": False,
        "simple_optimization_forced": False,
        "phase4_optimization": {
            "tv_spend": {"phase4_fallback_to_linear": False},
            "radio_spend": {"phase4_fallback_to_linear": True},
        },
    })
    self.assertEqual(
        rows["Simple Optimization"],
        "user selection: False | final decision: tv_spend: False,"
        " radio_spend: True (linear fallback: Phase 0 selected linear for"
        " radio_spend)",
    )
    # A treatment without a Phase 4 record (its optimization was skipped)
    # keeps the value in force and is still listed, in `treatment_cols`
    # order.
    rows = self._header_rows_of_run({
        **base,
        "treatment_cols": ["tv_spend", "radio_spend"],
        "treatment_types": {"tv_spend": "spend", "radio_spend": "spend"},
        "simple_optimization": False,
        "simple_optimization_requested": False,
        "simple_optimization_derived": False,
        "simple_optimization_forced": False,
        "phase4_optimization": {
            "tv_spend": {"phase4_fallback_to_linear": True}
        },
    })
    self.assertEqual(
        rows["Simple Optimization"],
        "user selection: False | final decision: tv_spend: True,"
        " radio_spend: False (linear fallback: Phase 0 selected linear for"
        " tv_spend)",
    )
    # Explicit False forced to True by a non-spend treatment.
    rows = self._header_rows_of_run({
        **base,
        "treatment_types": {"tv_spend": "impressions"},
        "simple_optimization": True,
        "simple_optimization_requested": False,
        "simple_optimization_derived": False,
        "simple_optimization_forced": True,
    })
    self.assertEqual(
        rows["Simple Optimization"],
        "user selection: False | final decision: True (forced: non-spend"
        " treatment)",
    )
    # Explicit True.
    rows = self._header_rows_of_run({
        **base,
        "simple_optimization": True,
        "simple_optimization_requested": True,
        "simple_optimization_derived": True,
        "simple_optimization_forced": False,
    })
    self.assertEqual(
        rows["Simple Optimization"],
        "user selection: True | final decision: True",
    )
    # Not set without a saturation family: derived True.
    rows = self._header_rows_of_run({
        **base,
        "saturation_family": None,
        "simple_optimization": True,
        "simple_optimization_requested": None,
        "simple_optimization_derived": True,
        "simple_optimization_forced": False,
    })
    self.assertEqual(
        rows["Simple Optimization"],
        "user selection: not set (default True: no saturation family)"
        " | final decision: True",
    )

  def test_header_names_the_price_benchmark_and_investment(self):
    """A price treatment shows its benchmark and how its spend column arose."""
    base = {
        "treatment_cols": ["tv_spend"],
        "outcome_col": "sales",
        "treatment_types": {"tv_spend": "price"},
        "treatment_spend_cols": {"tv_spend": "tv_spend_investment"},
        "x_cols_list": [["price"]],
        "date_col": "date",
    }
    record = {
        "benchmark": {
            "method": "rolling_max",
            "percentile": None,
            "window": "26W",
            "column": None,
            "value": None,
            "description": "rolling_max(window=26W)",
        },
        "benchmark_resolution": "explicit",
        "cost_source": "auto",
        "cost_column": "tv_spend_investment",
        "cost_input_column": "sales",
        "cost_formula": "sales / tv_price * max(tv_spend, 0)",
    }
    # rolling_max with its window; the investment computed from the
    # benchmark because no treatment_spend_cols entry was given.
    rows = self._header_rows_of_run(
        {**base, "price_benchmarks": {"tv_spend": record}}
    )
    self.assertEqual(
        rows["Treatment Types"],
        "tv_spend: price (benchmark: rolling_max, window: 26W)",
    )
    self.assertEqual(
        rows["Treatment Spend Col(s)"],
        "tv_spend: tv_spend_investment (not specified; computed from the"
        " benchmark: sales / tv_price * max(tv_spend, 0))",
    )
    # An automatic q_100 and a caller's cost column.
    rows = self._header_rows_of_run({
        **base,
        "price_benchmarks": {
            "tv_spend": {
                **record,
                "benchmark": {
                    "method": "quantile",
                    "percentile": 100,
                    "description": "q_100",
                },
                "benchmark_resolution": "auto",
                "cost_source": "user",
                "cost_input_column": "Cost",
                "cost_formula": "Cost where tv_spend > 0, else 0",
            }
        },
    })
    self.assertEqual(
        rows["Treatment Types"],
        "tv_spend: price (benchmark: q_100, automatic)",
    )
    self.assertEqual(
        rows["Treatment Spend Col(s)"],
        "tv_spend: tv_spend_investment (from treatment_spend_cols 'Cost':"
        " Cost where tv_spend > 0, else 0)",
    )
    # Older metadata without the record, and non-price treatments: unchanged.
    rows = self._header_rows_of_run(base)
    self.assertEqual(rows["Treatment Types"], "tv_spend: price")
    self.assertEqual(
        rows["Treatment Spend Col(s)"], "tv_spend: tv_spend_investment"
    )
    # The remaining benchmark kinds: a column, a constant and `as_is`.
    # pylint: disable=protected-access
    for benchmark, expected in (
        ({"method": "column", "column": "list_price"},
         " (benchmark: column 'list_price')"),
        ({"method": "constant", "value": 100.0},
         " (benchmark: constant 100)"),
        ({"method": "as_is"},
         " (benchmark: as_is, the column is a gap already)"),
    ):
      self.assertEqual(
          html_extraction._price_type_suffix(
              "price",
              {**record, "benchmark": benchmark, "benchmark_resolution": "x"},
          ),
          expected,
      )
    self.assertEqual(html_extraction._price_type_suffix("spend", record), "")
    # No investment column (`cost_source` = none): nothing to append.
    self.assertEqual(
        html_extraction._price_investment_suffix(
            {**record, "cost_source": "none", "cost_column": None}
        ),
        "",
    )

  def test_cate_images_share_one_scale(self):
    """The narrower CATE png gets a proportional width, the wider 100%."""
    with tempfile.TemporaryDirectory() as tmp:
      self._write_run(
          tmp,
          {
              "treatment_cols": ["tv_spend"],
              "outcome_col": "sales",
              "treatment_types": {"tv_spend": "spend"},
              "x_cols_list": [["price"]],
              "date_col": "date",
          },
      )
      cate_dir = os.path.join(
          tmp, "step4_consolidated_results", "graph", "004_cate"
      )
      os.makedirs(cate_dir, exist_ok=True)
      for name, width in (
          ("cate_scatter_matrix.png", 400),
          ("cate_response_curves.png", 300),
      ):
        fig = plt.figure(figsize=(width / 100.0, 1.0), dpi=100)
        fig.savefig(os.path.join(cate_dir, name), dpi=100)
        plt.close(fig)
      report_path = html_extraction.generate_html_report(tmp)
      with open(report_path, encoding="utf-8") as f:
        html = f.read()
    start = html.index('id="tab_004_cate"')
    end = html.index("</div>", start)
    styles = re.findall(
        r'<img class="chart-img" style="([^"]+)"', html[start:end]
    )
    self.assertEqual(styles, ["width: 100.00%", "width: 75.00%"])

  def test_png_width_and_cate_image_widths(self):
    """The PNG width comes from the header; missing files fall back."""
    # pylint: disable=protected-access
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "a.png")
      fig = plt.figure(figsize=(2.5, 1.0), dpi=100)
      fig.savefig(path, dpi=100)
      plt.close(fig)
      self.assertEqual(html_extraction._png_width(path), 250)
      missing = os.path.join(tmp, "missing.png")
      self.assertIsNone(html_extraction._png_width(missing))
      # A file that is not a PNG (wrong signature) or is truncated: no width.
      not_png = os.path.join(tmp, "not.png")
      with open(not_png, "wb") as handle:
        handle.write(b"GIF89a" + bytes(30))
      self.assertIsNone(html_extraction._png_width(not_png))
      truncated = os.path.join(tmp, "short.png")
      with open(truncated, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
      self.assertIsNone(html_extraction._png_width(truncated))
      self.assertEqual(
          html_extraction._cate_image_widths([path, missing]),
          ["width: 100.00%", "width: 100%"],
      )
      self.assertEqual(
          html_extraction._cate_image_widths([missing, missing]),
          ["width: 100%", "width: 100%"],
      )


if __name__ == "__main__":
  unittest.main()
