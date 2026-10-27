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

"""Consistency checks between the Markdown documentation and the code.

Guards ``README.md``, ``docs/api_reference.md`` and the other ``docs/*.md``
pages against drifting away from the package:

* every public symbol of the documented modules (module-level functions,
  classes, ``ALL_CAPS`` constants and the public methods of public classes)
  has an entry in ``docs/api_reference.md``;
* every signature heading of ``docs/api_reference.md`` (``#### `name(...)```)
  matches the live ``inspect.signature``: parameter names and order, defaults,
  ``*`` / ``**`` markers;
* the Python examples of the Markdown pages parse, the keyword arguments they
  pass to ``CausalWorkflowOrchestrator``, ``CausalDataScaler``,
  ``run_full_pipeline`` and ``create_lagged_features``
  exist, the README quick start passes every required argument, and the
  ``saturation_config`` examples pass the orchestrator's strict validation and
  show the real defaults;
* relative links and ``#anchors`` between the Markdown files resolve;
* the error and warning messages quoted in ``docs/faq.md`` exist in the source;
* obsolete wording (``Phase 5``, ``report.html`` as the report name) does not
  come back;
* the example notebooks linked from the README open with a title and an
  "Open in Colab" badge, and the links of their Markdown cells (GitHub URLs
  into this repository, raw data URLs) point at files and headings that exist.

The test only reads files; it is cheap and runs with the unit tests.
"""

import ast
import importlib
import inspect
import json
import os
import pathlib
import re
import unittest

from doubleml_pipeline import optimization
from doubleml_pipeline import workflow
from doubleml_pipeline.evaluation import alpha_selection
from doubleml_pipeline.models import saturation
from doubleml_pipeline.preprocessing import features
from doubleml_pipeline.preprocessing import scalers
from doubleml_pipeline.visualization import html_extraction


def _root_candidates():
  """Directories that may hold README.md and docs/.

  The repository root (``test/`` sits next to ``README.md`` on GitHub) is
  tried first, then the package directory itself (the Bazel runfiles place the
  Markdown data next to the modules) and the ``src/`` layout's parent.
  Symlinks are deliberately not resolved: the runfiles tree is made of
  symlinks into a content-addressed store.

  Returns:
    The candidate directories, most specific first.
  """
  package_dir = pathlib.Path(os.path.abspath(workflow.__file__)).parent
  return (
      pathlib.Path(os.path.abspath(__file__)).parent.parent,
      package_dir,
      package_dir.parent.parent,
  )


def _find_root():
  """The first candidate that holds README.md and docs/api_reference.md."""
  for candidate in _root_candidates():
    if (candidate / "README.md").is_file() and (
        candidate / "docs" / "api_reference.md"
    ).is_file():
      return candidate
  return None


_ROOT = _find_root()
_README = (_ROOT or pathlib.Path()) / "README.md"
_DOCS = (_ROOT or pathlib.Path()) / "docs"
_API_REFERENCE = _DOCS / "api_reference.md"


def setUpModule():  # pylint: disable=invalid-name
  """Skips the module when the Markdown sources are not available."""
  if _ROOT is None:
    raise unittest.SkipTest(
        "README.md and docs/ are not available (installed package?); looked in "
        + ", ".join(str(c) for c in _root_candidates())
    )


# Modules whose public API must be covered by docs/api_reference.md.
_DOCUMENTED_MODULES = (
    "doubleml_pipeline.workflow",
    "doubleml_pipeline.optimization",
    "doubleml_pipeline.preprocessing.scalers",
    "doubleml_pipeline.preprocessing.features",
    "doubleml_pipeline.evaluation.runner",
    "doubleml_pipeline.evaluation.hparam_search",
    "doubleml_pipeline.evaluation.selection",
    "doubleml_pipeline.evaluation.sensitivity",
    "doubleml_pipeline.evaluation.metrics",
    "doubleml_pipeline.evaluation.alpha_selection",
    "doubleml_pipeline.evaluation.saturation_evaluation",
    "doubleml_pipeline.models.flaml_dml",
    "doubleml_pipeline.models.fixed_hparam_regressor",
    "doubleml_pipeline.models.saturation",
    "doubleml_pipeline.utils.io",
    "doubleml_pipeline.visualization.plots",
    "doubleml_pipeline.visualization.html_extraction",
)

# Public names that are intentionally left out of the API reference.
_UNDOCUMENTED_ALLOWLIST = frozenset()

# Messages quoted in docs/faq.md and the module that emits them.
_FAQ_MESSAGES = (
    (
        "doubleml_pipeline.workflow",
        "Optimization with multiple treatments is not yet supported.",
    ),
    ("doubleml_pipeline.workflow", "The input DataFrame index has"),
    ("doubleml_pipeline.workflow", "df.reset_index(drop=True)"),
    (
        "doubleml_pipeline.workflow",
        "optimize=True with simple_optimization=False requires a saturation",
    ),
    ("doubleml_pipeline.workflow", "contains values outside [0.0, 1.0]"),
    (
        "doubleml_pipeline.workflow",
        (
            "simple_optimization is forced to True because the saturation-aware"
            " optimizer is only implemented for treatment_type='spend'"
        ),
    ),
    (
        "doubleml_pipeline.workflow",
        "has values in [0.0, 1.0] and name containing discount/pct/rate",
    ),
    (
        "doubleml_pipeline.workflow",
        "Skipping ROI output and optimization for treatment",
    ),
    (
        "doubleml_pipeline.evaluation.runner",
        (
            "Sample size is small (n < 1000). Set degree of continuous"
            " covariates to 2 or less."
        ),
    ),
    (
        "doubleml_pipeline.evaluation.alpha_selection",
        "which is incompatible with a saturation family",
    ),
)

# Files in which the literal `report.html` must not appear (the report is
# named `<treatment>_<outcome>[_<family>].html`; api_reference.md documents the
# empty-name fallback).
_NO_REPORT_HTML = (
    "README.md",
    "docs/advanced.md",
    "docs/concepts.md",
    "docs/faq.md",
    "docs/saturation.md",
    "docs/guide.md",
    "docs/output_csv_schema.md",
)

_SIGNATURE_HEADING = re.compile(
    r"^#### `(?:async )?(\w+)\((.*)\)\s*(->.*)?`\s*$"
)
_MODULE_HEADING = re.compile(r"^## Module: `([\w.]+)`")
_CLASS_HEADING = re.compile(r"^### Class: `(\w+)`")
_CODE_FENCE = re.compile(r"^(\s*)```(\w+)?\s*$")
_LINK = re.compile(r"\]\(([^)\s]+)\)")
_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$")

# The public repository the notebooks link to (absolute URLs work both on
# GitHub and in Colab, where relative links do not).
_GITHUB_REPO = "https://github.com/google/doubleml-pipeline"
_GITHUB_BLOB = _GITHUB_REPO + "/blob/main/"
_GITHUB_RAW = "https://raw.githubusercontent.com/google/doubleml-pipeline/main/"
_COLAB_PREFIX = (
    "https://colab.research.google.com/github/google/doubleml-pipeline/blob/main/"
)
_RAW_URL = re.compile(re.escape(_GITHUB_RAW) + r"([^\s)`'\"]+)")

_UNKNOWN = object()


def _read(path):
  return pathlib.Path(path).read_text(encoding="utf-8")


def _markdown_files():
  return [_README] + sorted(_DOCS.glob("*.md"))


def _python_blocks(path):
  """Yields (first_line_number, source) of every ```python block."""
  lines = _read(path).splitlines()
  in_block, lang, indent, start, buf = False, None, "", 0, []
  for number, line in enumerate(lines, start=1):
    match = _CODE_FENCE.match(line)
    if not in_block:
      if match and match.group(2):
        in_block, lang, indent, start, buf = (
            True,
            match.group(2).lower(),
            match.group(1),
            number + 1,
            [],
        )
    elif match and match.group(2) is None and match.group(1) == indent:
      if lang in ("python", "py"):
        yield start, "\n".join(
            l[len(indent) :] if l.startswith(indent) else l for l in buf
        )
      in_block = False
    else:
      buf.append(line)


def _call_name(node):
  """`Name` or `Attribute` callee of an `ast.Call`, else None."""
  func = node.func
  if isinstance(func, ast.Name):
    return func.id
  if isinstance(func, ast.Attribute):
    return func.attr
  return None


def _calls(source, names):
  """Yields `ast.Call` nodes whose callee name is in `names`."""
  for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.Call) and _call_name(node) in names:
      yield node


def _slug(heading):
  """GitHub-style anchor of a Markdown heading."""
  text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # link text only
  text = re.sub(r"[`*]", "", text).strip().lower()
  text = re.sub(r"[^\w\- ]", "", text)
  return re.sub(r"\s", "-", text)


def _anchors(path):
  """Set of heading anchors of a Markdown file (code fences excluded)."""
  anchors, counts, in_fence = set(), {}, False
  for line in _read(path).splitlines():
    if line.startswith("```"):
      in_fence = not in_fence
      continue
    match = None if in_fence else _HEADING.match(line)
    if match:
      slug = _slug(match.group(1))
      seen = counts.get(slug, 0)
      counts[slug] = seen + 1
      anchors.add(slug if seen == 0 else f"{slug}-{seen}")
  return anchors


def _string_constants(module):
  """Every string literal of a module's source (f-string parts included)."""
  tree = ast.parse(_read(module.__file__))
  return [
      node.value
      for node in ast.walk(tree)
      if isinstance(node, ast.Constant) and isinstance(node.value, str)
  ]


def _public_names(module):
  """Public module-level symbols defined in (or exported by) `module`."""
  names = set(getattr(module, "__all__", ()))
  for name, obj in vars(module).items():
    if name.startswith("_") or inspect.ismodule(obj):
      continue
    if inspect.isclass(obj) or inspect.isfunction(obj):
      if getattr(obj, "__module__", None) == module.__name__:
        names.add(name)
    elif name.isupper():
      names.add(name)
  return names


def _public_methods(cls):
  """Public methods / properties defined on `cls` itself."""
  return {
      name
      for name, obj in vars(cls).items()
      if not name.startswith("_")
      and (
          inspect.isfunction(obj)
          or isinstance(obj, (property, classmethod, staticmethod))
      )
  }


def _mentioned(name, text):
  return re.search(r"`" + re.escape(name) + r"\b", text) is not None


def _safe_getattr(obj, name):
  """`getattr(obj, name, None)` that tolerates lazily imported extras."""
  try:
    return getattr(obj, name, None)
  except ImportError:
    return None


def _eval_default(node, namespace):
  """Value of a documented default, or `_UNKNOWN` when it is not evaluable."""
  if isinstance(node, ast.Constant):
    return node.value
  try:
    return eval(  # pylint: disable=eval-used
        compile(ast.Expression(body=node), "<api_reference.md>", "eval"),
        dict(namespace),
    )
  except Exception:  # pylint: disable=broad-except
    return _UNKNOWN


def _same(a, b):
  try:
    result = a == b
  except Exception:  # pylint: disable=broad-except
    return False
  if hasattr(result, "all"):
    return bool(result.all())
  return bool(result)


def _documented_signatures():
  """Yields (line, module_name, class_name, name, ast.arguments, text)."""
  module_name, class_name = None, None
  for number, line in enumerate(_read(_API_REFERENCE).splitlines(), start=1):
    module_match = _MODULE_HEADING.match(line)
    if module_match:
      module_name, class_name = module_match.group(1), None
      continue
    if line.startswith("### "):
      class_match = _CLASS_HEADING.match(line)
      class_name = class_match.group(1) if class_match else None
      continue
    match = _SIGNATURE_HEADING.match(line)
    if not match or module_name is None:
      continue
    name, params = match.group(1), match.group(2)
    try:
      tree = ast.parse(f"def {name}({params}): pass")
    except SyntaxError:
      yield number, module_name, class_name, name, None, line
      continue
    yield number, module_name, class_name, name, tree.body[0].args, line


class ApiReferenceTest(unittest.TestCase):
  """docs/api_reference.md versus the live modules."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.text = _read(_API_REFERENCE)

  def test_public_api_is_documented(self):
    """Every public symbol and public method has an entry in the reference."""
    missing = []
    for module_name in _DOCUMENTED_MODULES:
      try:
        module = importlib.import_module(module_name)
      except ImportError:  # optional dependency
        continue
      for name in sorted(_public_names(module)):
        if name in _UNDOCUMENTED_ALLOWLIST:
          continue
        if not _mentioned(name, self.text):
          missing.append(f"{module_name}.{name}")
        obj = module.__dict__.get(name)
        if inspect.isclass(obj) and obj.__module__ == module_name:
          for method in sorted(_public_methods(obj)):
            qualified = f"{module_name}.{name}.{method}"
            if qualified in _UNDOCUMENTED_ALLOWLIST:
              continue
            if not _mentioned(method, self.text):
              missing.append(qualified)
    self.assertFalse(
        missing,
        "Public symbols without an entry in docs/api_reference.md:\n  "
        + "\n  ".join(missing),
    )

  def test_signature_headings_match_the_code(self):
    """Parameter names, order, markers and defaults of every heading."""
    problems, unparseable, checked = [], [], 0
    for (
        line,
        module_name,
        class_name,
        name,
        args,
        text,
    ) in _documented_signatures():
      if args is None:
        unparseable.append(f"L{line}: {text}")
        continue
      try:
        module = importlib.import_module(module_name)
      except ImportError:
        continue
      target, namespace = None, dict(vars(module))
      if class_name is not None:
        owner = _safe_getattr(module, class_name)
        if owner is None:
          continue  # class unavailable here (optional dependency)
        namespace.update(vars(owner))
        target = owner if name == class_name else _safe_getattr(owner, name)
        if target is None:
          problems.append(
              f"L{line}: `{name}` is not a member of {module_name}.{class_name}"
          )
          continue
      else:
        target = _safe_getattr(module, name)
        if target is None and hasattr(module, "__path__"):
          # A package section may document functions of its submodules.
          for sub in list(vars(module).values()):
            if inspect.ismodule(sub) and sub.__name__.startswith(
                module.__name__ + "."
            ):
              target = _safe_getattr(sub, name)
              if target is not None:
                namespace = dict(vars(sub))
                break
        if target is None:
          if name in getattr(module, "__all__", ()):
            continue  # lazily exported optional dependency
          problems.append(f"L{line}: `{name}` is not a member of {module_name}")
          continue
      if isinstance(target, property):
        continue
      try:
        actual = inspect.signature(target)
      except (TypeError, ValueError):
        continue
      checked += 1
      doc_params = [(a.arg, "pos") for a in args.posonlyargs + args.args]
      if args.vararg:
        doc_params.append((args.vararg.arg, "var"))
      doc_params += [(a.arg, "kw") for a in args.kwonlyargs]
      if args.kwarg:
        doc_params.append((args.kwarg.arg, "varkw"))
      kinds = {
          inspect.Parameter.POSITIONAL_ONLY: "pos",
          inspect.Parameter.POSITIONAL_OR_KEYWORD: "pos",
          inspect.Parameter.VAR_POSITIONAL: "var",
          inspect.Parameter.KEYWORD_ONLY: "kw",
          inspect.Parameter.VAR_KEYWORD: "varkw",
      }
      actual_params = [
          (p.name, kinds[p.kind]) for p in actual.parameters.values()
      ]
      if doc_params != actual_params:
        problems.append(
            f"L{line}: `{name}` parameters differ\n      doc:  "
            f"{[n for n, _ in doc_params]}\n      code: "
            f"{[n for n, _ in actual_params]}"
        )
        continue
      positional = args.posonlyargs + args.args
      defaults = dict(
          zip(
              [a.arg for a in positional][
                  len(positional) - len(args.defaults) :
              ],
              args.defaults,
          )
      )
      defaults.update({
          a.arg: d
          for a, d in zip(args.kwonlyargs, args.kw_defaults)
          if d is not None
      })
      for param in actual.parameters.values():
        has_doc_default = param.name in defaults
        has_code_default = param.default is not inspect.Parameter.empty
        if has_doc_default != has_code_default:
          problems.append(
              f"L{line}: `{name}({param.name}=...)` default "
              f"{'missing in the doc' if has_code_default else 'absent in the code'}"
          )
          continue
        if not has_code_default:
          continue
        documented = _eval_default(defaults[param.name], namespace)
        if documented is _UNKNOWN:
          continue
        if not _same(documented, param.default):
          problems.append(
              f"L{line}: `{name}({param.name}=...)` default "
              f"{documented!r} in the doc, {param.default!r} in the code"
          )
    self.assertGreater(
        checked, 100, "the heading parser found too few signatures"
    )
    self.assertFalse(
        unparseable,
        "Signature headings that are not valid Python:\n  "
        + "\n  ".join(unparseable),
    )
    self.assertFalse(
        problems, "Signature mismatches:\n  " + "\n  ".join(problems)
    )


class MarkdownExamplesTest(unittest.TestCase):
  """The Python examples of README.md and docs/*.md."""

  _CALLEES = {
      "CausalWorkflowOrchestrator": workflow.CausalWorkflowOrchestrator,
      "CausalDataScaler": scalers.CausalDataScaler,
      "run_full_pipeline": (
          workflow.CausalWorkflowOrchestrator.run_full_pipeline
      ),
      "create_lagged_features": features.create_lagged_features,
  }

  def _blocks(self):
    for path in _markdown_files():
      for line, source in _python_blocks(path):
        yield path, line, source

  def test_examples_parse_and_use_existing_keyword_arguments(self):
    """Every example parses and passes only keyword arguments that exist."""
    problems = []
    for path, line, source in self._blocks():
      where = f"{path.relative_to(_ROOT)}:{line}"
      try:
        ast.parse(source)
      except SyntaxError as err:
        problems.append(f"{where}: {err}")
        continue
      for call in _calls(source, self._CALLEES):
        allowed = set(
            inspect.signature(self._CALLEES[_call_name(call)]).parameters
        )
        for keyword in call.keywords:
          if keyword.arg is not None and keyword.arg not in allowed:
            problems.append(
                f"{where}: {_call_name(call)}({keyword.arg}=...) is not a"
                " parameter"
            )
    self.assertFalse(problems, "\n".join(problems))

  def test_readme_quick_start_passes_every_required_argument(self):
    """The README orchestrator examples name every required parameter."""
    required = {
        name
        for name, p in inspect.signature(
            workflow.CausalWorkflowOrchestrator
        ).parameters.items()
        if p.default is inspect.Parameter.empty
    }
    calls = []
    for _, source in _python_blocks(_README):
      calls.extend(_calls(source, {"CausalWorkflowOrchestrator"}))
    self.assertTrue(
        calls, "README.md has no CausalWorkflowOrchestrator example"
    )
    for call in calls:
      passed = {k.arg for k in call.keywords}
      self.assertLessEqual(
          required,
          passed,
          f"README example misses {sorted(required - passed)}",
      )

  def _saturation_configs(self):
    """Yields (where, dict) for every literal `saturation_config=` example."""
    for path, line, source in self._blocks():
      for call in _calls(source, {"CausalWorkflowOrchestrator"}):
        for keyword in call.keywords:
          if keyword.arg == "saturation_config":
            try:
              yield f"{path.relative_to(_ROOT)}:{line}", ast.literal_eval(
                  keyword.value
              )
            except ValueError:
              pass  # not a literal (e.g. a named constant)

  def test_saturation_config_examples_are_valid(self):
    configs = list(self._saturation_configs())
    self.assertTrue(configs, "no literal saturation_config example found")
    for where, config in configs:
      with self.subTest(where=where):
        workflow._validate_config(dict(config))  # pylint: disable=protected-access

  def test_full_key_example_shows_the_real_defaults(self):
    """The docs/saturation.md 'all defaults' example equals the code defaults."""
    full = [
        config
        for where, config in self._saturation_configs()
        if where.startswith("docs/saturation.md")
        and "learners" in config.get("alpha_selection", {})
    ]
    self.assertEqual(len(full), 1, "expected exactly one full-key example")
    config = full[0]
    self.assertEqual(
        config["alpha_selection"]["learners"],
        saturation.default_learner_config(),
    )
    self.assertEqual(config["saturation"]["family"], saturation.DEFAULT_FAMILY)
    self.assertEqual(
        config["saturation"]["t_ref_quantile"],
        saturation.DEFAULT_T_REF_QUANTILE,
    )
    self.assertEqual(
        config["alpha_selection"]["flatness_ratio"],
        alpha_selection.DEFAULT_FLATNESS_RATIO,
    )
    self.assertEqual(
        config["alpha_selection"]["t_ref_tolerance"],
        alpha_selection.DEFAULT_T_REF_TOLERANCE,
    )
    self.assertEqual(
        config["optimization"]["cap_quantile"],
        optimization.DEFAULT_CAP_QUANTILE,
    )
    self.assertIn(
        config["optimization"]["reallocate_within"],
        optimization.REALLOCATE_WITHIN,
    )
    self.assertIn(
        config["optimization"]["budget_mode"], optimization.BUDGET_MODES
    )


class CrossReferenceTest(unittest.TestCase):
  """Links, anchors, quoted messages and obsolete wording."""

  def test_relative_links_and_anchors_resolve(self):
    """Relative link targets exist and `#anchors` match a heading."""
    # `examples/` is a separate build package and is not shipped with the
    # test data in every environment; links into it are checked only when the
    # directory is present (always the case in a source checkout).
    skipped_roots = tuple(d for d in ("examples",) if not (_ROOT / d).exists())
    problems = []
    for path in _markdown_files():
      for target in _LINK.findall(_read(path)):
        if target.startswith(("http://", "https://", "mailto:")):
          continue
        file_part, _, anchor = target.partition("#")
        if file_part.lstrip("./").startswith(skipped_roots):
          continue
        resolved = (
            path
            if not file_part
            else pathlib.Path(os.path.normpath(path.parent / file_part))
        )
        if not resolved.exists():
          problems.append(f"{path.relative_to(_ROOT)}: {target} does not exist")
          continue
        if (
            anchor
            and resolved.suffix == ".md"
            and anchor not in _anchors(resolved)
        ):
          problems.append(
              f"{path.relative_to(_ROOT)}: no heading for #{anchor} "
              f"in {resolved.name}"
          )
    self.assertFalse(problems, "\n".join(problems))

  def test_faq_quotes_messages_that_exist_in_the_code(self):
    for module_name, fragment in _FAQ_MESSAGES:
      with self.subTest(fragment=fragment):
        constants = _string_constants(importlib.import_module(module_name))
        self.assertTrue(
            any(fragment in value for value in constants),
            f"{fragment!r} not found in {module_name}",
        )
        self.assertIn(fragment.split(" (")[0][:40], _read(_DOCS / "faq.md"))

  def test_report_file_name_example(self):
    name = html_extraction._report_filename(  # pylint: disable=protected-access
        ["discount_rate"], "sales_revenue", None
    )
    self.assertEqual(name, "discount_rate_sales_revenue.html")
    self.assertIn(name, _read(_README))
    self.assertIn("<treatment>_<outcome>[_<family>].html", _read(_README))

  def test_obsolete_wording_does_not_return(self):
    for path in _markdown_files():
      with self.subTest(file=path.name):
        self.assertNotIn("Phase 5", _read(path))
    for relative in _NO_REPORT_HTML:
      with self.subTest(file=relative):
        self.assertNotIn("report.html", _read(_ROOT / relative))

  def test_readme_links_every_docs_page(self):
    readme = _read(_README)
    for page in sorted(_DOCS.glob("*.md")):
      with self.subTest(page=page.name):
        self.assertIn(f"docs/{page.name}", readme)


def _readme_notebooks():
  """Repository-relative paths of the notebooks linked from README.md."""
  return sorted({
      target
      for target in _LINK.findall(_read(_README))
      if target.endswith(".ipynb")
      and not target.startswith(("http://", "https://"))
  })


def _load_notebook(path):
  with open(path, encoding="utf-8") as handle:
    return json.load(handle)


def _markdown_cells(notebook):
  return [
      "".join(cell.get("source", []))
      for cell in notebook.get("cells", [])
      if cell.get("cell_type") == "markdown"
  ]


class ExampleNotebooksTest(unittest.TestCase):
  """The example notebooks: Colab badges and links into the repository.

  `examples/` is a separate build package that is not shipped with the test
  data in every environment; the checks that read the notebooks are skipped
  when the files are absent (they always run in a source checkout).
  """

  def test_readme_has_a_colab_badge_per_notebook(self):
    notebooks = _readme_notebooks()
    self.assertTrue(notebooks, "README.md links no example notebook")
    readme = _read(_README)
    for relative in notebooks:
      with self.subTest(notebook=relative):
        self.assertIn(_COLAB_PREFIX + relative, readme)

  def _available_notebooks(self):
    available = [p for p in _readme_notebooks() if (_ROOT / p).is_file()]
    if not available:
      self.skipTest("examples/ is not available in this environment")
    return available

  def test_notebooks_start_with_a_title_and_a_colab_badge(self):
    for relative in self._available_notebooks():
      with self.subTest(notebook=relative):
        notebook = _load_notebook(_ROOT / relative)
        self.assertEqual(notebook.get("nbformat"), 4)
        first = notebook["cells"][0]
        self.assertEqual(first["cell_type"], "markdown")
        source = "".join(first["source"])
        self.assertTrue(source.startswith("# "), "first cell is not a title")
        self.assertIn(_COLAB_PREFIX + relative, source)

  def test_notebook_links_resolve(self):
    """GitHub, raw-data and relative links of the Markdown cells exist."""
    problems = []
    for relative in self._available_notebooks():
      text = "\n".join(_markdown_cells(_load_notebook(_ROOT / relative)))
      targets = set(_LINK.findall(text))
      targets.update(_GITHUB_RAW + m for m in _RAW_URL.findall(text))
      for target in sorted(targets):
        anchor = ""
        if target.startswith(_COLAB_PREFIX):
          path = target[len(_COLAB_PREFIX) :]
        elif target.startswith(_GITHUB_BLOB):
          path, _, anchor = target[len(_GITHUB_BLOB) :].partition("#")
        elif target.startswith(_GITHUB_RAW):
          path = target[len(_GITHUB_RAW) :]
        elif target.startswith(_GITHUB_REPO + "#"):
          path, anchor = "README.md", target.partition("#")[2]
        elif target.startswith(("http://", "https://", "mailto:")):
          continue  # external site
        else:  # relative to the notebook's directory
          path, _, anchor = target.partition("#")
          path = os.path.normpath(
              os.path.join(os.path.dirname(relative), path)
          )
        resolved = _ROOT / path
        if not resolved.exists():
          problems.append(f"{relative}: {target} does not exist")
        elif (
            anchor
            and resolved.suffix == ".md"
            and anchor not in _anchors(resolved)
        ):
          problems.append(
              f"{relative}: no heading for #{anchor} in {resolved.name}"
          )
    self.assertFalse(problems, "\n".join(problems))


if __name__ == "__main__":
  unittest.main()
