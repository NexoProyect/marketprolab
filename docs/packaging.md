# How the library is put together, how to install it, and how to publish it

Two things: **how a folder of `.py` files becomes an installable library**, and
**how to publish it** (or install it locally without publishing anything).

---

## Contents

1. [Project layout](#1-project-layout)
2. [What actually makes it a package](#2-what-actually-makes-it-a-package)
3. [`pyproject.toml`, line by line](#3-pyprojecttoml-line-by-line)
4. [Local installation](#4-local-installation)
5. [Building the package (wheel + sdist)](#5-building-the-package-wheel--sdist)
6. [Installing from the built file](#6-installing-from-the-built-file)
7. [Installing from GitHub, no PyPI](#7-installing-from-github-no-pypi)
8. [Publishing to TestPyPI first](#8-publishing-to-testpypi-first)
9. [Publishing to the real PyPI](#9-publishing-to-the-real-pypi)
10. [Publishing automatically with GitHub Actions](#10-publishing-automatically-with-github-actions)
11. [Versioning and updates](#11-versioning-and-updates)
12. [Pre-publish checklist](#12-pre-publish-checklist)
13. [Typical problems](#13-typical-problems)

---

## 1. Project layout

```
Backest All Strats/                 <- repository root
├── marketprolab/                   <- THE PACKAGE (what gets installed)
│   ├── __init__.py                 <- makes it a package and defines the public API
│   ├── py.typed                    <- marks the package as typed (PEP 561)
│   ├── enums.py
│   ├── sessions.py
│   ├── symbol.py                   <- SymbolSpec
│   ├── broker_profile.py           <- BrokerProfile, SymbolRegistry
│   ├── execution.py                <- spread / slippage / latency models
│   ├── orders.py
│   ├── broker.py                   <- the simulator
│   ├── data.py
│   ├── strategy.py
│   ├── indicators.py
│   ├── engine.py                   <- Backtest, BacktestResult
│   ├── metrics.py
│   ├── plotting.py
│   ├── report.py                   <- standalone HTML reports
│   ├── optimize.py
│   ├── montecarlo.py
│   └── presets.py
├── docs/
│   ├── usage.md
│   └── packaging.md                <- this file
├── examples/
│   ├── 01_basic_backtest.py
│   ├── 02_mt5_and_tick_data.py
│   ├── 03_optimization_and_walkforward.py
│   ├── 04_monte_carlo.py
│   ├── 05_any_instrument.py
│   └── 06_html_reports.py
├── tests/
│   └── test_marketprolab.py
├── pyproject.toml                  <- the packaging recipe
├── MANIFEST.in                     <- extra files that go into the sdist
├── README.md
├── LICENSE
└── .gitignore
```

Rule of thumb: **the package folder's name is the import name**
(`import marketprolab`), and `name` in `pyproject.toml` is the install name
(`pip install marketprolab`). Here they match, which is the sane choice, but it
is not mandatory.

---

## 2. What actually makes it a package

Only two things:

1. **A folder with an `__init__.py`.** That file is what Python executes on
   `import marketprolab`. In it we re-export the important classes so users can
   write `from marketprolab import Backtest` instead of
   `from marketprolab.engine import Backtest`.

2. **A `pyproject.toml` at the root.** This is the standard file (PEP 517/621)
   that tells `pip` how to build and install the project: name, version,
   dependencies and which folders to include.

That is all. `setup.py` has not been necessary for years.

About `py.typed`: an empty file telling editors and `mypy` that your type hints
are reliable and should be used. It is shipped via
`[tool.setuptools.package-data]`.

---

## 3. `pyproject.toml`, line by line

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]   # what pip needs in order to BUILD
build-backend = "setuptools.build_meta"  # who does the building
```

This is the first thing `pip` reads: it installs `setuptools` into an isolated
environment and asks it to build the package.

```toml
[project]
name = "marketprolab"      # the PyPI name: it must be AVAILABLE
version = "0.1.0"          # semantic version
description = "..."        # one line, shown in PyPI listings
readme = "README.md"       # rendered as the project page
requires-python = ">=3.9"
license = { text = "MIT" }
authors = [{ name = "Johangel" }]
keywords = ["backtesting", "metatrader5", ...]
classifiers = [...]        # standard tags: https://pypi.org/classifiers/
```

```toml
dependencies = [
    "numpy>=1.23",
    "pandas>=2.0",
    "matplotlib>=3.6",
]
```

Only what is **essential**. Everything listed here is force-installed on the
machine of anyone using your library, so be conservative.

```toml
[project.optional-dependencies]
mt5 = ["MetaTrader5>=5.0.40; platform_system=='Windows'"]
parquet = ["pyarrow>=12"]
dev = ["pytest>=7", "pytest-cov", "ruff", "build", "twine"]
all = [...]
```

**Extras** are installed with brackets: `pip install marketprolab[mt5]`.
Note the `; platform_system=='Windows'` marker: the `MetaTrader5` package only
exists on Windows, and without that marker installation would fail on Linux and
macOS. This is also why the library core never imports MT5 at module level -
only inside the functions that need it.

```toml
[tool.setuptools]
packages = ["marketprolab"]      # which folders get packaged

[tool.setuptools.package-data]
marketprolab = ["py.typed"]      # non-.py files to include
```

If you ever add subpackages (`marketprolab/reports/`, etc.), switch to automatic
discovery:

```toml
[tool.setuptools.packages.find]
include = ["marketprolab*"]
```

And `MANIFEST.in` controls what goes into the **sdist** (the `.tar.gz` source
archive): README, LICENSE, docs and examples yes; tests and data no.

---

## 4. Local installation

### a) Editable mode - what you want while developing

```bash
cd "c:\Users\Johangel\Desktop\VsCodeProjects\Backest All Strats"
pip install -e .
```

`-e` (editable) does **not** copy anything: it links to your folder. Edit a
`.py` and the change is live on the next `import`, no reinstall. This is the
right way to work on the library and use it from other projects at the same
time.

With extras:

```bash
pip install -e ".[dev]"     # + pytest, ruff, build, twine
pip install -e ".[all]"     # + MetaTrader5 and pyarrow
```

The quotes are needed in PowerShell and zsh.

### b) Normal installation (a copy)

```bash
pip install .
```

Copies the package into `site-packages`. Later edits to your folder are **not**
reflected; you have to reinstall.

### c) A virtual environment (strongly recommended)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -e ".[dev]"
```

Keeps the system Python clean and avoids version conflicts.

### d) Without installing anything

To just try it, run from the project root or put the folder on the path:

```python
import sys
sys.path.insert(0, r"c:\Users\Johangel\Desktop\VsCodeProjects\Backest All Strats")
import marketprolab
```

Or in the terminal: `set PYTHONPATH=.` (Windows) before launching the script.

### e) Verify it worked

```bash
pip show marketprolab
python -c "import marketprolab; print(marketprolab.__version__, marketprolab.__file__)"
pytest -q
```

### f) Uninstall

```bash
pip uninstall marketprolab
```

---

## 5. Building the package (wheel + sdist)

```bash
pip install --upgrade build
python -m build
```

This produces two files in `dist/`:

| File | What it is |
|---|---|
| `marketprolab-0.1.0-py3-none-any.whl` | **wheel**: prebuilt package, fast install. This is what `pip` uses almost always. |
| `marketprolab-0.1.0.tar.gz` | **sdist**: source code. A fallback for platforms without a wheel. |

`py3-none-any` means: any Python 3, any operating system, no compiled
extensions. That is the ideal case - your library is pure Python.

Check the contents before uploading anything:

```bash
pip install --upgrade twine
twine check dist/*
python -m zipfile -l dist/marketprolab-0.1.0-py3-none-any.whl   # see what is inside
```

Always delete `dist/` before rebuilding, or you will upload stale versions:

```bash
rmdir /s /q dist build            # Windows cmd
Remove-Item -Recurse -Force dist, build   # PowerShell
rm -rf dist build                 # bash
```

---

## 6. Installing from the built file

Useful to test the real package, or to hand it to someone without publishing:

```bash
pip install dist/marketprolab-0.1.0-py3-none-any.whl
```

This is exactly what `pip` will do when downloading from PyPI, so if it works in
a clean environment, publishing will work too.

---

## 7. Installing from GitHub, no PyPI

Very often you do not need PyPI at all:

```bash
# Latest commit on the default branch
pip install git+https://github.com/NexoProyect/marketprolab.git

# A specific tag
pip install git+https://github.com/NexoProyect/marketprolab.git@v0.1.0

# A private repository (it will ask for credentials, or use a token)
pip install git+https://TOKEN@github.com/NexoProyect/marketprolab.git
```

In a `requirements.txt`:

```
marketprolab @ git+https://github.com/NexoProyect/marketprolab.git@v0.1.0
```

---

## 8. Publishing to TestPyPI first

**Always do this.** TestPyPI is a copy of PyPI for rehearsals and, unlike the
real thing, **it forgives mistakes**: on PyPI a published version can never be
replaced.

1. Create an account at <https://test.pypi.org/account/register/>.
2. Enable two-factor authentication (mandatory).
3. Create a token under *Account settings -> API tokens*. It starts with `pypi-`.
4. Upload:

```bash
twine upload --repository testpypi dist/*
```

Username: `__token__` (literally). Password: the full token.

5. Test the install in a clean environment:

```bash
python -m venv /tmp/trial && /tmp/trial/Scripts/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ marketprolab
python -c "import marketprolab; print(marketprolab.__version__)"
```

The `--extra-index-url` is required because numpy, pandas and matplotlib are not
on TestPyPI and must come from the real one.

To avoid retyping the token, create `~/.pypirc`
(`C:\Users\Johangel\.pypirc` on Windows):

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...your-pypi-token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgENdGVzdC5weXBpLm9yZw...your-testpypi-token
```

That file holds credentials: never commit it.

---

## 9. Publishing to the real PyPI

1. Check the name is free: <https://pypi.org/project/marketprolab/>. If it is
   taken, change `name` in `pyproject.toml` (the import name can stay different).
2. Account at <https://pypi.org/account/register/> + 2FA + API token.
3. Upload:

```bash
python -m build
twine check dist/*
twine upload dist/*
```

4. Confirm by installing from PyPI:

```bash
pip install marketprolab
```

**Three things you cannot undo:**

- An uploaded version cannot be overwritten. If you get it wrong, publish `0.1.1`.
- You can *yank* a version so pip stops choosing it by default, but the file
  stays reachable.
- The project name stays reserved to you for as long as it exists.

---

## 10. Publishing automatically with GitHub Actions

The most comfortable setup long term: publish when you push a tag. With Trusted
Publishing there is no token to store in GitHub either.

`.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  push:
    tags: ["v*"]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[dev]"
      - run: pytest -q

  publish:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      id-token: write          # required for Trusted Publishing
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

Then, on PyPI -> your project -> *Publishing*, add the trusted publisher (your
GitHub user/repo and the workflow name). After that:

```bash
git tag v0.1.1
git push origin v0.1.1
```

and it publishes itself.

---

## 11. Versioning and updates

Semantic versioning `MAJOR.MINOR.PATCH`:

- **PATCH** (`0.1.0 -> 0.1.1`): you fixed a bug, nothing changes for users.
- **MINOR** (`0.1.0 -> 0.2.0`): you added functionality without breaking anything.
- **MAJOR** (`0.9.0 -> 1.0.0`): you changed the API and broke other people's code.

To publish a new version:

1. Bump the number in `pyproject.toml` **and** in `marketprolab/__init__.py`
   (`__version__`). They must match.
2. Note the changes in `CHANGELOG.md`.
3. `pytest -q` green.
4. Delete `dist/`, `python -m build`, `twine check dist/*`, `twine upload dist/*`.
5. Tag it in git: `git tag v0.2.0 && git push origin v0.2.0`.

If keeping the version in two places annoys you, read it from a single source:

```toml
[project]
dynamic = ["version"]

[tool.setuptools.dynamic]
version = { attr = "marketprolab.__version__" }
```

---

## 12. Pre-publish checklist

- [ ] `pytest -q` passes in full.
- [ ] `ruff check .` is clean.
- [ ] The version is bumped in `pyproject.toml` and `__init__.py`, and they match.
- [ ] `README.md` renders correctly (`twine check dist/*` validates it).
- [ ] `LICENSE` is present and matches the `license` field.
- [ ] Dependencies are minimal with realistic bounds.
- [ ] `MetaTrader5` is only in extras, with the platform marker.
- [ ] Nothing sensitive in the package: inspect the `.whl` contents.
- [ ] No heavy data files (the tick CSV does NOT belong inside).
- [ ] Tested from TestPyPI in a clean environment.
- [ ] Git tag created.

---

## 13. Typical problems

| Error | Cause and fix |
|---|---|
| `File already exists` on upload | That version is already on PyPI. Bump the version; overwriting is impossible. |
| `Invalid or non-existent authentication` | The username must be literally `__token__` and the password the full token, `pypi-` prefix included. |
| `The name 'x' is too similar to an existing project` | PyPI blocks names close to existing ones. Pick another `name`. |
| Installs fine but `import` fails | Missing `__init__.py`, or `packages` in `pyproject.toml` does not include the folder. |
| Files missing from the sdist | Add them in `MANIFEST.in`. |
| Install fails on Linux because of `MetaTrader5` | The `; platform_system=='Windows'` marker is missing, or you put it in `dependencies` instead of extras. |
| `pip install -e .` does not reflect changes | You are importing another installed copy. `pip uninstall marketprolab` and reinstall editable. |
| `__pycache__` sneaks into the package | Add `global-exclude __pycache__ *.py[cod]` to `MANIFEST.in`. |
| The README shows as plain text on PyPI | `readme = "README.md"` is missing, or the file is not valid markdown. |
| `ModuleNotFoundError: No module named 'marketprolab'` when running examples | You have not installed it: `pip install -e .`, or run from the root with `PYTHONPATH=.`. |

---

## Five commands, summarised

```bash
pip install -e ".[dev]"      # develop
pytest -q                    # verify
python -m build              # build
twine upload --repository testpypi dist/*    # rehearse the release
twine upload dist/*          # publish
```
