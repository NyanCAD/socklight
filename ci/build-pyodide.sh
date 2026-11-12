#!/usr/bin/env bash
# Build the Pyodide (WebAssembly) wheel into wheelhouse/ and run the tests in
# Pyodide, as VACASK's Pyodide job: cibuildwheel, platform pyodide, cp313
# (pyemscripten_2025_0, Emscripten 4.0.9).
#
#   ci/build-pyodide.sh [VACASK_PYODIDE_WHEEL]
#
# Environment:
#   VACASK_PYODIDE_WHEEL  VACASK's Pyodide wheel (not on PyPI; CI artifact
#                         "pyodide-wheels"); the tests need it. Without it, the
#                         wheel is built but not tested.
#   OPENVAF_COMPILER      openvaf-r with the wasm32-unknown-emscripten target
#                         (default: on PATH).
#   EMSDK                 an Emscripten SDK whose upstream/bin/wasm-ld links the
#                         va/*.va side modules (default: cibuildwheel's own).
#   CIBUILDWHEEL          the cibuildwheel command (default: uvx cibuildwheel==3.2.1).
set -euo pipefail

cd "$(dirname "$0")/.."

VACASK_PYODIDE_WHEEL="${1:-${VACASK_PYODIDE_WHEEL:-}}"
export OPENVAF_COMPILER="${OPENVAF_COMPILER:-openvaf-r}"
export CIBW_PLATFORM=pyodide
export CIBW_BUILD="${CIBW_BUILD:-cp313-pyodide_wasm32}"
if [[ -n "$VACASK_PYODIDE_WHEEL" ]]; then
  export CIBW_TEST_REQUIRES_PYODIDE="pytest $(realpath "$VACASK_PYODIDE_WHEEL")"
else
  echo "build-pyodide.sh: VACASK_PYODIDE_WHEEL not set, building without testing" >&2
  export CIBW_TEST_SKIP="*"
fi

read -r -a cibuildwheel <<< "${CIBUILDWHEEL:-uvx cibuildwheel==3.2.1}"
"${cibuildwheel[@]}" --output-dir wheelhouse .
