#!/usr/bin/env bash
# Build the Pyodide (WebAssembly) wheel into wheelhouse/ and run the tests in
# Pyodide: cibuildwheel, platform pyodide, cp314 (Pyodide 314, the one marimo
# ships; pyemscripten_2026_0, Emscripten 5.0.3). VACASK and the wasm-capable
# openvaf-r come from PyPI (vacask, vacask-bin).
#
#   ci/build-pyodide.sh
#
# Environment:
#   CIBUILDWHEEL  the cibuildwheel command (default: uvx cibuildwheel==4.3.5).
set -euo pipefail

cd "$(dirname "$0")/.."

export CIBW_PLATFORM=pyodide
read -r -a cibuildwheel <<< "${CIBUILDWHEEL:-uvx cibuildwheel==4.3.5}"
"${cibuildwheel[@]}" --output-dir wheelhouse .
