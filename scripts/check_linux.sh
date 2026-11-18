#!/usr/bin/env bash
# Coverage build + tests + lcov gate, meant for a plain Linux box or container:
#
#   docker run --rm -v "$PWD:/src" -w /src python:3.12 bash scripts/check_linux.sh
#
# Steps: configure with TINYINFER_COVERAGE=ON (-O0 -g --coverage), run ctest, collect line
# coverage with lcov, keep only src/, and fail if it drops below MIN_COVERAGE (percent).
# If pip is around it also builds the Python module and runs pytest.
#
# Env knobs: BUILD_DIR (default /tmp/build-cov), MIN_COVERAGE, SKIP_PYTHON=1, HTML=1 (genhtml).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-/tmp/build-cov}
# 90.6% measured with AVX-512 + VNNI (the AVX-512 / VNNI kernel files can't run on older CPUs,
# which costs roughly 5 points), so 85 passes everywhere and still catches real regressions
MIN_COVERAGE=${MIN_COVERAGE:-85}
cd "$ROOT"

need() { command -v "$1" >/dev/null 2>&1; }

# tools: try apt (container runs as root), else hope they are already installed
if ! need lcov || ! need ninja || ! need cmake; then
    if need apt-get && [ "$(id -u)" = 0 ]; then
        apt-get update -qq >/dev/null
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq lcov ninja-build cmake >/dev/null
    fi
fi
for t in cmake ninja lcov gcov; do
    need "$t" || { echo "missing tool: $t" >&2; exit 2; }
done

echo "== configure (coverage)"
rm -rf "$BUILD_DIR"
cmake -S . -B "$BUILD_DIR" -G Ninja -DTINYINFER_COVERAGE=ON -DTINYINFER_BUILD_TESTS=ON >/dev/null
cmake --build "$BUILD_DIR" 2>&1 | tail -1

echo "== ctest"
# zero the counters first so only this run is measured
lcov -q --zerocounters --directory "$BUILD_DIR" 2>/dev/null || true
ctest --test-dir "$BUILD_DIR" --output-on-failure -j"$(nproc)" | tail -4

echo "== lcov"
# newer lcov (2.x) is strict about gcc's slightly odd line data; older ones lack some flags
IGN=()
if lcov --version 2>&1 | grep -q "LCOV version 2"; then
    # (an error type listed twice also silences its warning, which is pure gcc / lcov noise here)
    IGN=(--ignore-errors mismatch,negative,inconsistent,inconsistent,unused,unused,count,count,empty)
fi
RAW="$BUILD_DIR/coverage.all.info"
OUT="$BUILD_DIR/coverage.info"
lcov -q --capture --directory "$BUILD_DIR" --output-file "$RAW" "${IGN[@]}"
# drop system headers, googletest (_deps), and our own tests / tools / bench, then keep src/ only
lcov -q --remove "$RAW" '/usr/*' '*/_deps/*' '*/tests/*' '*/tools/*' '*/bench/*' '*/python/*' \
    --output-file "$RAW.f" "${IGN[@]}"
lcov -q --extract "$RAW.f" "$ROOT/src/*" --output-file "$OUT" "${IGN[@]}"

# per-file table, then the number the gate uses
lcov --list "$OUT" "${IGN[@]}" 2>/dev/null | sed -e 's#'"$ROOT"'/##'
SUMMARY=$(lcov --summary "$OUT" "${IGN[@]}" 2>&1)
echo "$SUMMARY" | grep -E '(lines|functions)\.*:'
PCT=$(echo "$SUMMARY" | awk '/lines\.*:/ {gsub("%","",$2); print $2; exit}')
echo "src/ line coverage: ${PCT}% (minimum ${MIN_COVERAGE}%)"

if [ "${HTML:-0}" = 1 ] && need genhtml; then
    genhtml -q "$OUT" --output-directory "$BUILD_DIR/html" "${IGN[@]}" >/dev/null
    echo "html report: $BUILD_DIR/html/index.html"
fi

if ! awk -v p="$PCT" -v m="$MIN_COVERAGE" 'BEGIN { exit !(p + 0 >= m + 0) }'; then
    echo "FAIL: coverage ${PCT}% is below ${MIN_COVERAGE}%" >&2
    exit 1
fi

if [ "${SKIP_PYTHON:-0}" != 1 ] && need pip; then
    echo "== pytest"
    # the wheel is built separately (normal Release flags); only the C++ tests are instrumented
    pip install -q pytest numpy . --config-settings=build-dir=/tmp/py-build 2>&1 | tail -2
    python -m pytest -q python/tests
fi

echo "OK"
