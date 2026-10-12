#!/usr/bin/env bash
# Frozen from https://github.com/llvm/circt/issues/10568.
# Contract: exit 1 for the original assertion, 0 when fixed, and 2 for an
# unrelated failure. The pinned benchmark image was verified to return 0;
# isolation_benchmark.py records the exact script hash and expected result.
set -u

work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
input="$work_dir/issue-10568.mlir"
stderr_log="$work_dir/stderr.log"

cat > "$input" <<'MLIR'
hw.module @__pdr_transition(in %state_reg0 : i2, out constraint : i1, out bad : i1, out next_reg0 : i2) {
  %c0_i2 = hw.constant 0 : i2
  %0 = comb.icmp eq %state_reg0, %c0_i2 : i2
  %true = hw.constant true
  %1 = comb.xor %0, %true : i1
  %2 = comb.mul bin %state_reg0, %state_reg0 : i2
  hw.output %true, %1, %2 : i1, i1, i2
}
MLIR

if circt-opt "$input" -convert-comb-to-synth >/dev/null 2>"$stderr_log"; then
    echo "issue 10568 no longer reproduces"
    exit 0
fi
if grep -Fq "Assertion \`addends.size() > 2' failed" "$stderr_log"; then
    echo "issue 10568 reproduced: CompressorTree addends assertion" >&2
    exit 1
fi
cat "$stderr_log" >&2
echo "issue 10568 command failed without the expected assertion" >&2
exit 2
