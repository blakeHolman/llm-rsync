#!/usr/bin/env bash
# rsync_diff_sweep.sh  OLD NEW OUT.csv [BLOCK_SIZES...]
# Uses your rsync_diff.py to sweep many block sizes and save CSV.

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 OLD NEW OUT.csv [BLOCK_SIZES...]" >&2; exit 1
fi

OLD="$1"; NEW="$2"; OUTCSV="$3"; shift 3

[[ -f "rsync_diff.py" ]] || { echo "ERROR: rsync_diff.py not found"; exit 1; }
[[ -e "$OLD" ]] || { echo "ERROR: OLD not found: $OLD"; exit 1; }
[[ -e "$NEW" ]] || { echo "ERROR: NEW not found: $NEW"; exit 1; }

# Default block sizes (dense small + broader large; includes 300)
if [[ $# -eq 0 ]]; then
  BLOCKS=(
    64 80 96 112 128 144 160 176 192 208 224 240 256 272 288 300 304 320 336 352 368 384
    400 416 432 448 464 480 496 512
    544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 1024
    1152 1280 1408 1536 1664 1792 1920 2048 2304 2560 2816 3072 3328 3584 3840 4096
    4608 5120 5632 6144 6656 7168 7680 8192
    12288 16384 24576 32768 49152 65536 98304 131072 196608 262144 393216 524288 786432 1048576
  )
else
  BLOCKS=("$@")
fi

echo "[i] Sweeping ${#BLOCKS[@]} block sizes"
echo "block_size,total_new_bytes,matched_bytes,literal_bytes,residual_percent" > "$OUTCSV"
printf "%-8s | %-13s | %-13s | %-13s | %-9s\n" "BLOCK" "TOTAL(new)" "MATCHED" "LITERAL" "RESIDUAL"
printf "%-8s-+-%-13s-+-%-13s-+-%-13s-+-%-9s\n" "--------" "-------------" "-------------" "-------------" "---------"

for B in "${BLOCKS[@]}"; do
  echo "[i] block=$B …"
  OUT="$(python3 rsync_diff.py "$OLD" "$NEW" --block "$B")" || {
    echo "[warn] rsync_diff.py failed for block=$B" >&2
    continue
  }

  # Expected lines:
  # Total (new): N
  # Matched    : M
  # Literals   : L
  # Residual % : P (block=B)

  TOTAL=$(echo "$OUT" | awk -F': ' '/^Total \(new\)/{gsub(/,/,"",$2); print $2}')
  MATCHED=$(echo "$OUT" | awk -F': ' '/^Matched/{gsub(/,/,"",$2); print $2}')
  LITERALS=$(echo "$OUT" | awk -F': ' '/^Literals/{gsub(/,/,"",$2); print $2}')
  RESID=$(echo "$OUT" | awk -F': ' '/^Residual %/{sub(/ .*/,"",$2); print $2}')

  if [[ -z "$TOTAL" || -z "$MATCHED" || -z "$LITERALS" || -z "$RESID" ]]; then
    echo "[warn] parse failed for block=$B; raw output:" >&2
    echo "$OUT" >&2
    continue
  fi

  printf "%-8s | %-13s | %-13s | %-13s | %-9s\n" "$B" "$TOTAL" "$MATCHED" "$LITERALS" "${RESID}%"
  echo "$B,$TOTAL,$MATCHED,$LITERALS,$RESID" >> "$OUTCSV"
done

echo "[✓] Saved CSV → $OUTCSV"

