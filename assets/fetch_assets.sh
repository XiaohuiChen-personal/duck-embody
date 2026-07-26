#!/usr/bin/env bash
# Mirror every scene asset (and everything it references) into assets/.
#
# WHY a local mirror: the benchmark batch must never depend on a live S3 bucket.
# A mid-batch network hiccup or an upstream asset revision would silently break
# cross-model comparability (design doc 03 §5, AGENTS.md §4).
#
# WHAT it does, per root asset in asset_list.tsv:
#   1. download the root .usd
#   2. ask pxr which external files it references (scripts/usd_deps.py)
#   3. download those, recursively for referenced .usd files
#   4. record sha256 for everything into checksums.txt
#
# This is dependency-driven rather than prefix-driven on purpose: pulling whole
# S3 prefixes would drag in ~700 MB of textures for models we never place
# (Assets/.../Appliances/Oven/ alone is 243 MB across two ovens; we use one).
#
# IDEMPOTENT: files already present are skipped. Delete a file to re-fetch it.
# The USD binaries are gitignored; checksums.txt IS committed (AGENTS.md rule 7).
#
# Usage:  bash assets/fetch_assets.sh [--verify]
#           --verify   re-check existing files against checksums.txt and exit

set -euo pipefail

BUCKET="https://omniverse-content-production.s3-us-west-2.amazonaws.com"
ASSETS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$ASSETS_DIR")"
LIST="$ASSETS_DIR/asset_list.tsv"
CHECKSUMS="$ASSETS_DIR/checksums.txt"

# pxr is importable from neither default interpreter — use the pinned packman
# USD runtime (verified 2026-07-26: reports USD (0, 24, 5)).
USDROOT="$(ls -d "$HOME"/.cache/packman/chk/usd.py311.manylinux_2_35_aarch64.stock.release/* 2>/dev/null | head -1)"
if [[ -z "$USDROOT" ]]; then
  echo "FATAL: packman USD runtime not found. Has Isaac Sim been run at least once?" >&2
  exit 1
fi
usd_deps() {
  PYTHONPATH="$USDROOT/lib/python" \
  LD_LIBRARY_PATH="$USDROOT/lib:${LD_LIBRARY_PATH:-}" \
    "$HOME/IsaacLab/_isaac_sim/python.sh" "$REPO_ROOT/scripts/usd_deps.py" "$1" 2>/dev/null
}

# --- kit core MDL modules are NOT asset files -------------------------------
# ArchVis USDs reference shaders like `OmniPBR.mdl` by bare name. Those resolve
# from the *renderer's* MDL search path inside Isaac Sim, not from the asset's
# own directory — attempts to download one 404 (verified: OmniPBR.mdl is absent
# from every plausible bucket prefix but present at
# <kit-kernel>/mdl/core/Base/OmniPBR.mdl). Skipping them is correct, not a
# workaround; mirroring them would be mirroring part of Isaac Sim.
#
# The decision is DOWNLOAD-DRIVEN, not name-driven, and the distinction matters:
# `SimPBR.mdl` exists BOTH in the kit core tree and as a real bucket object
# (Assets/simready_content/materials/SimPBR.mdl, HTTP 200). A name-based skip
# would drop it, while a stale copy from an earlier run stayed in checksums.txt
# — so `--verify` on a clean clone would fail against a file the script never
# fetches. Rule: try the download; only a 404 on a name present in the kit core
# tree is treated as renderer-resolved. Anything else that fails is fatal.
KIT_MDL_DIR="$(ls -d "$HOME"/.cache/packman/chk/kit-kernel/*/mdl 2>/dev/null | head -1)"
declare -A CORE_MDL=()
if [[ -n "$KIT_MDL_DIR" ]]; then
  while IFS= read -r m; do CORE_MDL["$m"]=1; done < <(find "$KIT_MDL_DIR" -name '*.mdl' -printf '%f\n' | sort -u)
  echo "Kit core MDL modules known: ${#CORE_MDL[@]} (skipped only if the bucket 404s)"
else
  echo "WARNING: kit MDL core dir not found; core-MDL references may 404." >&2
fi

if [[ "${1:-}" == "--verify" ]]; then
  echo "Verifying against $CHECKSUMS ..."
  cd "$ASSETS_DIR" && sha256sum -c checksums.txt
  exit $?
fi

# ---------------------------------------------------------------------------
# fetch <s3_key> -> assets/<s3_key>, then recurse into its USD references.
# ---------------------------------------------------------------------------
declare -A FETCHED=()

fetch() {
  local key="$1"
  [[ -n "${FETCHED[$key]:-}" ]] && return 0
  FETCHED[$key]=1

  local dest="$ASSETS_DIR/$key"
  local base="${key##*/}"
  mkdir -p "$(dirname "$dest")"

  if [[ -f "$dest" ]]; then
    echo "  skip   $key"
  else
    # --fail so a 404 is an error instead of a saved error page masquerading as
    # an asset (which would surface much later as a confusing USD parse error).
    local code
    code="$(curl -sS -o "$dest.part" -w '%{http_code}' "$BUCKET/$key" || true)"
    if [[ "$code" == "200" ]]; then
      echo "  get    $key"
      mv "$dest.part" "$dest"
    else
      rm -f "$dest.part"
      if [[ "$code" == "404" && "$base" == *.mdl && -n "${CORE_MDL[$base]:-}" ]]; then
        # Renderer-resolved core shader module — not a file in this bucket.
        echo "  core   $base  (renderer-resolved, not mirrored)"
        return 0
      fi
      echo "FATAL: download failed (HTTP $code): $BUCKET/$key" >&2
      exit 1
    fi
  fi

  # Only USD files have references worth chasing.
  case "$key" in
    *.usd|*.usda|*.usdc)
      local dir rel resolved
      dir="$(dirname "$key")"
      while IFS= read -r rel; do
        [[ -z "$rel" ]] && continue
        # Resolve the reference against the referrer's directory, then normalise
        # away any ../ so it becomes a clean S3 key.
        resolved="$(python3 -c 'import posixpath,sys; print(posixpath.normpath(posixpath.join(sys.argv[1], sys.argv[2])))' "$dir" "$rel")"
        fetch "$resolved"
      done < <(usd_deps "$dest")
      ;;
  esac
}

echo "Mirroring assets into $ASSETS_DIR"
echo "Source bucket: $BUCKET"
echo

while IFS=$'\t' read -r name key collision; do
  [[ -z "${name// }" || "${name:0:1}" == "#" ]] && continue
  echo "[$name]  (expected collision: $collision)"
  fetch "$key"
done < "$LIST"

echo
echo "Writing $CHECKSUMS"
cd "$ASSETS_DIR"
# Deterministic order so the committed file diffs cleanly between runs.
find . -type f \( -name '*.usd' -o -name '*.usda' -o -name '*.usdc' \
                  -o -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \
                  -o -name '*.mdl' -o -name '*.json' \) \
  | LC_ALL=C sort | xargs sha256sum > checksums.txt

echo "Done: $(wc -l < checksums.txt) files, $(du -sh "$ASSETS_DIR" | cut -f1) on disk"
