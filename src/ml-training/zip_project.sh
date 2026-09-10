#!/usr/bin/env bash
# zip_project.sh
# ==============
# Packages the OceanXRay project + all downloaded raw data into a single
# zip ready to hand to the next Claude session.
#
# Usage:
#   bash zip_project.sh              # creates oceanxray_with_real_data.zip
#   bash zip_project.sh myname.zip   # custom output name
#
# What is included:
#   - All source code (src/, *.py, *.txt, *.md, *.sh)
#   - All downloaded raw data (data/raw/**/*.nc)
#   - Processed intermediates if present (data/processed/)
#   - Results if present (results/)
#   - Saved models if present (models/)
#
# What is excluded:
#   - __pycache__ and *.pyc files
#   - .git directory
#   - .DS_Store

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${1:-oceanxray_with_real_data.zip}"

# Make output path absolute relative to CWD where script is called
if [[ "${OUTPUT}" != /* ]]; then
    OUTPUT="$(pwd)/${OUTPUT}"
fi

echo "========================================================"
echo "  OceanXRay — Project Packaging"
echo "========================================================"
echo "  Project root : ${SCRIPT_DIR}"
echo "  Output zip   : ${OUTPUT}"
echo ""

# Check that raw data directories are not all empty
echo "[check] Raw data directories:"
for d in sst ssh currents winds glorys; do
    dir="${SCRIPT_DIR}/data/raw/${d}"
    if [ -d "${dir}" ]; then
        count=$(find "${dir}" -name "*.nc" 2>/dev/null | wc -l | tr -d ' ')
        echo "        data/raw/${d}/  -->  ${count} .nc file(s)"
    else
        echo "        data/raw/${d}/  -->  DIRECTORY NOT FOUND (run download_real_data.py first)"
    fi
done

echo ""
echo "[zip] Creating archive..."

cd "${SCRIPT_DIR}"

zip -r "${OUTPUT}" . \
    --exclude "*.git*" \
    --exclude "*__pycache__*" \
    --exclude "*.pyc" \
    --exclude ".DS_Store" \
    --exclude "*.zip"

SIZE_MB=$(du -m "${OUTPUT}" | cut -f1)
echo ""
echo "========================================================"
echo "  Done!"
echo "  Archive : ${OUTPUT}"
echo "  Size    : ~${SIZE_MB} MB"
echo "========================================================"
echo ""
echo "Next steps:"
echo "  1. Upload ${OUTPUT} to your Claude session."
echo "  2. Tell Claude: 'Here is the project with real data downloaded."
echo "     Run the Day 1-3 pipeline on the real data.'"
