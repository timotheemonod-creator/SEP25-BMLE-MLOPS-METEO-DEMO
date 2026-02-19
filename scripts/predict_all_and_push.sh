#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/env.local.sh"
cd "${PROJECT_ROOT}"


API_URL="${API_URL:-http://meteo-api:8000}"
AUTH_HEADER="Authorization: Bearer ${API_KEY}"

stations=(
  "Adelaide" "Albany" "Albury-Wodonga" "Alice Springs" "Ballarat" "Bendigo" "Brisbane"
  "Broome" "Cairns" "Canberra" "Casey" "Christmas Island" "Cocos Islands" "Darwin"
  "Davis" "Devonport" "Gold Coast" "Hobart" "Kalgoorlie-Boulder" "Launceston"
  "Lord Howe Island" "Macquarie Island" "Mawson" "Melbourne" "Mount Gambier"
  "Norfolk Island" "Penrith" "Perth" "Port Lincoln" "Renmark" "Sydney"
  "Tennant Creek" "Townsville" "Tuggeranong" "Wollongong" "Wynyard"
)

echo "Calling /predict for ${#stations[@]} stations..."

for s in "${stations[@]}"; do
  enc=$(python -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))' "$s")
  curl -s -X POST \
    "${API_URL}/predict?use_latest=true&station_name=${enc}" \
    -H "accept: application/json" \
    -H "$AUTH_HEADER" \
    >/dev/null
done

echo "Updating DVC + Git..."

if command -v dvc >/dev/null 2>&1; then
  dvc add outputs/preds_api.csv
  git add outputs/preds_api.csv.dvc
  git commit -m "Update API prediction log ($(date -u +%F))" || true
  git push || true
  dvc push || true
else
  echo "dvc not installed in this runtime, skipping dvc/git push steps."
fi

echo "Done."
