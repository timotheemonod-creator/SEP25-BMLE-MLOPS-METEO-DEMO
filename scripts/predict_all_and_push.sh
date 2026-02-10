#!/usr/bin/env bash
set -euo pipefail

source scripts/env.local.sh

API_URL="${API_URL:-http://127.0.0.1:8000}"
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
    -H "Content-Type: application/json" \
    -d '{}' >/dev/null
done

echo "Updating DVC + Git..."
dvc add outputs/preds_api.csv
git add outputs/preds_api.csv.dvc
git commit -m "Update API prediction log ($(date -u +%F))" || true
git push
dvc push

echo "Done."
