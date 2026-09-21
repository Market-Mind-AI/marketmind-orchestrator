#!/usr/bin/env bash
#
# Runs the step-8 parity check, and says what is missing when it cannot.
#
# Reads the configuration that already exists in the two .env files rather than
# asking anyone to copy values around: a token that has to be pasted in two
# places is a token that will be pasted wrong in one of them.
#
#   ./scripts/run_parity_check.sh "air fryer" ESP PRT BRA
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCORING_REPO="${SCORING_REPO:-$HERE/../marketmind-core}"

read_env() {  # read_env <file> <key>
  [ -f "$1" ] || return 1
  grep -E "^$2=" "$1" | tail -1 | cut -d= -f2- | sed 's/^"//;s/"$//'
}

: "${SCORING_API_URL:=http://localhost:8004}"
: "${ORCHESTRATOR_URL:=http://localhost:8081}"
: "${SCORING_INTERNAL_TOKEN:=$(read_env "$HERE/.env" SCORING_INTERNAL_TOKEN)}"
export SCORING_API_URL ORCHESTRATOR_URL SCORING_INTERNAL_TOKEN

fail=0
say_missing() { printf '  MISSING  %s\n     %s\n' "$1" "$2"; fail=1; }
say_ok()      { printf '  ok       %s\n' "$1"; }

echo "Checking what has to be up before anything is compared."
echo

# The secret, and that both sides hold the SAME one. Two different values fail
# every /internal call with a 401, which reads like a bug in the agent.
scoring_token="$(read_env "$SCORING_REPO/.env" INTERNAL_API_TOKEN)"
if [ -z "${SCORING_INTERNAL_TOKEN:-}" ]; then
  say_missing "the shared secret" "SCORING_INTERNAL_TOKEN is not in $HERE/.env"
elif [ -z "$scoring_token" ]; then
  say_missing "the shared secret on the scoring side" "INTERNAL_API_TOKEN is not in $SCORING_REPO/.env"
elif [ "$scoring_token" != "$SCORING_INTERNAL_TOKEN" ]; then
  say_missing "matching secrets" "the two .env files hold different values; every /internal call will answer 401"
else
  say_ok "both sides hold the same secret"
fi

if curl -fsS --max-time 5 "$SCORING_API_URL/health" >/dev/null 2>&1; then
  say_ok "the scoring service answers at $SCORING_API_URL"
  # Its own view of the four collection services, which it needs to read data.
  down="$(curl -fsS --max-time 5 "$SCORING_API_URL/health" \
    | python3 -c 'import json,sys; b=json.load(sys.stdin); print(" ".join(s["source"] for s in b.get("external_sources",[]) if s["status"]!="connected"))' 2>/dev/null)"
  if [ -n "$down" ]; then
    say_missing "the collection services: $down" "start each one; the analysis cannot read what they hold"
  else
    say_ok "all four collection services are reachable from it"
  fi
else
  say_missing "the scoring service" "cd $SCORING_REPO && docker compose up -d   (it publishes 8004)"
fi

if curl -fsS --max-time 5 "$ORCHESTRATOR_URL/healthz" >/dev/null 2>&1; then
  say_ok "the orchestrator answers at $ORCHESTRATOR_URL"
else
  say_missing "the orchestrator" "cd $HERE && .venv/bin/uvicorn app.main:app --port 8081"
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "Not ready. Start what is missing above and run this again."
  exit 2
fi

if [ "$#" -lt 2 ]; then
  echo "Ready. Now say what to compare:"
  echo "  $0 \"air fryer\" ESP PRT BRA"
  exit 0
fi

echo "Everything is up. Running the same analysis through both paths."
echo "The first one collects and can take minutes; the second reads what it left."
echo
exec "$HERE/.venv/bin/python" "$HERE/scripts/parity_check.py" "$@"
