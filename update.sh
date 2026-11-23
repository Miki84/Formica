#!/bin/sh
# Formica-Update: läuft im Hilfscontainer "formica-updater" (gestartet über den
# Button in der Oberfläche) oder auf dem Server. Holt die neue Version, baut die
# Verwaltungsoberfläche neu und startet sie. Kommt die neue Version nicht hoch,
# geht es auf die vorige zurück.
set -eu
DIR="${FORMICA_HOME:-/opt/formica}"
LOG="$DIR/update.log"
exec >"$LOG" 2>&1

echo "== Formica-Update $(date '+%Y-%m-%d %H:%M:%S')"
cd "$DIR/app"
git config --global --add safe.directory "$DIR/app" 2>/dev/null || true
OLD=$(cat VERSION)
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)
[ "$BRANCH" = HEAD ] && BRANCH=main

echo "-- Hole $BRANCH"
git fetch --depth 1 origin "$BRANCH"
git reset --hard FETCH_HEAD
NEW=$(cat VERSION)
echo "-- Version $OLD -> $NEW"

echo "-- Sichere die laufende Version"
docker tag formica-control:local formica-control:previous

echo "-- Baue die Verwaltungsoberfläche"
docker build -t "formica-control:$NEW" -t formica-control:local .

echo "-- Starte neu"
docker compose -f "$DIR/control.yaml" up -d --force-recreate

ok=0
for i in $(seq 1 30); do
  sleep 2
  if docker exec formica-control python3 /app/formica.py version >/dev/null 2>&1 \
     && [ "$(docker inspect -f '{{.State.Running}}' formica-control)" = true ]; then
    ok=1; break
  fi
done

if [ "$ok" = 1 ]; then
  echo "== Fertig: Formica $NEW läuft"
else
  echo "!! Die neue Version startet nicht. Zurück auf $OLD"
  docker logs --tail 50 formica-control || true
  docker tag formica-control:previous formica-control:local
  docker compose -f "$DIR/control.yaml" up -d --force-recreate
  git reset --hard HEAD@{1} 2>/dev/null || true
  echo "== Zurück auf $OLD"
  exit 1
fi
