#!/bin/sh
# Formica-Update: läuft im Hilfscontainer "formica-updater" (gestartet über den
# Button in der Oberfläche) oder auf dem Server. Holt die neue Version, baut die
# Verwaltungsoberfläche neu und startet sie. Kommt die neue Version nicht hoch,
# geht es auf die vorige zurück.
set -eu
DIR="${FORMICA_HOME:-/opt/formica}"
LOG="$DIR/update.log"
cd "$DIR/app"
git config --global --add safe.directory "$DIR/app" 2>/dev/null || true

if [ -z "${FORMICA_STAGE2:-}" ]; then
  exec >"$LOG" 2>&1
  echo "== Formica-Update $(date '+%Y-%m-%d %H:%M:%S')"
  OLD=$(cat VERSION)
  BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)
  [ "$BRANCH" = HEAD ] && BRANCH=main
  echo "-- Hole $BRANCH"
  git fetch --depth 1 origin "$BRANCH"
  git reset --hard FETCH_HEAD
  # Weiter mit dem Update-Skript der neuen Version, falls es sich geändert hat
  if ! cmp -s "$0" "$DIR/app/update.sh"; then
    FORMICA_STAGE2=1 FORMICA_OLD="$OLD" exec sh "$DIR/app/update.sh"
  fi
else
  exec >>"$LOG" 2>&1
  OLD="${FORMICA_OLD:-?}"
  echo "-- Neues Update-Skript übernommen"
fi
NEW=$(cat VERSION)
echo "-- Version $OLD -> $NEW"

echo "-- Sichere die laufende Version"
docker tag formica-control:local formica-control:previous

echo "-- Baue die Verwaltungsoberfläche"
docker build -t "formica-control:$NEW" -t formica-control:local .

echo "-- Schreibe control.yaml"
PORT=$(grep -oE 'FORMICA_PORT: "?[0-9]+' "$DIR/control.yaml" 2>/dev/null | grep -oE '[0-9]+$' || true)
[ -n "$PORT" ] || PORT=$(grep -oE '"[0-9]+:8095"' "$DIR/control.yaml" 2>/dev/null | grep -oE '^"[0-9]+' | tr -d '"' || true)
[ -n "$PORT" ] || PORT=8095
cp "$DIR/control.yaml" "$DIR/control.yaml.bak" 2>/dev/null || true
cat >"$DIR/control.yaml" <<EOF
# Formica-Verwaltungsoberfläche. Erzeugt von update.sh.
name: formica-control
services:
  formica:
    image: formica-control:local
    container_name: formica-control
    restart: unless-stopped
    # Host-Netz: der Monitor sieht die echten Netzwerkkarten des Servers
    network_mode: host
    environment:
      TZ: Europe/Berlin
      FORMICA_HOME: $DIR
      FORMICA_PORT: "$PORT"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - $DIR:$DIR
EOF

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
  [ -f "$DIR/control.yaml.bak" ] && cp "$DIR/control.yaml.bak" "$DIR/control.yaml"
  docker compose -f "$DIR/control.yaml" up -d --force-recreate
  git reset --hard HEAD@{1} 2>/dev/null || true
  echo "== Zurück auf $OLD"
  exit 1
fi
