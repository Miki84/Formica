#!/usr/bin/env bash
# Formica: Installation auf einem Linux-Server (VM oder Bare Metal).
#
#   curl -fsSL https://raw.githubusercontent.com/Miki84/Formica/main/install.sh | bash
#
# Oder aus einem geklonten Repo: ./install.sh
# Als normaler Benutzer mit sudo-Rechten oder als root starten. Das Skript kann
# gefahrlos erneut laufen; so wird Formica auch aktualisiert.
#
# Optionen:
#   --dir DIR        Datenverzeichnis (Standard /opt/formica)
#   --port N         Port der Verwaltungsoberfläche (Standard 8095)
#   --repo URL       anderes Git-Repo (Fork, Spiegel)
#   --branch NAME    Branch oder Tag (Standard main)
#   --src DIR        lokalen Klon als Quelle nehmen
#   --yes            keine Rückfragen
#   --help           diese Hilfe
set -euo pipefail

DIR=/opt/formica
PORT=8095
REPO=https://github.com/Miki84/Formica.git
BRANCH=main
SRC=""
YES=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --src) SRC="$2"; shift 2 ;;
    --yes|-y) YES=1; shift ;;
    --help|-h) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unbekannte Option: $1 (siehe --help)"; exit 1 ;;
  esac
done

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '  \033[36m•\033[0m %s\n' "$*"; }
fail() { printf '\033[31mFehler:\033[0m %s\n' "$*" >&2; exit 1; }

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  command -v sudo >/dev/null || fail "Bitte als root oder mit sudo-Rechten starten."
  SUDO="sudo"
fi

bold "Formica wird installiert"

# --- System prüfen ---------------------------------------------------------
[ "$(uname -s)" = Linux ] || fail "Formica läuft auf Linux."
. /etc/os-release 2>/dev/null || true
info "System: ${PRETTY_NAME:-unbekannt}, $(nproc) Kerne, $(awk '/MemTotal/{printf "%.0f GB RAM", $2/1048576}' /proc/meminfo)"
mem_gb=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)
[ "$mem_gb" -ge 7 ] || info "Achtung: weniger als 8 GB RAM. Docling und die Embeddings brauchen zusammen etwa 6 GB."

# --- Docker ----------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  info "Docker fehlt, wird installiert (offizielles Skript von get.docker.com)"
  case "${ID:-}" in
    ubuntu|debian|raspbian|fedora|centos|rhel|rocky|almalinux|linuxmint|pop)
      curl -fsSL https://get.docker.com | $SUDO sh ;;
    arch|manjaro|endeavouros)
      $SUDO pacman -Sy --noconfirm docker docker-compose ;;
    *) fail "Docker bitte zuerst installieren: https://docs.docker.com/engine/install/" ;;
  esac
  $SUDO systemctl enable --now docker
fi
$SUDO docker compose version >/dev/null 2>&1 || fail "Das Docker-Compose-Plugin fehlt (Paket docker-compose-plugin)."
info "Docker $($SUDO docker version --format '{{.Server.Version}}' 2>/dev/null || echo '?')"

if command -v nvidia-smi >/dev/null 2>&1; then
  info "NVIDIA-GPU gefunden: $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
fi

# --- Quellcode -------------------------------------------------------------
$SUDO mkdir -p "$DIR"
APP="$DIR/app"
if [ -n "$SRC" ]; then
  info "Quelle: lokaler Ordner $SRC"
  $SUDO rm -rf "$APP.new"
  $SUDO cp -a "$SRC" "$APP.new"
  $SUDO rm -rf "$APP"; $SUDO mv "$APP.new" "$APP"
elif [ -d "$APP/.git" ]; then
  info "Aktualisiere Quellcode ($BRANCH)"
  $SUDO git -C "$APP" fetch --depth 1 origin "$BRANCH"
  $SUDO git -C "$APP" reset --hard FETCH_HEAD
else
  command -v git >/dev/null 2>&1 || {
    info "git fehlt, wird installiert"
    if command -v apt-get >/dev/null; then $SUDO apt-get update -qq && $SUDO apt-get install -y -qq git
    elif command -v dnf >/dev/null; then $SUDO dnf install -y -q git
    elif command -v pacman >/dev/null; then $SUDO pacman -Sy --noconfirm git
    else fail "Bitte git installieren."; fi
  }
  info "Lade Formica von $REPO"
  $SUDO rm -rf "$APP"
  $SUDO git clone --depth 1 --branch "$BRANCH" "$REPO" "$APP"
fi
VERSION=$(cat "$APP/VERSION")
info "Version $VERSION"

# --- Verwaltungsoberfläche bauen und starten -------------------------------
info "Baue die Verwaltungsoberfläche"
$SUDO docker build -q -t "formica-control:$VERSION" -t formica-control:local "$APP" >/dev/null

$SUDO tee "$DIR/control.yaml" >/dev/null <<EOF
# Formica-Verwaltungsoberfläche. Erzeugt von install.sh.
name: formica-control
services:
  formica:
    image: formica-control:local
    container_name: formica-control
    restart: unless-stopped
    ports:
      - "$PORT:8095"
    environment:
      TZ: Europe/Berlin
      FORMICA_HOME: $DIR
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - $DIR:$DIR
EOF
$SUDO docker compose -f "$DIR/control.yaml" up -d --force-recreate >/dev/null

# --- Befehl "formica" auf dem Server ---------------------------------------
$SUDO tee /usr/local/bin/formica >/dev/null <<EOF
#!/usr/bin/env bash
# Formica auf diesem Server. Die Oberfläche: http://<server>:$PORT
DIR="$DIR"
case "\${1:-help}" in
  token)          docker exec formica-control python3 /app/formica.py token ;;
  reset-password) docker exec formica-control python3 /app/formica.py reset-password ;;
  version)        docker exec formica-control python3 /app/formica.py version ;;
  logs)           docker logs -f --tail 200 formica-control ;;
  status)         docker compose -p formica -f "\$DIR/stack/compose.yaml" ps ;;
  restart)        docker compose -f "\$DIR/control.yaml" restart ;;
  update)         shift; bash "\$DIR/app/install.sh" --dir "\$DIR" --port $PORT --yes "\$@" ;;
  *) echo "formica token | reset-password | version | logs | status | restart | update" ;;
esac
EOF
$SUDO chmod 755 /usr/local/bin/formica

# --- Warten und Adresse zeigen ---------------------------------------------
for _ in $(seq 1 30); do
  curl -fs "http://127.0.0.1:$PORT/api/state" >/dev/null 2>&1 && break
  sleep 1
done
curl -fs "http://127.0.0.1:$PORT/api/state" >/dev/null 2>&1 || fail "Die Oberfläche startet nicht. Logs: sudo formica logs"

IP=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
IP=${IP:-$(hostname -I | awk '{print $1}')}

if command -v ufw >/dev/null 2>&1 && $SUDO ufw status 2>/dev/null | grep -q "Status: active"; then
  info "Firewall (ufw) aktiv: öffne Ports $PORT, 80 und 443"
  $SUDO ufw allow "$PORT/tcp" >/dev/null; $SUDO ufw allow 80/tcp >/dev/null; $SUDO ufw allow 443/tcp >/dev/null
fi

echo
bold "Formica $VERSION läuft."
echo
echo "  Verwaltungsoberfläche:  http://$IP:$PORT"
TOKEN=$($SUDO docker exec formica-control python3 /app/formica.py token 2>/dev/null || true)
case "$TOKEN" in
  *eingerichtet*) echo "  (bereits eingerichtet, Anmeldung mit dem Admin-Passwort)" ;;
  *) echo "  Einrichtungscode:       $TOKEN" ;;
esac
echo
echo "  Auf dem Server:         formica token | status | logs | update"
echo
