# pai-status: read-only host status exporter

The bot runs in a container, so it cannot see host load, temperatures, GPU,
systemd units or the Ollama watchdog. This tiny exporter writes one JSON file
on the host; the bot reads it (bind-mounted **read-only**) for `/status`.

| What | Where |
| --- | --- |
| Script (root-owned, 0755) | `/usr/local/sbin/pai-status-export` |
| Units | `/etc/systemd/system/pai-status-export.service` + `.timer` (every 30 s, `Nice=19`, idle IO, 20 s hard timeout) |
| Output (atomic write, dir 0755 root, file 0644) | `/var/lib/pai-status/host.json` |
| CPU counter state (tmpfs) | `/run/pai-status-export/cpu.json` |

It only **reads**: `/proc`, `/sys` (hwmon, drm), `systemctl is-active`,
`docker inspect --format` (state/health fields only, never the environment) and
the `ollama-watchdog` journal. It never touches Ollama, models, the watchdog's
files or any other service, and the JSON contains no secrets or chat data.

## Install (on the host, once; re-run after `git pull` to update)

```bash
cd ~/PrivateAI/ops/pai-status
sudo install -m 0755 -o root -g root pai-status-export /usr/local/sbin/pai-status-export
sudo install -m 0644 -o root -g root pai-status-export.service pai-status-export.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pai-status-export.timer
sudo systemctl start pai-status-export.service      # first snapshot right away
python3 -m json.tool /var/lib/pai-status/host.json | head
```

The bot container mounts it through `docker-compose.yml`
(`/var/lib/pai-status:/host-status:ro`); no other host path is mounted.

## Disable / remove

```bash
sudo systemctl disable --now pai-status-export.timer     # stop exporting
# full removal
sudo rm /etc/systemd/system/pai-status-export.{service,timer} /usr/local/sbin/pai-status-export
sudo rm -rf /var/lib/pai-status && sudo systemctl daemon-reload
```

With the timer off `/status` still works: the host section shows
`n/a (exporter not running)` and counts as a WARN (never an error).

## Configuration (environment of the service, optional)

`PAI_DATA_PATH` (bot data dir, default `/home/jthornto/PrivateAI/data`),
`PAI_UNITS` (comma list, default `ollama.service,docker.service,ollama-watchdog.timer`),
`PAI_CONTAINERS` (default `privateai,privateai-memory`),
`PAI_WATCHDOG_UNIT` (default `ollama-watchdog`). Use a drop-in
(`systemctl edit pai-status-export.service`) to change them.
