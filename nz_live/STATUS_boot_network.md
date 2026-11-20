# nz_live: boot start + network (tasks 1–2, test A5)

*2026-11-16. Covers plan_reduced_shortterm.md §4.1, §4.5, work-order tasks 1–2 and acceptance test A5.*

## What was set up

| Where | Change | Undo |
|---|---|---|
| Laptop | NetworkManager profile `Hotspot` (SSID `QIDK_DevNet`, `ipv4.method shared`, 10.42.0.1/24): band `bg`, `autoconnect yes`, priority 100. Wi-Fi card is now the hotspot (no longer on `LanIsDown`). Laptop's own traffic still uses wired `IIIT-LAN` | `nmcli c down Hotspot; nmcli c modify Hotspot connection.autoconnect no` |
| Laptop | Forwarding fix (run by the user): `iptables -I DOCKER-USER -i wlp0s20f3 -j ACCEPT` + the RELATED,ESTABLISHED return rule. Needed because Docker 29 sets the FORWARD policy to DROP | Rules vanish on reboot / firewalld reload |
| Board | Joined `QIDK_DevNet` (saved network, 10.42.0.237 via DHCP) | `adb root; adb shell cmd wifi forget-network <id>` |
| Board | `settings put global captive_portal_mode 0`: without it Android marked the intranet-only hotspot "no internet (permanent)" and refused to reconnect after reboot | `adb shell settings put global captive_portal_mode 1` |
| Board | `/system/etc/init/nz_live.rc` ([board/nz_live.rc](board/nz_live.rc)): runs `/data/local/tmp/nz_live/start.sh` as `shell`, `u:r:shell:s0`, on `sys.boot_completed=1` | `adb root && adb remount && adb shell rm /system/etc/init/nz_live.rc && adb reboot` |

## Results

| Test | Result |
|---|---|
| Board → API through the hotspot (as `shell`) | ✅ HTTP 200, ~0.6 s (after the forwarding fix) |
| `adb reboot` with the `.rc` installed (07:44) | ✅ `init.svc.nz_live=running`; nz_dashboard, llama-server, Bun dashboard all started by init as `shell`; :8080 200, :8090 `/health` ok, :5173 200 |
| **A5: power cycle, USB unplugged, laptop only running the hotspot** (10:25) | ✅ **Pass.** Board down 10:25:11 → Wi-Fi back on the hotspot by itself at 10:29:41 (+270 s, incl. the time it was switched off) → `/api/live/status` with `api.ok` and `npu.ok` true at 10:30:43 (+332 s). Laptop → board :8080 and :5173 both 200. NPU 4.9–5.6 ms, `now` correct (IST, from the API, not the board's 2023 clock). Today's 8 injected-fault events kept across the reboot (`last_24h.injected_faults = 8`). First snapshot during start-up showed `last_reading_checked: null` / `next_check: 1970…`; a minute later it was normal (10:15 checked, next 10:47) |

Baseline before A5 (10:03): everything green, 8 lines in `events.jsonl`.

After A5, USB didn't come back until a replug at 10:34. Then: `init.svc.nz_live=running`, all 3 services up as `shell`, 8 lines in `events.jsonl`. final-af's fixed dashboard build (no "1970" `next_check` before the first sync) was deployed at 10:34 and verified: `now` correct and `next_check` null right after the restart; 14 s later api ok, npu ok (5.3 ms), next check 10:47.

## Still needed from the user

1. **Make the forwarding fix permanent**: the iptables rules are lost on a laptop reboot, and then the board loses the API. `echo '{"ip-forward-no-drop": true}' | sudo tee /etc/docker/daemon.json && sudo systemctl restart docker` (no containers were running on 16 Nov).
2. **Keep the laptop awake on power** while it's the router (lid close must not suspend).
3. *Optional:* adb over Wi-Fi (`adb tcpip 5555`): not enabled (the attempt was denied), so board-side checks need USB.

## Update, 19 Nov: USB fallback route

The board now has a second way to the API that doesn't depend on the hotspot or Docker's forwarding: the live loop
falls back to `http://127.0.0.1:19999` on the board, which `adb reverse` carries over the USB cable to a `socat` relay on
the laptop (`nz_live/laptop/usb_uplink.sh`, systemd user service `nz-usb-uplink.service`). Setup, tests and commands:
[SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) §9a.

| Where | Change | Undo |
|---|---|---|
| Laptop | `~/.config/systemd/user/nz-usb-uplink.service`, enabled (relay on 127.0.0.1:19999 → API; keeps `adb reverse tcp:19999 tcp:19999` set) | `systemctl --user disable --now nz-usb-uplink; rm ~/.config/systemd/user/nz-usb-uplink.service` |
| Board | `start.sh`: `LIVE_API` lists both routes (old copy `start.sh.bak-19nov`) | Copy the backup back |

On 19 Nov the laptop joined `wifi@iiith` as a client, so the hotspot is currently **off** and the board runs over USB.
To bring the hotspot back: `nmcli c up Hotspot` (the laptop then needs the wired LAN, or another uplink, for its own
traffic). Campus Wi-Fi seen from the board: `wifi@iiith` (WPA2-Enterprise, −74 to −76 dBm) and `esw-m19@iiith`
(WPA2-PSK, −36 dBm); not configured.
