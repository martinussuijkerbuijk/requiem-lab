# Reolink PTZ Camera — LAN Connection Guide

## Setup Overview

- **Camera MAC:** `14:14:16:37:42:4c`
- **Camera hostname:** `Reolink`
- **Host machine IP on LAN:** `192.168.1.50` (on `eth0`)
- **Camera DHCP range:** `192.168.1.100–200` (assigned by dnsmasq)
- **Target static IP:** `192.168.1.126` (to be set via Reolink app)
- **Connection type:** Direct LAN cable (no router), camera uses DHCP

---

## The Core Problem

The camera uses **DHCP** to get its IP on the wired interface. When connected directly to your computer (no router), there is no DHCP server — so the camera gets no IP and becomes unreachable.

**Solution:** Run `dnsmasq` as a DHCP server on `eth0` every time you connect the camera via LAN.

---

## Step 1 — Start the DHCP Server

Run this command **before** or **immediately after** connecting the camera's LAN cable:

```bash
sudo dnsmasq --interface=eth0 --bind-interfaces \
  --dhcp-range=192.168.1.100,192.168.1.200,255.255.255.0,12h \
  --port=0 --no-daemon --log-dhcp
```

- `--port=0` — disables DNS to avoid conflict with the system dnsmasq
- `--no-daemon` — runs in the foreground so you can see DHCP activity
- Keep this terminal open while using the camera

**Expected output when camera connects:**
```
dnsmasq-dhcp: client provides name: Reolink
dnsmasq-dhcp: DHCPACK(eth0) 192.168.1.176 14:14:16:37:42:4c Reolink
```
The IP shown (e.g. `192.168.1.176`) is the camera's current address.

---

## Step 2 — Verify Connectivity

```bash
ping -c 4 192.168.1.176   # use whatever IP dnsmasq assigned
```

---

## Step 3 — Set a Permanent Static IP (via Reolink App)

To avoid needing to check the DHCP log every time:

1. Open the **Reolink app** on your phone
2. Add the camera (it must be on the same network — use LAN or temporarily switch to WiFi)
3. Go to **Device Settings → Network → Wired Network**
4. Switch from DHCP to **Static**
5. Set:
   - IP: `192.168.1.126`
   - Subnet: `255.255.255.0`
   - Gateway: `192.168.1.50`
   - DNS: `8.8.8.8` / `8.8.4.4`
6. Save and reboot the camera

After this, the camera will always be at `192.168.1.126` and dnsmasq is no longer needed.

---

## Troubleshooting

### Camera not responding after connecting LAN cable

**Check 1 — Is the physical link up?**
```bash
ethtool eth0 | grep "Link detected"
# Expected: Link detected: yes
```

**Check 2 — Is dnsmasq running?**
```bash
pgrep -a dnsmasq
```
If not running, start it (see Step 1).

**Check 3 — Did the camera get an IP?**

Check the dnsmasq terminal output for a `DHCPACK` line, or scan the subnet:
```bash
nmap -sn 192.168.1.0/24 --send-eth -e eth0
```

**Check 4 — Is the camera in the ARP table?**
```bash
ip neigh show dev eth0
# Look for MAC 14:14:16:37:42:4c
```

**Check 5 — Power cycle the camera**

Unplug power, wait 10 seconds, plug back in. The camera sends a fresh DHCP request on boot.

---

### Camera disappeared mid-session

The DHCP lease is valid for 12 hours, but if `dnsmasq` was stopped and the camera rebooted, it won't get a new IP.

**Fix:** Restart dnsmasq (Step 1), then power cycle the camera.

---

### Can't reach camera via WiFi after connecting LAN cable

Reolink PTZ cameras **prioritize wired over WiFi** — when a LAN cable is plugged in, WiFi is disabled automatically. This is normal.

---

### Factory reset required

If the camera's network config is broken (e.g. wrong static IP was set):

1. Find the **reset pinhole** on the camera body
2. Hold with a paperclip for **10 seconds** while powered on
3. Wait 60 seconds for reboot
4. Note: factory reset does **not** reset network config on all models — if it persists, hold longer or check your model's manual
5. Start dnsmasq and power cycle the camera to get a fresh DHCP lease

---

### eth0 missing the 192.168.1.x address

Verify your host has an IP on the right subnet:
```bash
ip addr show eth0
# Should include: inet 192.168.1.50/24
```

If missing, add it:
```bash
sudo ip addr add 192.168.1.50/24 dev eth0
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Start DHCP server | `sudo dnsmasq --interface=eth0 --bind-interfaces --dhcp-range=192.168.1.100,192.168.1.200,255.255.255.0,12h --port=0 --no-daemon --log-dhcp` |
| Check link status | `ethtool eth0 \| grep "Link detected"` |
| Scan for camera | `nmap -sn 192.168.1.0/24 --send-eth -e eth0` |
| Ping camera | `ping -c 4 192.168.1.176` |
| Check ARP table | `ip neigh show dev eth0` |
| Check eth0 addresses | `ip addr show eth0` |
