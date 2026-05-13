"""
Relais USB: Arduino (ligne JSON sur le moniteur série) -> POST http://localhost:8008/barn_sensor
Utilise: pip install pyserial
Exemple: python serial_barn_relay.py --port COM3
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_API = "http://127.0.0.1:8008/barn_sensor"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True, help="Port série, ex. COM3 ou /dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--api", default=DEFAULT_API, help="URL complete POST /barn_sensor")
    args = p.parse_args()

    try:
        import serial
    except ImportError:
        print("Installez pyserial: pip install pyserial", file=sys.stderr)
        return 1

    ser = serial.Serial(args.port, args.baud, timeout=1)
    print(f"Relais: {args.port} -> {args.api}", flush=True)
    for raw in ser:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "temp_c" not in data or "humidity" not in data:
            continue
        body = json.dumps(
            {
                "temp_c": data["temp_c"],
                "humidity": data["humidity"],
                **({"thi": data["thi"]} if "thi" in data else {}),
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            args.api,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                print(r.status, line[:80], flush=True)
        except urllib.error.URLError as e:
            print("POST error:", e, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
