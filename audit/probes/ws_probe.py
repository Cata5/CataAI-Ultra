"""Read-only live probe: WebSocket message delivery + status progression. No state changes."""
import asyncio
import json
import sys
import time

import aiohttp

BASE = "http://127.0.0.1:8080"
WS = "ws://127.0.0.1:8080/ws"


async def main(duration=12.0):
    out = {"ws_connected": False, "messages": [], "event_types": {}, "status_samples": []}
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(WS) as ws:
            out["ws_connected"] = True
            t_end = time.time() + duration
            while time.time() < t_end:
                try:
                    msg = await ws.receive(timeout=2.0)
                except asyncio.TimeoutError:
                    continue
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        d = json.loads(msg.data)
                    except Exception:
                        continue
                    et = d.get("event_type", "?")
                    out["event_types"][et] = out["event_types"].get(et, 0) + 1
                    if len(out["messages"]) < 3:
                        out["messages"].append(d)
        for _ in range(2):
            async with s.get(BASE + "/api/system/status") as r:
                d = await r.json()
                out["status_samples"].append({
                    "state": d.get("state"),
                    "run_id": d.get("run_id"),
                    "step": (d.get("latest_metrics") or {}).get("step"),
                    "loss": (d.get("latest_metrics") or {}).get("train_loss"),
                    "tok_per_sec": (d.get("latest_metrics") or {}).get("tok_per_sec"),
                })
            await asyncio.sleep(3)
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    asyncio.run(main(float(sys.argv[1]) if len(sys.argv) > 1 else 12.0))
