import asyncio
import json
import base64
import subprocess
import time
from pathlib import Path
import urllib.request
import aiohttp

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
USER_DATA = r"C:\Users\Cata\AppData\Local\Temp\chrome_cdp_profile"
PORT = 9222
OUT_DIR = Path(r"D:\proiecte\CataAI Ultra\audit\screenshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TABS = [
    ("overview", "overview.png"),
    ("network_studio", "network_studio.png"),
    ("training", "training.png"),
    ("datasets", "datasets.png"),
    ("evaluation", "evaluation.png"),
    ("checkpoints", "checkpoints.png"),
    ("settings", "settings.png"),
]

proc = subprocess.Popen([
    CHROME,
    "--headless=new",
    f"--remote-debugging-port={PORT}",
    f"--user-data-dir={USER_DATA}",
    "--window-size=1600,1000",
    "http://localhost:8080/"
])

try:
    # Wait for DevTools port
    for _ in range(30):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json") as resp:
                targets = json.loads(resp.read().decode())
                if targets:
                    break
        except Exception:
            pass
    else:
        print("Failed to connect to Chrome DevTools")
        sys.exit(1)

    page_target = [t for t in targets if t.get("type") == "page"][0]
    ws_url = page_target["webSocketDebuggerUrl"]

    async def run():
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                msg_id = 1
                async def send_cmd(method, params=None):
                    nonlocal msg_id
                    mid = msg_id
                    msg_id += 1
                    await ws.send_json({"id": mid, "method": method, "params": params or {}})
                    while True:
                        msg = await ws.receive_json()
                        if msg.get("id") == mid:
                            return msg.get("result")

                # Wait for page load
                await asyncio.sleep(2)

                for tab_id, filename in TABS:
                    print(f"Switching to {tab_id}...")
                    await send_cmd("Runtime.evaluate", {
                        "expression": f"switchTab('{tab_id}');"
                    })
                    await asyncio.sleep(1.0)
                    res = await send_cmd("Page.captureScreenshot", {"format": "png"})
                    img_bytes = base64.b64decode(res["data"])
                    (OUT_DIR / filename).write_bytes(img_bytes)
                    print(f"Saved {filename} ({len(img_bytes)} bytes)")

    asyncio.run(run())

finally:
    proc.terminate()
    proc.wait()
