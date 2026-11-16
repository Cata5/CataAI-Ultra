import asyncio
import json
import subprocess
import time
from pathlib import Path
import urllib.request
import aiohttp

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
USER_DATA = r"C:\Users\Cata\AppData\Local\Temp\chrome_console_profile"
PORT = 9223

proc = subprocess.Popen([
    CHROME,
    "--headless=new",
    f"--remote-debugging-port={PORT}",
    f"--user-data-dir={USER_DATA}",
    "--window-size=1600,1000",
    "http://localhost:8080/"
])

try:
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
        print("Failed to connect")
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
                    return mid

                await send_cmd("Runtime.enable")
                await send_cmd("Log.enable")

                logs = []
                async def listen():
                    while True:
                        try:
                            msg = await asyncio.wait_for(ws.receive_json(), timeout=1.0)
                            method = msg.get("method")
                            if method == "Runtime.consoleAPICalled":
                                params = msg.get("params", {})
                                logs.append({"type": params.get("type"), "args": [a.get("value") for a in params.get("args", [])]})
                            elif method == "Runtime.exceptionThrown":
                                logs.append({"type": "exception", "details": msg.get("params", {}).get("exceptionDetails", {})})
                            elif method == "Log.entryAdded":
                                logs.append({"type": "log_entry", "entry": msg.get("params", {}).get("entry", {})})
                        except asyncio.TimeoutError:
                            break

                await asyncio.sleep(2)
                tabs = ['overview', 'network_studio', 'training', 'datasets', 'evaluation', 'checkpoints', 'settings']
                for t in tabs:
                    await send_cmd("Runtime.evaluate", {"expression": f"switchTab('{t}')"})
                    await asyncio.sleep(0.5)

                await listen()
                print("COLLECTED LOGS:")
                print(json.dumps(logs, indent=2))

    asyncio.run(run())

finally:
    proc.terminate()
    proc.wait()
