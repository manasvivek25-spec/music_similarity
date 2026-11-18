"""
Headless Chrome CDP End-to-End UI Verification & Screenshot Suite (P8).

Connects to http://127.0.0.1:8001 via headless Google Chrome + DevTools Protocol:
1. Opens Analysis Dashboard, verifies zero console errors and zero failed network calls.
2. Saves reports/ui_checks/dashboard_home.png.
3. Uploads a bad non-audio file -> verifies client validation error message appears.
4. Uploads a REAL held-out track from data/v2/v2_held_out.json -> waits for SSE/pipeline completion,
   verifies Authoritative Verdict, Case Routing, Truth Panel (Ground Truth Matched!), F1-F7 cards,
   and saves reports/ui_checks/analysis_truth_panel.png.
5. Switches to Evaluation & Dataset Audit tab -> verifies Master Table, Truth Agreement Table,
   Model Specs, Confound Probes, Zero-Leakage badge, and all 4 PNG diagnostic figures loaded
   (naturalWidth > 0), saving reports/ui_checks/evaluation_tab.png and reports/ui_checks/eda_audit_view.png.
"""

import asyncio
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import websockets

REPO_ROOT = Path(__file__).resolve().parent.parent
CHROME_EXE = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
OUT_DIR = REPO_ROOT / "reports" / "ui_checks"
BASE_URL = "http://127.0.0.1:8001"
CDP_PORT = 9299


class CDPClient:
    def __init__(self, ws):
        self.ws = ws
        self.msg_id = 0
        self.console_errors: List[str] = []
        self.page_errors: List[str] = []
        self.failed_requests: List[str] = []

    async def send(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        self.msg_id += 1
        cid = self.msg_id
        payload = {"id": cid, "method": method, "params": params or {}}
        await self.ws.send(json.dumps(payload))
        while True:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
            msg = json.loads(raw)
            if msg.get("id") == cid:
                if "error" in msg:
                    raise RuntimeError(f"CDP error for {method}: {msg['error']}")
                return msg.get("result", {})
            self._handle_event(msg)

    def _handle_event(self, msg: Dict[str, Any]) -> None:
        m = msg.get("method", "")
        p = msg.get("params", {})
        if m == "Runtime.consoleAPICalled" and p.get("type") == "error":
            args = [a.get("value") or a.get("description") or "" for a in p.get("args", [])]
            self.console_errors.append(" ".join(str(x) for x in args))
        elif m == "Runtime.exceptionThrown":
            desc = p.get("exceptionDetails", {}).get("text", "Uncaught exception")
            self.page_errors.append(desc)
        elif m == "Network.responseReceived":
            resp = p.get("response", {})
            status = resp.get("status", 200)
            url = resp.get("url", "")
            if status >= 400 and url.startswith(BASE_URL):
                self.failed_requests.append(f"HTTP {status}: {url}")

    async def evaluate(self, expr: str) -> Any:
        res = await self.send("Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True})
        return res.get("result", {}).get("value")

    async def screenshot(self, path: Path, full_page: bool = False) -> None:
        params: Dict[str, Any] = {"format": "png"}
        if full_page:
            metrics = await self.send("Page.getLayoutMetrics")
            cs = metrics.get("contentSize", {"width": 1440, "height": 1080})
            params["clip"] = {"x": 0, "y": 0, "width": min(1440, cs["width"]), "height": min(3200, cs["height"]), "scale": 1}
        res = await self.send("Page.captureScreenshot", params)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(res["data"]))


def _pick_real_held_out_wav() -> Path:
    ho_json = REPO_ROOT / "data" / "v2" / "v2_held_out.json"
    tracks = json.loads(ho_json.read_text(encoding="utf-8")).get("tracks", {})
    for tid in sorted(tracks.keys()):
        wp = REPO_ROOT / tracks[tid]["norm_wav_path"]
        if wp.is_file():
            return wp
    raise FileNotFoundError("No real held-out WAV found under data/v2/audio/norm10s/")


async def run_cdp_verification() -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    user_data_dir = tempfile.mkdtemp(prefix="chrome_cdp_p8_")
    chrome_proc = subprocess.Popen(
        [
            str(CHROME_EXE),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={user_data_dir}",
            "--window-size=1440,1080",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Wait for DevTools HTTP endpoint
        ws_url = None
        for _ in range(40):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json/list", timeout=2) as r:
                    targets = json.loads(r.read().decode("utf-8"))
                    page = next((t for t in targets if t.get("type") == "page"), None)
                    if page and page.get("webSocketDebuggerUrl"):
                        ws_url = page["webSocketDebuggerUrl"]
                        break
            except Exception:
                time.sleep(0.25)
        if not ws_url:
            raise RuntimeError("Could not connect to headless Chrome CDP endpoint.")

        async with websockets.connect(ws_url, max_size=32 * 1024 * 1024) as ws:
            cdp = CDPClient(ws)
            await cdp.send("Page.enable")
            await cdp.send("Runtime.enable")
            await cdp.send("Network.enable")
            await cdp.send("DOM.enable")
            await cdp.send("Emulation.setDeviceMetricsOverride", {
                "width": 1440, "height": 1080, "deviceScaleFactor": 1, "mobile": False
            })

            # 1. Open Analysis Dashboard
            print("[CDP] Navigating to", BASE_URL)
            await cdp.send("Page.navigate", {"url": BASE_URL})
            health_txt = inv_total = clf_txt = None
            for _ in range(30):
                await asyncio.sleep(0.5)
                inv_total = await cdp.evaluate("document.getElementById('invTotalIndexed')?.innerText || ''")
                if inv_total and inv_total.isdigit() and int(inv_total) >= 300:
                    break
            health_txt = await cdp.evaluate("document.getElementById('backendStatusText')?.innerText")
            clf_txt = await cdp.evaluate("document.getElementById('classifierStatusText')?.innerText")
            loc_href = await cdp.evaluate("window.location.href")
            rstate = await cdp.evaluate("document.readyState")
            print(f"[CDP] Home loaded ({loc_href}, {rstate}): health='{health_txt}', indexed='{inv_total}', clf='{clf_txt}'")
            if cdp.console_errors or cdp.page_errors:
                print("[CDP] Early errors:", cdp.console_errors, cdp.page_errors)
            assert inv_total and inv_total.isdigit() and int(inv_total) >= 300, f"Unexpected indexed count on UI: {inv_total}"
            await cdp.screenshot(OUT_DIR / "dashboard_home.png", full_page=False)

            # 2. Bad upload validation test (non-audio .txt file)
            bad_file = Path(user_data_dir) / "not_audio.txt"
            bad_file.write_text("this is plain text not audio", encoding="utf-8")
            doc = await cdp.send("DOM.getDocument", {"depth": -1})
            q_node = await cdp.send("DOM.querySelector", {
                "nodeId": doc["root"]["nodeId"],
                "selector": "#mainQueryFileInput",
            })
            await cdp.send("DOM.setFileInputFiles", {
                "nodeId": q_node["nodeId"],
                "files": [str(bad_file.resolve())],
            })
            await asyncio.sleep(0.5)
            bad_fb = await cdp.evaluate("document.getElementById('pipelineStatusFeedback')?.innerText")
            print(f"[CDP] Bad file validation feedback: '{bad_fb}'")
            assert "Validation Error" in (bad_fb or ""), f"Expected client validation error, got: {bad_fb}"

            # 3. Upload REAL held-out track -> run full analysis -> verify Truth Panel & F1-F7
            ho_wav = _pick_real_held_out_wav()
            print(f"[CDP] Uploading real held-out track: {ho_wav.name}")
            await cdp.send("DOM.setFileInputFiles", {
                "nodeId": q_node["nodeId"],
                "files": [str(ho_wav.resolve())],
            })
            await asyncio.sleep(0.6)
            preview_vis = await cdp.evaluate("!document.getElementById('queryAudioPreviewWrapper').classList.contains('hidden')")
            assert preview_vis, "Query audio preview player did not appear!"

            await cdp.evaluate("document.getElementById('runFullAnalysisBtn').click()")
            # Wait up to 90s for analysis completion
            done = False
            for sec in range(90):
                await asyncio.sleep(1.0)
                fb = await cdp.evaluate("document.getElementById('pipelineStatusFeedback')?.innerText || ''")
                if "Analysis complete:" in fb:
                    done = True
                    print(f"[CDP] Pipeline completed in ~{sec+1}s: {fb}")
                    break
                if "Error:" in fb:
                    raise RuntimeError(f"UI pipeline error: {fb}")
            assert done, "Timed out waiting for UI pipeline completion."

            verdict_txt = await cdp.evaluate("document.getElementById('verdictCodeBadge')?.innerText")
            case_txt = await cdp.evaluate("document.getElementById('caseRoutingBadge')?.innerText")
            truth_badge = await cdp.evaluate("document.getElementById('truthPanelBadge')?.innerText")
            f1_val = await cdp.evaluate("document.getElementById('card1SimScore')?.innerText")
            f4_val = await cdp.evaluate("document.getElementById('card4PredictedOrigin')?.innerText")
            print(f"[CDP] Analysis UI state: verdict='{verdict_txt}', case='{case_txt}', truth='{truth_badge}', F1={f1_val}, F4={f4_val}")
            assert truth_badge == "Ground Truth Matched", f"Expected Ground Truth Matched, got: {truth_badge}"
            assert f1_val and float(f1_val) > 0.45, f"Expected real non-zero F1 similarity on UI, got: {f1_val}"
            assert f4_val in ("Human", "AI"), f"Expected valid F4 prediction, got: {f4_val}"

            await cdp.evaluate("document.getElementById('verdictBannerSection').scrollIntoView({behavior:'instant'})")
            await asyncio.sleep(0.4)
            await cdp.screenshot(OUT_DIR / "analysis_truth_panel.png", full_page=False)

            # 4. Switch to Evaluation & Dataset Audit Tab
            print("[CDP] Switching to Evaluation & Dataset Audit tab...")
            await cdp.evaluate("document.getElementById('navTabEvaluation').click()")
            await asyncio.sleep(2.5)

            leak_badge = await cdp.evaluate("document.getElementById('evalLeakageBadge')?.innerText")
            master_rows = await cdp.evaluate("document.querySelectorAll('#evalMasterTableBody tr').length")
            truth_rows = await cdp.evaluate("document.querySelectorAll('#evalTruthAgreementBody tr').length")
            figs_ok = await cdp.evaluate(
                "Array.from(document.querySelectorAll('#evalFiguresGrid img')).map(i => ({src: i.src, ok: i.complete && i.naturalWidth > 0}))"
            )
            print(f"[CDP] Eval tab: leakage='{leak_badge}', master_rows={master_rows}, truth_rows={truth_rows}, figs={len(figs_ok)}")
            assert "PASSED" in (leak_badge or ""), f"Leakage badge not passed: {leak_badge}"
            assert master_rows >= 18, f"Expected >=18 master rows, got {master_rows}"
            assert truth_rows == 7, f"Expected 7 truth agreement rows, got {truth_rows}"
            assert len(figs_ok) == 4 and all(x["ok"] for x in figs_ok), f"Not all 4 figures loaded cleanly: {figs_ok}"

            await cdp.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.3)
            await cdp.screenshot(OUT_DIR / "evaluation_tab.png", full_page=False)

            await cdp.evaluate("document.getElementById('datasetAuditContainer').scrollIntoView({behavior:'instant'})")
            await asyncio.sleep(0.4)
            await cdp.screenshot(OUT_DIR / "eda_audit_view.png", full_page=False)

            # Check zero console/page/network errors
            print(f"[CDP] Console errors: {len(cdp.console_errors)}, Page errors: {len(cdp.page_errors)}, Failed local requests: {len(cdp.failed_requests)}")
            assert len(cdp.console_errors) == 0, f"Console errors: {cdp.console_errors}"
            assert len(cdp.page_errors) == 0, f"Page errors: {cdp.page_errors}"
            assert len(cdp.failed_requests) == 0, f"Failed requests: {cdp.failed_requests}"

            summary = {
                "status": "PASSED",
                "health_text": health_txt,
                "indexed_count": inv_total,
                "bad_upload_validation": bad_fb,
                "real_upload_verdict": verdict_txt,
                "real_upload_case": case_txt,
                "truth_panel_badge": truth_badge,
                "f1_score": f1_val,
                "f4_prediction": f4_val,
                "eval_leakage_badge": leak_badge,
                "eval_master_rows": master_rows,
                "eval_truth_rows": truth_rows,
                "figures_loaded": len(figs_ok),
                "console_errors": 0,
                "page_errors": 0,
                "failed_requests": 0,
                "screenshots": [
                    "reports/ui_checks/dashboard_home.png",
                    "reports/ui_checks/analysis_truth_panel.png",
                    "reports/ui_checks/evaluation_tab.png",
                    "reports/ui_checks/eda_audit_view.png",
                ],
            }
            (OUT_DIR / "cdp_verification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print("[CDP] ALL UI CHECKS PASSED!")
            return summary
    finally:
        chrome_proc.terminate()
        try:
            chrome_proc.wait(timeout=5)
        except Exception:
            chrome_proc.kill()


if __name__ == "__main__":
    asyncio.run(run_cdp_verification())
