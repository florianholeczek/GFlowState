"""
Benchmark C: dashboard rendering times and memory usage.

Starts the dashboard on a temporary copy of a run (the dashboard writes to the db),
drives it with Playwright (Chromium) and reports, per repeat (fresh server and
browser each time, so every repeat is a cold start):

    0. startup   server process start -> HTTP server answers
    1. cold      page navigation -> all initial callbacks answered and plots drawn
    2. iterations  lower bound of the iteration slider set to the middle of the range
                   -> sample ranking (bumpchart) and state projection redrawn
    3. umap      dimensionality reduction set to UMAP -> state projection redrawn
    4. select    lasso over all samples of the last iteration in the sample ranking
                 -> selection callback, linked views and the DAG elements/layout
    5. dag_tab   switch to the DAG tab -> DAG drawn

Timings are taken in the browser (performance.now()): callback round trips are
measured around the fetch of /_dash-update-component, plot rendering from
Plotly.react/newPlot until two animation frames after it resolved, the DAG from
cytoscape layout/render events. A phase ends after QUIET_MS without any callback,
render or layout activity; its duration is the time of the last activity.

Memory: resident set size of the dashboard server process tree and of the
Chromium process tree (sum of RSS, sampled every 50 ms), JS heap via CDP.

Usage (run with an interpreter that has playwright and psutil):
    python benchmarks/dashboard_metrics.py --run path/to/logs/run_dir
        [--launcher path/to/run_dashboard.py] [--python path/to/python.exe]
        [--repeats 3] [--port 8061] [--headed] [--out file.json]
"""

import argparse
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import psutil
from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAUNCHER = (
    REPO_ROOT.parent / "gflownet" / "gflownet" / "utils" / "vislogger" / "run_dashboard.py"
)
QUIET_MS = 1500
PHASE_TIMEOUT_S = 1800
NO_EFFECT_MS = 30000  # fail if an action causes no activity within this time
VIEWPORT = {"width": 1920, "height": 1080}

# Instrumentation injected before any page script runs.
INSTRUMENT_JS = r"""
(() => {
  const bm = window.__bm = {reqs: [], renders: [], cy: [], inflight: 0, last: 0,
                            selected: null};
  const mark = () => { bm.last = performance.now(); };

  const origFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const url = typeof input === 'string' ? input : input.url;
    if (!url.includes('_dash-update-component')) return origFetch(input, init);
    let out = null;
    try { out = JSON.parse(init.body).output; } catch (e) {}
    const t0 = performance.now();
    bm.inflight++; mark();
    try {
      const r = await origFetch(input, init);
      const buf = await r.clone().arrayBuffer();
      bm.reqs.push({out, t0, t1: performance.now(), bytes: buf.byteLength,
                    status: r.status});
      return r;
    } finally { bm.inflight--; mark(); }
  };

  const wrapPlotly = (P) => {
    if (!P || P.__bmWrapped) return P;
    for (const fn of ['react', 'newPlot']) {
      const orig = P[fn];
      if (typeof orig !== 'function') continue;
      P[fn] = function (gd, ...rest) {
        const t0 = performance.now(); mark();
        const el = typeof gd === 'string' ? document.getElementById(gd) : gd;
        const host = el && el.closest ? el.closest('.dash-graph') : null;
        const id = host ? host.id : (el ? el.id : null);
        return Promise.resolve(orig.call(this, gd, ...rest)).then((res) => {
          requestAnimationFrame(() => requestAnimationFrame(() => {
            bm.renders.push({id, fn, t0, t1: performance.now()}); mark();
          }));
          return res;
        });
      };
    }
    P.__bmWrapped = true;
    return P;
  };
  let _P = window.Plotly ? wrapPlotly(window.Plotly) : undefined;
  Object.defineProperty(window, 'Plotly', {
    configurable: true, get() { return _P; }, set(v) { _P = wrapPlotly(v); },
  });

  // cytoscape (dag-graph): attach as soon as the instance exists
  const hookCy = setInterval(() => {
    const el = document.getElementById('dag-graph');
    const cy = el && el._cyreg && el._cyreg.cy;
    if (!cy) return;
    clearInterval(hookCy);
    cy.on('layoutstart layoutstop', (e) => {
      bm.cy.push({ev: e.type, t: performance.now(), n: cy.elements().length});
      mark();
    });
    cy.on('render', () => {
      const t = performance.now();
      requestAnimationFrame(() => { bm.cy.push({ev: 'render', t, t1: performance.now(),
                                                n: cy.elements().length}); mark(); });
    });
  }, 20);
})();
"""


# ----------------------------------------------------------------------------- utils
def find_pythonpath(launcher):
    """Topmost directory above the launcher that is not a python package."""
    d = launcher.parent
    while (d / "__init__.py").exists():
        d = d.parent
    return d


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def tree(proc):
    try:
        return [proc] + proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def rss_sum(procs):
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


class MemoryMonitor(threading.Thread):
    """Samples RSS of the server tree and of the browser processes."""

    def __init__(self, server_pid, interval=0.05):
        super().__init__(daemon=True)
        self.server = psutil.Process(server_pid)
        self.me = psutil.Process()
        self.interval = interval
        self.phase = "startup"
        self.peaks = {}
        self.last = {}
        self._halt = threading.Event()

    def browser_procs(self):
        server_pids = {p.pid for p in tree(self.server)}
        return [
            p
            for p in self.me.children(recursive=True)
            if p.pid not in server_pids
            and any(k in p.name().lower() for k in ("chrome", "headless_shell"))
        ]

    def run(self):
        while not self._halt.is_set():
            try:
                s = rss_sum(tree(self.server))
                b = rss_sum(self.browser_procs())
            except psutil.Error:
                s, b = 0, 0
            pk = self.peaks.setdefault(self.phase, {"server": 0, "browser": 0})
            pk["server"] = max(pk["server"], s)
            pk["browser"] = max(pk["browser"], b)
            self.last[self.phase] = {"server": s, "browser": b}
            time.sleep(self.interval)

    def stop(self):
        self._halt.set()


def wait_quiet(page, t_start, timeout=PHASE_TIMEOUT_S):
    """Wait until no callback/render activity for QUIET_MS after t_start."""
    deadline = time.time() + timeout
    next_dbg = time.time() + 10
    while time.time() < deadline:
        s = page.evaluate(
            "() => ({now: performance.now(), inflight: window.__bm.inflight,"
            " last: window.__bm.last, title: document.title,"
            " nreq: window.__bm.reqs.length, nren: window.__bm.renders.length,"
            " ncy: window.__bm.cy.length})"
        )
        if os.environ.get("BENCH_DEBUG") and time.time() > next_dbg:
            print(f"  waiting: {s}", file=sys.stderr)
            next_dbg = time.time() + 10
        if s["last"] < t_start and s["now"] - t_start > NO_EFFECT_MS:
            raise RuntimeError("action triggered no callback or render")
        if (
            s["inflight"] == 0
            and s["title"] != "Updating..."
            and s["last"] >= t_start
            and s["now"] - s["last"] > QUIET_MS
        ):
            return
        time.sleep(0.1)
    raise TimeoutError("dashboard did not become idle")


def collect(page, t_start):
    """Events of the current phase, relative to t_start (ms)."""
    bm = page.evaluate("() => window.__bm")
    reqs = [r for r in bm["reqs"] if r["t0"] >= t_start]
    renders = [r for r in bm["renders"] if r["t0"] >= t_start]
    cy = [c for c in bm["cy"] if c["t"] >= t_start]

    def graph_end(gid):
        ts = [r["t1"] for r in renders if r["id"] == gid]
        return max(ts) - t_start if ts else None

    def cb(substr):
        rs = [r for r in reqs if r["out"] and substr in r["out"]]
        if not rs:
            return None
        return {
            "count": len(rs),
            "roundtrip_ms_total": sum(r["t1"] - r["t0"] for r in rs),
            "end_ms": max(r["t1"] for r in rs) - t_start,
            "response_bytes": sum(r["bytes"] for r in rs),
        }

    layout_stops = [c for c in cy if c["ev"] == "layoutstop"]
    cy_renders = [c for c in cy if c["ev"] == "render"]
    ends = [r["t1"] for r in reqs] + [r["t1"] for r in renders]
    ends += [c.get("t1", c["t"]) for c in cy]
    return {
        "end_to_end_ms": (max(ends) - t_start) if ends else 0.0,
        "n_callbacks": len(reqs),
        "callbacks": [
            {
                "output": r["out"],
                "start_ms": r["t0"] - t_start,
                "roundtrip_ms": r["t1"] - r["t0"],
                "bytes": r["bytes"],
                "status": r["status"],
            }
            for r in reqs
        ],
        "bumpchart_ms": graph_end("bumpchart"),
        "state_space_ms": graph_end("state-space-plot"),
        "dag_overview_ms": graph_end("dag-overview"),
        "cb_bumpchart": cb("bumpchart.figure"),
        "cb_state_space": cb("state-space-plot.figure"),
        "cb_selection": cb("selected-objects.data"),
        "cb_dag": cb("dag-graph.elements"),
        "dag_layoutstop_ms": (
            max(c["t"] for c in layout_stops) - t_start if layout_stops else None
        ),
        "dag_last_render_ms": (
            max(c["t1"] for c in cy_renders) - t_start if cy_renders else None
        ),
        "dag_elements": cy[-1]["n"] if cy else None,
    }


def now(page):
    return page.evaluate("() => performance.now()")


# ----------------------------------------------------------------------------- steps
def step_iterations(page):
    # drag the lower thumb to the middle of the track (pointer -> value is linear
    # over the slider root in the radix slider used by dcc.RangeSlider)
    thumb = page.locator("#iteration .dash-slider-thumb-1")
    vmin = int(thumb.get_attribute("aria-valuemin"))
    vmax = int(thumb.get_attribute("aria-valuemax"))
    root = page.locator("#iteration .dash-slider-root").bounding_box()
    tb = thumb.bounding_box()
    target_x = root["x"] + root["width"] * 0.5
    y = tb["y"] + tb["height"] / 2
    page.mouse.move(tb["x"] + tb["width"] / 2, y)
    page.mouse.down()
    page.mouse.move(target_x, y, steps=15)
    t0 = now(page)
    page.mouse.up()
    lo = int(thumb.get_attribute("aria-valuenow"))
    return t0, {
        "iteration_range": [lo, vmax],
        "iteration_full_range": [vmin, vmax],
    }


def step_umap(page):
    t0 = now(page)
    page.locator("#projection-method").click()
    page.get_by_role("option", name="UMAP", exact=True).click()
    return t0, {}


def step_select_last_iteration(page):
    gd = "#bumpchart .js-plotly-plot"
    page.hover(gd)
    page.locator(f'#bumpchart .modebar-btn[data-val="lasso"]').click()
    box = page.evaluate(
        """(sel) => {
        const gd = document.querySelector(sel);
        const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
        const r = gd.getBoundingClientRect();
        // iterations present in the plotted traces (axis may be linear or category)
        const xs = new Set();
        for (const trace of gd.calcdata) for (const pt of trace) {
          if (pt.x !== undefined && pt.x !== null && !Number.isNaN(pt.x)) xs.add(pt.x);
        }
        const its = [...xs].sort((a, b) => a - b);
        const last = its[its.length - 1];
        const prev = its.length > 1 ? its[its.length - 2] : last - 1;
        const step = Math.abs(xa.c2p(last) - xa.c2p(prev));
        // the redraw after selection fires plotly_selected again; keep the max
        gd.on('plotly_selected', (e) => {
          const n = e && e.points ? e.points.length : 0;
          window.__bm.selected = Math.max(window.__bm.selected || 0, n);
        });
        return {
          x0: r.left + xa._offset + xa.c2p(last) - step / 2,
          x1: r.left + xa._offset + xa._length - 1,
          y0: r.top + ya._offset + 1,
          y1: r.top + ya._offset + ya._length - 1,
          last_iteration: xa.type === 'category' ? xa._categories[last] : last,
        };
    }""",
        gd,
    )
    m = page.mouse
    m.move(box["x0"], box["y0"])
    m.down()
    for x, y in [
        (box["x1"], box["y0"]),
        (box["x1"], box["y1"]),
        (box["x0"], box["y1"]),
        (box["x0"], box["y0"] + 1),
    ]:
        m.move(x, y, steps=10)
    t0 = now(page)
    m.up()
    return t0, {"last_iteration": box["last_iteration"]}


def step_dag_tab(page):
    t0 = now(page)
    page.locator("#tab-dag-view").click()
    return t0, {}


STEPS = [
    ("iterations", step_iterations),
    ("umap", step_umap),
    ("select", step_select_last_iteration),
    ("dag_tab", step_dag_tab),
]


# ----------------------------------------------------------------------------- run
def prepare_run_copy(run_dir, tmp):
    dst = Path(tmp) / run_dir.name
    shutil.copytree(run_dir / ".hydra", dst / ".hydra")
    (dst / "visdata").mkdir(parents=True)
    shutil.copyfile(run_dir / "visdata" / "data.db", dst / "visdata" / "data.db")
    return dst


def one_repeat(args, run_dir, launcher, pythonpath, shot_dir, log_copy):
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        run_copy = prepare_run_copy(run_dir, tmp)
        env = dict(os.environ, PORT=str(args.port), PYTHONUNBUFFERED="1")
        # non-GUI matplotlib backend: with the default Tk backend the env image
        # functions crash the server when called from Flask worker threads
        env.setdefault("MPLBACKEND", "Agg")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(pythonpath)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        log = open(Path(tmp) / "server.log", "w")
        t_start = time.perf_counter()
        server = subprocess.Popen(
            [args.python, str(launcher), "--path", str(run_copy), *args.launcher_args],
            env=env,
            cwd=str(pythonpath),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        mon = MemoryMonitor(server.pid)
        mon.start()
        url = f"http://127.0.0.1:{args.port}/"
        try:
            while True:
                if server.poll() is not None:
                    log.close()
                    raise RuntimeError(
                        "dashboard exited:\n" + (Path(tmp) / "server.log").read_text()
                    )
                try:
                    urllib.request.urlopen(url, timeout=1)
                    break
                except Exception:
                    time.sleep(0.05)
            res["startup"] = {"server_ready_s": time.perf_counter() - t_start}
            time.sleep(1.0)  # let the sampler record the idle server
            res["startup"]["server_rss_idle"] = mon.last.get("startup", {}).get("server")

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=not args.headed)
                ctx = browser.new_context(viewport=VIEWPORT)
                ctx.add_init_script(INSTRUMENT_JS)
                page = ctx.new_page()
                cdp = ctx.new_cdp_session(page)
                cdp.send("Performance.enable")

                def heap():
                    m = cdp.send("Performance.getMetrics")["metrics"]
                    m = {x["name"]: x["value"] for x in m}
                    return {"js_heap_used": m.get("JSHeapUsedSize"),
                            "js_heap_total": m.get("JSHeapTotalSize")}

                def finish(name, t0, extra):
                    wait_quiet(page, t0)
                    r = collect(page, t0)
                    r.update(extra)
                    r.update(heap())
                    r["mem_peak"] = dict(mon.peaks.get(name, {}))
                    r["mem_end"] = dict(mon.last.get(name, {}))
                    if shot_dir:
                        page.screenshot(path=str(shot_dir / f"{name}.png"))
                    return r

                mon.phase = "cold"
                page.goto(url, wait_until="commit")
                res["cold"] = finish("cold", 0.0, {})
                for name, fn in STEPS:
                    mon.phase = name
                    t0, extra = fn(page)
                    res[name] = finish(name, t0, extra)
                    if name == "select":
                        res[name]["selected_points"] = page.evaluate(
                            "() => window.__bm.selected"
                        )
                browser.close()
        finally:
            mon.stop()
            exit_before_stop = server.poll()  # None = server was still alive
            server.terminate()
            try:
                server.wait(10)
            except subprocess.TimeoutExpired:
                server.kill()
            log.close()
            res["server_exit_code_before_stop"] = exit_before_stop
            if log_copy:
                shutil.copyfile(Path(tmp) / "server.log", log_copy)
    return res


# ----------------------------------------------------------------------------- report
# (phase, key path, label); times are from the user action to the event
REPORT = [
    ("startup", "server_ready_s", "server start -> ready [s]"),
    ("cold", "end_to_end_ms", "cold load: all views done [ms]"),
    ("cold", "bumpchart_ms", "cold load: sample ranking drawn [ms]"),
    ("cold", "state_space_ms", "cold load: state projection drawn [ms]"),
    ("iterations", "bumpchart_ms", "iterations: sample ranking drawn [ms]"),
    ("iterations", ("cb_bumpchart", "roundtrip_ms_total"),
     "iterations: sample ranking callback [ms]"),
    ("iterations", "state_space_ms", "iterations: state projection drawn [ms]"),
    ("iterations", ("cb_state_space", "roundtrip_ms_total"),
     "iterations: state projection callback [ms]"),
    ("iterations", "end_to_end_ms", "iterations: all views done [ms]"),
    ("umap", "state_space_ms", "UMAP: state projection drawn [ms]"),
    ("umap", ("cb_state_space", "roundtrip_ms_total"),
     "UMAP: state projection callback [ms]"),
    ("select", "selected_points", "selection: lasso points"),
    ("select", ("cb_dag", "end_ms"), "selection: DAG callback answered [ms]"),
    ("select", "dag_layoutstop_ms", "selection: DAG layout done [ms]"),
    ("select", "dag_elements", "selection: DAG elements"),
    ("select", "end_to_end_ms", "selection: all views done [ms]"),
    ("dag_tab", "end_to_end_ms", "DAG tab: DAG drawn [ms]"),
]


def get(d, *keys):
    flat = []
    for k in keys:
        flat.extend(k if isinstance(k, tuple) else (k,))
    for k in flat:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def summarize(repeats):
    summary = {}
    for phase, key, label in REPORT:
        summary[label] = stats([get(r, phase, key) for r in repeats])
    for phase in ["cold", "iterations", "umap", "select", "dag_tab"]:
        for who in ["server", "browser"]:
            summary[f"{phase}: peak RSS {who} [MB]"] = stats(
                [(get(r, phase, "mem_peak", who) or 0) / 2**20 or None for r in repeats]
            )
        summary[f"{phase}: JS heap used [MB]"] = stats(
            [(get(r, phase, "js_heap_used") or 0) / 2**20 or None for r in repeats]
        )
    summary["server RSS idle after start [MB]"] = stats(
        [(get(r, "startup", "server_rss_idle") or 0) / 2**20 or None for r in repeats]
    )
    return summary


def system_info():
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_logical": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "viewport": VIEWPORT,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="run dir with .hydra and visdata")
    ap.add_argument("--launcher", default=str(DEFAULT_LAUNCHER))
    ap.add_argument("--python", default=sys.executable, help="interpreter for server")
    ap.add_argument("--pythonpath", default=None, help="default: launcher repo root")
    ap.add_argument("--port", type=int, default=8061)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--screenshots", action="store_true",
                    help="save a screenshot after each phase of the first repeat")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("launcher_args", nargs="*", help="extra args after --, e.g. --default-s0")
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    launcher = Path(args.launcher).resolve()
    pythonpath = Path(args.pythonpath) if args.pythonpath else find_pythonpath(launcher)
    name = args.name or run_dir.name
    if not port_free(args.port):
        sys.exit(f"port {args.port} is in use, choose another one with --port")

    out = Path(args.out) if args.out else (
        REPO_ROOT / "benchmarks" / "results" / f"{name}_dashboard.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    repeats = []
    for i in range(args.repeats):
        shot_dir = None
        if args.screenshots and i == 0:
            shot_dir = out.parent / f"{name}_screenshots"
            shot_dir.mkdir(exist_ok=True)
        print(f"[{name}] repeat {i + 1}/{args.repeats}", file=sys.stderr)
        log_dir = out.parent / f"{name}_server_logs"
        log_dir.mkdir(exist_ok=True)
        repeats.append(one_repeat(args, run_dir, launcher, pythonpath, shot_dir,
                                  log_dir / f"repeat{i}.log"))

    # failed callbacks / crashed servers make the affected timings meaningless
    errors = []
    for i, r in enumerate(repeats):
        if r.get("server_exit_code_before_stop") is not None:
            errors.append(f"repeat {i}: server exited "
                          f"(code {r['server_exit_code_before_stop']})")
        for phase, v in r.items():
            if not isinstance(v, dict):
                continue
            for c in v.get("callbacks", []):
                if c["status"] != 200:
                    errors.append(f"repeat {i}, {phase}: HTTP {c['status']} for "
                                  f"{c['output'][:60]}")

    result = {
        "name": name,
        "run": str(run_dir),
        "launcher": str(launcher),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "system": system_info(),
        "quiet_ms": QUIET_MS,
        "errors": errors,
        "summary": summarize(repeats),
        "repeats": repeats,
    }
    out.write_text(json.dumps(result, indent=2))
    if errors:
        print(f"\nWARNING: {len(errors)} errors, see server logs in {log_dir}:",
              file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)

    print(f"\n== {name} ({args.repeats} repeats, mean +/- std) ==")
    for label, s in result["summary"].items():
        if s is None:
            print(f"{label:45s} n/a")
        else:
            print(f"{label:45s} {s['mean']:10.1f} +/- {s['std']:.1f}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
