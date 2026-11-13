"""
Benchmark B: data and preprocessing metrics of a logged run.

Reports, for one vislogger data.db:
    - database size (file, pages, per table if SQLite has the dbstat table)
    - number of trajectories, logged states, distinct states, final objects
    - number of nodes and edges of the graph (before / after truncation)
    - time and memory of create_graph_dbs (graph construction + truncation)

The graph is rebuilt on a temporary copy of the db (nodes/edges dropped), so the
input file is never modified.

Usage:
    python benchmarks/data_metrics.py --db path/to/visdata/data.db
        [--repeats 3] [--graph-module path/to/graphdbs_from_db.py] [--out file.json]
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import platform
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:  # memory is optional
    psutil = None

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_graph_module(path):
    spec = importlib.util.spec_from_file_location("graphdbs_bench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scalar(conn, query, params=()):
    return conn.execute(query, params).fetchone()[0]


def table_exists(conn, name):
    return (
        scalar(
            conn,
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        )
        > 0
    )


def size_metrics(conn, path):
    page_size = scalar(conn, "PRAGMA page_size")
    res = {
        "file_bytes": os.path.getsize(path),
        "page_size": page_size,
        "page_count": scalar(conn, "PRAGMA page_count"),
        "freelist_bytes": scalar(conn, "PRAGMA freelist_count") * page_size,
    }
    try:
        rows = conn.execute(
            "SELECT name, SUM(pgsize) FROM dbstat GROUP BY name ORDER BY 2 DESC"
        ).fetchall()
        tables = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT name, tbl_name FROM sqlite_master WHERE type IN ('table','index')"
            )
        }
        per_table = {}
        for name, size in rows:
            owner = tables.get(name, name)  # indexes are attributed to their table
            per_table[owner] = per_table.get(owner, 0) + size
        res["bytes_per_table_incl_indexes"] = per_table
    except sqlite3.OperationalError:
        res["bytes_per_table_incl_indexes"] = None  # dbstat not compiled in
    return res


def data_metrics(conn):
    q = lambda s: scalar(conn, s)  # noqa: E731
    res = {
        "trajectories": q("SELECT COUNT(DISTINCT final_id) FROM trajectories"),
        "logged_state_rows": q("SELECT COUNT(*) FROM trajectories"),
        "distinct_states": q("SELECT COUNT(DISTINCT text) FROM trajectories"),
        "final_objects": q("SELECT COUNT(*) FROM trajectories WHERE final_object=1"),
        "distinct_final_objects": q(
            "SELECT COUNT(DISTINCT text) FROM trajectories WHERE final_object=1"
        ),
        "logged_iterations": q("SELECT COUNT(DISTINCT iteration) FROM trajectories"),
        "iteration_min": q("SELECT MIN(iteration) FROM trajectories"),
        "iteration_max": q("SELECT MAX(iteration) FROM trajectories"),
        "max_trajectory_length": q(
            "SELECT MAX(n) FROM (SELECT COUNT(*) n FROM trajectories GROUP BY final_id)"
        ),
        "testset_rows": (
            q("SELECT COUNT(*) FROM testset") if table_exists(conn, "testset") else 0
        ),
    }
    cols = [r[1] for r in conn.execute("PRAGMA table_info(trajectories)")]
    if "features_valid" in cols:  # old layout: one column per feature
        res["feature_dims"] = len(cols) - cols.index("features_valid") - 1
    elif table_exists(conn, "feature_info"):  # one feature vector (BLOB) per object
        names = conn.execute("SELECT value FROM feature_info WHERE key = 'names'").fetchone()
        if names:
            res["feature_dims"] = len(json.loads(names[0]))
    return res


def graph_metrics(conn):
    if not (table_exists(conn, "nodes") and table_exists(conn, "edges")):
        return None
    q = lambda s: scalar(conn, s)  # noqa: E731
    return {
        "nodes": q("SELECT COUNT(*) FROM nodes"),
        "final_nodes": q("SELECT COUNT(*) FROM nodes WHERE node_type='final'"),
        "edges": q("SELECT COUNT(*) FROM edges"),
        # same transition in several trajectories/iterations is one edge in the DAG
        "distinct_edges": q(
            "SELECT COUNT(*) FROM (SELECT DISTINCT source, target FROM edges)"
        ),
    }


def untruncated_graph_metrics(conn):
    """Graph before truncation, computed from the trajectories (one node per
    distinct state plus the root, one edge per logged step)."""
    q = lambda s: scalar(conn, s)  # noqa: E731
    return {
        "nodes": q("SELECT COUNT(DISTINCT text) FROM trajectories") + 1,
        "final_nodes": q(
            "SELECT COUNT(DISTINCT text) FROM trajectories WHERE final_object=1"
        ),
        "edges": q("SELECT COUNT(*) FROM trajectories"),
        "distinct_edges": q("""
            SELECT COUNT(*) FROM (
                SELECT DISTINCT
                    COALESCE(LAG(text) OVER (PARTITION BY final_id ORDER BY step), '#'),
                    text
                FROM trajectories
            )"""),
    }


class PeakRSS:
    """Peak resident memory of this process (Windows: peak working set)."""

    def __init__(self):
        self.proc = psutil.Process() if psutil else None

    def rss(self):
        return self.proc.memory_info().rss if self.proc else None

    def peak(self):
        if not self.proc:
            return None
        mi = self.proc.memory_info()
        return getattr(mi, "peak_wset", None)  # only exposed on Windows


# stage functions of known create_graph_dbs implementations; timed if present
STAGES = {
    "truncate_graph": "truncate",  # old: build full graph in SQL, then truncate
    "_collect_structure": "pass1_structure",  # new: scan, find removable nodes
    "_write_compressed_edges": "pass2_edges",  # new: write truncated edges
}


def time_graph_build(src_db, graph_mod, workdir):
    """Rebuild nodes/edges on a copy of src_db. Returns timings and graph sizes."""
    db = Path(workdir) / "data.db"
    shutil.copyfile(src_db, db)
    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE IF EXISTS edges")
    conn.execute("DROP TABLE IF EXISTS nodes")
    conn.commit()
    conn.execute("VACUUM")  # size of the db without graph tables
    size_without_graph = os.path.getsize(db)

    stage_s = {}
    originals = {}

    def timed(fn, label):
        def wrapper(*a, **kw):
            t = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                stage_s[label] = stage_s.get(label, 0.0) + time.perf_counter() - t
        return wrapper

    for fname, label in STAGES.items():
        if hasattr(graph_mod, fname):
            originals[fname] = getattr(graph_mod, fname)
            setattr(graph_mod, fname, timed(originals[fname], label))

    mem = PeakRSS()
    rss_before = mem.rss()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            graph_mod.create_graph_dbs(conn)
            total = time.perf_counter() - t0
    finally:
        for fname, fn in originals.items():
            setattr(graph_mod, fname, fn)
    rss_after = mem.rss()

    truncated = graph_metrics(conn)
    conn.execute("VACUUM")
    conn.close()
    stage_s["other"] = total - sum(stage_s.values())
    return {
        "total_s": total,
        "stages_s": stage_s,
        "graph_truncated": truncated,
        "db_bytes_without_graph": size_without_graph,
        "db_bytes_with_graph": os.path.getsize(db),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "process_peak_rss_bytes": mem.peak(),
    }


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "runs": values,
    }


def system_info():
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
    }
    if psutil:
        info["cpu_logical"] = psutil.cpu_count()
        info["ram_bytes"] = psutil.virtual_memory().total
    return info


def fmt_bytes(b):
    if b is None:
        return "n/a"
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", required=True, help="vislogger data.db")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--graph-module",
        default=str(REPO_ROOT / "graphdbs_from_db.py"),
        help="file providing create_graph_dbs",
    )
    parser.add_argument("--name", default=None, help="run name for the output file")
    parser.add_argument("--out", default=None, help="output json")
    args = parser.parse_args()

    db = Path(args.db).resolve()
    name = args.name or db.parent.parent.name
    graph_mod = load_graph_module(args.graph_module)

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    result = {
        "name": name,
        "db": str(db),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "system": system_info(),
        "graph_module": str(Path(args.graph_module).resolve()),
        "size": size_metrics(conn, db),
        "data": data_metrics(conn),
        "graph_in_db": graph_metrics(conn),
        "graph_untruncated": untruncated_graph_metrics(conn),
    }
    conn.close()

    runs = []
    for i in range(args.repeats):
        with tempfile.TemporaryDirectory() as tmp:
            r = time_graph_build(db, graph_mod, tmp)
        runs.append(r)
        print(f"[{name}] create_graph_dbs run {i + 1}/{args.repeats}: "
              f"{r['total_s']:.2f} s", file=sys.stderr)

    last = runs[-1]
    stage_names = sorted({k for r in runs for k in r["stages_s"]})
    result["create_graph_dbs"] = {
        "total_s": summarize([r["total_s"] for r in runs]),
        "stages_s": {
            k: summarize([r["stages_s"].get(k, 0.0) for r in runs]) for k in stage_names
        },
        "graph_truncated": last["graph_truncated"],
        "db_bytes_without_graph": last["db_bytes_without_graph"],
        "db_bytes_with_graph": last["db_bytes_with_graph"],
        "rss_growth_bytes": [
            (r["rss_after_bytes"] - r["rss_before_bytes"])
            if r["rss_before_bytes"] is not None
            else None
            for r in runs
        ],
        "process_peak_rss_bytes": max(
            (r["process_peak_rss_bytes"] or 0) for r in runs
        )
        or None,
    }

    out = Path(args.out) if args.out else REPO_ROOT / "benchmarks" / "results" / (
        f"{name}_data.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    d, g, s = result["data"], result["create_graph_dbs"], result["size"]
    gu, gt = result["graph_untruncated"], g["graph_truncated"]
    print(f"\n== {name} ==")
    print(f"db size              {fmt_bytes(s['file_bytes'])}")
    print(f"trajectories         {d['trajectories']}")
    print(f"logged states        {d['logged_state_rows']}  "
          f"(distinct {d['distinct_states']})")
    print(f"final objects        {d['final_objects']}  "
          f"(distinct {d['distinct_final_objects']})")
    print(f"iterations logged    {d['logged_iterations']}  "
          f"[{d['iteration_min']}, {d['iteration_max']}]")
    print(f"graph untruncated    {gu['nodes']} nodes, {gu['edges']} edges "
          f"({gu['distinct_edges']} distinct)")
    print(f"graph truncated      {gt['nodes']} nodes, {gt['edges']} edges "
          f"({gt['distinct_edges']} distinct)")
    stages = ", ".join(
        f"{k} {v['mean']:.2f}" for k, v in g["stages_s"].items()
    )
    print(f"create_graph_dbs     {g['total_s']['mean']:.2f} +/- "
          f"{g['total_s']['std']:.2f} s  ({stages})")
    print(f"db w/o / with graph  {fmt_bytes(g['db_bytes_without_graph'])} / "
          f"{fmt_bytes(g['db_bytes_with_graph'])}")
    print(f"process peak RSS     {fmt_bytes(g['process_peak_rss_bytes'])}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
