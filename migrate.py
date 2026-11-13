"""
Converts a vis database from the old feature layout (one column per feature in the
trajectories and testset tables) to one feature vector per final object, stored as
BLOB (see "Feature storage" in logger.py).

The converted database replaces data.db, the original is kept as data_pre_blob.db in
the same folder. All other tables (nodes, edges, ...) and the indexes are copied
unchanged. Tables written by older dashboard versions (current_dp, temp_updates) are
dropped.

Self-contained (no torch needed), the encoding is the same as in logger.py.

Usage:
    python migrate.py path [path ...]
        path: folder containing data.db, or the data.db file itself
"""
import argparse
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def _choose_dtype(matrix):
    """uint8 if all values are integers in [0, 255], otherwise float32 (as in logger.py)."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.size == 0:
        return "uint8"
    if np.isfinite(m).all() and m.min() >= 0 and m.max() <= 255 and (m == np.round(m)).all():
        return "uint8"
    return "float32"


def _set_info(conn, key, value):
    if key == "names":
        value = json.dumps(list(value))
    conn.execute("INSERT OR REPLACE INTO feature_info (key, value) VALUES (?, ?)", (key, value))


def _ensure_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS features (
            final_id INTEGER PRIMARY KEY,
            features_valid INTEGER NOT NULL,
            vec BLOB
        )
    """)
    conn.execute("CREATE TABLE IF NOT EXISTS feature_info (key TEXT PRIMARY KEY, value TEXT)")


def _old_layout(conn):
    traj_cols = [r[1] for r in conn.execute("PRAGMA table_info(trajectories)")]
    testset_cols = [r[1] for r in conn.execute("PRAGMA table_info(testset)")]
    return "features_valid" in traj_cols or any(
        c.startswith("f_") and c[2:].isdigit() for c in testset_cols
    )


def _wider_dtype(a, b):
    return "float32" if "float32" in (a, b) else "uint8"


def _encode(matrix, dtype):
    """Each row as raw bytes (as encode_features in logger.py)."""
    return [row.tobytes() for row in np.ascontiguousarray(np.asarray(matrix).astype(dtype))]


BACKUP_NAME = "data_pre_blob.db"
SKIP_TABLES = {"current_dp", "temp_updates"}  # written by older dashboard versions
CHUNK = 2000


def _q(name):
    return '"' + str(name).replace('"', '""') + '"'


def _columns(conn, table):
    """(name, declared type, pk position) of each column"""
    return [(r[1], r[2], r[5]) for r in conn.execute(f"PRAGMA table_info({_q(table)})")]


def _create_table(conn, table, columns, extra=()):
    """Creates a table with the given (name, type, pk) columns plus extra column definitions."""
    pks = [c[0] for c in sorted(columns, key=lambda c: c[2]) if c[2] > 0]
    defs = [f"{_q(n)} {t}".strip() for n, t, _ in columns] + list(extra)
    if pks:
        defs.append(f"PRIMARY KEY ({', '.join(_q(p) for p in pks)})")
    conn.execute(f"CREATE TABLE {_q(table)} ({', '.join(defs)})")


def _feature_dtype(src, query):
    """dtype for the vectors: uint8 if all values are integers in [0, 255], else float32."""
    dtype = "uint8"
    for chunk in pd.read_sql_query(query, src, chunksize=CHUNK):
        dtype = _wider_dtype(dtype, _choose_dtype(chunk.to_numpy(dtype=float)))
        if dtype == "float32":
            break
    return dtype


def _migrate_trajectories(src, dst):
    """trajectories without feature columns, features of final objects as BLOBs."""
    columns = _columns(src, "trajectories")
    names = [c[0] for c in columns]
    i_valid = names.index("features_valid")
    kept, feature_names = columns[:i_valid], names[i_valid + 1:]

    _create_table(dst, "trajectories", kept)
    kept_sql = ", ".join(_q(c[0]) for c in kept)
    dst.execute(f"INSERT INTO main.trajectories ({kept_sql}) SELECT {kept_sql} FROM old.trajectories")

    if not feature_names:
        return 0, None
    feature_sql = ", ".join(_q(n) for n in feature_names)
    dtype = _feature_dtype(src, f"SELECT {feature_sql} FROM trajectories WHERE final_object = 1")
    _ensure_tables(dst)
    _set_info(dst, "names", feature_names)
    _set_info(dst, "dtype", dtype)
    query = f"SELECT final_id, features_valid, {feature_sql} FROM trajectories WHERE final_object = 1"
    n = 0
    for chunk in pd.read_sql_query(query, src, chunksize=CHUNK):
        vecs = _encode(chunk[feature_names].to_numpy(dtype=float), dtype)
        rows = zip(
            (int(i) for i in chunk["final_id"]),
            (int(v == 1) for v in chunk["features_valid"]),
            vecs,
        )
        dst.executemany("INSERT INTO features (final_id, features_valid, vec) VALUES (?, ?, ?)", rows)
        n += len(chunk)
    return n, dtype


def _migrate_testset(src, dst):
    """testset with one BLOB column 'features' instead of the f_i columns."""
    columns = _columns(src, "testset")
    f_cols = sorted(
        (c[0] for c in columns if c[0].startswith("f_") and c[0][2:].isdigit()),
        key=lambda c: int(c[2:]),
    )
    kept = [c for c in columns if c[0] not in f_cols and c[0] != "features"]
    _create_table(dst, "testset", kept, extra=['"features" BLOB'])
    kept_names = [c[0] for c in kept]
    if not f_cols:
        kept_sql = ", ".join(_q(n) for n in kept_names)
        dst.execute(f"INSERT INTO main.testset ({kept_sql}) SELECT {kept_sql} FROM old.testset")
        return 0, None

    f_sql = ", ".join(_q(c) for c in f_cols)
    dtype = _feature_dtype(src, f"SELECT {f_sql} FROM testset")
    _ensure_tables(dst)
    _set_info(dst, "testset_dtype", dtype)
    query = f"SELECT {', '.join(_q(n) for n in kept_names)}, {f_sql} FROM testset"
    n = 0
    for chunk in pd.read_sql_query(query, src, chunksize=CHUNK):
        out = chunk[kept_names].copy()
        out["features"] = _encode(chunk[f_cols].to_numpy(dtype=float), dtype)
        out.to_sql("testset", dst, if_exists="append", index=False)
        n += len(chunk)
    return n, dtype


def migrate(path):
    """
    Converts one database. Returns True if it was converted, False if it already uses
    the new layout.
    :param path: folder containing data.db or the db file
    """
    db = Path(path)
    if db.is_dir():
        db = db / "data.db"
    if not db.is_file():
        raise FileNotFoundError(db)
    src_uri = f"{db.resolve().as_uri()}?mode=ro"
    src = sqlite3.connect(src_uri, uri=True)
    if not _old_layout(src):
        src.close()
        print(f"{db}: already in the new layout, nothing to do")
        return False

    backup = db.with_name(BACKUP_NAME)
    tmp = db.with_name(db.name + ".migrating")
    if backup.exists():
        src.close()
        raise FileExistsError(f"{backup} exists, remove or rename it first")
    if tmp.exists():
        tmp.unlink()

    dst = sqlite3.connect(tmp.resolve().as_uri(), uri=True)
    dst.execute("ATTACH DATABASE ? AS old", (src_uri,))
    tables = [
        (name, sql) for name, sql in src.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    report = []
    for name, sql in tables:
        if name in SKIP_TABLES:
            report.append(f"dropped {name}")
        elif name == "trajectories":
            n, dtype = _migrate_trajectories(src, dst)
            report.append(f"trajectories: {n} feature vectors ({dtype})")
        elif name == "testset":
            n, dtype = _migrate_testset(src, dst)
            report.append(f"testset: {n} feature vectors ({dtype})")
        else:
            dst.execute(sql)
            dst.execute(f"INSERT INTO main.{_q(name)} SELECT * FROM old.{_q(name)}")
    for name, table, sql in src.execute(
        "SELECT name, tbl_name, sql FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL"
    ):
        if table in SKIP_TABLES:
            continue
        try:
            dst.execute(sql)
        except sqlite3.OperationalError as e:  # index on a removed feature column
            report.append(f"skipped index {name}: {e}")
    dst.commit()
    dst.execute("DETACH DATABASE old")
    dst.close()
    src.close()

    try:
        os.replace(db, backup)
    except PermissionError as e:  # Windows: the db is open in another program
        tmp.unlink()
        raise RuntimeError(
            f"{db} is opened by another program (e.g. a running dashboard or an IDE), "
            "close it and run again"
        ) from e
    os.replace(tmp, db)
    print(f"{db}: converted ({'; '.join(report)}), original kept as {backup}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="folders containing data.db, or db files")
    args = parser.parse_args()
    for path in args.paths:
        migrate(path)


if __name__ == "__main__":
    main()
