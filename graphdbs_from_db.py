
# Sentinel marking a node with more than one distinct predecessor/successor
_MULTI = object()

# Number of edge rows buffered before an executemany
_BATCH_SIZE = 50_000

_TRAJECTORY_QUERY = """
    SELECT
        final_id,
        step,
        final_object,
        text,
        iteration,
        total_reward,
        logprobs_forward,
        logprobs_backward
    FROM trajectories
    ORDER BY final_id, step
"""


def create_tables(conn):
    """
    (Re)create nodes and edges tables. The graph is always rebuilt from all trajectories,
    since new trajectories can change which nodes are removable, so existing tables are dropped.
    """
    conn.execute("DROP TABLE IF EXISTS edges")
    conn.execute("DROP TABLE IF EXISTS nodes")
    conn.execute("""
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            node_type TEXT,
            reward REAL
        )
    """)

    conn.execute("""
        CREATE TABLE edges (
            id TEXT,
            source TEXT,
            target TEXT,
            trajectory_id INTEGER,
            iteration INTEGER,
            logprobs_forward REAL,
            logprobs_backward REAL
        )
    """)
    conn.commit()


def insert_root_node(conn):
    conn.execute("""
        INSERT OR IGNORE INTO nodes (id, node_type, reward)
        VALUES ('#', 'start', NULL)
    """)
    conn.commit()


def create_indexes(conn):
    """Create indexes on the graph tables"""
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)")
    conn.commit()


def _add_neighbor(neighbors, node, other):
    """Record `other` as neighbor of `node`; collapse to _MULTI once a second distinct one appears."""
    current = neighbors.get(node)
    if current is None:
        neighbors[node] = other
    elif current is not _MULTI and current != other:
        neighbors[node] = _MULTI


def _collect_structure(conn):
    """
    Pass 1: scan all trajectories once and collect
      - nodes: text -> reward (reward of the last final occurrence, else None)
      - final: set of nodes that are final in at least one trajectory
      - pred / succ: node -> unique predecessor / successor, or _MULTI
    """
    nodes = {}
    final = set()
    pred = {}
    succ = {}

    current_trajectory = None
    prev_text = None
    for trajectory_id, _, final_object, text, _, total_reward, _, _ in conn.execute(_TRAJECTORY_QUERY):
        if trajectory_id != current_trajectory:
            current_trajectory = trajectory_id
            prev_text = "#"

        if final_object == 1:
            final.add(text)
            nodes[text] = total_reward
        elif text not in nodes:
            nodes[text] = None

        _add_neighbor(succ, prev_text, text)
        _add_neighbor(pred, text, prev_text)
        prev_text = text

    return nodes, final, pred, succ


def _find_removable(nodes, final, pred, succ):
    """A node is removable if it is not final and has exactly one distinct predecessor and successor."""
    removable = set()
    for node in nodes:
        if node in final:
            continue
        p = pred.get(node)
        s = succ.get(node)
        if p is not None and p is not _MULTI and s is not None and s is not _MULTI:
            removable.add(node)
    return removable


def _resolve_kept(node, removable, succ):
    """Follow unique successors from `node` until a kept node is reached (None on a cycle)."""
    visited = set()
    while node in removable:
        if node in visited:
            return None
        visited.add(node)
        node = succ[node]
    return node


def _write_compressed_edges(conn, removable, succ):
    """
    Pass 2: re-scan trajectories and write one edge per segment between consecutive kept
    nodes. Removable nodes are skipped; their logprobs are summed into the segment edge.
    The segment edge keeps id and iteration of the edge leaving the kept source node.
    Returns the number of edges written.
    """
    insert_sql = """
        INSERT INTO edges (
            id,
            source,
            target,
            trajectory_id,
            iteration,
            logprobs_forward,
            logprobs_backward
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    batch = []
    written = 0

    def flush_tail(trajectory_id, seg, last_text):
        # Trajectory ended on a removable node: extend to the next kept node on its chain
        target = _resolve_kept(last_text, removable, succ)
        if target is not None:
            batch.append((seg[0], seg[1], target, trajectory_id, seg[2], seg[3], seg[4]))

    current_trajectory = None
    source = None
    seg = None  # [edge_id, source, iteration, logpf, logpb] of the open segment
    last_text = None

    for trajectory_id, step, _, text, iteration, _, logpf, logpb in conn.execute(_TRAJECTORY_QUERY):
        if trajectory_id != current_trajectory:
            if seg is not None:
                flush_tail(current_trajectory, seg, last_text)
            current_trajectory = trajectory_id
            source = "#"
            seg = None

        if seg is None:
            seg = [f"{trajectory_id}_{step}", source, iteration, logpf, logpb]
        else:
            # Same semantics as SQL `a + COALESCE(b, 0)`: NULL in the first edge propagates
            if seg[3] is not None:
                seg[3] += logpf if logpf is not None else 0
            if seg[4] is not None:
                seg[4] += logpb if logpb is not None else 0

        if text not in removable:
            batch.append((seg[0], seg[1], text, trajectory_id, seg[2], seg[3], seg[4]))
            source = text
            seg = None

        last_text = text

        if len(batch) >= _BATCH_SIZE:
            conn.executemany(insert_sql, batch)
            written += len(batch)
            batch.clear()

    if seg is not None:
        flush_tail(current_trajectory, seg, last_text)

    if batch:
        conn.executemany(insert_sql, batch)
        written += len(batch)

    return written


def print_stats(conn):
    """Print graph statistics"""
    node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    node_types = conn.execute("""
        SELECT node_type, COUNT(*)
        FROM nodes
        GROUP BY node_type
    """).fetchall()

    print("\n=== Graph Statistics ===")
    print(f"Total nodes: {node_count}")
    print(f"Total edges: {edge_count}")
    print("Node types:")
    for node_type, count in node_types:
        print(f"  {node_type}: {count}")


def create_graph_dbs(conn):
    """
    Build the truncated trajectory graph in two passes over the trajectories table.
    Nodes with exactly one distinct predecessor and successor (and never final) are
    bypassed directly, so the untruncated graph is never written.
    """
    create_tables(conn)
    insert_root_node(conn)

    print("Pass 1: collecting graph structure...")
    nodes, final, pred, succ = _collect_structure(conn)
    removable = _find_removable(nodes, final, pred, succ)
    print(f"  Found {len(nodes)} nodes, {len(removable)} removable")

    print("Writing nodes...")
    node_rows = (
        (text, "final", reward) if text in final else (text, "standard", None)
        for text, reward in nodes.items()
        if text not in removable
    )
    conn.executemany("""
        INSERT OR IGNORE INTO nodes (id, node_type, reward)
        VALUES (?, ?, ?)
    """, node_rows)

    print("Pass 2: writing truncated edges...")
    edges_written = _write_compressed_edges(conn, removable, succ)
    print(f"  Wrote {edges_written} edges")
    conn.commit()

    print("\nCreating indexes...")
    create_indexes(conn)

    print_stats(conn)

    print("\nDone!")


if __name__ == "__main__":
    import sqlite3
    # Connect to your database
    conn = sqlite3.connect("debugdata/data.db")

    try:
        create_graph_dbs(conn)
    finally:
        conn.close()
