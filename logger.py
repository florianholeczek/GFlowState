import sqlite3
import warnings
import pandas as pd
from typing import Callable
from datetime import datetime
import os
import torch
import numpy as np
from graphdbs_from_db import create_graph_dbs
import json

"""
Use the provided logger to log n on-policy samples every m iterations.
The logger saves the data in a SQLite database to be queried by the dashboard.
Note that for running the dashboard you will also need a text-to-image function 
to calculate the image representation of a state.

Functions:
    - log() stores all given data in a df.
    - write_to_db() writes the logged data to the database after logging a batch is done.
    - compute_graph() computes the dag from the trajectories. 
        By default this is done each time write_to_db is called.
        You can disable this behavior to save resources and call compute_graph() after training is finished if you do not need the data while training.
    create_and_append_testset() allows you to write batches of data to the testset in the expected format


How to use:
    Before starting training create a VisLogger instance.
    You will need:
        -a path where the data is stored
        -a function that converts your logged states to a text representation that allows for uniquely identifying each state
        -(optional) a function to compute features from the text representations
        -(optional) a list with the names of additional metrics to log (by default loss and total_reward are logged)
        -(optional) a list with features you will log
        -set the flag s0_included to show if you will pass your trajectories with the empty first state or if they start with s1
    
    in your training loop, call log() every m iterations. 
    You can use your existing training samples or sample new ones if you are training off policy.
        if iteration % m ==0:
            batch_idx, states, logprobs_forward, logprobs_backward = my_sampling(...)
            logger.log(batch_idx=batch_idx, states=states, logprobs_forward=logprobs_forward, logprobs_backward=logprobs_backward)
            loss = calc_my_loss(states)
            reward = calc_my_reward(states)
            logger.log(loss=loss, total_reward=reward)
            # If all neccessary data is logged:
            logger.write_to_db(compute_graph = False)
    As we did flagged compute_graph=False, we need to call compute_graph() once after training is done.
    If you have a testset you want to include you can then also call create_and_append_testset().
    Further detail on the expected parameters and shapes is in the function documentation.
    
Scalability: 
    The complete trajectories get saved, so the resulting rowcount is n * (total iterations / m) * average trajectory length.
    Keeping this below 1e6 should work fine.
    Rule of thumb: 1000 samples at 20 timepoints during training should work fine for trajectories below length 50.
"""

# ================= Feature storage =================
# Features are only stored for final objects (the dashboard only projects these),
# one row per object in the table
#     features(final_id INTEGER PRIMARY KEY, features_valid INTEGER, vec BLOB)
# vec holds the whole feature vector as raw bytes: the logged features first, then the
# computed ones. The testset stores its vectors the same way in the BLOB column
# testset.features.
# The table feature_info holds the feature names and the dtype of the vectors:
# uint8 if all values are integers in [0, 255] (e.g. fingerprints), otherwise float32.
# If a later batch does not fit uint8, the stored vectors are converted to float32.
# One BLOB per object avoids SQLite's limit of 2000 columns per table and keeps the
# database small. The dashboard reads it with decode_features() in plot_utils.py.
# Databases with the old layout (one column per feature) are converted by migrate.py.

FEATURES_UINT8 = "uint8"
FEATURES_FLOAT32 = "float32"

OLD_LAYOUT_MSG = (
    "{db} stores features in one column per feature (old layout). "
    "Convert it first: python migrate.py {db}"
)

_FEATURES_TABLE = """
    CREATE TABLE IF NOT EXISTS features (
        final_id INTEGER PRIMARY KEY,
        features_valid INTEGER NOT NULL,
        vec BLOB
    )
"""
_FEATURE_INFO_TABLE = "CREATE TABLE IF NOT EXISTS feature_info (key TEXT PRIMARY KEY, value TEXT)"


def features_to_matrix(x):
    """
    Converts features (np array, torch Tensor or nested list) to a 2D np array.
    :param x: features of shape (n_objects, n_features)
    :return: np array
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def choose_feature_dtype(matrix):
    """
    uint8 if all values are integers in [0, 255], otherwise float32.
    :param matrix: np array
    :return: dtype name
    """
    m = np.asarray(matrix)
    if m.dtype == bool:
        return FEATURES_UINT8
    m = m.astype(np.float64)
    if m.size == 0:
        return FEATURES_UINT8
    if np.isfinite(m).all() and m.min() >= 0 and m.max() <= 255 and (m == np.round(m)).all():
        return FEATURES_UINT8
    return FEATURES_FLOAT32


def wider_feature_dtype(a, b):
    """The dtype that can hold values of both dtypes."""
    return FEATURES_FLOAT32 if FEATURES_FLOAT32 in (a, b) else FEATURES_UINT8


def encode_features(matrix, dtype):
    """
    Encodes each row of the matrix as raw bytes.
    :param matrix: np array (n_objects, n_features)
    :param dtype: uint8 or float32
    :return: list of bytes
    """
    m = np.ascontiguousarray(np.asarray(matrix).astype(dtype))
    return [row.tobytes() for row in m]


def read_feature_info(conn):
    """
    Reads the feature info (names, dtype, testset_dtype).
    :param conn: sqlite connection
    :return: dict, empty if no features were stored
    """
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='feature_info'"
    ).fetchone()
    if not exists:
        return {}
    info = dict(conn.execute("SELECT key, value FROM feature_info").fetchall())
    if "names" in info:
        info["names"] = json.loads(info["names"])
    return info


def ensure_feature_tables(conn):
    """Creates the features and feature_info tables if they do not exist."""
    conn.execute(_FEATURES_TABLE)
    conn.execute(_FEATURE_INFO_TABLE)


def set_feature_info(conn, key, value):
    """Stores a feature info entry (names are stored as JSON)."""
    if key == "names":
        value = json.dumps(list(value))
    conn.execute("INSERT OR REPLACE INTO feature_info (key, value) VALUES (?, ?)", (key, value))


def _convert_stored_features(conn, table, id_col, blob_col, old, new):
    """Re-encodes the stored vectors of a table with a wider dtype."""
    rows = conn.execute(
        f"SELECT {id_col}, {blob_col} FROM {table} WHERE {blob_col} IS NOT NULL"
    ).fetchall()
    conn.executemany(
        f"UPDATE {table} SET {blob_col} = ? WHERE {id_col} = ?",
        [(np.frombuffer(b, dtype=old).astype(new).tobytes(), i) for i, b in rows],
    )


def write_features(conn, final_ids, features_valid, matrix, names):
    """
    Appends the feature vectors of final objects to the features table.
    Does not commit.
    :param conn: sqlite connection
    :param final_ids: final ids of the objects (as stored in trajectories)
    :param features_valid: bool per object
    :param matrix: features, shape (n_objects, n_features)
    :param names: feature names, must be the same for every batch
    :return: None
    """
    ensure_feature_tables(conn)
    info = read_feature_info(conn)
    names = list(names)
    if "names" in info and info["names"] != names:
        raise ValueError(f"Features differ from the ones logged before: {names} vs {info['names']}")
    dtype = choose_feature_dtype(matrix)
    if "dtype" in info:
        new = wider_feature_dtype(info["dtype"], dtype)
        if new != info["dtype"]:
            _convert_stored_features(conn, "features", "final_id", "vec", info["dtype"], new)
        dtype = new
    set_feature_info(conn, "names", names)
    set_feature_info(conn, "dtype", dtype)
    rows = zip(
        (int(i) for i in final_ids),
        (int(bool(v)) for v in features_valid),
        encode_features(matrix, dtype),
    )
    conn.executemany("INSERT INTO features (final_id, features_valid, vec) VALUES (?, ?, ?)", rows)


def encode_testset_features(conn, matrix, testset_exists):
    """
    Encodes testset feature vectors, converting already stored testset vectors to a
    wider dtype if needed. Does not commit.
    :param conn: sqlite connection
    :param matrix: features, shape (n_objects, n_features)
    :param testset_exists: whether the testset table exists already
    :return: list of bytes, one per object
    """
    conn.execute(_FEATURE_INFO_TABLE)
    info = read_feature_info(conn)
    dtype = choose_feature_dtype(matrix)
    if "testset_dtype" in info:
        new = wider_feature_dtype(info["testset_dtype"], dtype)
        if new != info["testset_dtype"] and testset_exists:
            _convert_stored_features(conn, "testset", "id", "features", info["testset_dtype"], new)
        dtype = new
    set_feature_info(conn, "testset_dtype", dtype)
    return encode_features(matrix, dtype)


def old_feature_layout(conn):
    """True if the db stores features in one column per feature (convert with migrate.py)."""
    traj_cols = [r[1] for r in conn.execute("PRAGMA table_info(trajectories)")]
    testset_cols = [r[1] for r in conn.execute("PRAGMA table_info(testset)")]
    return "features_valid" in traj_cols or any(
        c.startswith("f_") and c[2:].isdigit() for c in testset_cols
    )


class VisLogger:
    def __init__(
            self,
            path: str | None = None,
            s0_included: bool = True,
            fn_state_to_text: Callable | None = None,
            fn_compute_features: Callable | None = None,
            metrics: list[str] | None = None,
            features: list[str] | None = None
    ):
        """
        A Logger for the visualizations.
        :param path: path to a folder to save the data. If none creates one based on datetime in root.
        :param s0_included: if True, the trajectories are expected to have the (empty) start state included.
            The start states will then be removed before writing to the database.
             s0 will be specified as '#' in the visualizations.
        :param fn_state_to_text:
            Function to convert a batch of states to a list of readable strings to identify a single state.
            Neccessary, used to distinguish states.
            Consecutive states with the same text will be merged (logprobs will be added).
            s0 will be specified as '#' in the visualizations, make sure no state has the same identifier.
        :param fn_compute_features: Optional.
            Function to compute features from the states.
            It is only called with the final states (the objects) of each batch.
            Should take the states in the same format as given (torch Tensor, np array or list)
            and return a tuple consisting of:
                1. A np array or tensor of size (n_states, n_features) containing the features
                2. A list(bool) that tells for each state if the feature computation was successful.
                Objects with unsuccessful feature computation are not projected.
            These will be used for the downprojections.
            Features are stored as one vector per final object (see "Feature storage" above).
            Additional features can also be logged with the features parameter
        :param metrics: Optional.
            Names of additional metrics for final objects. Might be different losses or rewards.
            If the reward or loss function consists of multiple parts you can specify all of them here (list of strings).
            They will need to be logged each iteration.
            Otherwise only the total reward and the loss will be logged.
        :param features: Optional.
            If you want to log features, specify them here (list of strings).
            They will need to be logged each iteration.
            The features will be used for the downprojections.
            If features can be calculated from the states you can additionally use the fn_compute_features parameter.
        """

        if path:
            self.path = path
        else:
            self.path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(self.path)

        self.db = f"{self.path}/data.db"
        conn = sqlite3.connect(self.db)
        conn.close()
        self.metrics = metrics
        self.features = features
        self. s0_included = s0_included
        self.fn_state_to_text = fn_state_to_text
        self.fn_compute_features = fn_compute_features
        self.cols = [
            'final_id',
            'step',
            'final_object',
            'text',
            'iteration',
            'total_reward',
            'loss',
            *(metrics if metrics is not None else []),
            'logprobs_forward',
            'logprobs_backward',
        ]
        self.df = pd.DataFrame(columns=self.cols)

        # tracks the logged data
        self._reset_current()

        # Checks and Warnings
        assert self.fn_state_to_text is not None, "No fn_state_to_text provided. This is neccessary to distinguish states."

    def _reset_current(self):
        """
        Resets the logged data of the current block.
        Every key must be logged again before the next write_to_db().
        :return: None
        """
        self.current = {
            "batch_idx": None,
            "states": None,
            "total_reward": None,
            "loss": None,
            "iteration": None,
            "logprobs_forward": None,
            "logprobs_backward": None,
        }
        self.current.update({name: None for name in self.metrics or []})
        self.current.update({name: None for name in self.features or []})
        if self.features:
            self.current["features_valid_provided"] = None

    def attach_fns(self, fn_state_to_text=None, fn_compute_features=None):
        """
        Allows attaching the functions after initialization.
        """
        if fn_state_to_text is not None:
            self.fn_state_to_text = fn_state_to_text
        if fn_compute_features is not None:
            self.fn_compute_features = fn_compute_features

    def log(
            self,
            batch_idx: np.ndarray | torch.Tensor | None,
            states: np.ndarray | torch.Tensor | list | None,
            total_reward: np.ndarray | torch.Tensor | None,
            loss: np.ndarray | torch.Tensor | None,
            iteration: int | None,
            logprobs_forward: np.ndarray | torch.Tensor | None,
            logprobs_backward: np.ndarray | torch.Tensor | None,
            metrics: np.ndarray | torch.Tensor | list[list[float]] | None = None,
            features_valid_provided: np.ndarray | torch.Tensor | list[bool] = None,
            features: np.ndarray | torch.Tensor | list[list[float]] | None = None,
    ):
        """
        Logs everything provided for the current iteration.
        :param batch_idx: indices of final object of the batch.
            Each final object should have one index repeating for the length of its trajectory.
            [0,0,0,0,1,1,2,...] -> Object 0, trajectory length 4, Object 1, trajectory length 2 ...
            Trajectory data is expected to be logged from first state to last.
            In the example above the final objects would be at index 3 for the first Trajectory and
            index 5 for the second.
            Expected size: (sum(trajectory_lengths), ).
        :param total_reward: array or tensor of total rewards.
            Expected size: (batchsize, ).
        :param loss: array or tensor of losses.
            Expected size: (batchsize, ).
        :param iteration: current iteration, scalar
        :param states: array or tensor of states (full trajectories).
            Expected size: (sum(trajectory_lengths), ).
        :param logprobs_forward: array or tensor of forward logprobabilities.
            Expected size: (sum(trajectory_lengths), ).
            The logprob of a state s is expected to be the logprob of reaching s, see example in lobprobs_backward.
        :param logprobs_backward: array or tensor of backward logprobabilities.
            Expected size: (sum(trajectory_lengths), ).
            The lobprob of a state s is expected to be the logprob of reaching s, eg.:

            state   | logprob_forward       | logprobs_backward
            --------|-----------------------|------------------
            s0      | 0                     | logprob(s1->s0)=0
            s1      | logprob(s0->s1)       | logprob(s2->s1)
            s2      | logprob(s1->s2)       | logprob(s3->s2)
            st      | logprob(s(t-1)->st)   | logprob(s(t+1)->st) = 0, not applicable

        :param metrics: Optional. Additionally logged metrics of final objects based on the initialized metrics.
            Total reward and loss are logged seperately.
            A torch Tensor or np array of shape (len(metrics), batchsize,) or
            a nested list of the same shape.
        :param features_valid_provided: boolean array of shape(sum(trajectory_lengths), ).
            Flags if the provided features are valid. Otherwise they are ignored.
        :param features: Optional. Additionally logged features based on the initialized features.
            A torch Tensor or np array of shape (sum(trajectory_lengths), len(features)) or
            a nested list of the same shape. Only the rows of the final states are stored.
        :return: None
        """

        if batch_idx is not None:
            self.current["batch_idx"] = self.__to_np__(batch_idx, np.int64)
        if states is not None:
            if isinstance(states, list):
                self.current["states"] = states
            elif isinstance(states, torch.Tensor):
                self.current["states"] = states.detach().cpu()
            else:
                self.current["states"] = self.__to_np__(states, np.float32)
        if total_reward is not None:
            self.current["total_reward"] = self.__to_np__(total_reward, np.float32)
        if loss is not None:
            self.current["loss"] = self.__to_np__(loss, np.float32)
        if iteration is not None:
            self.current["iteration"] = int(iteration)
        if logprobs_forward is not None:
            self.current["logprobs_forward"] = self.__to_np__(logprobs_forward, np.float32)
        if logprobs_backward is not None:
            self.current["logprobs_backward"] = self.__to_np__(logprobs_backward, np.float32)
        if metrics is not None:
            for i,r in enumerate(self.metrics):
                self.current[r] = metrics[i]
        if features_valid_provided is not None:
            self.current["features_valid_provided"] = self.__to_np__(features_valid_provided, np.bool_)
        if features is not None:
            if isinstance(features, torch.Tensor):
                features = features.detach().cpu().numpy()
            features = np.asarray(features)
            for i,r in enumerate(self.features):
                self.current[r] = features[:, i]

    def __check_data__(self):
        """
        Checks the data for completeness before writing to db.
        :return: None
        """
        for k, v in self.current.items():
            assert v is not None, f"{k} has not been logged"

        datalength = len(self.current["batch_idx"])
        for i in [self.current["states"], self.current["logprobs_forward"], self.current["logprobs_backward"]]:
            assert len(i) == datalength, f"lengths of batch_idx, logprobs and states must match"

    def __to_np__(self, x, dtype):
        """
        Converts to np array with given dtype
        :param x: tensor or np array
        :param dtype: expected dtype
        :return:
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x.astype(dtype, copy=False)

    def __select_states__(self, idx):
        """
        Selects states by row index, keeping the format they were logged in
        (list, np array or torch Tensor).
        :param idx: np array of row indices
        :return: selected states
        """
        states = self.current["states"]
        if isinstance(states, list):
            return [states[i] for i in idx]
        if isinstance(states, torch.Tensor):
            return states[torch.as_tensor(idx, dtype=torch.long)]
        return states[idx]


    def write_to_db(self, compute_graph=True):
        """
        Writes the data of the current block to the database.
        Should be called every m iterations.
        :param compute_graph: bool, whether to compute the graph when writing to the db.
            When true, the graph dbs neccessary to view the DAG gets computed.
            As this takes some time, memory and computation power,
            this makes sense only if you want to view the Graph during running training.
            When false you will need to call compute_graph() after training is done.
            If you write the samples to the db in batches instead of all at once, write all batches first with compute_graph = False and call
            compute_graph() afterwards.
        :return: None
        """

        self.__check_data__()
        data = pd.DataFrame(columns=self.cols)

        # calculate the step for each state of a trajectory and if a state is final
        _, data["final_id"] = np.unique(self.current["batch_idx"], return_inverse=True)

        change = self.current["batch_idx"][1:] != self.current["batch_idx"][:-1]
        step = np.arange(len(self.current["batch_idx"])) - np.maximum.accumulate(
            np.r_[0, np.flatnonzero(change) + 1]
            .repeat(np.diff(np.r_[0, np.flatnonzero(change) + 1, len(self.current["batch_idx"])]))
        )
        last = np.r_[step[1:] == 0, True]
        data["step"] = step
        data["final_object"] = last

        # expand scalar iteration and add iteration and logprobs
        datalength = len(self.current["batch_idx"])
        data["iteration"] = np.array([self.current["iteration"]]*datalength)
        data["logprobs_forward"] = self.current["logprobs_forward"]
        data["logprobs_backward"] = self.current["logprobs_backward"]

        # expand arrays of length batchsize to db size (total_reward, loss, provided additional metrics)
        nan_prefill = np.full(len(self.current["batch_idx"]), np.nan)
        metric_list = ["total_reward", "loss"]
        if self.metrics is not None:
            metric_list += [i for i in self.metrics]
        for m in metric_list:
            ar = nan_prefill.copy()
            ar[last] = self.current[m]
            data[m] = ar

        # compute texts
        data["text"] = self.fn_state_to_text(self.current["states"])

        # keep the row index of each state to allow feature computation only for neccessary ones
        # (states themselves may be multi-dimensional and can not be stored in a column)
        data["state_idx"] = np.arange(datalength)

        # delete s0 or update step to start with 1
        if self.s0_included:
            data = data[data["step"]!=0]
        else:
            data["step"] += 1

        # collapse consecutive identical texts of each trajectory (take last row and sum logprobs)
        data = data.sort_values(["final_id", "text", "step"])
        logprob_sums = (
            data.groupby(["final_id", "text"], as_index=False)[["logprobs_forward", "logprobs_backward"]]
            .sum()
        )
        last_rows = (
            data.groupby(["final_id", "text"], as_index=False)
            .last()
            .drop(columns=["logprobs_forward", "logprobs_backward"])
        )
        data = last_rows.merge(logprob_sums, on=["final_id", "text"], how="left")

        # features of the final objects (the dashboard only projects these):
        # logged features first, then computed ones
        finals = data[data["final_object"].astype(bool)]
        final_idx = finals["state_idx"].to_numpy()
        feature_parts, feature_names = [], []
        features_valid = np.ones(len(finals), dtype=bool)
        if self.features is not None:
            feature_parts.append(
                np.column_stack([np.asarray(self.current[n])[final_idx] for n in self.features])
            )
            feature_names += list(self.features)
            features_valid &= np.asarray(self.current["features_valid_provided"], dtype=bool)[final_idx]
        if self.fn_compute_features is not None:
            computed, computed_valid = self.fn_compute_features(self.__select_states__(final_idx))
            computed = features_to_matrix(computed)
            feature_parts.append(computed)
            feature_names += [f"computed_features_{i}" for i in range(computed.shape[1])]
            features_valid &= features_to_matrix(computed_valid).astype(bool).reshape(-1)

        column_list = [
            "final_id",
            "text",
            "step",
            "final_object",
            "iteration",
            "total_reward",
            "loss",
            *(self.metrics if self.metrics is not None else []),
            "logprobs_forward",
            "logprobs_backward",
        ]
        data = data[column_list]
        final_ids = finals["final_id"].to_numpy()

        # create db if not existing, shift final ids and save to db
        conn = sqlite3.connect(self.db)
        cur = conn.cursor()
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name='trajectories'"
        cur.execute(query)
        table_exists = cur.fetchone() is not None
        if old_feature_layout(conn):
            conn.close()
            raise RuntimeError(OLD_LAYOUT_MSG.format(db=self.db))
        if table_exists:
            query = "SELECT COALESCE(MAX(final_id), 0) AS max FROM trajectories"
            offset = pd.read_sql_query(query, conn)["max"][0]
            data["final_id"] = data["final_id"] + offset + 1
            final_ids = final_ids + offset + 1
        data.to_sql(
            "trajectories",
            conn,
            if_exists="append",
            index=False
        )
        if feature_parts:
            write_features(
                conn, final_ids, features_valid, np.column_stack(feature_parts), feature_names
            )
            conn.commit()

        # indexing
        if not table_exists:
            cur.execute("CREATE INDEX idx_points_finalid ON trajectories(final_id)")
            cur.execute("CREATE INDEX idx_points_text ON trajectories(text)")
            cur.execute("CREATE INDEX idx_points_iteration ON trajectories(iteration)")
            cur.execute("CREATE INDEX idx_points_reward ON trajectories(total_reward)")
            cur.execute("CREATE INDEX idx_points_loss ON trajectories(loss)")

        # compute graphs and save nodes and edges db
        if compute_graph:
            create_graph_dbs(conn)
        conn.close()

        # reset current
        self._reset_current()

    def compute_graph(self):
        conn = sqlite3.connect(self.db)
        create_graph_dbs(conn)
        conn.close()

    def create_and_append_testset(
            self,
            texts: list | None = None,
            total_reward: np.ndarray | torch.Tensor | None = None,
            metrics: dict | None = None,
            features: np.ndarray | torch.Tensor | None = None,
            features_valid: np.ndarray | torch.Tensor | None = None,
    ):
        """
        Function to create the testset in the expected format.
        Expects final states, their reward and their features.
        Provide the same features as in the logged training data.
        Allows for passing the whole data at once or in chunks.
        If passed in chunks just call the function repeatedly.
        It then appends to the created testset each time.
        Neccessary:
            - Unique string representations of each state, use the same function to convert states to text as in logging
            - total_reward
            - features and features_valid
        Optional: Additional metrics of the final states
        :param texts: array of shape (chunksize,) containing unique string representations of states.
            Use the same function to generate texts as in logging
        :param total_reward: total rewards of the states, shape (chunksize,)
        :param metrics: Optional: If there are more metrics of the final states specify them here.
            Expects a dict with the title of the metric as key and the metric for all states as array, tensor or list.
        :param features: array or tensor of shape (len(features), chunksize) for the downprojections.
            Dtype should be int or float.
        :param features_valid: bool array, tensor or list of shape (chunksize,),
            Specifying if the features of a state are valid.
        :return: None
        """

        assert texts is not None , "Specify text representation of states"
        assert total_reward is not None, "Specify rewards of the objects"
        assert features is not None and features_valid is not None, "Specify features of the objects"
        assert len(texts) == len(total_reward) == features.shape[1] == len(features_valid), "lengths do not match"

        df = pd.DataFrame({"text": texts, "total_reward": total_reward})
        if metrics is not None:
            for k,v in metrics.items():
                df[k] = v
        df["features_valid"] = features_to_matrix(features_valid).astype(bool).reshape(-1)
        df.insert(0, "id", -df.index-1)

        # create db or append to it
        conn = sqlite3.connect(self.db)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='testset'")
        table_exists = cur.fetchone() is not None
        if old_feature_layout(conn):
            conn.close()
            raise RuntimeError(OLD_LAYOUT_MSG.format(db=self.db))

        # one feature vector (BLOB) per object, features are given as (n_features, chunksize)
        df["features"] = encode_testset_features(
            conn, features_to_matrix(features).T, table_exists
        )
        if table_exists:
            existing_cols = {
                row[1] for row in cur.execute("PRAGMA table_info(testset)")
            }
            for col in df.columns:
                if col not in existing_cols:
                    dtype = df[col].dtype
                    if pd.api.types.is_integer_dtype(dtype):
                        sql_type = "INTEGER"
                    elif pd.api.types.is_float_dtype(dtype):
                        sql_type = "REAL"
                    else:
                        sql_type = "TEXT"
                    cur.execute(f'ALTER TABLE testset ADD COLUMN "{col}" {sql_type}')
            conn.commit()
            query = "SELECT COALESCE(MIN(id), 0) AS min FROM testset"
            offset = pd.read_sql_query(query, conn)["min"][0]
            df["id"] = df["id"] + offset
        df.to_sql(
            "testset",
            conn,
            if_exists="append",
            index=False,
            dtype={"features": "BLOB"},
        )

        # create indices the first time

        if not table_exists:
            cur.execute("CREATE INDEX idx_testset_text ON testset(text)")
            cur.execute("CREATE INDEX idx_testset_reward ON testset(total_reward)")
        conn.commit()
        conn.close()




if __name__ == "__main__":
    """
    create debug data
    Minimal working example: States are int<9
    """
    def get_action(s):
        p=np.random.rand()
        if s==0:
            if p>0.5:
                return 1, False
            else:
                return 2, False
        elif s==1:
            return 3, False
        elif s==2:
            if p<0.1:
                return 3, False
            elif p<0.4:
                return 4, False
            else:
                return 7, True
        elif s==3:
            return 5, False
        elif s==4:
            if p>0.3:
                return 6, True
            else:
                return 7, True
        elif s==5:
            return 8, False
        elif s==8:
            return 9, True

    def get_trajectory():
        b=False
        t=[0]
        while not b:
            s, b = get_action(t[-1])
            t.append(s)
        return t

    def get_trajectories(n):
        batch_ids=[]
        trajectories=[]
        rewards=[]
        losses=[]
        c_metric=[]
        for count in range(n):
            t=get_trajectory()
            batch_ids +=[count]*len(t)
            trajectories += t
            rewards.append(t[-1])
            losses.append(np.random.rand())
            c_metric.append(np.random.rand())

        return np.array(batch_ids),np.array(trajectories),np.array(rewards),np.array(losses),c_metric

    def to_t(ss):
        return [str(int(s)) for s in ss]

    def to_f(ss):
        return np.random.uniform(1, 2, (len(ss),5)), np.array([True]*len(ss))

    import base64

    def to_i(ss):
        out = []
        for s in ss:
            dots = [(i%3, i//3) for i in range(s)]
            svg = '<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60">' + \
                  ''.join(f'<circle cx="{10+x*20}" cy="{10+y*20}" r="5" fill="black"/>' for x,y in dots) + \
                  '</svg>'
            out.append(base64.b64encode(svg.encode()).decode())
        return out


    logger = VisLogger(
        path="./debugdata",
        s0_included=True,
        fn_state_to_text=to_t,
        fn_compute_features=to_f,
        metrics=["custom_metric"],
        features=["f1", "f2", "f3"],
    )

    for i in range(10):
        b, t, r, l, l2 = get_trajectories(10)
        d_length = len(b)
        lpf = np.random.uniform(-10, 0, (d_length,))
        lpb = np.random.uniform(-10, 0, (d_length,))
        features = np.random.uniform(0, 1, (d_length, 3))
        f_valid = np.array([True] * d_length)

        logger.log(b, t, r, l, i, lpf, lpb, metrics= [l2], features_valid_provided= f_valid, features= features)
        logger.write_to_db()

    print("logging done")

    r = [7]*25 + [9]*20 + [6]*5
    t = [str(i) for i in r]
    m = {"custom_metric": np.random.uniform(0,1,(len(r),))}
    f_valid = [True]*(len(r)-1) + [False]
    f = np.random.uniform(0,1,(8, len(r)))
    logger.create_and_append_testset(t, r, m, f, f_valid)
    logger.create_and_append_testset(t, r, m, f, f_valid)
    logger.create_and_append_testset(t, r, m, f, f_valid)

    print("Testset done")


