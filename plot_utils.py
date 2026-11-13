import hashlib
import json
import heapq
import os
import sqlite3
import tempfile
from pathlib import Path

import diskcache
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.colors import make_colorscale
from sklearn import manifold
from umap import UMAP
from dash import html


# Allowed values for client-supplied choices that end up in SQL
DIRECTIONS = ("forward", "backward")
DAG_METRICS = ("highest", "lowest", "variance", "frequency")
PROJECTION_METHODS = ("tsne", "umap")
HEX_STYLES = ("Hex Ratio", "Hex Obj. Metric", "Hex Correlation")
RANK_METRICS = ("highest over all", "highest per iter", "lowest over all", "lowest per iter")


def connect_ro(path):
    """
    Open the data db read-only. The dashboard never writes to it.
    TEMP tables still work, they live outside the db file and are private to the connection.
    """
    return sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)


def check_choice(value, allowed, what):
    """Raise if a client-supplied value is not one of the allowed ones."""
    if value not in allowed:
        raise ValueError(f"Invalid {what}: {value!r}")
    return value


def quote_ident(name, allowed):
    """
    Quote a column name for use in SQL. Only names from `allowed` (taken from the db schema)
    are accepted, so client-supplied names can not inject SQL.
    """
    check_choice(name, allowed, "column name")
    return '"' + str(name).replace('"', '""') + '"'


def _table_columns(conn, table):
    """Column names of a table, empty list if the table does not exist."""
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


# Feature storage: one vector (BLOB) per final object, written by the logger
# (see "Feature storage" in logger.py). The dashboard only reads it.

def read_feature_info(conn):
    """
    Reads the feature info (names, dtype, testset_dtype).
    :param conn: sqlite connection
    :return: dict, empty if no features were stored
    """
    if not _table_columns(conn, "feature_info"):
        return {}
    info = dict(conn.execute("SELECT key, value FROM feature_info").fetchall())
    if "names" in info:
        info["names"] = json.loads(info["names"])
    return info


def decode_features(blobs, dtype):
    """
    Decodes feature vectors.
    :param blobs: iterable of bytes, all of the same length
    :param dtype: dtype the vectors were stored with (None if no features were stored)
    :return: float32 np array (n_objects, n_features)
    """
    blobs = list(blobs)
    if dtype is None or not blobs:
        return np.zeros((len(blobs), 0), dtype=np.float32)
    if len({len(b) for b in blobs}) != 1:
        raise ValueError("Stored feature vectors have different lengths")
    flat = np.frombuffer(b"".join(blobs), dtype=dtype)
    return flat.reshape(len(blobs), -1).astype(np.float32)


def old_feature_layout(conn):
    """True if the db stores features in one column per feature (convert with migrate.py)."""
    traj_cols = _table_columns(conn, "trajectories")
    testset_cols = _table_columns(conn, "testset")
    return "features_valid" in traj_cols or any(
        c.startswith("f_") and c[2:].isdigit() for c in testset_cols
    )


class Plotter:

    def __init__(self, data, image_fn, state_aggregation_fn, s0):
        self.data = data
        self.s0 = s0
        self.image_fn = image_fn
        self.agg_fn = state_aggregation_fn

        # schema: metric columns of the logged data and the testset, feature storage
        conn = connect_ro(data)
        if old_feature_layout(conn):
            conn.close()
            raise RuntimeError(
                f"{data} stores features in one column per feature (old layout). "
                f"Convert it first: python migrate.py {data}"
            )
        traj_cols = _table_columns(conn, "trajectories")
        self.traj_cols = set(traj_cols)
        self.final_object_metrics = traj_cols[traj_cols.index("total_reward"): traj_cols.index("logprobs_forward")]
        self.has_features = bool(_table_columns(conn, "features"))
        feature_info = read_feature_info(conn)
        self.feature_dtype = feature_info.get("dtype")
        self.testset_feature_dtype = feature_info.get("testset_dtype")
        testset_cols = _table_columns(conn, "testset")
        self.testset_cols = set(testset_cols)
        if testset_cols:
            self.testset_metrics = testset_cols[testset_cols.index("total_reward"): testset_cols.index("features_valid")]
            self.has_testset = conn.execute("SELECT EXISTS (SELECT 1 FROM testset)").fetchone()[0] == 1
        else:
            self.testset_metrics = []
            self.has_testset = False
        conn.close()

        # Projection cache shared by all sessions and worker processes.
        # Keyed by the db state at startup: restart the dashboard to see newly logged data.
        stat = os.stat(data)
        fingerprint = f"{Path(data).resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
        cache_dir = Path(tempfile.gettempdir()) / "gflowstate_cache" / hashlib.sha1(fingerprint.encode()).hexdigest()[:16]
        self.cache = diskcache.Cache(str(cache_dir), size_limit=2 ** 30)

        #colorscales
        self.cs_main = px.colors.sequential.speed
        self.cs_iteration = px.colors.sequential.solar
        self.cs_diverging_testset = px.colors.diverging.RdYlGn  # Earth
        self.cs_diverging_edgechange = px.colors.diverging.RdYlBu_r
        self.cs_diverging_dir = px.colors.diverging.BrBG
        self.plotly_template = "plotly_white"

    def html_from_imagefn(self, text):
        """
        Create the html for a hover when passing a text to the imagefn
        :param text:
        :return:
        """
        out=self.image_fn(text)
        if type(out) is list:
            out = [html.Div(i, style={"color": "black", "marginTop": "5px"}) for i in out]
        elif type(out) is str:
            out = [html.Img(src=out, style={"width": "150px", "height": "150px"})]
        return out

    def hex_distplot(self, data, testdata, name):
        x_min, x_max = np.inf, -np.inf
        if len(data) > 0:
            x_min = min(x_min, data.min())
            x_max = max(x_max, data.max())
        if testdata is not None and len(testdata) > 0:
            x_min = min(x_min, testdata.min())
            x_max = max(x_max, testdata.max())
        if x_min == np.inf:
            return None
        if x_max == x_min:
            delta = 0.5 if x_min == 0 else abs(x_min) * 0.05 or 0.5
            x_min -= delta
            x_max += delta

        fig = go.Figure()
        if len(data) > 0 and testdata is not None and len(testdata) > 0:
            opacity = 0.6
        else:
            opacity = 1

        if len(data) > 0:
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name="Samples",
                    marker=dict(color=self.cs_diverging_testset[-1]),
                    opacity=opacity,
                    nbinsx=30,
                    autobinx=False,
                    xbins=dict(start=x_min, end=x_max, size=(x_max - x_min) / 20),
                    histnorm="probability density",
                )
            )

        if testdata is not None and len(testdata) > 0:
            fig.add_trace(
                go.Histogram(
                    x=testdata,
                    name="Testset Objects",
                    marker=dict(color=self.cs_diverging_testset[0]),
                    opacity=opacity,
                    nbinsx=30,
                    autobinx=False,
                    xbins=dict(start=x_min, end=x_max, size=(x_max - x_min) / 20),
                    histnorm="probability density",
                )
            )

        fig.update_layout(
            title=f"Histogram of {name}",
            xaxis_title=name,
            yaxis_title="Probability Density",
            template=self.plotly_template,
            height=200,
            width=450,
            margin=dict(l=20, r=20, t=30, b=20),
            barmode="overlay",
            showlegend=True,
        )
        fig.update_xaxes(range=[x_min, x_max])

        return fig

    def hex_hover_figures(self, dp, hex_q, hex_r, metric, ss_style, metric_in_testset, usetestset):
        """
        Calculate the figures for the tooltips for the hex state space
        :param dp: projection dataframe from get_projection
        :return:
        """
        check_choice(ss_style, HEX_STYLES, "state space style")
        metric_col = quote_ident(metric, self.final_object_metrics)
        lossfig = None
        metricfig = None

        # get texts
        in_hex = dp[(dp["hex_q"] == int(hex_q)) & (dp["hex_r"] == int(hex_r))]
        texts = in_hex.loc[~in_hex["istestset"], "text"].tolist()
        placeholders = ",".join("?" for _ in texts)
        if usetestset:
            texts_t = in_hex.loc[in_hex["istestset"], "text"].tolist()
            placeholders_t = ",".join("?" for _ in texts_t)

        conn = connect_ro(self.data)

        # get loss table
        query = f"""
                                SELECT
                                    iteration,
                                    AVG(loss) AS mean,
                                    MIN(loss) AS min,
                                    MAX(loss) AS max
                                FROM trajectories
                                WHERE final_object = 1
                                AND text in ({placeholders})
                                GROUP BY iteration
                            """
        loss_df = pd.read_sql_query(query, conn, params=texts)

        # get reward data
        query = f"""
                    SELECT total_reward
                    FROM trajectories
                    WHERE final_object = 1
                    AND text in ({placeholders})
                """
        rewards_samples = (
            pd.read_sql_query(query, conn, params=texts).to_numpy().flatten()
        )
        if usetestset:
            if self.has_testset:
                query = f"SELECT total_reward FROM testset WHERE text in ({placeholders_t})"
                rewards_testset = (
                    pd.read_sql_query(query, conn, params=texts_t).to_numpy().flatten()
                )
            else:
                rewards_testset = np.array([])

        # if metric custom metric is choosen, give dist as well
        if (
                ss_style == "Hex Obj. Metric"
                and metric != "total_reward"
                and metric != "loss"
        ):
            query = f"""
                        SELECT {metric_col}
                        FROM trajectories
                        WHERE final_object = 1 AND text in ({placeholders})
                    """
            metric_samples = (
                pd.read_sql_query(query, conn, params=texts).to_numpy().flatten()
            )
            if metric_in_testset and usetestset and self.has_testset:
                metric_col_t = quote_ident(metric, self.testset_metrics)
                query = f"SELECT {metric_col_t} FROM testset WHERE text in ({placeholders_t})"
                metric_testset = (
                    pd.read_sql_query(query, conn, params=texts_t).to_numpy().flatten()
                )
            else:
                metric_testset = np.array([])

        # Use metric plot for correlation scatter
        if ss_style == "Hex Correlation":
            corr_df = (
                in_hex.loc[~in_hex["istestset"], ["id", "total_reward", "log_pf"]]
                .rename(columns={"total_reward": "reward", "log_pf": "pf"})
                .reset_index(drop=True)
            )
            corr_df["reward"] = np.log(corr_df["reward"])
            # check if correlation is similar
            # print("corr", corr_df["reward"].corr(corr_df["pf"]))
        conn.close()

        # create loss figure
        if len(loss_df) != 0:
            lossfig = go.Figure()
            lossfig.add_trace(
                go.Scatter(
                    x=loss_df["iteration"],
                    y=loss_df["max"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            lossfig.add_trace(
                go.Scatter(
                    x=loss_df["iteration"],
                    y=loss_df["min"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor=self.cs_diverging_testset[-1]
                    .replace("rgb", "rgba")
                    .replace(")", ", 0.2)"),
                    opacity=0.2,
                    line=dict(width=0),
                    showlegend=True,
                    name="Range",
                )
            )
            lossfig.add_trace(
                go.Scatter(
                    x=loss_df["iteration"],
                    y=loss_df["mean"],
                    mode="lines" if len(loss_df) > 1 else "markers",
                    line=dict(color=self.cs_diverging_testset[-1], width=2),
                    marker=dict(size=8, color=self.cs_diverging_testset[-1]),
                    name="Mean",
                    showlegend=True,
                )
            )
            lossfig.update_layout(
                title="Average Loss per Iteration (Samples)",
                xaxis_title="Iteration",
                yaxis_title="Loss",
                template=self.plotly_template,
                height=200,
                width=450,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis=dict(nticks=4),
            )
            lossfig.update_xaxes(range=[0, loss_df["iteration"].max() * 1.05])
            lossfig.update_yaxes(range=[0, loss_df["max"].max() * 1.05])

        # create reward distribution
        if usetestset and len(rewards_testset) != 0:
            rewardfig = self.hex_distplot(
                rewards_samples, rewards_testset, "Total Reward"
            )
        else:
            rewardfig = self.hex_distplot(rewards_samples, None, "Total Reward")

        if (
                ss_style == "Hex Obj. Metric"
                and metric != "total_reward"
                and metric != "loss"
        ):
            if metric_in_testset and usetestset and len(metric_testset) != 0:
                metricfig = self.hex_distplot(metric_samples, metric_testset, metric)
            else:
                metricfig = self.hex_distplot(metric_samples, None, metric)
        if ss_style == "Hex Correlation":
            metricfig = make_subplots(
                rows=1,
                cols=2,
                horizontal_spacing=0.15,
            )
            metricfig.add_trace(
                go.Scatter(
                    x=corr_df["reward"],
                    y=corr_df["pf"],
                    mode="markers",
                    marker=dict(color=self.cs_diverging_testset[-1]),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
            metricfig.add_trace(
                go.Scatter(
                    x=np.exp(corr_df["reward"]),
                    y=np.exp(corr_df["pf"]),
                    mode="markers",
                    marker=dict(color=self.cs_diverging_testset[-1]),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=2,
            )
            metricfig.update_layout(
                title="(Log) Reward vs Forward (Log) Probabilities for Samples",
                template=self.plotly_template,
                height=200,
                width=550,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            metricfig.update_xaxes(row=1, col=1, title_text=" Log Reward")
            metricfig.update_yaxes(row=1, col=1, title_text="Logprobabilities")
            metricfig.update_xaxes(row=1, col=2, title_text="Reward")  # , range=[,])
            metricfig.update_yaxes(row=1, col=2, title_text="Probabilities")

        if usetestset:
            texts += texts_t
        return (lossfig, rewardfig, metricfig), texts


    def update_hex(self, df, dp, selected_ids, ss_style, metric, usetestset, size=8.0):
        """
        Update the state space plot for hex style
        :param df: df with grouped data for hex (hex_r, hex_q, metric, n_samples, n_test)
        :param dp: projection dataframe from get_projection
        :return: figure
        """
        check_choice(metric, self.final_object_metrics, "metric")
        fig = go.Figure()

        # colorscales and titles
        colorscale = self.cs_main
        title = "State Projection - "
        metric_min = df["metric"].min()
        metric_max = df["metric"].max()
        if ss_style == "Hex Ratio":
            if usetestset:
                colorscale = self.cs_diverging_testset
                metric_max = 1
                metric_min = -1
                legend_title = "Score"
                title += "Log Odds Ratio of Samples to Testset Objects (Scaled to [-1,1])"
                title += "<br><sup>1: Only Testset Objects. -1: Only Samples. 0: Ratio Samples/Testset Objects Is the Same as the Global Ratio.</sup>"
            else:
                legend_title = "N Samples"
                title += "Number of Sampled Objects"
        elif ss_style == "Hex Obj. Metric":
            legend_title = f"Mean {metric}"
            title += f"Mean {metric} of the Sampled Objects"
        elif ss_style == "Hex Correlation":
            metric_max = 1
            metric_min = 0
            legend_title = "R"
            title += "Correlation Between the Sum of the Forward Logprobabilities and Reward for bins with > 10 Samples"
        else:
            raise NotImplementedError("Unknown ss_style")
        title += "<br><sup>Select a hex or hover over it to see its details</sup>"

        def hex_corners(x, y, s):
            angles = np.deg2rad(np.arange(0, 360, 60) - 30)  # pointy-top
            return (
                x + s * np.cos(angles),
                y + s * np.sin(angles),
            )
        def map_color(v, vmin, vmax, cs, opacity):
            if v is None:
                return "black"
            if np.isnan(v) or v == float("nan"):
                return "grey"
            if v < vmin:
                idx=0
            elif v > vmax:
                idx=-1
            else:
                norm = (v - vmin) / (vmax - vmin) if vmax > vmin else 0
                idx = int(norm * (len(cs) - 1))
            return cs[idx].replace("rgb", "rgba").replace(")", f", {opacity})")

        if selected_ids:
            # get texts first as ids in dp table might not be the same (multiple final objects)
            conn = connect_ro(self.data)
            placeholders = ",".join("?" for _ in selected_ids)
            query = f"SELECT DISTINCT text FROM trajectories WHERE final_object = 1 AND final_id IN ({placeholders})"
            selected_texts = pd.read_sql_query(query, conn, params=selected_ids)["text"].tolist()
            if usetestset and self.has_testset:
                query = f"SELECT DISTINCT text FROM testset WHERE id IN ({placeholders})"
                selected_texts_t = pd.read_sql_query(query, conn, params=selected_ids)["text"].tolist()
                selected_texts = selected_texts + selected_texts_t
            conn.close()
            selected_df = (
                dp.loc[dp["text"].isin(selected_texts), ["id", "text", "iteration", metric, "x", "y", "hex_q", "hex_r"]]
                .rename(columns={metric: "metric"})
                .reset_index(drop=True)
            )
            if ss_style == "Hex Ratio":
                selected_df["metric"] = np.where(selected_df["id"] >= 0, 1, -1)
            # check if all selected ids are in one hex: in this case zoom in and disable hex hover
            zoom = selected_df['hex_q'].nunique() == 1 and selected_df['hex_r'].nunique() == 1
            if zoom:
                cx = size * np.sqrt(3) * (selected_df['hex_q'][0] + selected_df['hex_r'][0] / 2)
                cy = size * 1.5 * selected_df['hex_r'][0]
                zoom_corners = hex_corners(cx, cy, size)
                offset = size*0.2
                zoom_xrange = [zoom_corners[0].min() - offset,  zoom_corners[0].max() + offset]
                zoom_yrange = [zoom_corners[1].min() - offset, zoom_corners[1].max() + offset]

        hex_opacity = 0.3 if selected_ids else 1
        for _, row in df.iterrows():
            cx = size * np.sqrt(3) * (row["hex_q"] + row["hex_r"] / 2)
            cy = size * 1.5 * row["hex_r"]
            xs, ys = hex_corners(cx, cy, size)
            customdata = ["usehex", row["hex_q"], row["hex_r"], row["metric"], row["n_samples"], row["n_test"]]

            fig.add_trace(
                go.Scatter(
                    x=np.append(xs, xs[0]),
                    y=np.append(ys, ys[0]),
                    mode="lines",
                    fill="toself",
                    line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
                    fillcolor=map_color(row["metric"], metric_min, metric_max, colorscale, hex_opacity),
                    hoverinfo="none",
                    hovertemplate=None,
                    customdata=customdata,
                    showlegend=False,
                )
            )

        # selected ids scatter
        if selected_ids:
            selected_df_s = selected_df[selected_df["id"]>=0]
            fig.add_trace(go.Scatter(
                x=selected_df_s['x'],
                y=selected_df_s['y'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=selected_df_s['metric'],
                    colorscale=colorscale,
                    line=dict(color='white', width=1),
                    showscale=False,
                    cmin=metric_min,
                    cmax=metric_max,
                ),
                customdata=selected_df_s[['id', 'iteration', 'metric', 'text']].values,
                hoverinfo='none',
                name="Selected Samples"
            ))
            if usetestset:
                selected_df_t = selected_df[selected_df["id"]<0]
                selected_df_t["color"] = -1
                fig.add_trace(go.Scatter(
                    x=selected_df_t['x'],
                    y=selected_df_t['y'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=selected_df_t['color'],
                        colorscale=self.cs_diverging_testset,
                        line=dict(color='white', width=1),
                        showscale=False,
                        cmin=metric_min,
                        cmax=metric_max,
                    ),
                    customdata=selected_df_t[['id', 'iteration', 'metric', 'text']].values,
                    hoverinfo='none',
                    name="Testset"
                ))

        # dummy trace for legend
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=0,
                    color=[metric_min, metric_max],
                    colorscale=colorscale,
                    cmin=metric_min,
                    cmax=metric_max,
                    showscale=True,
                    colorbar=dict(
                        title=legend_title,
                        len=0.9
                    ),
                ),
                hoverinfo="none",
                showlegend=False,
            )
        )

        fig.update_layout(
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="y",
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            margin=dict(l=40, r=40, t=40, b=40),
            autosize=True,
            template=self.plotly_template,
            title=title,
            modebar_remove=["lasso2d", "select2d"]

        )

        if selected_ids and zoom:
            fig.update_xaxes(range=zoom_xrange)
            fig.update_yaxes(range=zoom_yrange)

        return fig

    def get_hexbin_data(self, dp, method, metric):
        """
        Get the data for the hex plots
        :param dp: projection dataframe from get_projection
        """
        check_choice(method, HEX_STYLES, "state space style")
        check_choice(metric, self.final_object_metrics, "metric")
        is_test = dp["istestset"].astype(bool)
        df = (
            dp.assign(n_test=is_test.astype(int), n_samples=(~is_test).astype(int))
            .groupby(["hex_r", "hex_q"], as_index=False)[["n_test", "n_samples"]]
            .sum()
        )
        samples = dp[~is_test]

        if method == "Hex Ratio":
            if df["n_test"].sum() == 0: #no testset used, return frequency
                df["metric"] = df["n_samples"]
            else:
                # Log Odds ratio scaled to [-1,1]
                # Identical to tanh(log((samples/test)/(sum(samples)/sum(test))))
                eps = 0.0001
                ratio = (df["n_samples"].sum()+eps) / (df["n_test"].sum()+eps)
                df["metric"] = (df["n_samples"]+eps)/(df["n_test"]+eps)
                df["metric"] = (df["metric"]**2-ratio**2)/(df["metric"]**2+ratio**2)
        elif method == "Hex Obj. Metric":
            means = samples.groupby(["hex_r", "hex_q"])[metric].mean().rename("metric").reset_index()
            df = df.merge(means, on=["hex_r", "hex_q"], how="left")
            df["metric"] = df["metric"].fillna(0)
        else:  # Hex Correlation
            df_corr = (
                samples.groupby(["hex_r", "hex_q"])[["log_pf", "total_reward"]]
                .apply(lambda g: g["log_pf"].corr(np.log(g["total_reward"])))
                .rename("metric")
                .reset_index()
            )
            df = df.merge(df_corr, on=["hex_r", "hex_q"], how="outer")
            df.loc[df["n_samples"] <= 10, "metric"] = np.nan
        return df

    @staticmethod
    def calculate_hexbins(df, size=8.0):
        """
        Assign pointy-top hex axial coordinates (q, r) to points in df.
        :param df: df with columns x and y
        :param size: center to corner hex radius
        :return: q, r: hex coordinates as np array
        """

        x = df["x"].to_numpy()
        y = df["y"].to_numpy()

        # Axial coordinates (fractional)
        q = (np.sqrt(3) / 3 * x - 1 / 3 * y) / size
        r = (2 / 3 * y) / size

        # Convert to cube coordinates
        cx = q
        cz = r
        cy = -cx - cz

        # Round cube coordinates
        rx = np.round(cx)
        ry = np.round(cy)
        rz = np.round(cz)

        # Fix rounding errors
        dx = np.abs(rx - cx)
        dy = np.abs(ry - cy)
        dz = np.abs(rz - cz)

        mask_x = (dx > dy) & (dx > dz)
        mask_y = (~mask_x) & (dy > dz)
        mask_z = ~(mask_x | mask_y)

        rx[mask_x] = -ry[mask_x] - rz[mask_x]
        ry[mask_y] = -rx[mask_y] - rz[mask_y]
        rz[mask_z] = -rx[mask_z] - ry[mask_z]

        # Axial coordinates are (q, r) = (x, z)
        return rx.astype(np.int64), rz.astype(np.int64)

    def get_projection(self, iteration, use_testset, method="tsne", param_value=15):
        """
        Downprojection of the final objects in the iteration range (and the testset).
        Results are cached on disk, shared by all sessions and worker processes.
        :return: (dataframe with one row per projected object, hexbin size)
        """
        check_choice(method, PROJECTION_METHODS, "projection method")
        key = ("dp", int(iteration[0]), int(iteration[1]), bool(use_testset), method, int(param_value))
        result = self.cache.get(key)
        if result is None:
            # one process computes, the others wait and read the cached result
            with diskcache.Lock(self.cache, repr(("lock",) + key), expire=3600):
                result = self.cache.get(key)
                if result is None:
                    result = self._compute_projection([key[1], key[2]], key[3], method, key[5])
                    self.cache.set(key, result)
        return result

    def _compute_projection(self, iteration, use_testset, method, param_value):
        conn = connect_ro(self.data)
        metric_sql = ", ".join(quote_ident(c, self.traj_cols) for c in self.final_object_metrics)
        metric_sql_outer = ", ".join(f"t.{quote_ident(c, self.traj_cols)}" for c in self.final_object_metrics)
        if self.has_features:
            feature_join = "LEFT JOIN features f ON f.final_id = t.final_id"
            feature_cols = "f.features_valid, f.vec"
        else:
            feature_join = ""
            feature_cols = "NULL AS features_valid, NULL AS vec"

        # Get logged data (unique texts, latest iteration) and their feature vectors
        query = f"""
                    SELECT
                        t.final_id AS id,
                        t.text,
                        {metric_sql_outer},
                        t.iteration,
                        {feature_cols}
                    FROM (
                        SELECT
                            final_id,
                            text,
                            {metric_sql},
                            iteration,
                            ROW_NUMBER() OVER (
                                PARTITION BY text
                                ORDER BY iteration DESC
                            ) AS rn
                        FROM trajectories
                        WHERE iteration BETWEEN ? AND ?
                          AND final_object = 1
                    ) t
                    {feature_join}
                    WHERE t.rn = 1
                """
        logged = pd.read_sql_query(query, conn, params=iteration)
        logged = logged[logged["features_valid"] == 1]
        if self.has_features:
            query = """
                        SELECT COUNT(*) AS count
                        FROM trajectories t
                        JOIN features f ON f.final_id = t.final_id
                        WHERE t.final_object = 1
                        AND t.iteration BETWEEN ? AND ?
                        AND f.features_valid = 0
                    """
            non_valid = pd.read_sql_query(query, conn, params=iteration)["count"][0]
            if non_valid:
                print(f"""
                            {non_valid} objects of the logged data have not been
                            downprojected due to invalid features.
                            """)

        features = decode_features(logged["vec"], self.feature_dtype)
        df_dp = logged.drop(columns=["vec"]).reset_index(drop=True)

        # Get testset data
        if use_testset and self.has_testset:
            metric_sql_t = ", ".join(quote_ident(c, self.testset_cols) for c in self.testset_metrics)
            query = f"""
                        SELECT
                            id,
                            text,
                            {metric_sql_t},
                            features
                        FROM testset
                        WHERE features_valid = 1
                    """
            testset = pd.read_sql_query(query, conn)
            query = "SELECT COUNT(*) AS count FROM testset WHERE features_valid = 0"
            non_valid_t = pd.read_sql_query(query, conn)["count"][0]
            if non_valid_t:
                print(f"""
                            {non_valid_t} objects of the testset have not been
                            downprojected due to invalid features.
                        """)

            # concat features
            features_t = decode_features(testset["features"], self.testset_feature_dtype)
            testset = testset.drop(columns=["features"])
            assert features.shape[1] == features_t.shape[1], f"""
                            Testset and Logged data have a different amout of features.\n
                            Testset: {features_t.shape[1]}
                            Logged: {features.shape[1]}
                        """
            features = np.concatenate((features, features_t), axis=0)
            df_dp = pd.concat([df_dp, testset], axis=0, ignore_index=True)

        # Downprojection
        if features.shape[1] <= 1:
            raise ValueError("Not enough features")
        proj_s = self._project(features, method, param_value)

        df_dp["x"] = proj_s[:, 0]
        df_dp["y"] = proj_s[:, 1]
        df_dp["istestset"] = df_dp["id"] < 0

        # calc hexbins (fallback size if there are no points or all of them coincide)
        extent = float(np.max(proj_s.max(axis=0) - proj_s.min(axis=0))) if len(proj_s) else 0.0
        hexbin_size = extent / 16 if extent > 0 else 1.0
        df_dp["hex_q"], df_dp["hex_r"] = self.calculate_hexbins(df_dp, size=hexbin_size)

        # add P_F for correlation data
        # Load all states (in step order) of the trajectories in the iteration range that end
        # in one of the projected samples.
        # The texts of the projected samples go into a TEMP table (private to this connection)
        conn.execute("CREATE TEMP TABLE dp_texts (text TEXT PRIMARY KEY)")
        conn.executemany(
            "INSERT OR IGNORE INTO temp.dp_texts (text) VALUES (?)",
            ((t,) for t in df_dp.loc[~df_dp["istestset"], "text"]),
        )
        query = """
            SELECT
                t.final_id,
                t.text,
                t.iteration,
                t.logprobs_forward,
                rf.final_text
            FROM trajectories t
            JOIN (
                SELECT tr.final_id, tr.text AS final_text
                FROM temp.dp_texts dp
                JOIN trajectories tr
                    ON tr.text = dp.text
                   AND tr.final_object = 1
                   AND tr.iteration BETWEEN ? AND ?
            ) rf ON rf.final_id = t.final_id
            ORDER BY t.final_id, t.step
            """
        steps = pd.read_sql_query(query, conn, params=iteration)
        conn.close()

        df_dp = df_dp.merge(self._log_pf(steps), on="text", how="left")

        return df_dp, hexbin_size

    # Below this many samples t-SNE / UMAP are not defined or not meaningful
    MIN_SAMPLES_DR = 5

    @staticmethod
    def _project(features, method, param_value):
        """
        Project the features to 2D.
        :param features: np array (n_samples, n_features)
        :param method: tsne or umap
        :param param_value: perplexity (tsne) or n_neighbors (umap)
        :return: np array (n_samples, 2)
        """
        n = features.shape[0]
        if features.shape[1] == 2:
            return features
        if n < Plotter.MIN_SAMPLES_DR:
            # too few samples: first two principal components (zero padded)
            proj = np.zeros((n, 2))
            if n > 1:
                centered = features - features.mean(axis=0)
                u, s, _ = np.linalg.svd(centered, full_matrices=False)
                k = min(2, len(s))
                proj[:, :k] = u[:, :k] * s[:k]
            return proj
        if method == "tsne":
            return manifold.TSNE(
                perplexity=min(param_value, n - 1),
                init="pca",
                learning_rate="auto",
            ).fit_transform(features)
        if method == "umap":
            return UMAP(n_neighbors=min(param_value, n - 1)).fit_transform(features)
        raise NotImplementedError("Method not implemented")

    @staticmethod
    def _log_pf(steps):
        """
        Log probability to sample each final object under the forward policy:
        logsumexp over its distinct trajectories of the summed forward logprobs.
        If the same trajectory (same sequence of states) occurs in several iterations,
        only the latest one is used.
        :param steps: df with final_id, text, iteration, logprobs_forward, final_text,
            ordered by final_id and step
        :return: df with text and log_pf
        """
        if steps.empty:
            return pd.DataFrame({"text": pd.Series(dtype=object), "log_pf": pd.Series(dtype=float)})
        by_traj = steps.groupby("final_id", sort=False)
        traj = by_traj.agg(
            text=("final_text", "first"),
            iteration=("iteration", "max"),
            sequence=("text", lambda s: "||".join(map(str, s))),
        )
        traj["logpf"] = by_traj["logprobs_forward"].sum(min_count=1)
        traj = (
            traj.sort_values("iteration", ascending=False, kind="stable")
            .drop_duplicates(["text", "sequence"])
        )
        by_text = traj["logpf"].groupby(traj["text"])
        max_lp = by_text.transform("max")
        sum_exp = np.exp(traj["logpf"] - max_lp).groupby(traj["text"]).sum(min_count=1)
        log_pf = by_text.max() + np.log(sum_exp)
        return log_pf.rename("log_pf").reset_index()

    def update_DAG(
            self,
            iteration,
            direction="forward",
            metric="highest",
            max_freq=0,
            add_handlers=True,
            build_ids=[]
    ):
        """
        Updates the DAG based on the given metric and direction.
        :param iteration: iteration range
        :param direction: forward/backward
        :param metric: lowest, highest, variance, frequency
        :param max_freq: Highest frequency in the data
        :param add_handlers: add handlers if in expanding mode, dont add in selection mode
        :param build_ids: states to build the dag from
        :return: dict with elements and stylesheets
        """

        check_choice(metric, DAG_METRICS, "DAG metric")
        column = "logprobs_" + check_choice(direction, DIRECTIONS, "direction")
        conn = connect_ro(self.data)
        placeholders = ",".join("?" for _ in build_ids)
        base_where = f"""
            source IN ({placeholders})
            AND target IN ({placeholders})
            AND iteration BETWEEN ? AND ?
        """
        params = build_ids + build_ids + [iteration[0], iteration[1]]

        if metric in ["highest", "lowest"]:
            query = f"""
                WITH edge_ids AS (
                    SELECT
                        source,
                        target,
                        GROUP_CONCAT(id, '-') AS id
                    FROM edges
                    WHERE {base_where}
                    GROUP BY source, target
                )
                SELECT
                    ei.id,
                    e.source,
                    e.target,
                    {"MAX" if metric == "highest" else "MIN"}(e.{column}) AS metric
                FROM edges e
                JOIN edge_ids ei
                  ON ei.source = e.source
                 AND ei.target = e.target
                WHERE
                    e.source IN ({placeholders})
                    AND e.target IN ({placeholders})
                    AND e.iteration BETWEEN ? AND ?
                GROUP BY e.source, e.target
            """
            params *= 2
        elif metric == "variance":
            query = f"""
            WITH edge_groups AS (
                SELECT 
                    source,
                    target,
                    MAX(iteration) AS max_iteration,
                    AVG({column}) AS mean_metric
                FROM edges
                WHERE source IN ({placeholders})
                  AND target IN ({placeholders})
                  AND iteration BETWEEN ? AND ?
                GROUP BY source, target
            ),
            edge_ids AS (
                SELECT 
                    source,
                    target,
                    GROUP_CONCAT(id, '-') AS id
                FROM edges
                WHERE source IN ({placeholders})
                  AND target IN ({placeholders})
                  AND iteration BETWEEN ? AND ?
                GROUP BY source, target
            )
            SELECT 
                ei.id,
                e.source,
                e.target,
                eg.max_iteration AS iteration,
                e.{column} - eg.mean_metric AS metric
            FROM edge_groups eg
            JOIN edges e 
              ON e.source = eg.source
             AND e.target = eg.target
             AND e.iteration = eg.max_iteration
            JOIN edge_ids ei 
              ON ei.source = eg.source
             AND ei.target = eg.target
            WHERE e.source IN ({placeholders})
              AND e.target IN ({placeholders})
              AND e.iteration BETWEEN ? AND ?
            """
            params *= 3
        else:
            query = f"""
                WITH edge_ids AS (
                    SELECT
                        source,
                        target,
                        GROUP_CONCAT(id, '-') AS id
                    FROM edges
                    WHERE {base_where}
                    GROUP BY source, target
                )
                SELECT
                    ei.id,
                    e.source,
                    e.target,
                    COUNT(*) AS metric
                FROM edges e
                JOIN edge_ids ei
                  ON ei.source = e.source
                 AND ei.target = e.target
                WHERE
                    e.source IN ({placeholders})
                    AND e.target IN ({placeholders})
                    AND e.iteration BETWEEN ? AND ?
                GROUP BY e.source, e.target
            """
            params *= 2

        edges = pd.read_sql_query(query, conn, params=params)

        query = f"""
            SELECT *
            FROM nodes
            WHERE id IN ({placeholders})
            """
        nodes = pd.read_sql_query(query, conn, params=build_ids)

        nodes['image'] = nodes['id'].apply(self.image_fn)
        nodes['has_image'] = nodes['image'].apply(lambda x: "str" if isinstance(x, str) else "list")
        nodes['image'] = nodes['image'].apply(lambda x: "\n".join(x) if isinstance(x, list) else x)

        if add_handlers:
            # get number of children
            query = f"""
                            SELECT
                                source,
                                COUNT(DISTINCT target) AS n_children
                            FROM edges
                            WHERE source IN ({placeholders})
                              AND target NOT IN ({placeholders})
                            GROUP BY source
                        """
            counts = pd.read_sql_query(query, conn, params=build_ids + build_ids)
            nodes = nodes.merge(
                counts,
                left_on='id',
                right_on='source',
                how='left'
            )
            nodes["n_children"] = nodes["n_children"].fillna(0)

            # create handlers
            handler_nodes = (nodes[(nodes['node_type'] != "final") | (nodes['n_children'] > 0)].copy().drop(["reward"], axis=1))
            handler_nodes["node_type"] = "handler"
            handler_nodes["id"] = "handler_" + handler_nodes["id"]
            handler_nodes["label"] = "Select children: " + handler_nodes["n_children"].astype(int).astype(str)
            handler_nodes["metric"] = metric
            handler_nodes["direction"] = direction
            handler_edges = handler_nodes["id"].copy().to_frame().rename(columns={"id": "target"})
            handler_edges["source"] = handler_edges["target"].str.removeprefix("handler_")

            nodes = pd.concat([nodes, handler_nodes], ignore_index=True)
            edges = pd.concat([edges, handler_edges], ignore_index=True)

        nodes["iteration0"] = iteration[0]
        nodes["iteration1"] = iteration[1]
        if direction == "backward":
            edges.rename(columns={"source": "target", "target": "source"}, inplace=True)

        if self.s0 != "#":
            # same handling as for the other nodes: lists are joined to one string
            s0_image = self.image_fn(self.s0)
            is_root = nodes["id"] == "#"
            nodes.loc[is_root, "has_image"] = (
                "str" if isinstance(s0_image, str) else "list"
            )
            nodes.loc[is_root, "image"] = (
                "\n".join(s0_image) if isinstance(s0_image, list) else s0_image
            )

        # convert to cytoscape structure
        nodes = [{"data": row} for row in nodes.to_dict(orient="records")]
        edges = [{"data": row} for row in edges.to_dict(orient="records")]

        conn.close()

        # Compute color scale
        if metric in ['highest', 'lowest']:
            vmin, vmax = -10, 0
            colorscale = self.cs_main
        elif metric == "variance":
            vmin, vmax = -2.5, 2.5
            colorscale = self.cs_diverging_edgechange
        elif metric == "frequency":
            vmin, vmax = 0, max_freq
            colorscale = self.cs_main
        else:
            vmin, vmax = -1, 1  # fallback
            colorscale = px.colors.sequential.Viridis

        # Create color mapping
        def get_color(value, vmin, vmax, colorscale):
            if np.isnan(value):
                return "#8e8e8e"
            if value < vmin:
                return colorscale[0]  # lowest color
            if value > vmax:
                return colorscale[-1]  # highest color
            if vmax == vmin:
                norm = 0.5
            else:
                norm = (value - vmin) / (vmax - vmin)
            idx = int(norm * (len(colorscale) - 1))
            return colorscale[idx]

        # Color of each edge goes into its data and is mapped by one stylesheet rule
        # (one rule per edge made Cytoscape match every rule against every edge)
        # Rows with the same id (variance: several edges in the latest iteration) get the
        # color of the last one, as with the former per-id rules
        color_by_id = {
            edge['data'].get('id'): get_color(edge['data'].get("metric", 0), vmin, vmax, colorscale)
            for edge in edges
        }
        for edge in edges:
            edge['data']['color'] = color_by_id[edge['data'].get('id')]

        # Build stylesheet
        elements = nodes + edges

        stylesheet = [
            # Default node style
            {
                'selector': 'node',
                'style': {
                    'background-color': '#fff',
                    'background-image': 'data(image)',
                    'background-fit': 'contain',
                    'background-clip': 'none',
                    'shape': 'round-rectangle',
                    'width': '50px',
                    'height': '50px',
                    'border-width': '0px',
                    'border-color': '#000000'
                }
            },
            # default if no image available
            {
                'selector': 'node[has_image = "list"]',
                'style': {
                    'background-image': 'none',
                    'label': 'data(image)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'text-wrap': 'wrap',
                    'text-max-width': '110px',
                    'font-size': '8px',
                    'width': '150px',
                    "border-width": "1px",
                }
            },
            # START node default "#" (keep text label)
            {
                'selector': 'node[node_type = "start"]',
                'style': {
                    'background-color': '#0e5a2c',
                    'background-image': 'none',
                    'label': f'data(id)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'font-size': '12px',
                    'shape': 'diamond',
                    'width': '40px',
                    'height': '40px',
                    'border-color': '#000000',
                    'border-width': '0px',
                    'font-weight': 'bold',
                    'text-wrap': 'wrap',
                    'text-max-width': '55px'
                }
            },
            # Final node
            {
                'selector': 'node[node_type = "final"]',
                'style': {
                    'background-color': '#fff',
                    'height': '60px',
                    'border-width': '0px',
                    'border-color': '#000000'
                }
            },
            # Handler node
            {
                'selector': 'node[node_type = "handler"]',
                'style': {
                    'background-color': '#fff',
                    'background-image': 'none',
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'font-size': '8px',
                    'shape': 'round-rectangle',
                    'width': '70px',
                    'height': '18px',
                    'border-width': '0.5px',
                    'border-color': '#000000',
                    # 'font-weight': 'bold',
                    'text-wrap': 'wrap',
                    'text-max-width': '90px'
                }
            },
            # Default edge style
            {
                'selector': 'edge',
                'style': {
                    'width': 2,
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'arrow-scale': 1,
                    'line-color': 'data(color)',
                    'target-arrow-color': 'data(color)',
                }
            }
        ]

        if self.s0 != "#":
            stylesheet.append(
                # START node custom (display image)
                {
                    'selector': 'node[node_type = "start"]',
                    'style': {
                        'label': '',
                        'background-color': '#fff',
                        'background-image': 'data(image)',
                        'background-fit': 'contain',
                        'background-clip': 'none',
                        'shape': 'round-rectangle',
                        'border-width': '0px',
                        'border-color': '#0e5a2c'
                    }
                },
            )

        return {
            'elements': elements,
            'stylesheet': stylesheet,
        }

    def dag_remove_node_prune(self, nodelist, node_to_remove):
        """
        removes the node from the list and all unconnected children
        :return: new list
        """
        conn = connect_ro(self.data)
        remaining = set(nodelist)
        if node_to_remove not in remaining:
            return remaining
        stack = [node_to_remove]
        while stack:
            node = stack.pop()
            if node not in remaining:
                continue
            remaining.remove(node)
            query = "SELECT DISTINCT target FROM edges WHERE source = ?"
            children = pd.read_sql_query(query, conn, params=[node])["target"].tolist()
            for child in children:
                if child not in remaining:
                    continue
                query = "SELECT DISTINCT source FROM edges WHERE target = ?"
                parents = pd.read_sql_query(query, conn, params=[child])["source"].tolist()
                if not any(parent in remaining for parent in parents):
                    stack.append(child)

        conn.close()
        return remaining

    def update_DAG_overview(self, direction, metric, iteration, page):
        """
        edge heatmap plot
        pagination is implemented with offset.
        This is the fast implementation and might break for scrolling far down.
        might need to implement with keyset later on.
        """
        check_choice(metric, DAG_METRICS, "DAG metric")
        column = "logprobs_" + check_choice(direction, DIRECTIONS, "direction")
        top_n = 150
        page = max(int(page), 0)
        offset = top_n * page

        conn = connect_ro(self.data)

        # Build query based on metric
        if metric == "highest":
            query = f"""
                    WITH edge_max AS (
                        SELECT source, target, MAX({column}) as max_val
                        FROM edges
                        WHERE iteration BETWEEN ? AND ?
                        GROUP BY source, target
                        ORDER BY max_val DESC
                        LIMIT {top_n} OFFSET {offset}
                    )
                    SELECT e.source, e.target, e.iteration, e.{column} as value, em.max_val as metric_val
                    FROM edges e
                    INNER JOIN edge_max em ON e.source = em.source AND e.target = em.target
                    WHERE e.iteration BETWEEN ? AND ?
                    ORDER BY em.max_val DESC, e.source, e.target, e.iteration
                """
        elif metric == "lowest":
            query = f"""
                    WITH edge_min AS (
                        SELECT source, target, MIN({column}) as min_val
                        FROM edges
                        WHERE iteration BETWEEN ? AND ?
                        GROUP BY source, target
                        ORDER BY min_val ASC
                        LIMIT {top_n} OFFSET {offset}
                    )
                    SELECT e.source, e.target, e.iteration, e.{column} as value, em.min_val as metric_val
                    FROM edges e
                    INNER JOIN edge_min em ON e.source = em.source AND e.target = em.target
                    WHERE e.iteration BETWEEN ? AND ?
                    ORDER BY em.min_val ASC, e.source, e.target, e.iteration
                """
        elif metric == "variance":
            query = f"""
                    WITH edge_stats AS (
                        SELECT 
                            source, 
                            target,
                            AVG({column}) as mean_val,
                            COUNT(*) as cnt
                        FROM edges
                        WHERE iteration BETWEEN ? AND ?
                        GROUP BY source, target
                        HAVING cnt > 1
                    ),
                    edge_variance AS (
                        SELECT 
                            es.source,
                            es.target,
                            es.mean_val,
                            SUM((e.{column} - es.mean_val) * (e.{column} - es.mean_val)) / es.cnt as variance
                        FROM edges e
                        INNER JOIN edge_stats es ON e.source = es.source AND e.target = es.target
                        WHERE e.iteration BETWEEN ? AND ?
                        GROUP BY es.source, es.target, es.mean_val, es.cnt
                        ORDER BY variance DESC
                        LIMIT {top_n} OFFSET {offset}
                    )
                    SELECT e.source, e.target, e.iteration, e.{column} as value, ev.mean_val, ev.variance as metric_val
                    FROM edges e
                    INNER JOIN edge_variance ev ON e.source = ev.source AND e.target = ev.target
                    WHERE e.iteration BETWEEN ? AND ?
                    ORDER BY ev.variance DESC, e.source, e.target, e.iteration
                """
        elif metric == "frequency":
            query = f"""
                    WITH edge_freq AS (
                        SELECT source, target, COUNT(*) as freq
                        FROM edges
                        WHERE iteration BETWEEN ? AND ?
                        GROUP BY source, target
                        ORDER BY freq DESC
                        LIMIT {top_n} OFFSET {offset}
                    )
                    SELECT e.source, e.target, e.iteration, e.{column} as value, ef.freq as metric_val
                    FROM edges e
                    INNER JOIN edge_freq ef ON e.source = ef.source AND e.target = ef.target
                    WHERE e.iteration BETWEEN ? AND ?
                    ORDER BY ef.freq DESC, e.source, e.target, e.iteration
                """

        params = iteration * 3 if metric == "variance" else iteration * 2
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            fig = go.Figure().add_annotation(text="No data found", showarrow=False)
            return fig, None, []

        # Create edge identifier and preserve order
        df['edge_id'] = df['source'].astype(str) + '-' + df['target'].astype(str)

        # Get unique edges in order (already ordered by metric in SQL)
        unique_edges = df['edge_id'].unique()
        edge_to_idx = {edge: idx for idx, edge in enumerate(unique_edges)}
        df['edge_idx'] = df['edge_id'].map(edge_to_idx)

        # For variance metric, adjust values to be zero-based (value - mean)
        # For frequency metric, use the frequency value itself for coloring
        if metric == "variance":
            edge_means = df.groupby('edge_id')['value'].transform('mean')
            df['plot_value'] = df['value'] - edge_means
        elif metric == "frequency":
            df['plot_value'] = df['metric_val']  # Use frequency for coloring
        else:
            df['plot_value'] = df['value']

        # create edge list (source, target per row) for hover and selection
        edge_list =df[['source', "target", "edge_idx"]]
        edge_list = edge_list.drop_duplicates().reset_index(drop=True)
        edge_list = list(edge_list[['source', 'target']].itertuples(index=False, name=None))

        # Create pivot table for heatmap
        heatmap_data = df.pivot_table(
            index='iteration',
            columns='edge_idx',
            values='plot_value',
            aggfunc='first'
        ).sort_index()

        if metric == "variance":
            color_scale = self.cs_diverging_edgechange
            zmin, zmax, zmid = -2.5, 2.5, 0
            colorbar_title = "Value - Mean"
            title = f"Transition Heatmap<br><sup>Difference: {direction.capitalize()} Logprobability - Mean of Transition</sup>"
        elif metric == "frequency":
            color_scale = self.cs_main
            zmin = 0
            zmax = df['metric_val'].max()
            zmid = None
            colorbar_title = "Frequency"
            title = f"Transition Heatmap<br><sup>Highest frequency</sup>"
        else:  # highest or lowest
            color_scale = self.cs_main
            zmin, zmax, zmid = -10, 0, None
            colorbar_title = "Value"
            title = f"Transition Heatmap<br><sup>{metric.capitalize()} Value of {direction.capitalize()} Logprobabilities</sup>"

        #heatmap_data = heatmap_data.sort_index(axis=1)

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values.T,
            x=heatmap_data.index,
            y=heatmap_data.columns,
            colorscale=color_scale,
            showscale=True,
            zmin=zmin,
            zmax=zmax,
            zmid=zmid,
            # customdata=customdata,
            # hovertemplate = "<<%{customdata}>><extra></extra>",
            colorbar=dict(
                title=dict(text=None),
                orientation="h",
                y=1.01,
                yanchor="bottom",
                thickness=24,
                outlinecolor="black",
                outlinewidth=1,
            ),
        ))
        ticks = 10*(np.arange(15)+1)+(150*page)
        fig.update_layout(
            autosize=True,
            template=self.plotly_template,
            margin=dict(l=40, r=40, t=40, b=40),
            dragmode="select",
            title=dict(text = title, ),
            xaxis=dict(
                title=f"Iteration",
                ticks="outside",
                showticklabels=True,
                showline=False,
                zeroline=False,
                showgrid=False,
                showspikes=False,
            ),
            yaxis=dict(
                title=f"Transition Rank ",
                ticks="outside",
                tickvals=heatmap_data.columns[9::10],
                ticktext=[str(int(v)) for v in ticks],
                showticklabels=True,
                showgrid=False,
                showline=False,
                zeroline=False,
                showspikes=False,
            ),
        )
        fig.update_traces(hoverinfo="none", hovertemplate=None)
        fig.update_yaxes(autorange="reversed")

        if metric == "frequency":
            return fig, zmax, edge_list
        return fig, None, edge_list

    def edge_hover_fig(self, edge_data):
        fig = go.Figure()
        mode = "markers" if len(edge_data) == 1 else "lines"

        fig.add_trace(go.Scatter(
            x=edge_data["iteration"],
            y=edge_data["logprobs_forward"],
            mode=mode,
            name="forward",
            line=dict(color=self.cs_diverging_dir[-3]),
            marker=dict(color=self.cs_diverging_dir[-3])
        ))

        fig.add_trace(go.Scatter(
            x=edge_data["iteration"],
            y=edge_data["logprobs_backward"],
            mode=mode,
            name="backward",
            line=dict(color=self.cs_diverging_dir[2]),
            marker=dict(color=self.cs_diverging_dir[2])
        ))

        fig.update_layout(
            height=200,
            width=250,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True,
            legend=dict(x=0.3, y=1.1, orientation="h", xanchor="center", yanchor="bottom"),
            xaxis_title="Iteration",
            yaxis_title="Logprobs",
            template=self.plotly_template
        )

        return fig

    def update_state_space(self, df, selected_ids=[], metric="total_reward"):
        """
        Updates the state space for final objects
        :param df: dataframe with required columns text, x, y, metric, istetset, iteration
        :param selected_ids: list of selected texts
        :param metric: metric for coloring
        :return: updated plot
        """
        check_choice(metric, self.final_object_metrics, "metric")
        # Normalize metric, scale to range 6-30px, set size=4 for missing values (no metric in testset)
        m_min = df[metric].min()
        m_max = df[metric].max()
        if m_max > m_min:
            df["metric_norm"] = 6 + (df[metric] - m_min) / (m_max - m_min) * (30 - 6)
        else:
            # all values equal (or missing): medium size
            df["metric_norm"] = np.where(df[metric].notna(), 18.0, np.nan)
        df["metric_norm"] = df["metric_norm"].fillna(4)

        if selected_ids:
            df["opacity"] = df["id"].isin(selected_ids)*0.9 +0.1
        else:
            df["opacity"] = 0.5


        # Separate test set and normal points
        df_test = df[df['istestset']==1]
        df_normal = df[df['istestset']==0]

        fig = go.Figure()

        if df_test.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_normal["x"],
                    y=df_normal["y"],
                    mode="markers",
                    marker=dict(
                        size=df_normal["metric_norm"],
                        color=df_normal["iteration"],
                        colorscale=self.cs_iteration,
                        line=dict(color="white", width=1),
                        showscale=True,
                        colorbar=dict(title="Iteration", thickness=15, len=0.7),
                        opacity=df_normal["opacity"],
                    ),
                    customdata=df_normal[["id", "iteration", metric, "text"]].values,
                    hoverinfo="none",
                    name="Samples",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df_test["x"],
                    y=df_test["y"],
                    mode="markers",
                    marker=dict(
                        size=df_test["metric_norm"],
                        color=self.cs_diverging_testset[0],
                        line=dict(color="white", width=1),
                        opacity=df_test["opacity"],
                    ),
                    customdata=df_test[["id", "iteration", metric, "text"]].values,
                    hoverinfo="none",
                    name="Test Set",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_normal["x"],
                    y=df_normal["y"],
                    mode="markers",
                    marker=dict(
                        size=df_normal["metric_norm"],
                        color=self.cs_diverging_testset[-1],
                        line=dict(color="white", width=1),
                        opacity=df_normal["opacity"],
                    ),
                    customdata=df_normal[["id", "iteration", metric, "text"]].values,
                    hoverinfo="none",
                    name="Samples",
                )
            )

        fig.update_layout(
            autosize=True,
            title=f"State Projection<br><sup>Size shows {metric} for the latest iteration the object occured",
            template=self.plotly_template,
            legend=dict(
                itemsizing='constant',  # ensures marker size is not scaled
            ),
            margin=dict(l=40, r=40, t=40, b=40)
        )

        return fig

    def prepare_bump_all(self, df, n_top, order):
        """
        Selection-independent part of the cumulative bump chart: for each iteration, rank the
        n_top best objects seen so far (an object keeps its latest metric value).
        The result can be cached and reused for different selections.

        :param df: prepared dataframe (should NOT include testset rows)
        :param n_top: number of top objects to display
        :param order: highest or lowest for highest/lowest rank
        :return: dict with the ranked points ("points") and the iteration range ("it_min", "it_max")
        """
        # Sort key: NaN last, then metric (desc for "highest"), then text as tiebreaker
        sign = -1 if order == "highest" else 1

        def rank_key(item):
            text, metric = item
            is_nan = metric != metric
            return (is_nan, 0 if is_nan else sign * metric, text)

        records = []
        seen_objects = {}
        for it, group in df.groupby("iteration", sort=True):
            # Add objects from this iteration (later rows overwrite earlier ones)
            seen_objects.update(zip(group["text"], group["metric"]))
            top = heapq.nsmallest(n_top, seen_objects.items(), key=rank_key)
            records.extend(
                (it, text, rank, metric)
                for rank, (text, metric) in enumerate(top, start=1)
            )

        pts = pd.DataFrame(records, columns=["iteration", "text", "value", "metric"])

        # Attach IDs
        pts = pts.merge(
            df[["final_id", "text"]].drop_duplicates(subset="text"),
            on="text",
            how="left",
        )

        # Marker size: circle if the object was sampled in that iteration
        sampled_pairs = pd.MultiIndex.from_arrays([df["text"], df["iteration"]])
        pts_pairs = pd.MultiIndex.from_arrays([pts["text"], pts["iteration"]])
        pts["sampled"] = np.where(pts_pairs.isin(sampled_pairs), 6, 0).astype(np.int8)

        # Color by iteration of first appearance (normalized index into the colorscale)
        iterations = np.sort(df["iteration"].unique())
        iter_to_idx = {it: i for i, it in enumerate(iterations)}
        n_colors = len(self.cs_iteration)
        first_iter = pts.groupby("text")["iteration"].min()
        color_idx = first_iter.map(
            lambda it: int(iter_to_idx[it] / max(1, len(iterations) - 1) * (n_colors - 1))
        )
        pts["color_idx"] = pts["text"].map(color_idx)

        # Order: by color, then by best rank (consistent line ordering), then along iterations
        pts["best_rank"] = pts["text"].map(pts.groupby("text")["value"].min())
        pts = pts.sort_values(["color_idx", "best_rank", "text", "iteration"], ignore_index=True)

        return {"points": pts, "it_min": iterations.min(), "it_max": iterations.max()}

    def bump_all_traces(self, prepared, selected_ids):
        """
        Traces for the cumulative bump chart. Objects sharing a color and selection state are
        merged into one trace, separated by NaN gaps, so the number of traces is bounded by
        2 * len(cs_iteration) instead of growing with the number of objects.

        :param prepared: output of prepare_bump_all
        :param selected_ids: list of final_ids to highlight
        :return: list of trace dicts
        """
        pts = prepared["points"]
        if selected_ids:
            selected = pts["final_id"].isin(selected_ids).to_numpy()
        else:
            selected = np.ones(len(pts), dtype=bool)

        traces = []
        # Unselected (faded) first so that selected lines are drawn on top
        for is_selected, opacity in [(False, 0.1), (True, 1)]:
            sub = pts[selected == is_selected]
            for color_idx, g in sub.groupby("color_idx", sort=True):
                # Insert a gap before each new object
                text = g["text"].to_numpy()
                gaps = np.flatnonzero(text[1:] != text[:-1]) + 1
                # float32 (exact for iterations/ranks) keeps the payload small; NaN marks gaps
                x = np.insert(g["iteration"].to_numpy(dtype=np.float32), gaps, np.nan)
                y = np.insert(g["value"].to_numpy(dtype=np.float32), gaps, np.nan)
                size = np.insert(g["sampled"].to_numpy(), gaps, 0)
                customdata = g[["final_id", "value", "metric", "text"]].to_numpy(dtype=object)
                customdata = np.insert(customdata, gaps, None, axis=0)
                color = self.cs_iteration[color_idx]
                traces.append(dict(
                    type="scatter",
                    x=x,
                    y=y,
                    mode="lines+markers",
                    marker=dict(symbol="circle", size=size, color=color),
                    line=dict(width=1, color=color),
                    opacity=opacity,
                    customdata=customdata,
                    showlegend=False,
                    hoverinfo="none",
                ))

        # dummy trace for the colorbar
        traces.append(dict(
            type="scatter",
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=0,
                color=[prepared["it_min"], prepared["it_max"]],
                # explicit [[pos, color], ...]: raw trace dicts (Patch) are not converted by plotly.py
                colorscale=make_colorscale(self.cs_iteration),
                cmin=prepared["it_min"],
                cmax=prepared["it_max"],
                showscale=True,
                colorbar=dict(title=dict(text="Iteration of<br>First Sample")),
            ),
            hoverinfo="none",
            showlegend=False,
        ))
        return traces

    @staticmethod
    def bump_title(fo_metric, order, per_iter):
        """Title of the bump chart."""
        if per_iter:
            sub = "For each Iteration the highest ranking objects of this iteration are shown."
        else:
            sub = (
                "For each Iteration the highest ranking objects so far are shown, "
                "objects from previous iterations persist as long as their rank is "
                "high enough. "
                "Markers show if an object was actually sampled in this iteration."
            )
        return f"Sample Ranking by {fo_metric}, {order} object has highest rank<br><sup>{sub}</sup>"

    def bump_figure(self, traces, title):
        """Wrap bump chart traces into a figure with the shared layout."""
        fig = go.Figure(data=traces)
        fig.update_layout(
            autosize=True,
            title=title,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
            xaxis_title="Iteration",
            yaxis_title="Rank",
            template=self.plotly_template,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        fig.update_yaxes(autorange="reversed")
        return fig

    def update_bump_all(self, df, n_top, selected_ids, fo_metric, order, prepared=None):
        """
        Bump chart for cumulative top-ranked objects where rank evolves as new objects appear.

        :param df: prepared dataframe (should NOT include testset rows)
        :param n_top: number of top objects to display
        :param selected_ids: list of final_ids to highlight
        :param order: highest or lowest for highest/lowest rank
        :param prepared: optional cached output of prepare_bump_all (df and n_top are then ignored)
        :return: Plotly figure
        """
        if prepared is None:
            prepared = self.prepare_bump_all(df, n_top, order)
        return self.bump_figure(
            self.bump_all_traces(prepared, selected_ids),
            self.bump_title(fo_metric, order, per_iter=False),
        )

    @staticmethod
    def prepare_bump_iter(df):
        """
        Selection-independent part of the per-iteration bump chart: dense rank per iteration
        after dropping duplicate objects. The result can be cached.

        :param df: prepared dataframe (should NOT include testset rows)
        :return: dataframe with columns iteration, text, metric, final_id, rank
        """
        df = df.drop_duplicates(subset=["iteration", "text", "metric"])
        df = df.rename(columns={"rank": "oldrank"})
        df["rank"] = df.groupby("iteration")["oldrank"].rank(method="dense", ascending=True).astype(int)
        return df

    def bump_iter_traces(self, prepared, selected_ids):
        """
        Trace for the per-iteration bump chart.

        :param prepared: output of prepare_bump_iter
        :param selected_ids: list of final_ids to highlight
        :return: list with one trace dict
        """
        if selected_ids:
            opacity = prepared["final_id"].isin(selected_ids).to_numpy() * 0.9 + 0.1
        else:
            opacity = 1
        return [dict(
            type="scatter",
            x=prepared["iteration"].to_numpy(),
            y=prepared["rank"].to_numpy(),
            mode="markers",
            marker=dict(
                color=prepared["iteration"].to_numpy(),
                # explicit [[pos, color], ...]: raw trace dicts (Patch) are not converted by plotly.py
                colorscale=make_colorscale(self.cs_iteration),
                colorbar=dict(title=dict(text="Iteration of<br>First Sample")),
                symbol="square",
                line=dict(width=1, color="black"),
                size=10,
                opacity=opacity,
            ),
            line=dict(width=1),
            customdata=prepared[["final_id", "rank", "metric", "text"]].to_numpy(dtype=object),
            showlegend=False,
            hoverinfo="none",
        )]

    def update_bump_iter(self, df, selected_ids, fo_metric, order, prepared=None):
        """
        Bump chart showing the top-ranked objects of each iteration.

        :param df: prepared dataframe (should NOT include testset rows)
        :param selected_ids: list of final_ids to highlight
        :param prepared: optional cached output of prepare_bump_iter (df is then ignored)
        :return: Plotly figure
        """
        if prepared is None:
            prepared = self.prepare_bump_iter(df)
        return self.bump_figure(
            self.bump_iter_traces(prepared, selected_ids),
            self.bump_title(fo_metric, order, per_iter=True),
        )

    def select_all_ids_from_ids(self, selected_ids, iterations):
        """
        When selecting final objects in bump /state space, only one item is displayed for final_ids with the same texts.
        This selects all texts to the selected_ids and all final_ids to these texts.
        :param selected_ids: list of ids
        :param iterations: iteration range
        :return: list of ids
        """
        conn = connect_ro(self.data)
        placeholders = ",".join("?" for _ in selected_ids)
        query = f"""
        
        SELECT DISTINCT t2.final_id
        FROM trajectories t2
        JOIN trajectories t1
          ON t1.text = t2.text
        WHERE t1.final_id IN ({placeholders})
          AND t1.final_object = 1
          AND t2.final_object = 1
          AND t1.iteration BETWEEN ? AND ?
          AND t2.iteration BETWEEN ? AND ?;
        """
        params = selected_ids + iterations + iterations
        result = pd.read_sql_query(query, conn, params=params)["final_id"].tolist()
        return result