import threading
import warnings
from functools import lru_cache
import dash
from dash import dcc, html, Input, Output, State, no_update, dash_table, ctx, Patch
from dash.dash_table.Format import Format, Scheme
import dash_cytoscape as cyto
from plot_utils import *
from pathlib import Path

# Page shell with the dashboard's stylesheet. The accent is Dash 4's own
# interactive color (--Dash-Fill-Interactive-Strong) so buttons match the dcc components.
INDEX_STRING = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
:root {
    --gfs-accent: var(--Dash-Fill-Interactive-Strong, #7f4bc4);
    --gfs-accent-weak: rgba(127, 75, 196, 0.10);
    --gfs-text: rgba(0, 18, 77, 0.87);
    --gfs-text-weak: rgba(0, 12, 51, 0.65);
    --gfs-border: rgba(0, 24, 102, 0.12);
    --gfs-bg: #f7f7fa;
    accent-color: var(--gfs-accent);
}
html, body { margin: 0; height: 100%; }
body {
    font-family: "Open Sans", verdana, arial, sans-serif;
    font-size: 13px;
    color: var(--gfs-text);
    background: #fff;
}
.gfs-hidden { display: none !important; }
.gfs-app { display: flex; width: 100vw; height: 100vh; overflow: hidden; }

/* ---------- sidebar ---------- */
.gfs-sidebar {
    flex: 0 0 200px; height: 100vh; overflow-y: auto; box-sizing: border-box;
    padding: 14px 12px 20px; background: var(--gfs-bg); border-right: 1px solid var(--gfs-border);
    display: flex; flex-direction: column; gap: 20px;
}
.gfs-brand { display: flex; align-items: baseline; justify-content: space-between; }
.gfs-brand-title { font-size: 15px; font-weight: 700; }
.gfs-help { font-size: 12px; font-weight: 600; color: var(--gfs-accent); text-decoration: none; }
.gfs-help:hover { text-decoration: underline; }
.gfs-section { display: flex; flex-direction: column; gap: 10px; }
.gfs-section-title {
    margin: 0; font-size: 11px; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--gfs-text-weak);
}
.gfs-control { display: flex; flex-direction: column; gap: 4px; }
.gfs-label { font-size: 12px; font-weight: 600; }

/* ---------- buttons ---------- */
.gfs-btn {
    font: inherit; font-size: 12px; font-weight: 600; padding: 7px 10px; cursor: pointer;
    color: var(--gfs-text); background: #fff; border: 1px solid var(--gfs-border); border-radius: 6px;
    transition: background-color 0.15s, border-color 0.15s, color 0.15s;
}
.gfs-btn:hover { border-color: var(--gfs-accent); color: var(--gfs-accent); }
.gfs-btn.active { background: var(--gfs-accent); border-color: var(--gfs-accent); color: #fff; }
.gfs-segmented {
    display: flex; gap: 3px; padding: 3px; background: #fff;
    border: 1px solid var(--gfs-border); border-radius: 8px;
}
.gfs-segmented .gfs-btn { flex: 1; padding: 7px 4px; white-space: nowrap; border-color: transparent; }
.gfs-segmented .gfs-btn.active { border-color: var(--gfs-accent); }
.gfs-btn-row { display: flex; gap: 6px; }
.gfs-btn-row .gfs-btn { flex: 1; white-space: nowrap; }

/* ---------- content ---------- */
.gfs-main { flex: 1 1 auto; min-width: 0; height: 100vh; overflow: hidden; }
.gfs-tab { height: 100%; box-sizing: border-box; display: flex; }
.gfs-fill { flex: 1 1 auto; min-height: 0; }
.gfs-tab-ss { flex-direction: column; padding: 5px; }
.gfs-bump { flex: 35 1 0; min-height: 0; }
.gfs-statespace { flex: 65 1 0; min-height: 0; }

/* DAG tab: the three columns shrink down to their min-width, below that the tab scrolls */
.gfs-tab-dag { flex-direction: row; gap: 12px; padding: 10px 12px; overflow-x: auto; }
.gfs-dag-overview { flex: 0 1 380px; min-width: 240px; display: flex; flex-direction: column; gap: 8px; }
.gfs-dag-main { flex: 1 1 360px; min-width: 320px; display: flex; flex-direction: column; }
.gfs-dag-title { font-size: 13px; font-weight: 700; }
.gfs-dag-subtitle { font-size: 11px; color: var(--gfs-text-weak); white-space: pre-line; margin: 2px 0 8px; }
.gfs-dag-canvas { border: 1px solid var(--gfs-border); border-radius: 8px; overflow: hidden; }
.gfs-dag-table { flex: 0 1 460px; min-width: 360px; min-height: 0; display: flex; flex-direction: column; }
.gfs-dag-table > div, .gfs-dag-table .dash-table-container {
    display: flex; flex-direction: column; flex: 1 1 auto; min-height: 0;
}
/* long state texts / large images stay inside their cell */
.gfs-dag-table td[data-dash-column="image"] .dash-cell-value {
    max-height: 120px; overflow: auto; overflow-wrap: anywhere;
}
.gfs-dag-table td[data-dash-column="image"] p { margin: 0; }
/* DataTable styles its own `.dash-tooltip` with a border, which dcc.Tooltip's zero-height
   anchor box shares: without this a line shows on the hovered point */
.dcc-tooltip-bounding-box.dash-tooltip { border: none; }
/* the state space tooltip is placed freely next to the hex, so its caret would point elsewhere */
.hover:has(> #image-tooltip1)::before, .hover:has(> #image-tooltip1)::after { display: none; }
.gfs-dag-table td[data-dash-column="image"] img { max-width: 100%; max-height: 80px; height: auto; display: block; }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
"""

def run_dashboard(
        data: str,
        text_to_img_fn: callable,
        state_aggregation_fn: callable,
        s0: str = "#",
        debug_mode: bool = False):
    """
    Runs the dashboard on http://127.0.0.1:8050/
    :param data: folder of the logged data
    :param text_to_img_fn: function to convert the texts representing the states to base64 encoded svg images.
        Used to identify states on the dashboard
    :param state_aggregation_fn: function to aggregate states.
        In the state space it might help to see what states in a bin have in common.
        This function provides this aggregation.
        This can be eg the Maximum Common Substructure in Molecules / Graphs, paths that must have been taken for all states in Games, etc.
        Must take the states as a list of strings and return one state as a string.
        If not specified the longest common substring of all strings will be used. There is no guarantee that this is a valid state.
    :param s0: Gives the option to specify the start state. By default '#' is used and treated as an empty state.
        If s0 is meaningful in your environment (e.g. the position [0, 0] in a grid) you can specify how your start state looks here.
        In that case it will be displayed via the text_to_img_fn.
    :param debug_mode: whether to display in debug mode for error handling
    :return: None
    """

    data_path = Path(data)
    data_folder = data_path.parent if data_path.suffix else data_path
    if data_folder.exists() and data_folder.is_dir():
        data_path = data_folder / "data.db"
    else:
        raise FileNotFoundError(f"Folder does not exist: {data_folder}")
    if not data_path.exists() or (not data_path.is_file()):
        raise FileNotFoundError(f"Data does not exist: {data_path}")

    # Callbacks run in parallel threads. Image and aggregation functions often draw with
    # matplotlib's pyplot, whose current figure is shared by all threads: concurrent calls
    # can save another thread's unfinished figure (wrong crop / zoom). Serialize them.
    render_lock = threading.RLock()

    if text_to_img_fn is None:
        warnings.warn(
            "No text-to-image function provided. "
            "Identifying states will not be possible. "
            "Provide text_to_img_fn to allow converting text representations of states to images."
        )
        def image_fn(state):
            return None
    else:
        # Cached: images are requested repeatedly for the same states (DAG updates, hovers).
        # Returned lists are shared between calls and must not be mutated.
        @lru_cache(maxsize=4096)
        def image_fn(state):
            if state == "#":
                return None
            with render_lock:
                result = text_to_img_fn(state)
            if isinstance(result, list):
                return result
            return f"data:image/svg+xml;base64,{result}"

    if state_aggregation_fn is None:
        warnings.warn(
            "No state-aggregation-function provided. "
        )
        def agg_fn(states):
            return None
    else:
        # Cached by the (ordered) states, e.g. a molecule MCS is expensive.
        # Returned lists are shared between calls and must not be mutated.
        @lru_cache(maxsize=256)
        def _agg_cached(states):
            with render_lock:
                agg = state_aggregation_fn(list(states))
            if isinstance(agg, str):
                return f"data:image/svg+xml;base64,{agg}"
            return agg

        def agg_fn(states):
            if len(states) == 0:
                return None
            elif len(states) == 1:
                return image_fn(states[0])
            else:
                return _agg_cached(tuple(states))


    plotter = Plotter(data_path, image_fn, agg_fn, s0)

    # provided metrics (read from the db schema, also used as whitelist for SQL column names)
    final_object_metrics = plotter.final_object_metrics
    testset_metrics = plotter.testset_metrics

    # get iteration range
    conn = connect_ro(data_path)
    query = "SELECT MIN(iteration) AS min, MAX(iteration) AS max FROM trajectories"
    iteration_range = pd.read_sql_query(query, conn)
    iteration_step = iteration_range["max"][0] - iteration_range["min"][0]
    iteration_marks = [0, 0.25, 0.5, 0.75, 1]
    iteration_marks = [i*iteration_step for i in iteration_marks]
    iteration_marks = [i+iteration_range["min"][0]for i in iteration_marks]
    iteration_marks = [int(i) for i in iteration_marks]
    iteration_marks_dict = dict()
    for i in iteration_marks:
        iteration_marks_dict[i] = str(i)

    conn.close()

    app = dash.Dash(__name__)
    app.title = "GFlowState"
    app.index_string = INDEX_STRING

    # Load extra layouts for cytoscape
    cyto.load_extra_layouts()

    def control(label, component, label_id=None):
        """Sidebar control: small label above the input."""
        return html.Div([
            html.Label(label, id=label_id, className="gfs-label") if label_id
            else html.Label(label, className="gfs-label"),
            component,
        ], className="gfs-control")

    def section(title, children, title_id=None, section_id=None, class_name="gfs-section"):
        header = html.H4(title, id=title_id, className="gfs-section-title") if title_id \
            else html.H4(title, className="gfs-section-title")
        if section_id:
            return html.Div([header, *children], id=section_id, className=class_name)
        return html.Div([header, *children], className=class_name)

    app.layout = html.Div([
        dcc.Store(id="selected-objects", data=[]),
        dcc.Store(id="build-ids", data= ["#"]),
        dcc.Store(id="max-frequency", data= 0),
        dcc.Store(id="dag-overview-edge-list", data= []),
        dcc.Store(id="dag-overview-page", data=0),
        dcc.Store(id="dag-fit-dummy"),
        dcc.Store(id="active-tab", data="state-space"),

        # ================= LEFT SIDEBAR =================
        html.Div([
            html.Div([
                html.Span("GFlowState", className="gfs-brand-title"),
                html.A(
                    "How to Use ↗",
                    href="https://github.com/florianholeczek/GFlowNet_Training_Vis_Pilot/blob/master/Dashboard_Introduction.md",
                    target="_blank",
                    className="gfs-help",
                ),
            ], className="gfs-brand"),

            # -------- View toggle --------
            section("View", [
                html.Div([
                    html.Button("Final Objects", id="tab-state-space", n_clicks=0, className="gfs-btn active"),
                    html.Button("DAG", id="tab-dag-view", n_clicks=0, className="gfs-btn"),
                ], className="gfs-segmented"),
            ]),

            # -------- General --------
            section("General", [
                control("Iterations", dcc.RangeSlider(
                    id="iteration",
                    min=iteration_range["min"][0],
                    max=iteration_range["max"][0],
                    step=1,
                    value=[iteration_range["min"][0], iteration_range["max"][0]],
                    marks=iteration_marks_dict,
                    allow_direct_input=False,
                    tooltip={"placement": "bottom", "always_visible": False}
                )),
                html.Button("Clear selection", id="clear-selection", n_clicks=0, className="gfs-btn"),
            ]),

            # -------- Projection controls --------
            section("Final Objects", [
                control("Object Metric", dcc.Dropdown(
                    id="fo-metric",
                    options=final_object_metrics,
                    value="total_reward",
                    clearable=False,
                )),
                control("Ranking Metric", dcc.Dropdown(
                    id="ranking-metric",
                    options=["highest over all", "highest per iter", "lowest over all", "lowest per iter"],
                    value="highest over all",
                    clearable=False,
                )),
                control("State Space Style", dcc.Dropdown(
                    id="state-space-style",
                    options=["Hex Ratio", "Hex Obj. Metric", "Hex Correlation", "Scatter"],
                    value="Hex Ratio",
                    clearable=False,
                )),
                control("Dimensionality Reduction", dcc.Dropdown(
                    id="projection-method",
                    options=[
                        {"label": "UMAP", "value": "umap"},
                        {"label": "t-SNE", "value": "tsne"}
                    ],
                    value="tsne",
                    clearable=False,
                )),
                control("", dcc.Slider(
                    id="projection-param",
                    min=5,
                    max=50,
                    step=1,
                    value=15,
                    marks=None,
                    allow_direct_input=False,
                    tooltip={"placement": "bottom", "always_visible": False}
                ), label_id="projection-param-label"),
                dcc.Checklist([" Use Testset"], [], id="use-testset"),
            ], title_id="sidebar-tab-header", section_id="sidebar-projection"),

            # -------- DAG controls --------
            section("DAG", [
                control("Layout", dcc.Dropdown(
                    id="dag-layout",
                    options=[
                        {"label": "Klay", "value": "klay"},
                        {"label": "Dagre", "value": "dagre"},
                        {"label": "Breadthfirst", "value": "breadthfirst"}
                    ],
                    value="klay",
                    clearable=False,
                )),
                control("Metric", dcc.Dropdown(
                    id="dag-metric",
                    options=["highest", "lowest", "variance", "frequency"],
                    value="highest",
                    clearable=False,
                )),
                control("Direction", dcc.RadioItems(
                    id="dag-direction",
                    options=["forward", "backward"],
                    value="forward",
                    inline=True,
                )),
            ], section_id="sidebar-dag", class_name="gfs-section gfs-hidden"),
        ], className="gfs-sidebar"),

        # ================= RIGHT CONTENT AREA =================
        html.Div([

            # ================= STATE-SPACE TAB =================
            html.Div([
                html.Div(
                    dcc.Graph(
                        id="bumpchart",
                        clear_on_unhover=True,
                        style={"height": "100%", "width": "100%"},
                        config={"responsive": True},
                    ),
                    className="gfs-bump",
                ),
                html.Div(
                    dcc.Graph(
                        id="state-space-plot",
                        clear_on_unhover=True,
                        style={"height": "100%", "width": "100%"},
                        config={"responsive": True},
                    ),
                    className="gfs-statespace",
                ),
            ], id="state-space-tab", className="gfs-tab gfs-tab-ss"),

            # ================= DAG TAB =================
            html.Div([
                # DAG Overview
                html.Div([
                    html.Div([
                        html.Button("Top 150", id="dag-overview-button-top", className="gfs-btn"),
                        html.Button("Previous 150", id="dag-overview-button-prev", className="gfs-btn"),
                    ], className="gfs-btn-row"),
                    html.Div(
                        dcc.Graph(
                            id="dag-overview",
                            clear_on_unhover=True,
                            style={"height": "100%", "width": "100%"},
                            config={"responsive": True},
                        ),
                        className="gfs-fill",
                    ),
                    html.Button("Next 150", id="dag-overview-button-next", className="gfs-btn"),
                ], className="gfs-dag-overview"),

                # DAG area
                html.Div([
                    html.Div("DAG-title", id="dag-title", className="gfs-dag-title"),
                    html.Div("DAG-subtitle", id="dag-subtitle", className="gfs-dag-subtitle"),
                    html.Div(
                        cyto.Cytoscape(
                            id='dag-graph',
                            layout={
                                'name': 'klay',
                                'directed': True,
                                'spacingFactor': 0.5,
                                'animate': False
                            },
                            style={'width': '100%', 'height': '100%', 'background-color': '#FFFFFF'},
                            elements=[],
                            stylesheet=[]
                        ),
                        className="gfs-fill gfs-dag-canvas",
                    ),
                ], className="gfs-dag-main"),

                # Data table
                html.Div(
                    dash_table.DataTable(
                        id='dag-table',
                        columns=[
                            {
                                "name": "State",
                                "id": "image",
                                "presentation": "markdown",
                            },
                            {
                                "name": "Final",
                                "id": "node_type",
                                "type": "any",
                            },
                            {
                                "name": "Metric",
                                "id": "logprobs",
                                "type": "numeric",
                                "format": Format(precision=4, scheme=Scheme.fixed),
                            },
                            {
                                "name": "Reward",
                                "id": "reward",
                                "type": "numeric",
                                "format": Format(precision=4, scheme=Scheme.fixed),
                            },
                        ],
                        row_selectable="multi",
                        filter_action="native",
                        sort_action="native",
                        selected_row_ids=[],
                        page_size=10,
                        markdown_options={"html": True},
                        style_table={"overflow": "auto", "flex": "1 1 auto", "minHeight": 0},
                        style_cell={
                            "fontFamily": "\"Open Sans\", verdana, arial, sans-serif",
                            "fontSize": "12px",
                            "padding": "4px 6px",
                            "whiteSpace": "normal",
                            "height": "auto",
                            "backgroundColor": "#FFFFFF",
                            "border": "1px solid var(--gfs-border)",
                        },
                        style_cell_conditional=[
                            {"if": {"column_id": "image"}, "textAlign": "left",
                             "minWidth": "90px", "width": "50%", "maxWidth": "220px"},
                            {"if": {"column_id": "node_type"}, "width": "48px"},
                        ],
                        style_header={
                            "backgroundColor": "var(--gfs-bg)",
                            "fontWeight": "600",
                        },
                        style_data_conditional=[
                            {"if": {"state": "active"},
                             "backgroundColor": "var(--gfs-accent-weak)",
                             "border": "1px solid var(--gfs-accent)"},
                            {"if": {"state": "selected"},
                             "backgroundColor": "var(--gfs-accent-weak)",
                             "border": "1px solid var(--gfs-border)"},
                        ],
                    ),
                    className="gfs-dag-table",
                ),
            ], id="dag-tab", className="gfs-tab gfs-tab-dag gfs-hidden"),

        ], className="gfs-main"),
        dcc.Tooltip(
            id="image-tooltip4",
            direction="right",
            style = {"zIndex": 999, "pointerEvents": "none", "overflow": "visible"}
        ),
        dcc.Tooltip(id="image-tooltip3", direction="bottom"),
        dcc.Tooltip(id="image-tooltip1", direction="top"),

    ], className="gfs-app")


    # ================= CALLBACK TO SWITCH TABS =================
    @app.callback(
        [Output("state-space-tab", "className"),
         Output("dag-tab", "className"),
         Output("active-tab", "data"),
         Output("tab-state-space", "className"),
         Output("tab-dag-view", "className"),
         Output("sidebar-projection", "className"),
         Output("sidebar-dag", "className")],
        [Input("tab-state-space", "n_clicks"),
         Input("tab-dag-view", "n_clicks")],
        [State("active-tab", "data")]
    )
    def switch_tabs(state_clicks, dag_clicks, current_tab):
        button_id = ctx.triggered_id or "tab-state-space"
        if button_id == "tab-state-space":
            active_tab = "state-space"
        elif button_id == "tab-dag-view":
            active_tab = "dag-view"
        else:
            active_tab = current_tab

        show, hide = "", " gfs-hidden"
        ss, dag = (show, hide) if active_tab == "state-space" else (hide, show)
        return (
            "gfs-tab gfs-tab-ss" + ss,
            "gfs-tab gfs-tab-dag" + dag,
            active_tab,
            "gfs-btn active" if active_tab == "state-space" else "gfs-btn",
            "gfs-btn active" if active_tab == "dag-view" else "gfs-btn",
            "gfs-section" + ss,
            "gfs-section" + dag,
        )


    # The DAG is laid out while its tab may be hidden (container width 0), so the
    # viewport is not fitted. Resize and fit it when the DAG tab is shown.
    app.clientside_callback(
        """
        function(activeTab) {
            if (activeTab === "dag-view") {
                requestAnimationFrame(() => {
                    const el = document.getElementById("dag-graph");
                    const cy = el && el._cyreg && el._cyreg.cy;
                    if (cy) {
                        cy.resize();
                        cy.fit(undefined, 30);
                    }
                });
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("dag-fit-dummy", "data"),
        Input("active-tab", "data"),
    )

    # Downprojection parameter header
    @app.callback(
        Output("projection-param-label", "children"),
        Output("projection-param", "min"),
        Output("projection-param", "max"),
        Output("projection-param", "step"),
        Output("projection-param", "marks"),
        Output("projection-param", "value"),
        Input("projection-method", "value")
    )
    def update_projection_param(method):
        if method == "umap":
            return "n_neighbors", 2, 200, 1, {2: "2", 50: "50", 100: "100", 200: "200"}, 15
        else:
            return "perplexity", 5, 50, 1, {5: "5", 25: "25", 50: "50"}, 30

    Input("dag-overview", "selectedData"),

    # Main selection update
    @app.callback(
        Output("selected-objects", "data"),
        Output("dag-table", "data"),
        Output("dag-table", "selected_rows"),
        Input("clear-selection", "n_clicks"),
        Input("state-space-plot", "selectedData"),
        Input("state-space-plot", "clickData"),
        Input("bumpchart", "selectedData"),
        Input("dag-graph", "tapNodeData"),
        Input("dag-overview", "selectedData"),
        State("selected-objects", "data"),
        State("build-ids", "data"),
        State("dag-overview-edge-list", "data"),
        State("iteration", "value"),
        State("projection-method", "value"),
        State("projection-param", "value"),
        State("use-testset", "value"),
        prevent_initial_call=True
    )
    def update_selected_objects(clear_clicks, ss_select, ss_click, bump_select, dag_node, selected_tids, current_ids, build_ids, edge_list, iterations, method, param_value, use_testset):

        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update

        trigger = ctx.triggered[0]["prop_id"]

        # -------- Clear button --------
        if "clear-selection" in trigger:
            return [], None, []

        # ---------- State-space lasso ----------
        if "state-space-plot.selectedData" in trigger:
            if not ss_select or not ss_select.get("points"):
                return no_update

            selected_ids = {pt["customdata"][0] for pt in ss_select["points"]}
            selected_ids = list(selected_ids)
            selected_ids = plotter.select_all_ids_from_ids(selected_ids, iterations)
            return selected_ids, None, []

        # ---------- State-space click ----------
        if "state-space-plot.clickData" in trigger:
            if not ss_click or not ss_click["points"] or ss_click["points"][0]["customdata"][0] != "usehex":
                return no_update

            _, hex_q, hex_r, _, _, _ = ss_click["points"][0]["customdata"]
            dp, _ = plotter.get_projection(iterations, use_testset, method, param_value)
            in_hex = (dp["hex_q"] == int(hex_q)) & (dp["hex_r"] == int(hex_r))
            selected_ids = dp.loc[in_hex, "id"].astype(int).tolist()
            selected_ids = plotter.select_all_ids_from_ids(selected_ids, iterations)

            return selected_ids, None, []

        # ---------- Bump chart lasso ----------
        elif "bumpchart.selectedData" in trigger:
            if not bump_select or not bump_select.get("points"):
                return no_update, None, []

            selected_ids = {pt["customdata"][0] for pt in bump_select["points"]}
            selected_ids = list(selected_ids)
            selected_ids = plotter.select_all_ids_from_ids(selected_ids, iterations)
            return selected_ids, None, []

        # ---------- DAG node click ----------
        elif "dag-graph.tapNodeData" in trigger:
            if not dag_node:
                return no_update
            if dag_node.get("id") == "#":
                # root selection clears selection
                return [], None, []

            iteration0 = int(dag_node.get("iteration0"))
            iteration1 = int(dag_node.get("iteration1"))

            # expand dag, build table
            if dag_node.get("node_type") == 'handler':
                text = dag_node.get("id")[8:]
                # node data comes back from the client: validate before it reaches SQL
                node_metric = check_choice(dag_node.get("metric"), DAG_METRICS, "DAG metric")
                node_direction = check_choice(dag_node.get("direction"), DIRECTIONS, "direction")
                conn = connect_ro(data_path)

                if node_metric in ["highest", "lowest", "variance"]:
                    col = "logprobs_" + node_direction
                    query = f"""
                        WITH ranked AS (
                            SELECT
                                target,
                                iteration,
                                {col} AS logprobs,
                                AVG({col}) OVER (PARTITION BY target) AS avg_logprobs,
                                ROW_NUMBER() OVER (
                                    PARTITION BY target
                                    ORDER BY iteration DESC
                                ) AS rn
                            FROM edges
                            WHERE source = ?
                            AND iteration BETWEEN ? AND ?
                        )
                        SELECT
                            target,
                            logprobs AS latest_logprobs,
                            avg_logprobs
                        FROM ranked
                        WHERE rn = 1;
                        """
                else: #frequency
                    query = f"""
                        SELECT
                            target,
                            COUNT(*) AS latest_logprobs
                        FROM edges
                        WHERE source = ?
                          AND iteration BETWEEN ? AND ?
                        GROUP BY target;
                        """
                children_e = pd.read_sql_query(query, conn, params=[text, iteration0, iteration1])

                if node_metric == "variance":
                    children_e["latest_logprobs"] -= children_e["avg_logprobs"]
                children_e.rename(columns={"target": "id", "latest_logprobs": "logprobs"}, inplace=True)
                children_e = children_e[["id", "logprobs"]]

                targets = list(children_e["id"])
                placeholders = ",".join("?" for _ in targets)
                query = f"""
                            SELECT DISTINCT
                                id, node_type, reward
                            FROM nodes
                            WHERE id IN ({placeholders})
                        """
                children_n = pd.read_sql_query(query, conn, params=targets)
                conn.close()
                children = pd.merge(children_n, children_e, on="id")

                def image_cell(id):
                    state = image_fn(id)
                    if not state:
                        return ""
                    if type(state) is list:
                        return "<br>".join(state)
                    return (
                        f'<img src="{state}" '
                        f'style="height:80px; max-width:100%; object-fit:contain;" />'
                    )

                children["image"] = children["id"].apply(image_cell)
                children["node_type"] = children["node_type"].eq("final")
                selected_row_ids = list(set.intersection(set(build_ids), set(list(children["id"]))))
                selected_rows = [
                    idx for idx, row in enumerate(children.to_dict("records"))
                    if row["id"] in selected_row_ids
                ]
                return no_update, children.to_dict("records"), selected_rows

            # selected trajectories
            else:
                text = dag_node.get("id")
                conn = connect_ro(data_path)
                query = f"""
                            SELECT DISTINCT
                                trajectory_id
                            FROM edges
                            WHERE target = ?
                              AND iteration BETWEEN ? AND ?
                        """
                selected_ids = pd.read_sql_query(query, conn, params=[text, iteration0, iteration1])
                selected_ids = list(selected_ids["trajectory_id"])
                conn.close()
                return (selected_ids, None, []) if selected_ids else ([], None, [])

        elif "dag-overview.selectedData" in trigger:
            if not selected_tids or "range" not in selected_tids:
                return no_update

            # selected rows of the heatmap are edges (source, target),
            # the selected x range is an iteration range within the slider range
            y_range = np.round(selected_tids["range"]["y"]).astype(int)
            selected_edges = (edge_list or [])[max(y_range[0], 0):y_range[1] + 1]
            if not selected_edges:
                return [], None, []
            x_range = np.round(selected_tids["range"]["x"]).astype(int).tolist()
            it0 = max(x_range[0], int(iterations[0]))
            it1 = min(x_range[1], int(iterations[1]))

            # trajectories that use one of the selected edges in the iteration range
            conn = connect_ro(data_path)
            pairs = ",".join("(?, ?)" for _ in selected_edges)
            query = f"""
            SELECT DISTINCT trajectory_id
            FROM edges
            WHERE (source, target) IN (VALUES {pairs})
                AND iteration BETWEEN ? AND ?
            """
            params = [v for edge in selected_edges for v in edge] + [it0, it1]
            selected_ids = pd.read_sql_query(query, conn, params=params)
            selected_ids = selected_ids["trajectory_id"].to_list()
            conn.close()
            return selected_ids, None, []


        return no_update

    # Bump Callback
    @app.callback(
        Output("bumpchart", "figure"),
        Input("iteration", "value"),
        Input("selected-objects", "data"),
        Input("fo-metric", "value"),
        Input("ranking-metric", "value"),
    )
    def bump_callback(iteration, selected_ids, fo_metric, rank_metric):
        order = "highest" if "highest" in rank_metric else "lowest"
        per_iter = "iter" in rank_metric
        # Ranking is independent of the selection, so it is cached
        prepared = bump_data(iteration[0], iteration[1], fo_metric, rank_metric)
        if per_iter:
            traces = plotter.bump_iter_traces(prepared, selected_ids)
        else:
            traces = plotter.bump_all_traces(prepared, selected_ids)
        title = plotter.bump_title(fo_metric, order, per_iter)

        # Only the selection changed: replace the traces, keep the layout
        if set(ctx.triggered_prop_ids) == {"selected-objects.data"}:
            patched = Patch()
            patched["data"] = traces
            patched["layout"]["title"]["text"] = title
            # Remove the lasso/box outline, as a full figure update would
            patched["layout"]["selections"] = []
            return patched
        return plotter.bump_figure(traces, title)

    @lru_cache(maxsize=16)
    def bump_data(it_min, it_max, fo_metric, rank_metric):
        """
        Load and rank the data for the bump chart. Cached; the trajectories table is not
        modified while the dashboard runs. The returned object must not be mutated.
        """
        n_top = 30
        check_choice(rank_metric, RANK_METRICS, "ranking metric")
        fo_col = quote_ident(fo_metric, final_object_metrics)
        order = "DESC" if "highest" in rank_metric else "ASC"
        conn = connect_ro(data_path)
        query = f"""
                SELECT iteration, text, metric, final_id, rank
                FROM (
                    SELECT
                        iteration,
                        text,
                        {fo_col} as metric,
                        final_id,
                        ROW_NUMBER() OVER (PARTITION BY iteration ORDER BY {fo_col} {order}) AS rank
                    FROM trajectories
                    WHERE iteration BETWEEN ? AND ?
                    AND final_object = 1
                ) sub
                WHERE rank <= {n_top}
                ORDER BY iteration, rank;
            """
        df = pd.read_sql_query(query, conn, params=(it_min, it_max))
        conn.close()

        if "iter" in rank_metric:
            return plotter.prepare_bump_iter(df)
        order = "highest" if "highest" in rank_metric else "lowest"
        return plotter.prepare_bump_all(df, n_top, order)

    @lru_cache(maxsize=256)
    def hex_hover_data(iteration, usetestset, method, param_value, hex_q, hex_r, metric, ss_style):
        """
        Figures and state aggregation for the hover of one hex. Cached: the projection for
        these inputs is fixed, so the db queries and the (possibly expensive) aggregation
        only run on the first hover. The returned objects must not be mutated.
        """
        dp, _ = plotter.get_projection(list(iteration), usetestset, method, param_value)
        figures, texts = plotter.hex_hover_figures(
            dp,
            hex_q,
            hex_r,
            metric,
            ss_style,
            metric in testset_metrics,
            usetestset
        )
        return figures, plotter.agg_fn(texts)


    # State Space Callback
    @app.callback(
        Output("state-space-plot", "figure"),
        Input("selected-objects", "data"),
        Input("projection-method", "value"),
        Input("projection-param", "value"),
        Input("iteration", "value"),
        Input("use-testset", "value"),
        Input("fo-metric", "value"),
        Input("state-space-style", "value"),
        Input("active-tab", "data"),
    )
    def update_projection_plots(selected_ids, method, param_value, iteration, use_testset, metric, ss_style, active_tab):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update
        # Not visible on the DAG tab: skip, switching back to this tab triggers the update
        if active_tab != "state-space":
            return no_update
        check_choice(ss_style, (*HEX_STYLES, "Scatter"), "state space style")

        # downprojection, cached for all sessions (only computed if the inputs are new)
        dp, hexbin_size = plotter.get_projection(iteration, use_testset, method, param_value)

        if "Hex" in ss_style:
            df = plotter.get_hexbin_data(dp, ss_style, metric)
            return plotter.update_hex(df, dp, selected_ids, ss_style, metric, use_testset, hexbin_size)
        cols = ["id", "x", "y", "hex_q", "hex_r", "text", "iteration", "istestset", *final_object_metrics]
        return plotter.update_state_space(dp[cols].copy(), selected_ids, metric)



    # DAG Callback
    @app.callback(
        [Output("dag-graph", "elements"),
         Output("dag-graph", "stylesheet"),
         Output("dag-graph", "layout"),
         Output("dag-title", "children"),
         Output("dag-subtitle", "children")],
        Input("dag-layout", "value"),
        Input("dag-direction", "value"),
        Input("dag-metric", "value"),
        Input(  "iteration", "value"),
        Input("selected-objects", "data"),
        Input("build-ids", "data"),
        Input("max-frequency", "data")
    )
    def update_dag(layout_name, direction, metric, iteration, selected_objects, build_ids, max_freq):
        add_handlers = True
        if selected_objects:
            # If final objects are selected via another vis, display the full dag of these
            conn = connect_ro(data_path)
            placeholders = ",".join("?" for _ in selected_objects)
            query = f"""
                        SELECT DISTINCT
                            target
                        FROM edges
                        WHERE trajectory_id IN ({placeholders})
                          AND iteration BETWEEN ? AND ?
                    """
            build_ids = pd.read_sql_query(query, conn, params=selected_objects +  [iteration[0], iteration[1]])
            build_ids = list(build_ids['target']) + ['#']
            conn.close()
            add_handlers = False
        elif not build_ids:
            build_ids = ["#"]

        result = plotter.update_DAG(
            iteration,
            direction,
            metric,
            max_freq,
            add_handlers,
            build_ids=build_ids,
        )

        # Configure layout based on selection
        layout_config = {
            'name': layout_name,
            'directed': True,
            'animate': False
        }

        # Add layout-specific parameters
        if layout_name == 'klay':
            layout_config['spacingFactor'] = 1.2
        elif layout_name == 'dagre':
            layout_config['spacingFactor'] = 1
            layout_config['rankDir'] = 'LR'  # Left to right
        elif layout_name == 'breadthfirst':
            layout_config['spacingFactor'] = 1.2
            layout_config['roots'] = '[id = "#"]'

        if add_handlers:
            title = "DAG-View, Mode: Expand"
            subtitle = "Click on 'Select children' nodes to expand the Graph and click on the root to collapse it. Select a node or items from other visuals to switch to selection mode. Edge coloring: "
        else:
            title = "DAG-View, Mode: Selection"
            subtitle = "Shows all trajectories going through the selected items. Clear selection or select the root to switch to expanding mode. Edge coloring: "
        if metric in ["highest", "lowest"]:
            subtitle += f"{metric.capitalize()} {direction} logprobabilities of the edge over selected iterations."
        elif metric == "variance":
            subtitle += f"Latest {direction} logprobability of the edge (in selected iterations) - mean ({direction} logprobabilities) over selected iterations."
        elif metric == "frequency":
            subtitle += "Frequency of the edge over selected iterations."
        return result['elements'], result['stylesheet'], layout_config, title, subtitle

    # Callback for dag-table
    @app.callback(
        Output('build-ids', 'data'),
        Input('dag-table', 'selected_row_ids'),
        Input("dag-graph", "tapNodeData"),
        State('dag-table', 'data'),
        State("build-ids", "data")
    )
    def save_selected_rows(selected_rows, node_select, table_data, build_ids):
        #reset build ids if root selected
        if "dag-graph.tapNodeData" in dash.callback_context.triggered[0]["prop_id"]:
            if node_select.get("id") == "#":
                return ["#"]
            else:
                return no_update
        # update build ids from table
        if selected_rows or table_data:
            children = set([r["id"] for r in table_data])
            unselected = children - set(selected_rows)
            new_build_ids = set(build_ids) - unselected
            if len(new_build_ids) < len(build_ids):
                removed_node = set(build_ids) - new_build_ids
                assert len(removed_node)==1, "Bug in DAG selection"
                build_ids = plotter.dag_remove_node_prune(build_ids, list(removed_node)[0])
            build_ids = set(selected_rows) | set(build_ids) | set(["#"])
            return list(build_ids)
        else:
            return ["#"]

    #dag overview
    @app.callback(
        Output("dag-overview", "figure"),
        Output("max-frequency", "data"),
        Output("dag-overview-edge-list", "data"),
        Input("dag-direction", "value"),
        Input("dag-metric", "value"),
        Input("iteration", "value"),
        Input("dag-overview-page", "data")
    )
    def update_dag_overview(direction, metric, iteration, page):
        fig, max_freq, edge_list = plotter.update_DAG_overview(direction, metric, iteration, page)
        if max_freq:
            return fig, max_freq, edge_list
        return fig, no_update, edge_list

    # Position of the state space tooltip, computed in the browser where window and
    # tooltip sizes are known: it opens to the side of the hex with more room and is
    # shifted so it stays inside the window. The content size is only known once the
    # server callback's content has rendered, so a ResizeObserver re-places it then.
    app.clientside_callback(
        """
        function(hoverData) {
            const nu = window.dash_clientside.no_update;
            const b = hoverData && hoverData.points.length && hoverData.points[0].bbox;
            const el = document.getElementById("image-tooltip1");
            if (!b || !el) { return [nu, nu]; }
            const cx = (b.x0 + b.x1) / 2;
            const cy = (b.y0 + b.y1) / 2;
            const graph = document.getElementById("state-space-plot").getBoundingClientRect();
            const direction = window.innerWidth - cx >= cx - graph.left ? "right" : "left";
            window.gfsTooltip = {cx: cx, cy: cy, direction: direction, hh: (b.y1 - b.y0) / 2};

            const place = () => {
                const s = window.gfsTooltip;
                const W = window.innerWidth, H = window.innerHeight;
                const w = el.offsetWidth, h = el.offsetHeight, pad = 8, gap = 10;
                if (!s || !w) { return; }
                const clamp = (v, size, max) => Math.max(pad, Math.min(v, max - size - pad));
                let x, y;
                const side = s.direction === "right" ? W - s.cx : s.cx;
                if (side >= w + gap + pad) {
                    // beside the hex, vertically centered
                    x = s.direction === "right" ? s.cx + gap : s.cx - gap - w;
                    y = clamp(s.cy - h / 2, h, H);
                } else {
                    // too wide for either side: above or below the hex, whichever has more room
                    x = clamp(s.cx - w / 2, w, W);
                    y = H - s.cy >= s.cy ? s.cy + s.hh + gap : s.cy - s.hh - gap - h;
                    y = clamp(y, h, H);
                }
                el.style.transform = "none";
                const natural = el.getBoundingClientRect();
                el.style.transform = `translate(${x - natural.left}px, ${y - natural.top}px)`;
            };
            if (!window.gfsTooltipObserver) {
                window.gfsTooltipObserver = new ResizeObserver(place);
                window.gfsTooltipObserver.observe(el);
            }
            // after Dash has applied bbox and direction
            setTimeout(() => requestAnimationFrame(place), 0);
            return [b, direction];
        }
        """,
        Output("image-tooltip1", "bbox"),
        Output("image-tooltip1", "direction"),
        Input("state-space-plot", "hoverData"),
    )

    #hover state space
    @app.callback(
        Output("image-tooltip1", "show"),
        Output("image-tooltip1", "children"),
        Input("state-space-plot", "hoverData"),
        State("fo-metric", "value"),
        State("state-space-style", "value"),
        State("use-testset", "value"),
        State("iteration", "value"),
        State("projection-method", "value"),
        State("projection-param", "value"),
    )
    def display_image_tooltip1(hoverData, metric, ss_style, usetestset, iteration, method, param_value):
        if hoverData is None:
            return False, None
        customdata = hoverData["points"][0]["customdata"]
        if customdata is None:
            return False, None

        if customdata[0] == "usehex":
            _, hex_q, hex_r, metric_value, n_samples, n_test = customdata
            if ss_style == "Hex Ratio":
                if usetestset:
                    metric_title = f"Log Odds Ratio: {metric_value:.2f}"
                else:
                    metric_title = ""
            elif ss_style == "Hex Obj. Metric":
                metric_title = f"Average {metric} (Samples): {metric_value:.4f}"
            else:
                if metric_value is None or np.isnan(metric_value):
                    metric_title = "Not enough samples for Correlation"
                else:
                    metric_title = f"Correlation log reward and logprobs: {metric_value:.2f}"


            figures, aggregation = hex_hover_data(
                (int(iteration[0]), int(iteration[1])),
                bool(usetestset),
                method,
                int(param_value),
                int(hex_q),
                int(hex_r),
                metric,
                ss_style,
            )
            # explicit size so the tooltip has its final height on first render (used for positioning)
            fig_graphs = [
                dcc.Graph(
                    figure=fig,
                    config={"displayModeBar": False},
                    style={"height": f"{fig.layout.height}px", "width": f"{fig.layout.width}px"}
                )
                for fig in figures
                if fig is not None
            ]
            if isinstance(aggregation, list):
                aggregation = [html.Div(i, style={"color": "black", "marginTop": "5px"}) for i in aggregation]
            elif isinstance(aggregation, str):
                aggregation = html.Img(src=aggregation, style={"width": "150px", "height": "150px"})

            info = html.Div([
                html.Div(
                    aggregation,
                    style={"marginRight": "15px"}
                ),
                html.Div([
                    html.Div(f" Unique Samples: {n_samples}", style={"color": "black", "marginTop": "5px"}),
                    html.Div(f" Testset Objects: {n_test}", style={"color": "black", "marginTop": "5px"}),
                    html.Div(metric_title, style={"color": "black", "marginTop": "5px"}),
                ]),
            ], style={
                "display": "flex",
                "alignItems": "center"  # vertical alignment
            })

            # 2-column grid keeps the tooltip short enough to fit next to the hovered hex
            children = [
                html.Div([*fig_graphs, info], style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(2, auto)",
                    "gap": "8px 12px",
                    "alignItems": "center",
                })
            ]

        else:
            _, iteration, metric_data, text = customdata
            metric_data = float('nan') if metric_data is None else metric_data
            if ss_style == "Hex Ratio":
                texts = [html.Div(f"Iteration: {iteration}", style={"color": "black"}),]
            else:
                texts = [
                    html.Div(f"Iteration: {iteration}", style={"color": "black"}),
                    html.Div(f"{metric}: {metric_data:.4f}", style={"color": "black"}),
                ]

            children = [
                html.Div([
                    *plotter.html_from_imagefn(text),
                    *texts
                ])
            ]

        return True, children


    #hover bump plot
    @app.callback(
        Output("image-tooltip3", "show"),
        Output("image-tooltip3", "bbox"),
        Output("image-tooltip3", "children"),
        Input("bumpchart", "hoverData"),
        State("fo-metric", "value")
    )
    def display_image_tooltip3(hoverData, fo_metric):
        if hoverData is None:
            return False, None, None

        point = hoverData["points"][0]

        # Check if this point has customdata (skip shading area)
        if "customdata" not in point or point["customdata"] is None:
            return False, None, None

        bbox = point["bbox"]
        _, rank, metric, text = point["customdata"]

        children = [
            html.Div([
                *plotter.html_from_imagefn(text),
                html.Div(f"Rank: {rank}", style={"color": "black"}),
                html.Div(f"{fo_metric}: {metric:.4f}", style={"color": "black"}),
            ])
        ]

        return True, bbox, children

    @app.callback(
        Output("image-tooltip4", "show"),
        Output("image-tooltip4", "bbox"),
        Output("image-tooltip4", "children"),
        Input("dag-overview", "hoverData"),
        State("dag-overview-edge-list", "data")
    )
    def display_image_tooltip4(hoverData, edge_list):
        if hoverData is None or hoverData["points"][0]["z"] is None:
            return False, None, None

        value = hoverData["points"][0]["z"]
        bbox = hoverData["points"][0]["bbox"]
        idx = hoverData["points"][0]["y"]
        source, target = edge_list[idx]

        # get data
        conn = connect_ro(data_path)
        query = "SELECT DISTINCT iteration, logprobs_forward, logprobs_backward FROM edges WHERE source = ? AND target = ?"
        edge_data = pd.read_sql_query(query, conn, params=[source, target])
        conn.close()

        # make images
        source_img = plotter.html_from_imagefn(source)
        target_img = plotter.html_from_imagefn(target)

        def make_img(img):
            if img is None:
                return html.Div("s0")
            return img

        children = html.Div(
            [
                html.Div(
                    f"Value: {value:.2f}",
                    style={"fontWeight": "bold", "fontFamily": "sans-serif"},
                ),
                html.Div(
                    [
                        html.Div(
                            [html.Div("Source:"), *make_img(source_img)],
                            style={
                                "marginTop": "5px",
                                "marginRight": "50px",
                                "display": "flex",
                                "alignItems": "center",
                                "flexDirection": "column",
                                "minHeight": "80px",
                                "justifyContent": "flex-start",
                            }
                        ),
                        html.Div(
                            [html.Div("Target:"), *make_img(target_img)],
                            style={"marginTop": "5px"}
                        ),
                    ],
                    style={
                        "display": "flex",
                        "marginTop": "5px",
                        "alignItems": "flex-start",
                    },
                ),
                dcc.Graph(
                    figure=plotter.edge_hover_fig(edge_data),
                    config={"displayModeBar": False},
                    style={"marginTop": "10px"}
                ),
            ],
            style={
                "color": "black",
                "backgroundColor": "white",
                "padding": "10px",
                "height": "auto",
                #"width": "260px",
            },
        )

        return True, bbox, children

    @app.callback(
        Output("dag-overview-page", "data"),
        Output("dag-overview-button-top", "disabled"),
        Output("dag-overview-button-prev", "disabled"),
        Output("dag-overview-button-next", "disabled"),
        Input("dag-overview-button-top", "n_clicks"),
        Input("dag-overview-button-next", "n_clicks"),
        Input("dag-overview-button-prev", "n_clicks"),
        Input("iteration", "value"),
        Input("dag-metric", "value"),
        State("dag-overview-page", "data"),
    )
    def dag_overview_buttons(top, nxt, prev, iterations, metric, page):
        trigger = dash.callback_context.triggered[0]["prop_id"]
        check_choice(metric, DAG_METRICS, "DAG metric")
        page = max(int(page or 0), 0)
        if "top" in trigger or "iteration" in trigger or "metric" in trigger:
            page = 0
        elif "prev" in trigger:
            page = max(page - 1, 0)
        elif "next" in trigger:
            page += 1
        disabled_top = page ==0
        conn = connect_ro(data_path)
        if metric == "variance":
            query = """
                SELECT COUNT(*) AS count
                FROM (
                    SELECT source, target
                    FROM edges
                    WHERE iteration BETWEEN ? AND ?
                    GROUP BY source, target
                    HAVING COUNT(*) > 1
                ) t;
            """
        else:
            query = "SELECT COUNT(*) AS count FROM (SELECT DISTINCT source, target FROM edges WHERE iteration BETWEEN ? AND ?);"

        n_edges = pd.read_sql_query(query, conn, params=iterations)["count"].tolist()[0]
        conn.close()
        disabled_bottom = (page+1)*150 >= n_edges
        return page, disabled_top, disabled_top, disabled_bottom

    # Run the dashboard
    #app.run(debug=debug_mode)
    return app