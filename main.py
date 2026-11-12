import ast
import io

import matplotlib
import numpy as np

# non-GUI backend: the image functions run in Flask worker threads, where the
# default Tk backend crashes the server (Tcl_AsyncDelete)
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from dashboard import run_dashboard

import base64



"""
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS

def imagefn_seh(smiles):
    if smiles is None:
        return None

    def smiles_to_mol(smiles):
        mol=None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return mol
        except Exception:
            pass
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                return mol
        except Exception:
            pass
        try:
            mol = Chem.MolFromSmarts(smiles)
            if mol is not None:
                return mol
        except Exception:
            pass

        if mol is None:
           return None
    mol = smiles_to_mol(smiles)
    svg = Draw.MolsToGridImage(
        [mol],
        molsPerRow=1,
        subImgSize=(200, 200),
        useSVG=True
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")

    return b64

#state aggregation seh
def state_agg_fn_seh(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    #find mcs
    mcs_result = rdFMCS.FindMCS(mols)
    #convert
    mcs_smarts = mcs_result.smartsString
    mcs_mol = Chem.MolFromSmarts(mcs_smarts)
    mcs_smiles = Chem.MolToSmiles(mcs_mol)
    mcs_smiles = imagefn_seh(mcs_smiles)
    return mcs_smiles
    
"""

#plotting function for debugdata
def imagefn_debugdata(s):
    dots = [(i%3, i//3) for i in range(int(s))]
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="60" height="100">' + \
          ''.join(f'<circle cx="{10+x*20}" cy="{10+y*20}" r="5" fill="black"/>' for x,y in dots) + \
          '</svg>'
    b64 = base64.b64encode(svg.encode()).decode()
    return b64

#aggregation function for debugdata
def state_agg_fn_debugdata(texts):
    return imagefn_debugdata(min([int(i) for i in texts]))

def dummyimagefn(text):
    # to test text display
    a= "abcd"
    return [a*10, a*5, a]


#Grid
def grid_readable(states):
    return [ast.literal_eval(s) for s in states]

def grid_imagefn(state):
    state = grid_readable([state])[0]
    size = 20
    length = 20
    w = h = length * size
    x, y = state
    grid_lines = []
    for i in range(length + 1):
        pos = i * size
        grid_lines.append(
            f'<line x1="{pos}" y1="0" x2="{pos}" y2="{h}" stroke="lightgray"/>'
        )
        grid_lines.append(
            f'<line x1="0" y1="{pos}" x2="{w}" y2="{pos}" stroke="lightgray"/>'
        )
    highlight = (
        f'<rect x="{x * size}" y="{y * size}" '
        f'width="{size}" height="{size}" fill="black"/>'
    )
    border = (
        f'<rect x="0" y="0" width="{w}" height="{h}" '
        f'fill="none" stroke="black" stroke-width="2"/>'
    )
    svg = f"""
                <svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
                    <g transform="scale(1,-1) translate(0,-{h})">
                        <rect width="100%" height="100%" fill="white"/>
                        {''.join(grid_lines)}
                        {border}
                        {highlight}
                    </g>
                </svg>
                """
    return base64.b64encode(svg.encode()).decode()

def grid_aggregation(states):
    length = 20
    buffer = io.BytesIO()
    states = grid_readable(states)
    grid = np.ones((length, length))
    for x, y in states:
        grid[y, x] = 0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        grid,
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=[-0.5, length - 0.5, -0.5, length - 0.5],
        origin="lower",
    )
    for i in range(length + 1):
        pos = i - 0.5
        ax.axhline(pos, color="lightgrey", linewidth=1)
        ax.axvline(pos, color="lightgrey", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, length - 0.5)
    ax.set_ylim(-0.5, length - 0.5)
    ax.set_aspect("equal")
    plt.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    svg_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return svg_base64



app = run_dashboard(
    data="grid",
    text_to_img_fn=grid_imagefn,
    state_aggregation_fn=grid_aggregation,
    s0="[0, 0]",
    debug_mode=False)
server = app.server

"""
run_dashboard(
    data="seh_mid",
    text_to_img_fn=imagefn_seh,
    state_aggregation_fn=state_agg_fn_seh,
    s0="#",
    debug_mode=True)
"""

"""
run_dashboard(
    data="debugdata", 
    text_to_img_fn=imagefn_debugdata, 
    state_aggregation_fn= state_agg_fn_debugdata, 
    debug_mode=True
)
"""


if __name__ == "__main__":
    app.run(debug=True)