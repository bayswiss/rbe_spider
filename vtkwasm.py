"""Interactive VTK.wasm scenes for Jupyter Book pages.

`scene_data` packs a PyVista grid into a JSON payload and `vtkwasm_view` returns
the HTML widget that runs the filters in the browser through VTK.wasm.
"""

import base64
import gzip
import json
import uuid

import matplotlib
import numpy as np
import pyvista as pv
from IPython.display import HTML

# VTK Lagrange cell type -> (linear cell on the corner vertices, number of corners)
LAGRANGE = {
    68: (pv.CellType.LINE, 2),
    69: (pv.CellType.TRIANGLE, 3),
    70: (pv.CellType.QUAD, 4),
    71: (pv.CellType.TETRA, 4),
    72: (pv.CellType.HEXAHEDRON, 8),
    73: (pv.CellType.WEDGE, 6),
}


def _pack(array, dtype) -> str:
    raw = np.ascontiguousarray(array, dtype=dtype).tobytes()
    return base64.b64encode(gzip.compress(raw)).decode("ascii")


def _cmap(name: str, n: int = 16) -> list:
    rgba = matplotlib.colormaps[name](np.linspace(0.0, 1.0, n))
    return [[float(t), *map(float, c[:3])] for t, c in zip(np.linspace(0, 1, n), rgba)]


def scene_data(
    grid: pv.UnstructuredGrid,
    cmap: str = "viridis",
    subdivision: int = 2,
    initial: dict | None = None,
    spider: pv.PolyData | None = None,
    frames: int = 40,
) -> dict:
    """Pack `grid` for `vtkwasm_view`.

    `initial` preselects controls, e.g. `{"color": "|u_3|", "warp": "by u_3"}`
    using the labels of the colour and warp menus, and `{"opacity": 0.4}` for
    the other controls.

    `spider` is an optional set of line cells, such as the connections of a
    multi-point constraint. It is shown with its nodes, and moves with the
    point arrays of `grid` that it has too; the others count as zero there.

    A mode is played as cos(2 pi step / `frames`) times its displacement, in `frames` steps.
    """
    surface = grid.extract_surface(
        nonlinear_subdivision=subdivision, algorithm="dataset_surface"
    )
    surface = surface.triangulate()

    ctypes = np.unique(grid.celltypes)
    if ctypes.max() < 68:  # linear cells: the surface edges are the cell edges
        edges = surface.extract_all_edges()
    else:  # Lagrange cells: edges between the cell vertices only
        # (grid.cells_dict fails for Lagrange cells: pyvista can't size them)
        if not np.isin(ctypes, list(LAGRANGE)).all():
            raise ValueError(f"Cannot mix Lagrange and other cell types: {ctypes}")
        # (`offset` was renamed `cell_offsets` in recent pyvista)
        offsets = getattr(grid, "cell_offsets", None)
        offsets = grid.offset if offsets is None else offsets
        conn_all = grid.cell_connectivity
        corner_cells = {}
        for ct in ctypes:
            linear_type, n = LAGRANGE[int(ct)]
            starts = offsets[:-1][grid.celltypes == ct]
            corner_cells[linear_type] = conn_all[starts[:, None] + np.arange(n)]
        linear = pv.UnstructuredGrid(corner_cells, grid.points)
        linear.point_data.update(grid.point_data)
        # In 3D only the edges on the surface are of interest
        solid = any(int(ct) in (71, 72, 73) for ct in ctypes)
        edges = (
            linear.extract_surface(algorithm="dataset_surface") if solid else linear
        ).extract_all_edges()

    arrays, colors, warps = [], [], [{"label": "None", "expr": "coords", "scale": 0}]
    diag = float(np.linalg.norm(np.ptp(grid.points, axis=0)))
    for k, name in enumerate(grid.point_data.keys()):
        values = np.asarray(grid.point_data[name], dtype=float)
        ncomp = 1 if values.ndim == 1 else values.shape[1]
        if ncomp == 2:  # DOLFINx 2D vectors: pad to 3 components for VTK
            values = np.c_[values, np.zeros(len(values))]
            ncomp = 3
        vid = f"a{k}"
        arrays.append({"id": vid, "name": name, "ncomp": ncomp})
        if ncomp == 1:
            colors.append(
                {"label": name, "expr": vid, "range": [values.min(), values.max()]}
            )
            peak = np.abs(values).max() or 1.0
            warps.append(
                {
                    "label": f"{name} (normal to plane)",
                    "expr": f"coords + F*{vid}*kHat",
                    "scale": 0.15 * diag / peak,
                }
            )
        else:
            mag = np.linalg.norm(values, axis=1)
            colors.append(
                {
                    "label": f"|{name}|",
                    "expr": f"mag({vid})",
                    "range": [mag.min(), mag.max()],
                }
            )
            for c, axis in enumerate("XYZ"[:ncomp]):
                colors.append(
                    {
                        "label": f"{name} {axis}",
                        "expr": f"{vid}_{c}",
                        "range": [values[:, c].min(), values[:, c].max()],
                    }
                )
            warps.append(
                {
                    "label": f"by {name}",
                    "expr": f"coords + F*{vid}",
                    "scale": 0.1 * diag / (mag.max() or 1.0),
                }
            )

    def point_arrays(ds):
        return {a["name"]: _pack(ds.point_data[a["name"]], np.float32) for a in arrays}

    tris = surface.regular_faces
    lines = edges.lines.reshape(-1, 3)[:, 1:]  # every edge is a two-point line
    labels = {
        "color": [c["label"] for c in colors],
        "warp": [w["label"] for w in warps],
    }
    start = {
        k: labels[k].index(value) if k in labels else value
        for k, value in (initial or {}).items()
    }
    spider_part = None
    if spider is not None:
        segments = spider.lines.reshape(-1, 3)[:, 1:]  # every connection is a two-point line
        zeros = lambda a: np.zeros((spider.n_points, a["ncomp"]))
        padded = {
            a["name"]: np.asarray(spider.point_data[a["name"]])
            if a["name"] in spider.point_data
            else zeros(a)
            for a in arrays
        }
        spider_part = {
            "points": _pack(spider.points, np.float32),
            "n_cells": len(segments),
            "connectivity": _pack(segments.ravel(), np.int32),
            "point_data": {k: _pack(x, np.float32) for k, x in padded.items()},
        }
    return {
        "frames": frames,
        "spider": spider_part,
        "initial": start,
        "bounds": list(map(float, grid.bounds)),
        "arrays": arrays,
        "colors": [{**c, "range": list(map(float, c["range"]))} for c in colors],
        "warps": [{**w, "scale": float(w["scale"])} for w in warps],
        "cmap": _cmap(cmap),
        "surface": {
            "points": _pack(surface.points, np.float32),
            "n_cells": len(tris),
            "connectivity": _pack(tris.ravel(), np.int32),
            "point_data": point_arrays(surface),
        },
        "edges": {
            "points": _pack(edges.points, np.float32),
            "n_cells": len(lines),
            "connectivity": _pack(lines.ravel(), np.int32),
            "point_data": point_arrays(edges),
        },
    }


JS_LIB = r"""
(function () {
  if (window.VtkWasmDemo) return;
  const LOADER = "https://cdn.jsdelivr.net/npm/@kitware/vtk-wasm@3.0.6/dist/umd/vtk.umd.js";
  // The bundle is pinned to a commit of the dist branch, and both files are checked against their hashes
  const BUNDLE = "https://raw.githack.com/Kitware/vtk-wasm/4499bb4085cb77a9685f3845b7451d36d6bce13d/releases/9.7.1/vtk-9.7.1-wasm32-emscripten.tar.gz";
  const LOADER_HASH = "sha384-tuHyZJc0/V4SGO7F1KWSeE4xi0iIn1bcQSr4I/ZubmxKRe1wH9P42+IzWEB3nC2R";
  const BUNDLE_HASH = "BFRn3JABpjdDFCzKLY1iP93UoeaRLpDAGyWgadgsgi7eel9UviXgcGiWFhFSvPvu";
  let runtime = null;
 
  function loadScript(src, integrity) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src; s.integrity = integrity; s.crossOrigin = "anonymous"; s.onload = resolve; s.onerror = () => reject(new Error("Could not load " + src));
      document.head.appendChild(s);
    });
  }
  // Download the bundle and check its hash. That needs a secure context (https, localhost, file): on a page
  // served over plain http the bundle is loaded unchecked, from the same pinned commit.
  async function verifiedBundle() {
    if (!window.crypto?.subtle) return BUNDLE;
    const response = await fetch(BUNDLE);
    if (!response.ok) throw new Error("Could not load " + BUNDLE);
    const bytes = await response.arrayBuffer();
    const digest = new Uint8Array(await crypto.subtle.digest("SHA-384", bytes));
    if (btoa(String.fromCharCode(...digest)) !== BUNDLE_HASH) throw new Error("The VTK.wasm bundle does not match its hash");
    return URL.createObjectURL(new Blob([bytes], { type: "application/gzip" }));
  }
  function getRuntime() {
    runtime ??= (async () => {
      if (!window.vtkwasm) await loadScript(LOADER, LOADER_HASH);
      return window.vtkwasm.loadAsync({ url: await verifiedBundle() });
    })();
    return runtime;
  }
  async function unpack(b64, Type) {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Type(await new Response(stream).arrayBuffer());
  }
 
  // Newer bundles copy through the wasm heap; 9.7.1 accepts the TypedArray directly.
  const CLASS = { Float32Array: "vtkTypeFloat32Array", Int32Array: "vtkTypeInt32Array" };
  function toVtk(session, typed, ncomp = 1, name = "") {
    try { return session.typedArrayInterface.toVTKAoSArray(typed, ncomp, name); }
    catch (e) {
      const a = session.vtk[CLASS[typed.constructor.name]]({ numberOfComponents: ncomp, name });
      a.setArray(typed);
      return a;
    }
  }
 
  async function polyData(session, part, arrays, kind) {
    const vtk = session.vtk;
    const nper = kind === "polys" ? 3 : 2;
    const points = vtk.vtkPoints();
    await points.setData(toVtk(session, await unpack(part.points, Float32Array), 3));
    const offsets = new Int32Array(part.n_cells + 1).map((_, k) => nper * k);
    const cells = vtk.vtkCellArray();
    await cells.setData(toVtk(session, offsets), toVtk(session, await unpack(part.connectivity, Int32Array)));
    const pd = vtk.vtkPolyData();
    await pd.setPoints(points);
    await (kind === "polys" ? pd.setPolys(cells) : pd.setLines(cells));
    const pointData = await pd.getPointData();
    for (const a of arrays) {
      await pointData.addArray(toVtk(session, await unpack(part.point_data[a.name], Float32Array), a.ncomp, a.name));
    }
    return pd;
  }
 
  async function calculator(vtk, arrays, input, coordinates) {
    const calc = vtk.vtkArrayCalculator();
    await calc.setInputConnection(input);
    await calc.addCoordinateVectorVariable("coords");
    for (const a of arrays) {
      if (a.ncomp === 1) { await calc.addScalarVariable(a.id, a.name, 0); continue; }
      await calc.addVectorVariable(a.id, a.name);
      for (let c = 0; c < a.ncomp; c++) await calc.addScalarVariable(`${a.id}_${c}`, a.name, c);
    }
    if (coordinates) await calc.setCoordinateResults(1);
    else await calc.setResultArrayName("__s");
    await calc.setFunction("coords");
    return calc;
  }
 
  // Pure pipeline, no DOM: surface -> warp -> colour scalar -> [plane clip] -> [range] -> mapper
  async function buildPipeline(session, data) {
    const vtk = session.vtk;
    const surface = await polyData(session, data.surface, data.arrays, "polys");
    const edges = await polyData(session, data.edges, data.arrays, "lines");
    const source = vtk.vtkTrivialProducer(); await source.setOutput(surface);
    const edgeSource = vtk.vtkTrivialProducer(); await edgeSource.setOutput(edges);
    let spiderWarp = null;
    if (data.spider) {
      const spider = await polyData(session, data.spider, data.arrays, "lines");
      const spiderSource = vtk.vtkTrivialProducer(); await spiderSource.setOutput(spider);
      spiderWarp = await calculator(vtk, data.arrays, await spiderSource.getOutputPort(), true);
    }
 
    const warp = await calculator(vtk, data.arrays, await source.getOutputPort(), true);
    const scalar = await calculator(vtk, data.arrays, await warp.getOutputPort(), false);
    const edgeWarp = await calculator(vtk, data.arrays, await edgeSource.getOutputPort(), true);
 
    const plane = vtk.vtkPlane();
    const clip = vtk.vtkClipPolyData(); await clip.setClipFunction(plane);
    const edgeClip = vtk.vtkClipPolyData(); await edgeClip.setClipFunction(plane);
    const above = vtk.vtkClipPolyData(); await above.setInputArrayToProcess(0, 0, 0, 0, "__s");
    const below = vtk.vtkClipPolyData(); await below.setInputArrayToProcess(0, 0, 0, 0, "__s");
    await below.setInsideOut(1);
    await below.setInputConnection(await above.getOutputPort());
    const contour = vtk.vtkContourFilter(); await contour.setInputArrayToProcess(0, 0, 0, 0, "__s");
 
    const lut = vtk.vtkColorTransferFunction();
    const mapper = vtk.vtkPolyDataMapper({ lookupTable: lut });
    await mapper.setScalarModeToUsePointFieldData();
    await mapper.selectColorArray("__s");
    const actor = vtk.vtkActor({ mapper });
    const contourMapper = vtk.vtkPolyDataMapper(); await contourMapper.setScalarVisibility(0);
    await contourMapper.setInputConnection(await contour.getOutputPort());
    const contourActor = vtk.vtkActor({ mapper: contourMapper });
    const edgeMapper = vtk.vtkPolyDataMapper(); await edgeMapper.setScalarVisibility(0);
    const edgeActor = vtk.vtkActor({ mapper: edgeMapper });
    // The spider: the connections as lines, and their nodes as points, moving with the solid
    const extra = [], extraProps = {};
    if (spiderWarp) {
      for (const [key, color, draw] of [["spider", [0.85, 0.33, 0.1], (p) => p.setLineWidth(2)],
                                        ["nodes", [0.1, 0.35, 0.85], async (p) => { await p.setRepresentationToPoints(); await p.setPointSize(5); }]]) {
        const m = vtk.vtkPolyDataMapper(); await m.setScalarVisibility(0);
        await m.setInputConnection(await spiderWarp.getOutputPort());
        const a = vtk.vtkActor({ mapper: m });
        const p = await a.getProperty(); await p.setColor(color); await draw(p);
        await a.setVisibility(0);
        extra.push(a); extraProps[key] = a;
      }
    }
    const props = {
      surface: await actor.getProperty(),
      contour: await contourActor.getProperty(),
      edges: await edgeActor.getProperty(),
    };
 
    async function apply(s) {
      const w = data.warps[s.warp];
      // "F" in the expression is a placeholder for the scale of the warp: the scale from the data, times the slider,
      // times the phase of the animation. The expressions are made in scene_data, which has no other capital F.
      const fn = w.expr.replace("F", String(w.scale * s.warpFactor * s.phase));
      await warp.setFunction(fn); await edgeWarp.setFunction(fn);
      if (spiderWarp) {
        await spiderWarp.setFunction(fn);
        await extraProps.spider.setVisibility(s.showSpider ? 1 : 0);
        await extraProps.nodes.setVisibility(s.showNodes ? 1 : 0);
      }
      const col = data.colors[s.color];
      await scalar.setFunction(col.expr);
 
      const b = data.bounds, axis = s.clipAxis;
      const origin = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2];
      origin[axis] = b[2 * axis] + s.clipAt * (b[2 * axis + 1] - b[2 * axis]);
      const normal = [0, 0, 0]; normal[axis] = s.clipFlip ? -1 : 1;
      await plane.setOrigin(origin); await plane.setNormal(normal);
 
      let port = await scalar.getOutputPort();
      let edgePort = await edgeWarp.getOutputPort();
      if (s.clip) {
        await clip.setInputConnection(port); port = await clip.getOutputPort();
        await edgeClip.setInputConnection(edgePort); edgePort = await edgeClip.getOutputPort();
      }
      await contour.setInputConnection(port);
      if (s.range) {
        await above.setInputConnection(port); await above.setValue(s.rangeMin);
        await below.setValue(s.rangeMax); port = await below.getOutputPort();
      }
      await mapper.setInputConnection(port);
      await edgeMapper.setInputConnection(edgePort);
 
      const [lo, hi] = col.range;
      await lut.removeAllPoints();
      for (const [t, r, g, bl] of data.cmap) await lut.addRGBPoint(lo + t * (hi - lo), r, g, bl);
      await mapper.setScalarRange(lo, hi);
      const pad = (hi - lo) / (2 * s.contours);
      await contour.generateValues(s.contours, [lo + pad, hi - pad]);
 
      await contourActor.setVisibility(s.showContours ? 1 : 0);
      await edgeActor.setVisibility(s.showEdges ? 1 : 0);
      await props.surface.setOpacity(s.opacity);
    }
    return { apply, actors: [actor, contourActor, edgeActor, ...extra], props, mapper, contour, below, clip, scalar };
  }
 
  function cssRgb(el, varNames, fallback) {
    const probe = document.createElement("span");
    const style = getComputedStyle(el);
    probe.style.color = varNames.map((n) => style.getPropertyValue(n).trim()).find(Boolean) || fallback;
    el.appendChild(probe);
    const m = getComputedStyle(probe).color.match(/[\d.]+/g).map(Number);
    probe.remove();
    return m.slice(0, 3).map((v) => v / 255);
  }
 
  function controls(data, state, onChange) {
    const frames = data.frames || 40;
    const opts = (list) => list.map((o, i) => `<option value="${i}">${o.label}</option>`).join("");
    const panel = document.createElement("div");
    panel.className = "vwd-controls";
    panel.innerHTML = `
      <div class="vwd-steps">
        <button type="button" data-a="play" aria-label="Play">▶</button>
        <input type="range" data-k="frame" min="0" max="${frames - 1}" step="1" value="0" aria-label="Step">
        <output data-o="step"></output>
        <output data-o="anim"></output>
      </div>
      <label>Colour by <select data-k="color">${opts(data.colors)}</select></label>
      <label>Warp <select data-k="warp">${opts(data.warps)}</select></label>
      <label>Warp scale <input type="range" data-k="warpFactor" min="0" max="4" step="0.05" value="1"> <output data-o="warpFactor"></output></label>
      <label>Opacity <input type="range" data-k="opacity" min="0.1" max="1" step="0.05" value="1"></label>
      <fieldset><legend><label><input type="checkbox" data-k="clip"> Clip</label></legend>
        <select data-k="clipAxis"><option value="0">x</option><option value="1">y</option><option value="2">z</option></select>
        <input type="range" data-k="clipAt" min="0" max="1" step="0.01" value="0.5">
        <label><input type="checkbox" data-k="clipFlip"> Flip</label></fieldset>
      <fieldset><legend><label><input type="checkbox" data-k="range"> Threshold</label></legend>
        <input type="range" data-k="rangeMin" step="any"> <input type="range" data-k="rangeMax" step="any">
        <output data-o="range"></output></fieldset>
      <fieldset><legend><label><input type="checkbox" data-k="showContours"> Contours</label></legend>
        <input type="number" data-k="contours" min="1" max="40" value="${state.contours}"></fieldset>
      <label><input type="checkbox" data-k="showEdges"> Mesh edges</label>
      ${data.spider ? `<label><input type="checkbox" data-k="showSpider"> Spider lines</label>
      <label><input type="checkbox" data-k="showNodes"> Spider nodes</label>` : ""}
      <button type="button" data-a="reset">Reset view</button>`;
    const legend = document.createElement("div");
    legend.className = "vwd-legend";
    const stops = data.cmap.map(([t, r, g, b]) => `rgb(${r * 255},${g * 255},${b * 255}) ${t * 100}%`).join(",");
    legend.innerHTML = `<span></span><i style="background:linear-gradient(90deg,${stops})"></i><span></span>`;
 
    const fmt = (v) => Number(v).toPrecision(3);
    function readout() {
      panel.querySelector("[data-o=warpFactor]").textContent = "×" + state.warpFactor.toFixed(2);
      const phase = Math.abs(state.phase) < 0.005 ? 0 : state.phase;  // not "-0.00"
      panel.querySelector("[data-o=step]").textContent = `step ${state.frame} / ${frames - 1}`;
      panel.querySelector("[data-o=anim]").textContent = state.warp === 0
        ? "Choose a warp to see the motion"
        : `${data.warps[state.warp].label} · cos(2π·${state.frame}/${frames}) = ${(phase >= 0 ? "+" : "") + phase.toFixed(2)}`;
    }
    function syncRange() {
      const [lo, hi] = data.colors[state.color].range;
      for (const k of ["rangeMin", "rangeMax"]) {
        const el = panel.querySelector(`[data-k=${k}]`);
        el.min = lo; el.max = hi; el.step = (hi - lo) / 200;
      }
      state.rangeMin = Math.max(lo, Math.min(state.rangeMin, hi));
      state.rangeMax = Math.max(state.rangeMin, Math.min(state.rangeMax, hi));
      panel.querySelector("[data-k=rangeMin]").value = state.rangeMin;
      panel.querySelector("[data-k=rangeMax]").value = state.rangeMax;
      panel.querySelector("[data-o=range]").textContent = `${fmt(state.rangeMin)} to ${fmt(state.rangeMax)}`;
      const [l, h] = legend.querySelectorAll("span");
      l.textContent = fmt(lo); h.textContent = fmt(hi);
    }
    panel.addEventListener("input", (e) => {
      const k = e.target.dataset.k;
      if (!k) return;
      const v = e.target.type === "checkbox" ? e.target.checked : Number(e.target.value);
      state[k] = v;
      if (k === "frame") state.phase = Math.cos((2 * Math.PI * v) / frames);
      if (k === "color") { [state.rangeMin, state.rangeMax] = data.colors[v].range; }
      if (k === "rangeMin" && v > state.rangeMax) state.rangeMax = v;
      if (k === "rangeMax" && v < state.rangeMin) state.rangeMin = v;
      syncRange();
      readout();
      onChange();
    });
    for (const el of panel.querySelectorAll("[data-k]")) {
      const v = state[el.dataset.k];
      if (el.type === "checkbox") el.checked = !!v; else if (v !== undefined) el.value = v;
    }
    syncRange();
    readout();
    return { panel, legend, readout };
  }
 
  async function mount(root, data, autoload = false) {
    const id = "vwd-" + Math.random().toString(36).slice(2);
    root.innerHTML = `<div class="vwd-view"><div class="vwd-stage"><canvas id="${id}" tabindex="0"></canvas></div>
      <div class="vwd-poster"><button type="button">Load interactive view</button>
      <small>Downloads VTK.wasm, about 24 MB the first time</small></div></div>`;
    const view = root.querySelector(".vwd-view");
    const poster = root.querySelector(".vwd-poster");
    const canvas = root.querySelector("canvas");
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
 
    let started = false;
    const start = async () => {
      if (started) return;
      started = true;
      poster.innerHTML = "<small>Loading VTK.wasm…</small>";
      try {
        const session = (await getRuntime()).createStandaloneSession();
        const vtk = session.vtk;
        const pipe = await buildPipeline(session, data);
        const renderer = vtk.vtkRenderer({ background: cssRgb(root, ["--pst-color-background", "--jp-layout-color0"], "#fff") });
        for (const a of pipe.actors) await renderer.addActor(a);
        const ink = cssRgb(root, ["--pst-color-text-base", "--jp-ui-font-color1"], "#222");
        await pipe.props.contour.setColor(ink);
        await pipe.props.edges.setColor(ink);
        await pipe.props.edges.setOpacity(0.35);
        const canvasSelector = "#" + id;
        const renderWindow = vtk.vtkRenderWindow({ canvasSelector });
        await renderWindow.addRenderer(renderer);
        const interactor = vtk.vtkRenderWindowInteractor({ canvasSelector, renderWindow });
        await interactor.setInteractorStyle(vtk.vtkInteractorStyleTrackballCamera());
 
        const b = data.bounds;
        async function resetView() {
          const camera = await renderer.getActiveCamera();
          const c = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2];
          await camera.setFocalPoint(c);
          await camera.setPosition([c[0], c[1], c[2] + 1]);
          await camera.setViewUp([0, 1, 0]);
          await renderer.resetCamera();
        }
        const state = {
          color: 0, warp: 0, warpFactor: 1, opacity: 1, clip: false, clipAxis: 0, clipAt: 0.5,
          clipFlip: false, range: false, rangeMin: data.colors[0].range[0], rangeMax: data.colors[0].range[1],
          showContours: false, contours: 10, showEdges: false, showSpider: false, showNodes: false,
          frame: 0, playing: false, phase: 1, ...data.initial,
        };
        [state.rangeMin, state.rangeMax] = data.colors[state.color].range;
        let busy = false, pending = false;
        async function update() {
          if (busy) { pending = true; return; }
          busy = true;
          try { await pipe.apply(state); await renderWindow.render(); }
          finally { busy = false; }
          if (pending) { pending = false; update(); }
        }
        const { panel, legend, readout } = controls(data, state, update);
        panel.querySelector("[data-a=reset]").addEventListener("click", async () => { await resetView(); update(); });
        // Play and pause, like the VCR controls of ParaView: the step runs at a fixed rate and loops
        const frames = data.frames || 40, fps = 15;
        const playButton = panel.querySelector("[data-a=play]");
        const slider = panel.querySelector("[data-k=frame]");
        let raf = 0, last = 0, elapsed = 0;
        function setStep(step) {
          state.frame = ((step % frames) + frames) % frames;
          state.phase = Math.cos((2 * Math.PI * state.frame) / frames);
          slider.value = state.frame;
          readout();
        }
        function tick(t) {
          if (!state.playing) { raf = 0; return; }
          elapsed += last ? t - last : 0;
          last = t;
          const steps = Math.floor((elapsed * fps) / 1000);
          if (steps === 0) { raf = requestAnimationFrame(tick); return; }
          elapsed -= (steps * 1000) / fps;
          setStep(state.frame + steps);
          update().finally(() => { raf = requestAnimationFrame(tick); });
        }
        playButton.addEventListener("click", () => {
          state.playing = !state.playing;
          playButton.textContent = state.playing ? "⏸" : "▶";
          playButton.setAttribute("aria-label", state.playing ? "Pause" : "Play");
          last = 0; elapsed = 0;
          if (state.playing && !raf) raf = requestAnimationFrame(tick);
        });
        view.after(legend, panel);
        await pipe.apply(state);
        await resetView();
        poster.remove();
        await renderWindow.render();
        await interactor.start();
      } catch (err) {
        console.error(err);
        poster.innerHTML = `<small>The interactive view could not start: ${err.message}</small>`;
      }
    };
    poster.querySelector("button").addEventListener("click", start);
    if (autoload) {
      // Wait until the scene is on screen, so pages that show it further down load nothing up front
      if ("IntersectionObserver" in window) {
        const seen = new IntersectionObserver((entries) => {
          if (entries.some((e) => e.isIntersecting)) { seen.disconnect(); start(); }
        });
        seen.observe(root);
      } else start();
    }
  }
 
  window.VtkWasmDemo = { mount, buildPipeline, unpack };
})();
"""

CSS = """
.vwd { font: inherit; font-size: 1rem; color: var(--pst-color-text-base, var(--jp-ui-font-color1, inherit));
  background: var(--pst-color-background, var(--jp-layout-color0, transparent)); margin: 1rem 0; padding: .5rem; border-radius: 6px; }
.vwd legend { font-size: 1em; width: auto; margin: 0; padding: 0 .3rem; float: none; }
.vwd-view { position: relative; border: 1px solid var(--pst-color-border, var(--jp-border-color1, #ccc));
  border-radius: 6px; overflow: hidden; background: var(--pst-color-background, var(--jp-layout-color0, #fff)); }
.vwd-stage { position: relative !important; width: 100% !important; height: 460px !important; }
.vwd-view canvas { width: 100%; height: 100%; display: block; outline: none; }
.vwd-poster { position: absolute; inset: 0; display: flex; flex-direction: column; gap: .5rem;
  align-items: center; justify-content: center; text-align: center; padding: 1rem; }
.vwd-poster button, .vwd-controls button { font: inherit; padding: .4rem .9rem; cursor: pointer;
  border: 1px solid var(--pst-color-primary, var(--jp-brand-color1, #2563eb)); border-radius: 4px;
  background: var(--pst-color-primary, var(--jp-brand-color1, #2563eb)); color: var(--pst-color-background, var(--jp-layout-color0, #fff)); }
.vwd-legend { display: grid; grid-template-columns: auto 1fr auto; gap: .5rem; align-items: center;
  margin: .5rem 0; font-size: .85em; font-variant-numeric: tabular-nums; }
.vwd-legend i { height: .6rem; border-radius: 3px; }
.vwd-controls { display: grid; grid-template-columns: repeat(auto-fill, minmax(14rem, 1fr));
  gap: .5rem 1rem; font-size: .9em; align-items: start; }
.vwd-controls label { display: flex; gap: .4rem; align-items: center; flex-wrap: wrap; }
.vwd-controls fieldset { border: 1px solid var(--pst-color-border, var(--jp-border-color1, #ccc)); border-radius: 4px;
  padding: .3rem .6rem .5rem; margin: 0; display: flex; gap: .4rem; align-items: center; flex-wrap: wrap; }
.vwd-controls input[type=range] { flex: 1; min-width: 6rem; }
.vwd-controls input[type=number] { width: 4.5rem; }
.vwd-controls select, .vwd-controls input { font: inherit; }
.vwd-controls output { font-variant-numeric: tabular-nums; }
.vwd-steps { grid-column: 1 / -1; display: flex; flex-wrap: wrap; align-items: center; gap: .3rem .6rem; }
.vwd-steps button { min-width: 2.4rem; }
.vwd-steps input[type=range] { flex: 1; min-width: 8rem; }
.vwd-steps output { white-space: nowrap; }
.vwd-steps output[data-o=anim] { flex-basis: 100%; }
"""


def vtkwasm_view(data: dict, autoload: bool = False) -> HTML:
    """HTML for one interactive scene; works in Jupyter and in a built Jupyter Book page.

    With `autoload`, VTK.wasm is fetched as soon as the scene scrolls into view
    instead of waiting for a click on the poster.
    """
    uid = uuid.uuid4().hex
    payload = json.dumps(data).replace("</", "<\\/")
    return HTML(
        f"<style>{CSS}</style>"
        f'<div class="vwd" id="vwd-{uid}"></div>'
        f'<script type="application/json" id="vwd-data-{uid}">{payload}</script>'
        f"<script>{JS_LIB}</script>"
        "<script>window.VtkWasmDemo.mount("
        f'document.getElementById("vwd-{uid}"),'
        f'JSON.parse(document.getElementById("vwd-data-{uid}").textContent),{json.dumps(autoload)});</script>'
    )
