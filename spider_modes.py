# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Implementation
#
# Copyright (C) 2026 Antonio Baiano Svizzero and Jørgen S. Dokken
#
# SPDX-License-Identifier: MIT
#
# Modal analysis of a spider: 17 separate parts joined by 16 pins, standing on 8 pinned feet.
# Pin j: the lug bore (tag 101+j) and the clevis bores (tag 201+j) are tied by RBE3 to two spiders
# at the pin centre, joined by a spring that lets the pin turn (K_HINGE) and holds everything else.
# Foot k: the pad bottom (tag 301+k) is tied by RBE2 to a spider held by a spring to the ground.
# Units: mm, N, t, s.

#  We start by importing the relevant libraries

# +
import warnings
from contextlib import ExitStack
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

import basix.ufl
import dolfinx.fem.petsc
import dolfinx_mpc
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.io import VTXWriter, gmsh
# -

# Next we define the various material and model parameters.

E, NU, RHO = 210e3, 0.3, 7.85e-9  # steel
K_PIN, K_TILT, K_HINGE = (
    2.1e8,
    5.25e9,
    1.0,
)  # pin spring: translation, rotation across and about the pin
K_GROUND = 2.1e8
DEGREE = 1  # polynomial degree of the displacement field, the mesh itself is linear
N_MODES = 6  # number of modes to extract

# ## Mesh and spider generation

# We read the mesh from [GMSH](https://gmsh.info/) and find the centres and axes of the pin bores, and the centres of the feet.

# +
comm = MPI.COMM_WORLD
msh = gmsh.read_from_msh("spider.msh", comm, gdim=3)
domain, tags = msh.mesh, msh.facet_tags


def vertices(tag):
    nodes = mesh.entities_to_geometry(domain, 2, tags.find(tag)).ravel()
    return np.unique(np.vstack(comm.allgather(domain.geometry.x[nodes])), axis=0)


def fit_pin(lug, clevis):
    # The bores lie on a cylinder x.Q.x + b.x + c = 0: its axis is the eigenvector of Q with eigenvalue 0
    o = lug.mean(axis=0)
    x, y, z = (np.vstack([lug, clevis]) - o).T
    q = np.linalg.svd(
        np.column_stack([x * x, y * y, z * z, x * y, x * z, y * z, x, y, z, 1 + 0 * x])
    )[2][-1]
    eigenvalues, eigenvectors = np.linalg.eigh(
        [[2 * q[0], q[3], q[4]], [q[3], 2 * q[1], q[5]], [q[4], q[5], 2 * q[2]]]
    )
    axis = eigenvectors[:, np.argmin(abs(eigenvalues))]
    # Across the axis the bores are a circle u^2 + v^2 = a u + b v + c, centred at (a/2, b/2)
    e1, e2 = np.linalg.svd([axis])[2][1:]
    u, v = (lug - o) @ e1, (lug - o) @ e2
    a, b, c = np.linalg.lstsq(np.column_stack([u, v, 1 + 0 * u]), u * u + v * v)[0]
    centre = o + a / 2 * e1 + b / 2 * e2
    s = (lug - centre) @ axis
    return centre + (s.min() + s.max()) / 2 * axis, axis


centres, axes, feet = [], [], []
for j in range(16):
    centre, axis = fit_pin(vertices(101 + j), vertices(201 + j))
    centres.append(centre)
    axes.append(axis)
for k in range(8):
    feet.append(vertices(301 + k).mean(axis=0))
# -

# We define the function space for all the components, and separate nodal meshes for each of the spiders, that will be
# used to connect the various parts of `domain`.

V = fem.functionspace(domain, ("Lagrange", DEGREE, (3,)))
spider = basix.ufl.element("DG", "point", 0, shape=(6,))  # translation and rotation
lug_mesh = dolfinx_mpc.create_spider_mesh(comm, centres)
clevis_mesh = dolfinx_mpc.create_spider_mesh(comm, centres)
foot_mesh = dolfinx_mpc.create_spider_mesh(comm, feet)
W_lug = fem.functionspace(lug_mesh, spider)
W_clevis = fem.functionspace(clevis_mesh, spider)
W_foot = fem.functionspace(foot_mesh, spider)

# Next, we define a {py:class}`dolfinx_mpc.MultiPointConstraint` for each of the four function spaces,
# and add the RBE2 and RBE3 constraints that tie the spiders to the solid parts.

mpc_u = dolfinx_mpc.MultiPointConstraint(V)
mpc_lug = dolfinx_mpc.MultiPointConstraint(W_lug)
mpc_clevis = dolfinx_mpc.MultiPointConstraint(W_clevis)
mpc_foot = dolfinx_mpc.MultiPointConstraint(W_foot)
mpc_lug.add_rbe3_topological(V, 2, [tags.find(101 + j) for j in range(16)])
mpc_clevis.add_rbe3_topological(V, 2, [tags.find(201 + j) for j in range(16)])
mpc_u.add_rbe2_topological(2, [tags.find(301 + k) for k in range(8)], W_foot)
mpcs = [mpc_u, mpc_lug, mpc_clevis, mpc_foot]
dolfinx_mpc.finalize_multipointconstraints(mpcs)

# ## Variational formulation
# As we have function spaces on multiple meshes, we use {py:class}`ufl.MixedFunctionSpace`
# to create the coupled problem

unknowns = ufl.MixedFunctionSpace(V, W_lug, W_clevis, W_foot)
u, w_lug, w_clevis, w_foot = ufl.TrialFunctions(unknowns)
v, z_lug, z_clevis, z_foot = ufl.TestFunctions(unknowns)

# We start by defining the elastic energy on each of the components of the spider

mu, lmbda = E / (2 * (1 + NU)), E * NU / ((1 + NU) * (1 - 2 * NU))
eps_u, eps_v = ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))
elastic = (
    2 * mu * ufl.inner(eps_u, eps_v) + lmbda * ufl.tr(eps_u) * ufl.tr(eps_v)
) * ufl.dx(domain)

# Next, we define the bilinear forms for the pin and ground springs.
# The pins are modelled as a spring between the two spiders at the pin centre, with a stiffness K_PIN in the direction of the pin axis, and a stiffness K_TILT across the axis. The ground springs are modelled as a spring between the spider at the foot centre and the ground,
# with a stiffness K_GROUND in all directions.

axis = fem.Function(
    fem.functionspace(lug_mesh, basix.ufl.element("DG", "point", 0, shape=(3,)))
)
axis.interpolate(
    lambda x: (
        np.array(axes)[
            np.argmin(np.linalg.norm(x.T[:, None] - centres, axis=2), axis=1)
        ].T
    )
)
along = ufl.outer(axis, axis)
across = ufl.Identity(3) - along
gap, gap_v = w_clevis - w_lug, z_clevis - z_lug
t = lambda w: ufl.as_vector([w[0], w[1], w[2]])
r = lambda w: ufl.as_vector([w[3], w[4], w[5]])
pins = (
    K_PIN * ufl.dot(t(gap), t(gap_v))
    + ufl.dot((K_TILT * across + K_HINGE * along) * r(gap), r(gap_v))
) * ufl.dx(lug_mesh)
ground = K_GROUND * ufl.dot(t(w_foot), t(z_foot)) * ufl.dx(foot_mesh)

# The total stiffness is the sum of the elastic, pin and ground contributions

stiffness = elastic + pins + ground

# ## Assembly and eigenvalue problem
# The mass is defined only on the solid part of the spider.
# The stiffness and mass matrices are assembled with {py:func}`dolfinx_mpc.assemble_matrix`

stiffness = ufl.extract_blocks(stiffness)
mass = RHO * ufl.inner(ufl.TrialFunction(V), ufl.TestFunction(V)) * ufl.dx
mass = [
    [mass, None, None, None],  # only the solid has mass
    [None, None, None, None],
    [None, None, None, None],
    [None, None, None, None],
]
pair = dolfinx_mpc.create_spider_pair(lug_mesh, clevis_mesh)
K = dolfinx_mpc.assemble_matrix(
    fem.form(stiffness, entity_maps=[pair]), mpcs, diagval=1.0, kind="mpi"
)
M = dolfinx_mpc.assemble_matrix(fem.form(mass), mpcs, diagval=1e-14, kind="mpi")

# The eigenvalue problem $K x = \lambda M x$, with $\lambda = \omega^2$, is solved with {py:class}`slepc4py.SLEPc.EPS`.
# A few things are subtle here:
#
# - The constrained degrees of freedom stay in the matrices, as rows with the diagonal value `diagval`: 1 in $K$ and
#   $10^{-14}$ in $M$. That gives them the eigenvalue $10^{14}$, far from the ones we look for, so they never show up among the modes.
# - The mass form is only defined on the solid, so the spiders have no mass of their own. The constraints move the mass of the solid
#   dofs they tie to the spider that is their master, as for the pads of the feet, but the mass matrix is still singular for the
#   spiders that are only dependent. The problem is therefore solved with shift-and-invert around a target, rather than directly.
#   The target is negative, so that $K - \sigma M$ is well defined also for the modes with $\lambda \approx 0$,
#   and the modes closest to it, the lowest ones, are found first.
# - `N_MODES` modes are requested, but the solver may find fewer. {py:meth}`getConverged<slepc4py.SLEPc.EPS.getConverged>` tells
#   how many converged, and only those are read below.

eigensolver = SLEPc.EPS().create(comm)
eigensolver.setOperators(K, M)
eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eigensolver.setDimensions(N_MODES)
eigensolver.setTarget(-1e3)
eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
eigensolver.getST().setType("sinvert")
eigensolver.getST().getKSP().getPC().setType("lu")
eigensolver.getST().getKSP().getPC().setFactorSolverType("mumps")
eigensolver.solve()

# ## Visualization
# ### Files for ParaView
# For visualizing the spiders, we create a mesh of line cells
# connecting the spider points to the nodes of the solid parts.
# The displacements of the spiders are computed from the displacements of the solid parts with {py:func}`dolfinx_mpc.spider_values`, and the displacements of the line cells are gathered on process 0 to be written to a file.

# + tags = ["hide-input"]
# Spider lines: each spider is a point joined to the nodes of its surface
X = V.tabulate_dof_coordinates()
owned = V.dofmap.index_map.size_local
spiders = []  # (point, surface tag, block, index)
for j in range(16):
    spiders.append((centres[j], 101 + j, 1, j))
    spiders.append((centres[j], 201 + j, 2, j))
for k in range(8):
    spiders.append((feet[k], 301 + k, 3, k))
line_points, line_cells, line_dofs = [], [], []
for point, tag, block, index in spiders:
    dofs = fem.locate_dofs_topological(V, 2, tags.find(tag))
    line_dofs.append(dofs[dofs < owned])
    first = len(line_points)
    line_points.append(point)
    line_points.extend(np.vstack(comm.allgather(X[line_dofs[-1]])))
    for n in range(first + 1, len(line_points)):
        line_cells.append([first, n])

functions = [fem.Function(m.function_space) for m in mpcs]

# The eigenvectors are stored in a vector made by DOLFINx for the four function spaces. It is ghosted, with the owned
# dofs of the four spaces one after the other on each process, which is the layout of K, followed by their ghost values.
# {py:func}`dolfinx.fem.petsc.assign` copies between a vector of this layout and the four functions. The blocks of K
# follow the order of the list `mpcs` that it was assembled with, so the spaces are taken from that list too.

# +
spaces = [m.function_space for m in mpcs]
x = dolfinx.fem.petsc.create_vector(
    spaces, kind="mpi"
)  # the kind of K and M, assembled above
# The sizes at least are those of K
assert x.getSizes() == K.getSizes()[0], (
    f"The vector {x.getSizes()} does not have the layout of K {K.getSizes()[0]}"
)

n_modes = min(N_MODES, eigensolver.getConverged())
if n_modes < N_MODES:
    warnings.warn(f"Only {n_modes} of the {N_MODES} requested modes converged")
assert n_modes > 0, "No modes converged"
# -

modes, line_modes, frequencies = [], {}, []
for i in range(n_modes):
    eigenvalue = eigensolver.getEigenpair(i, x).real
    # The solver only sets the owned entries: get the ghost values from their owners, and copy the
    # entries of the four spaces to the four functions. The constrained dofs are part of x too, but get
    # no value from the solver: backsubstitution below computes them from the dofs they are tied to.
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    dolfinx.fem.petsc.assign(x, functions)
    for m in mpcs:
        m.backsubstitution(functions)
    # The eigenvalue is omega^2. It can be a tiny negative number from round-off for a mode that costs no energy,
    # such as the hinge modes below, so it is clipped at 0 before the square root.
    frequencies.append(np.sqrt(max(eigenvalue, 0)) / (2 * np.pi))
    if comm.rank == 0:
        print(f"mode {i}: {frequencies[-1]:.2f} Hz")

    mode = fem.Function(V, name=f"u_{i}")
    mode.x.array[: 3 * owned] = functions[0].x.array[: 3 * owned]
    mode.x.scatter_forward()
    # An eigenvector has no natural size, nor sign: scale it so that the largest displacement is 10 mm,
    # which is only for display. The spiders are scaled by the same factor to stay consistent with the solid.
    scale = 10 / comm.allreduce(np.abs(mode.x.array).max(), op=MPI.MAX)
    mode.x.array[:] *= scale
    modes.append(mode)
    values = []
    for (point, tag, block, index), dofs in zip(spiders, line_dofs):
        values.append(scale * dolfinx_mpc.spider_values(functions[block], index)[:3])
        values.extend(np.vstack(comm.allgather(mode.x.array.reshape(-1, 3)[dofs])))
    line_modes[mode.name] = np.array(values)

# The spider lines as a mesh of line cells on a single process (the points are the same on all)
if comm.rank == 0:
    line_element = basix.ufl.element("Lagrange", "interval", 1, shape=(3,))
    line_mesh = mesh.create_mesh(
        MPI.COMM_SELF,
        np.array(line_cells, dtype=np.int64),
        line_element,
        np.array(line_points),
    )
    W_lines = fem.functionspace(line_mesh, ("Lagrange", 1, (3,)))
    # create_mesh reorders points and cells: find the dof of each of the original points
    line_dof = np.empty(len(line_points), dtype=np.int32)
    line_dof[line_mesh.geometry.input_global_indices[line_mesh.geometry.dofmap]] = (
        W_lines.dofmaps[0].list
    )

    def line_function(name, values):
        f = fem.Function(W_lines, name=name)
        f.x.array.reshape(-1, 3)[line_dof] = values
        return f
# -

# The modes of the solid are written as one file, with each mode as a function of its own at a single time.
# The spider lines are written by process 0 only, on a mesh that lives on that process alone.

Path("results/animation").mkdir(parents=True, exist_ok=True)
with VTXWriter(comm, "results/spider_modes.bp", modes) as writer:
    writer.write(0.0)
if comm.rank == 0:
    with VTXWriter(
        MPI.COMM_SELF,
        "results/spider_lines.bp",
        [line_function(k, v) for k, v in line_modes.items()],
    ) as writer:
        writer.write(0.0)

# A mode is a standing wave, $u(t) = \cos(\omega t) \phi$. To animate it in ParaView we write one period in 40 steps,
# each the mode multiplied by $\cos(2 \pi k / 40)$, with the step number as the time. The spider lines are written in the same
# steps, to a file of their own, and the writer of that file only exists on process 0. The `ExitStack` closes whichever writers
# were opened, also if the loop fails.

ANIMATE_MODE = min(2, len(modes) - 1)  # mode number to animate
u_t = fem.Function(V, name="u")
with ExitStack() as stack:  # the spider lines are written by process 0 only
    writer = stack.enter_context(VTXWriter(comm, "results/animation/spider.bp", [u_t]))
    if comm.rank == 0:
        u_lines = line_function("u", line_modes[f"u_{ANIMATE_MODE}"])
        lines_writer = stack.enter_context(
            VTXWriter(MPI.COMM_SELF, "results/animation/spider_lines.bp", [u_lines])
        )
    for frame in range(40):
        c = np.cos(2 * np.pi * frame / 40)
        u_t.x.array[:] = c * modes[ANIMATE_MODE].x.array
        writer.write(float(frame))
        if comm.rank == 0:
            u_lines.x.array.reshape(-1, 3)[line_dof] = (
                c * line_modes[f"u_{ANIMATE_MODE}"]
            )
            lines_writer.write(float(frame))

# %% [markdown]
# ### Interactive mode shapes
# The modes can be explored in the browser, through [VTK.wasm](https://docs.vtk.org/en/latest/getting_started/using_webassembly.html).
# Choose a mode under *Warp*, press play, or move the step slider, to oscillate it, and use *Clip* to look inside.
# The first three modes are the pins turning: the hinge stiffness `K_HINGE` is 1, so the hinges are
# almost free and these modes cost almost no energy, at about 0.1 Hz. With `K_HINGE = 1e6` they
# move to 61 to 74 Hz, while softening the ground springs does not change them. The view opens
# on the first mode above 1 Hz, which depends on `K_HINGE`.
#
# The view is built in five steps: a grid of the solid with the modes attached,
# the pieces of that grid from all processes joined into one, the spider as lines and points,
# the packing of the scene for the browser, and finally the display.

# %% [markdown]
# #### The solid and its modes
# {py:func}`dolfinx.plot.vtk_mesh` turns the function space `V` into the cells and points of a
# {py:class}`pyvista.UnstructuredGrid`. The points are the degrees of freedom, so with `DEGREE = 2`
# the cells are quadratic tetrahedra with the same straight edges, and values at the midpoints. Each mode is attached as point data, one 3-component array
# per mode, named by its number and frequency so that the viewer can offer it in its menus.

# %%
import pyvista as pv
from dolfinx import plot
from IPython.display import HTML, display

from vtkwasm import scene_data, vtkwasm_view

grid = pv.UnstructuredGrid(*plot.vtk_mesh(V))
names = [f"mode {i}, {f:.1f} Hz" for i, f in enumerate(frequencies)]
for name, mode in zip(names, modes):
    grid.point_data[name] = mode.x.array.reshape(
        -1, 3
    )  # all local dofs, as vtk_mesh(V) has

# %% [markdown]
# #### Joining the pieces of the grid
# When the program runs on several processes, each one only has the cells it owns.
# The grids are gathered on process 0, which merges them into a single grid, so that the
# viewer gets one mesh with one colour bar. Points that neighbouring pieces share are welded
# back together. As the processes compute their coordinates independently these can differ in
# the last bits, so the welding distance is a tiny fraction of the size of the spider.
# On a single process there is just one piece, and nothing is merged.

# + tags=["hide-input"]
pieces = comm.gather(grid, root=0)
root = pieces is not None  # only process 0 receives the pieces
if root:
    # The coordinates are in mm, so the distance is relative to the size of the spider
    tolerance = 1e-9 * max(np.ptp(piece.points, axis=0).max() for piece in pieces)
    merged = (
        pieces[0].merge(pieces[1:], tolerance=tolerance)
        if len(pieces) > 1
        else pieces[0]
    )
# -

# %% [markdown]
# #### The spider
# (`spider` above is the element of the spider meshes, so the grid of lines gets another name.)
# The spiders are the lines that join each pin centre or foot centre to the nodes of its
# surface. They were collected as `line_points` and `line_cells` above, and are made into a
# {py:class}`pyvista.PolyData` of line cells. Each mode is attached at their points too, as
# the displacement of the points of the spider, so that the lines and nodes move with the solid.

# + tags=["hide-input"]
if root:
    spider_lines = pv.PolyData(
        np.array(line_points),
        lines=np.column_stack([np.full(len(line_cells), 2), line_cells]).ravel(),
    )
    for i, name in enumerate(names):
        spider_lines.point_data[name] = line_modes[f"u_{i}"]
# -

# %% [markdown]
# #### Packing the scene
# {py:func}`vtkwasm.scene_data` turns the grids into what the browser needs: the outer surface of
# the solid, drawn as triangles (`subdivision` splits each triangle of a higher order cell, so
# that the surface follows the curved displacement), the edges of the surface, the spider, and every point array. They are
# stored as single precision, compressed and embedded in the page. The value range of each
# colouring and the scale of each warp are computed here, so that the browser never has to read
# data back from WebAssembly.
#
# `initial` chooses what the view starts with: the first flexible mode, a slightly transparent
# solid, and the spider shown.

# +
if root:
    first = next(
        (i for i, f in enumerate(frequencies) if f > 1.0), 0
    )  # skip the hinge modes
    scene = scene_data(
        merged,
        spider=spider_lines,
        subdivision=DEGREE - 1,  # nothing to split for linear cells
        initial={
            "color": f"|{names[first]}|",
            "warp": f"by {names[first]}",
            "opacity": 0.8,
            "showSpider": True,
            "showNodes": True,
        },
    )
# -

# %% [markdown]
# #### The view
# {py:func}`vtkwasm.vtkwasm_view` returns the HTML of the viewer. When it is shown, the browser
# downloads VTK.wasm (about 24 MB, cached afterwards), which runs the filters of the viewer: the
# warp, the clip along a plane, the threshold and the contours. With `autoload=True` this starts
# when the view scrolls into sight, and without it when the reader presses a button.

# +
if root:
    view = vtkwasm_view(scene, autoload=True)
    display(view)
# -

# %% [markdown]
# #### Sharing the view with the front page
# The front page of the book shows the same view and the table of frequencies, without running
# the simulation again. They are stored under a name with {py:func}`myst_nb.glue`, which the
# front page refers to.

# + tags=["hide-input"]
if root:
    try:
        from myst_nb import glue
    except ImportError:  # run outside of the book
        pass
    else:
        rows = "".join(
            f"<tr><td>{i}</td><td>{f:.2f}</td></tr>" for i, f in enumerate(frequencies)
        )
        glue("scene_view", view, display=False)
        glue(
            "frequency_table",
            HTML(
                "<table><thead><tr><th>Mode</th><th>Frequency (Hz)</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>"
            ),
            display=False,
        )
# -
