# Copyright (C) 2026 Antonio Baiano Svizzero
# SPDX-License-Identifier: MIT
#
# Modal analysis of a spider: 17 separate parts joined by 16 pins, standing on 8 pinned feet.
# Pin j: the lug bore (tag 101+j) and the clevis bores (tag 201+j) are tied by RBE3 to two spiders
# at the pin centre, joined by a spring that lets the pin turn (K_HINGE) and holds everything else.
# Foot k: the pad bottom (tag 301+k) is tied by RBE2 to a spider held by a spring to the ground.
# Units: mm, N, t, s.
from pathlib import Path

from mpi4py import MPI
from slepc4py import SLEPc

import basix.ufl
import dolfinx_mpc
import meshio
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.io import VTXWriter, gmsh

E, NU, RHO = 210e3, 0.3, 7.85e-9  # steel
K_PIN, K_TILT, K_HINGE = 2.1e8, 5.25e9, 1.0  # pin spring: translation, rotation across and about the pin
K_GROUND = 2.1e8
N_MODES= 6 # number of modes to extract
ANIMATE_MODE = 2 # mode number to animate 

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
    q = np.linalg.svd(np.column_stack([x * x, y * y, z * z, x * y, x * z, y * z, x, y, z, 1 + 0 * x]))[2][-1]
    eigenvalues, eigenvectors = np.linalg.eigh([[2 * q[0], q[3], q[4]], [q[3], 2 * q[1], q[5]], [q[4], q[5], 2 * q[2]]])
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

V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
spider = basix.ufl.element("DG", "point", 0, shape=(6,))  # translation and rotation
lug_mesh = dolfinx_mpc.create_spider_mesh(comm, centres)
clevis_mesh = dolfinx_mpc.create_spider_mesh(comm, centres)
foot_mesh = dolfinx_mpc.create_spider_mesh(comm, feet)
W_lug = fem.functionspace(lug_mesh, spider)
W_clevis = fem.functionspace(clevis_mesh, spider)
W_foot = fem.functionspace(foot_mesh, spider)

mpc_u = dolfinx_mpc.MultiPointConstraint(V)
mpc_lug = dolfinx_mpc.MultiPointConstraint(W_lug)
mpc_clevis = dolfinx_mpc.MultiPointConstraint(W_clevis)
mpc_foot = dolfinx_mpc.MultiPointConstraint(W_foot)
mpc_lug.add_rbe3_topological(V, 2, [tags.find(101 + j) for j in range(16)])
mpc_clevis.add_rbe3_topological(V, 2, [tags.find(201 + j) for j in range(16)])
mpc_u.add_rbe2_topological(2, [tags.find(301 + k) for k in range(8)], W_foot)
mpcs = [mpc_u, mpc_lug, mpc_clevis, mpc_foot]
dolfinx_mpc.finalize_multipointconstraints(mpcs)

unknowns = ufl.MixedFunctionSpace(V, W_lug, W_clevis, W_foot)
u, w_lug, w_clevis, w_foot = ufl.TrialFunctions(unknowns)
v, z_lug, z_clevis, z_foot = ufl.TestFunctions(unknowns)
t = lambda w: ufl.as_vector([w[0], w[1], w[2]])
r = lambda w: ufl.as_vector([w[3], w[4], w[5]])

mu, lmbda = E / (2 * (1 + NU)), E * NU / ((1 + NU) * (1 - 2 * NU))
eps_u, eps_v = ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))
elastic = (2 * mu * ufl.inner(eps_u, eps_v) + lmbda * ufl.tr(eps_u) * ufl.tr(eps_v)) * ufl.dx(domain)

axis = fem.Function(fem.functionspace(lug_mesh, basix.ufl.element("DG", "point", 0, shape=(3,))))
axis.interpolate(lambda x: np.array(axes)[np.argmin(np.linalg.norm(x.T[:, None] - centres, axis=2), axis=1)].T)
along = ufl.outer(axis, axis)
across = ufl.Identity(3) - along
gap, gap_v = w_clevis - w_lug, z_clevis - z_lug
pins = (K_PIN * ufl.dot(t(gap), t(gap_v)) + ufl.dot((K_TILT * across + K_HINGE * along) * r(gap), r(gap_v))) * ufl.dx(lug_mesh)
ground = K_GROUND * ufl.dot(t(w_foot), t(z_foot)) * ufl.dx(foot_mesh)

stiffness = ufl.extract_blocks(elastic + pins + ground)
mass = RHO * ufl.inner(ufl.TrialFunction(V), ufl.TestFunction(V)) * ufl.dx
mass = [[mass, None, None, None],  # only the solid has mass
        [None, None, None, None],
        [None, None, None, None],
        [None, None, None, None]]
pair = dolfinx_mpc.create_spider_pair(lug_mesh, clevis_mesh)
K = dolfinx_mpc.assemble_matrix(fem.form(stiffness, entity_maps=[pair]), mpcs, diagval=1.0, kind="mpi")
M = dolfinx_mpc.assemble_matrix(fem.form(mass), mpcs, diagval=1e-14, kind="mpi")

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
x = K.createVecRight()
modes, line_modes = [], {}
for i in range(N_MODES):
    eigenvalue = eigensolver.getEigenpair(i, x).real
    start = 0
    for f in functions:  # x holds the dofs of the 4 blocks one after the other
        n = f.x.index_map.size_local * f.x.block_size
        f.x.array[:n] = x.array_r[start:start + n]
        f.x.scatter_forward()
        start += n
    for m in mpcs:
        m.backsubstitution(functions)
    if comm.rank == 0:
        print(f"mode {i}: {np.sqrt(max(eigenvalue, 0)) / (2 * np.pi):.2f} Hz")

    mode = fem.Function(V, name=f"u_{i}")
    mode.x.array[:3 * owned] = functions[0].x.array[:3 * owned]
    mode.x.scatter_forward()
    scale = 10 / comm.allreduce(np.abs(mode.x.array).max(), op=MPI.MAX)  # largest displacement: 10 mm
    mode.x.array[:] *= scale
    modes.append(mode)
    values = []
    for (point, tag, block, index), dofs in zip(spiders, line_dofs):
        values.append(scale * dolfinx_mpc.spider_values(functions[block], index)[:3])
        values.extend(np.vstack(comm.allgather(mode.x.array.reshape(-1, 3)[dofs])))
    line_modes[mode.name] = np.array(values)

Path("results/animation").mkdir(parents=True, exist_ok=True)
with VTXWriter(comm, "results/spider_modes.bp", modes) as writer:
    writer.write(0.0)
if comm.rank == 0:
    meshio.write_points_cells("results/spider_lines.vtu", np.array(line_points), [("line", line_cells)], point_data=line_modes)

u_t = fem.Function(V, name="u")
with VTXWriter(comm, "results/animation/spider.bp", [u_t]) as writer:
    for frame in range(40):
        c = np.cos(2 * np.pi * frame / 40)
        u_t.x.array[:] = c * modes[ANIMATE_MODE].x.array
        writer.write(float(frame))
        if comm.rank == 0:
            meshio.write_points_cells(f"results/animation/spider_lines_{frame:02d}.vtu", np.array(line_points), [("line", line_cells)],
                                      point_data={"u": c * line_modes[f"u_{ANIMATE_MODE}"]})