"""
Tests for Dyablo frontend.
"""

import os

import h5py
import numpy as np
import numpy.typing as npt
import pytest

import yt
from yt.utilities.lib.geometry_utils import compute_morton


def create_test_file(
    tmpdir,
    n_blocks: npt.NDArray[np.int_],
    block_size: npt.NDArray[np.int_],
    left_edge: npt.NDArray[np.float64],
    right_edge: npt.NDArray[np.float64],
    seed: int = 42,
    units: list | None = None,
):
    """Create a small test HDF5 file for Dyablo."""
    n_blocks = np.asarray(n_blocks)
    block_size = np.asarray(block_size)
    left_edge = np.asarray(left_edge)
    right_edge = np.asarray(right_edge)

    iteration = 1234567

    fluid_fname = os.path.join(tmpdir, f"test_dyablo_iter{iteration}.h5")

    n_cells = np.prod(n_blocks) * np.prod(block_size)

    generator = np.random.default_rng(seed=seed)

    # Compute vertices of the blocks
    vertex_coordinates = np.mgrid[
        left_edge[0] : right_edge[0] : (block_size[0] * n_blocks[0] + 1) * 1j,
        left_edge[1] : right_edge[1] : (block_size[1] * n_blocks[1] + 1) * 1j,
        left_edge[2] : right_edge[2] : (block_size[2] * n_blocks[2] + 1) * 1j,
    ]

    # Block positions
    dx = (right_edge - left_edge) / n_blocks
    block_inds = np.mgrid[0 : n_blocks[0], 0 : n_blocks[1], 0 : n_blocks[2]]
    block_positions = (
        left_edge[:, None, None, None] + (block_inds + 0.5) * dx[:, None, None, None]
    )

    # Compute block order **along Morton curve**
    # Dyablo's Morton order uses x as the least-significant bit; yt's
    # compute_morton uses z as LSB, so we swap x and z.
    morton_index = compute_morton(
        block_positions[2].flatten(),
        block_positions[1].flatten(),
        block_positions[0].flatten(),
        left_edge[::-1],
        right_edge[::-1],
    )
    block_order = np.argsort(morton_index)

    # Compute cell ordering within each block **in column-major (Fortran) order**.
    # For each F-order output position, find the corresponding C-order column index
    # in cell_inds.reshape(3,-1). This is the F→C permutation (not C→F).
    ix_f, iy_f, iz_f = np.unravel_index(
        np.arange(np.prod(block_size)), block_size, order="F"
    )

    # Offset w.r.t. the bottom-left-front corner of each cell
    vertex_offset = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    ).T

    vertex_inds = (
        block_inds.reshape(3, -1)[:, block_order][:, :, None, None]
        * block_size[:, None, None, None]
        + np.array([ix_f, iy_f, iz_f])[:, None, :, None]
        + vertex_offset[:, None, None, :]
    )
    vertex_order = np.ravel_multi_index(
        vertex_inds, vertex_coordinates.shape[1:], order="C"
    )

    with h5py.File(fluid_fname, "w") as f:
        # Create fluid fields
        f["rho"] = generator.random(n_cells)
        f["rho_vx"] = generator.random(n_cells)
        f["rho_vy"] = generator.random(n_cells)
        f["rho_vz"] = generator.random(n_cells)
        f["e_tot"] = generator.random(n_cells)
        f["test_field"] = generator.random(n_cells)

        # Create vertices of the blocks
        f["coordinates"] = vertex_coordinates.reshape(3, -1).T
        f["connectivity"] = vertex_order.reshape(-1, 8)

        # Create scalar_data
        f.create_group("scalar_data")
        f["scalar_data"].attrs.update({"aexp": 1.0, "iter": iteration, "time": 123.0})

        if units:
            f["units"] = units


# Expected fluid fields written by create_test_file
_FLUID_FIELDS = ["rho", "rho_vx", "rho_vy", "rho_vz", "e_tot", "test_field"]


@pytest.mark.parametrize("units", [None, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]])
@pytest.mark.parametrize(
    "n_blocks, block_size",
    [
        # cubic, minimal 2×2×2 (block_size=1 is ambiguous to inference, start from 2)
        ([2, 2, 2], [2, 2, 2]),
        ([2, 2, 2], [4, 4, 4]),
        # non-cubic (N≠M≠L), uniform n_blocks
        ([4, 4, 4], [3, 4, 5]),
        # asymmetric n_blocks (all dims ≥ 2; avoid z-neighbor-as-block-1 orderings)
        ([3, 4, 6], [4, 4, 4]),
        ([2, 4, 6], [3, 5, 7]),
    ],
)
def test_dyablo_mock_dataset(tmp_path, n_blocks, block_size, units):
    n_blocks = np.asarray(n_blocks)
    block_size = np.asarray(block_size)

    create_test_file(
        str(tmp_path),
        n_blocks=n_blocks,
        block_size=block_size,
        left_edge=np.zeros(3),
        right_edge=np.ones(3),
        units=units,
    )

    fpath = next(tmp_path.glob("*.h5"))
    yt.mylog.setLevel(50)
    ds = yt.load(str(fpath))

    # Check inferred block_size and n_blocks
    np.testing.assert_array_equal(
        ds.block_size,
        block_size,
        err_msg=f"block_size mismatch: got {ds.block_size}, expected {block_size}",
    )
    np.testing.assert_array_equal(
        ds.domain_dimensions,
        n_blocks * block_size,
        err_msg=f"domain_dimensions mismatch: got {ds.domain_dimensions}, expected {n_blocks * block_size}",
    )

    ad = ds.all_data()
    expected_n_cells = int(np.prod(n_blocks) * np.prod(block_size))

    # Check all fluid fields are readable and have correct size
    for fname in _FLUID_FIELDS:
        data = ad["dyablo", fname]
        assert data.shape == (
            expected_n_cells,
        ), f"Field {fname}: expected {expected_n_cells} cells, got {data.shape}"
        assert not np.any(np.isnan(data)), f"Field {fname} contains NaN values"

    # Check derived density field
    rho = ad["gas", "density"]
    assert rho.shape == (expected_n_cells,)
    assert not np.any(np.isnan(rho))

    if units is not None:
        assert (
            ds.length_unit == units[0]
        ), f"Expected length unit {units[0]}, got {ds.length_unit}"
        assert (
            ds.mass_unit == units[1]
        ), f"Expected mass unit {units[1]}, got {ds.mass_unit}"
        assert (
            ds.time_unit == units[2]
        ), f"Expected time unit {units[2]}, got {ds.time_unit}"

        # Make sure we can convert properly
        length = ds.quan(1, "code_length").to("cm")
        assert (
            length == units[0]
        ), f"Length unit conversion failed: expected {units[0]}, got {length}"
        mass = ds.quan(1, "code_mass").to("g")
        assert (
            mass == units[1]
        ), f"Mass unit conversion failed: expected {units[1]}, got {mass}"
        time = ds.quan(1, "code_time").to("s")
        assert (
            time == units[2]
        ), f"Time unit conversion failed: expected {units[2]}, got {time}"


def test_dyablo_mock_dataset_edges(tmp_path):
    """Test that arbitrary left_edge and right_edge are preserved correctly."""
    n_blocks = np.asarray([2, 2, 2])
    block_size = np.asarray([4, 4, 4])
    left_edge = np.array([-3.0, 0.5, 1.0])
    right_edge = np.array([5.0, 4.5, 7.0])

    create_test_file(
        str(tmp_path),
        n_blocks=n_blocks,
        block_size=block_size,
        left_edge=left_edge,
        right_edge=right_edge,
    )

    fpath = next(tmp_path.glob("*.h5"))
    yt.mylog.setLevel(50)
    ds = yt.load(str(fpath))

    np.testing.assert_allclose(ds.domain_left_edge.d, left_edge)
    np.testing.assert_allclose(ds.domain_right_edge.d, right_edge)

    ad = ds.all_data()
    rho = ad["gas", "density"]
    assert rho.shape == (int(np.prod(n_blocks) * np.prod(block_size)),)
    assert not np.any(np.isnan(rho))

# ---------------------------------------------------------------------------
# AMR (2D and 3D) following the exact Dyablo format dyablo/core/src/io/IOManager_hdf5.cpp
# ---------------------------------------------------------------------------


def _morton_key(ix, iy, iz, nbits, ndim):
    """Morton key as used by Dyablo/PABLO: x is the least significant bit."""
    key = 0
    for b in range(nbits):
        key |= ((ix >> b) & 1) << (ndim * b)
        key |= ((iy >> b) & 1) << (ndim * b + 1)
        if ndim == 3:
            key |= ((iz >> b) & 1) << (ndim * b + 2)
    return key


def create_amr_test_file(
    tmpdir, ndim, block_size, coarse_level, max_level, refine, left_edge, right_edge, field
):
    """Write an AMR snapshot following IOManager_hdf5::save_snapshot_aux.

    - octree in [0, 1]^ndim, leaves sorted along the Morton curve;
    - (bx+1)(by+1)(bz+1) nodes per octant (not shared between octants), bz = 1 in 2D;
    - cells of a block stored as i + bx*(j + by*k);
    - connectivity of shape (n_cells, 4 | 8), using node_shuffle {0,1,3,2,4,5,7,6};
    - in 2D, all z coordinates equal zmin.

    Returns the file name, and the true cell centers, cell widths and levels
    (in file order).
    """
    bx, by, bz = block_size if ndim == 3 else (block_size[0], block_size[1], 1)
    left_edge = np.asarray(left_edge, dtype="float64")
    width = np.asarray(right_edge, dtype="float64") - left_edge

    # Octree: (level, i, j, k) of the leaves
    n0 = 2**coarse_level
    todo = [
        (coarse_level, i, j, k)
        for i in range(n0)
        for j in range(n0)
        for k in (range(n0) if ndim == 3 else [0])
    ]
    leaves = []
    while todo:
        lvl, i, j, k = todo.pop()
        s = 0.5**lvl
        center = np.array([(i + 0.5) * s, (j + 0.5) * s, (k + 0.5) * s if ndim == 3 else 0.0])
        if lvl < max_level and refine(center):
            for di, dj, dk in np.ndindex(2, 2, 2 if ndim == 3 else 1):
                todo.append((lvl + 1, 2 * i + di, 2 * j + dj, 2 * k + dk))
        else:
            leaves.append((lvl, i, j, k))
    leaves.sort(
        key=lambda o: _morton_key(*(c << (max_level - o[0]) for c in o[1:]), max_level, ndim)
    )
    n_oct = len(leaves)
    lvl = np.array([o[0] for o in leaves])
    corner = np.array([o[1:] for o in leaves], dtype="float64") * (0.5**lvl)[:, None]
    oct_size = 0.5**lvl

    # Nodes: iNode = i + Bx*(j + By*k)
    Bx, By, Bz = bx + 1, by + 1, (bz + 1 if ndim == 3 else 1)
    k, j, i = (a.ravel() for a in np.meshgrid(np.arange(Bz), np.arange(By), np.arange(Bx), indexing="ij"))
    coords = np.empty((n_oct, Bx * By * Bz, 3))
    for idim, (ind, b) in enumerate(((i, bx), (j, by), (k, bz))):
        coords[:, :, idim] = left_edge[idim] + (
            corner[:, None, idim] + ind[None] * oct_size[:, None] / b
        ) * width[idim]
    coords = coords.reshape(-1, 3)

    # Connectivity
    nvert = 4 if ndim == 2 else 8
    shuffle = np.array([0, 1, 3, 2, 4, 5, 7, 6])[:nvert]
    ic, jc, kc = shuffle % 2, (shuffle // 2) % 2, shuffle // 4
    cell = np.arange(bx * by * bz)
    ib, jb, kb = cell % bx, (cell // bx) % by, cell // (bx * by)
    connectivity = (
        np.arange(n_oct)[:, None, None] * (Bx * By * Bz)
        + (ic + ib[:, None])
        + (jc + jb[:, None]) * Bx
        + (kc + kb[:, None]) * Bx * By
    ).reshape(-1, nvert)

    cmin = coords[connectivity[:, 0]]
    cmax = coords[connectivity[:, nvert - 2]]  # opposite vertex: 2 in 2D, 6 in 3D
    centers, widths = 0.5 * (cmin + cmax), cmax - cmin

    fname = os.path.join(tmpdir, "test_amr_iter0000042.h5")
    with h5py.File(fname, "w") as f:
        f["coordinates"] = coords.astype("float32")  # output_real_type defaults to float
        f["connectivity"] = connectivity.astype("uint64")
        f["rho"] = field(*centers.T)
        f["level"] = np.repeat(lvl, bx * by * bz).astype("int32")
        f.create_group("scalar_data")
        f["scalar_data"].attrs.update({"iter": 42, "time": 0.5, "dt": 1e-3})
    return fname, centers, widths, np.repeat(lvl, bx * by * bz)


def _corner_refinement(center):
    # Deliberately NON-symmetric refinement (corner of small x and y): a
    # level-dependent weighting error cannot cancel out
    return center[0] < 0.3 and center[1] < 0.4


def _rho(x, y, z):
    return 1.0 + x**2 + 3.0 * y + 0.5 * z


@pytest.mark.parametrize(
    "ndim, block_size, left_edge, right_edge",
    [
        (2, (8, 8, 1), (0, 0, 0), (1, 1, 0)),
        (2, (8, 4, 1), (-1, 0, 0), (3, 1, 0)),  # rectangular domain and blocks
        (3, (4, 4, 4), (0, 0, 0), (1, 1, 1)),
        (3, (8, 4, 2), (0, -1, 0), (2, 1, 4)),
    ],
)
def test_dyablo_amr_dataset(tmp_path, ndim, block_size, left_edge, right_edge):
    fname, centers, widths, levels = create_amr_test_file(
        str(tmp_path), ndim, block_size, coarse_level=2, max_level=4,
        refine=_corner_refinement, left_edge=left_edge, right_edge=right_edge, field=_rho,
    )
    assert np.unique(levels).size == 3  # the test must actually contain AMR

    yt.mylog.setLevel(50)
    ds = yt.load(fname)
    ad = ds.all_data()

    assert ds.dimensionality == ndim
    np.testing.assert_array_equal(ds.block_size, block_size)
    assert float(ds.current_time) == 0.5
    assert ds.max_level - ds.min_level == 2

    # Cell ordering: each value read must be the one of the center where yt puts it
    x, y, z = (ad["index", ax].d for ax in "xyz")
    rho = ad["dyablo", "rho"].d
    assert rho.size == len(centers)
    np.testing.assert_allclose(rho, _rho(x, y, z if ndim == 3 else 0.0), rtol=1e-6)

    # In-plane cell sizes: identical to the ones in the file
    yt_cells = np.c_[x, y, ad["index", "dx"].d, ad["index", "dy"].d]
    ref_cells = np.c_[centers[:, :2], widths[:, :2]]
    sort = lambda a: a[np.lexsort(np.round(a, 6).T[::-1])]
    np.testing.assert_allclose(sort(yt_cells), sort(ref_cells), rtol=1e-5, atol=1e-6)

    # Volumes: sum of cell_volume = yt domain volume, and exact weighted mean
    dV = ad["index", "cell_volume"].d
    np.testing.assert_allclose(dV.sum(), np.prod(ds.domain_width.d), rtol=1e-6)
    true_dV = np.prod(widths[:, :ndim], axis=1)
    true_mean = (_rho(*centers.T) * true_dV).sum() / true_dV.sum()
    yt_mean = ad.quantities.weighted_average_quantity(("dyablo", "rho"), ("index", "cell_volume"))
    np.testing.assert_allclose(float(yt_mean), true_mean, rtol=1e-6)

    if ndim == 2:
        # 2D: dz must not depend on the refinement level (see select_fwidth)
        dz = ad["index", "dz"].d
        np.testing.assert_allclose(dz, ds.domain_width[2].d)
        # the "z" slice (the only useful one in 2D) covers the whole domain
        frb = ds.slice("z", ds.domain_center[2]).to_frb(
            ds.domain_width[0], (64, 32), height=ds.domain_width[1]
        )
        assert np.all(np.asarray(frb["dyablo", "rho"]) > 0)