# yt_dyablo

A [yt](https://yt-project.org/) frontend for [Dyablo](https://gitlab.maisondelasimulation.fr/dyablo/dyablo), packaged as an external plugin that auto-registers itself with yt.

## Installation

```bash
pip install yt_dyablo
```

For development, or to install from source:

```bash
git clone git@drf-gitlab.cea.fr:dyablo/yt_dyablo.git
cd yt_dyablo
pip install -e ".[tests]"
```

> **Note:** `yt_dyablo` depends on a development branch of yt that supports non-cubic AMR blocks (`feature/support-non-cubic-zones`). This is installed automatically as a git dependency.

## How it works

`yt_dyablo` registers itself as a yt frontend via the `yt.frontends` [entry point](https://yt-project.org/doc/developing/creating_frontend.html). Once the package is installed, **no explicit import is needed** — `yt.load()` will automatically recognise Dyablo output files.

## File naming convention

Dyablo writes output files following this pattern:

| File type | Pattern |
|-----------|---------|
| Hydro fields | `{name}_iter{NNNNNNN}.h5` |
| Particles | `{name}_particles_{type}_iter{NNNNNNN}.h5` |

`yt.load()` identifies Dyablo files by the hydro file pattern and the presence of `/rho` and `/coordinates` datasets inside the HDF5 file. Particle files matching the same name and iteration are discovered automatically.

## Usage

### Basic

```python
import yt
import yt_dyablo  # registers the frontend

ds = yt.load("my_sim_iter0001234.h5")
ad = ds.all_data()

# Access fluid fields
rho = ad["gas", "density"]          # code_density
vx  = ad["gas", "velocity_x"]       # code_velocity
P   = ad["gas", "pressure"]         # code_pressure

# Access raw Dyablo fields
e_tot = ad["dyablo", "e_tot"]
```

### Projections and slices

```python
import yt
import yt_dyablo

ds = yt.load("my_sim_iter0001234.h5")
p = yt.ProjectionPlot(ds, "z", ("gas", "density"), weight_field="density")
p.save()
```

### Overriding units

When physical units are not embedded in the HDF5 file, pass them explicitly:

```python
import unyt
import yt
import yt_dyablo

ds = yt.load(
    "my_sim_iter0001234.h5",
    units_override={
        "length_unit": (1.0, "kpc"),
        "mass_unit":   (1e4, "Msun"),
        "time_unit":   (1.0, "Gyr"),
    },
)
```

### Advanced constructor options

You can also load the dataset directly via `DyabloDataset` to pass extra overrides:

```python
from yt_dyablo.api import DyabloDataset

ds = DyabloDataset(
    "my_sim_iter0001234.h5",
    units_override={"length_unit": (1.0, "kpc"), ...},
    # Override domain geometry (inferred from file by default)
    domain_left_edge=[0.0, 0.0, 0.0],
    domain_right_edge=[1.0, 1.0, 1.0],
    periodicity=(True, True, True),
    # Override AMR block size (inferred from file by default)
    block_size=(4, 4, 4),
    max_level=5,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `units_override` | `dict` | `None` | Override code units (keys: `length_unit`, `mass_unit`, `time_unit`) |
| `unit_system` | `str` | `"code"` | yt unit system (`"code"`, `"cgs"`, `"mks"`) |
| `domain_left_edge` | array | inferred | Override domain left edge |
| `domain_right_edge` | array | inferred | Override domain right edge |
| `periodicity` | tuple of bool | inferred | Periodicity per dimension |
| `block_size` | tuple of int | inferred | AMR block shape `(N, M, L)` |
| `max_level` | `int` | inferred | Maximum refinement level |

## Available fields

### Fluid fields (`dyablo` field type — raw from file)

| Field | Aliases | Units |
|-------|---------|-------|
| `rho` | `density` | `code_density` |
| `rho_vx` | `density_times_velocity_x` | `code_density * code_velocity` |
| `rho_vy` | `density_times_velocity_y` | `code_density * code_velocity` |
| `rho_vz` | `density_times_velocity_z` | `code_density * code_velocity` |
| `e_tot` | `total_energy_density` | `code_density * code_velocity²` |
| `metallicity` | `metallicity` | dimensionless |

> **Note:** Any extra field found in the hydro HDF5 file that is not listed above is treated as **dimensionless** by default.

### Derived fields (`gas` field type)

| Field | Units | Description |
|-------|-------|-------------|
| `velocity_x/y/z` | `code_velocity` | Velocity components |
| `velocity_magnitude` | `code_velocity` | Speed |
| `pressure` | `code_pressure` | Thermal pressure (requires `gamma` in file or parameters) |
| `temperature_over_mu` | K | Temperature divided by mean molecular weight |

### Particle fields

| Field | Units | Description |
|-------|-------|-------------|
| `particle_position_x/y/z` | `code_length` | Position |
| `particle_vx/vy/vz` | `code_velocity` | Velocity |
| `particle_mass` | `code_mass` | Mass |
| `particle_birth_time` | `code_time` | Birth time (star particles) |
| `particle_identity` | — | Unique particle index |

> **Note:** Any extra field found in a particle HDF5 file that is not listed above is treated as **dimensionless** by default.

## Running tests

```bash
pip install ".[tests]"
pytest src/yt_dyablo/tests/
```
