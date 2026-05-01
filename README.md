# Planetary Magnetic Field Explorer

An interactive 3D viewer of real planetary magnetic field lines for Earth, Jupiter, Saturn, Uranus, and Neptune.

**Live viewer:** open `index.html` in a browser (or host on GitHub Pages).

## What you're seeing

Each planet's field lines are traced from real scientific models:

| Planet  | Model        | Notes |
|---------|-------------|-------|
| Earth   | IGRF 2020   | 26 models available from 1900–2025 |
| Jupiter | JRM33 + Con2020 | Internal (Juno 2022) + external magnetodisc |
| Saturn  | Cassini 11  | Dougherty 2018 Grand Finale |
| Uranus  | GSFC Q3     | Voyager 2 quadrupole |
| Neptune | GSFC O8     | Voyager 2 octupole |

The viewer also shows:
- Aurora footprints (Earth, Jupiter, Saturn)
- Moon flux tubes (Io, Enceladus, Miranda, Ariel, Triton)
- Io plasma torus / Enceladus E-ring
- Axial tilt (IAU convention)
- Orbital position slider — drag the Sun around the planet to simulate seasons and see day/night lighting change

## Repository structure

```
index.html                  # Self-contained Three.js viewer (no build needed)
field_lines_*.json          # Pre-generated field line data for each planet
generate_all_fields.py      # Script that generates the JSON files
internal_field_models.py    # Python wrapper around libinternalfield (provides get_field())
planetary_field_class.py    # Object-oriented class interface (alternative API)
requirements.txt            # Python dependencies for the generation script
```

## Regenerating the field line data

The JSON files are pre-computed and committed, so you only need to do this if you want to change resolution, add planets, or update models.

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Build `libinternalfield`

The field models use [mattkjames7/libinternalfield](https://github.com/mattkjames7/libinternalfield), a compiled C++ library:

```bash
git clone https://github.com/mattkjames7/libinternalfield.git
cd libinternalfield
make
sudo make install   # optional; installs to /usr/local/lib/
```

`internal_field_models.py` searches these locations automatically:
- `/usr/local/lib/libinternalfield.{dylib,so,dll}`
- `./libinternalfield/lib/`
- `~/jupitermag_temp/libinternalfield/lib/`

### 3. Run the generator

```bash
python generate_all_fields.py
```

This writes `field_lines_earth.json`, `field_lines_jupiter.json`, etc. (warning: large files, ~100 MB total).
