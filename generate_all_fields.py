#!/usr/bin/env python3
"""
Generate magnetic field lines for multiple planets using real field models.

Models used:
- Earth: igrf2020
- Jupiter: jrm33 + Con2020 (internal + external magnetodisc)
- Saturn: cassini11
- Uranus: gsfcq3
- Neptune: gsfco8

Output: JSON files with field line data for each planet
"""

import numpy as np
import json
import sys
import os

# Add path to magnetic field library
sys.path.append('/Users/jamesodonoghue/jupitermag_temp')
from planetary_magfield_simple import get_field

# Import Con2020 for Jupiter's external field
import con2020
jupiter_con2020 = con2020.Model(equation_type='analytic')

def get_jupiter_field(x, y, z):
    """Combined Jupiter field: JRM33 (internal) + Con2020 (external)"""
    B_int = get_field('jrm33', x, y, z)
    B_ext = jupiter_con2020.Field(np.array([x]), np.array([y]), np.array([z]))
    return np.array([B_int[0] + B_ext[0, 0], B_int[1] + B_ext[0, 1], B_int[2] + B_ext[0, 2]])

# Planet configurations
# Axial tilts are obliquity to orbit (angle between rotation axis and orbital plane normal)
# IAU convention: North pole is the one in the +Z hemisphere of the invariable plane
# For planets with tilt > 90°, the north pole is effectively "upside down" relative to orbit
PLANETS = {
    'earth': {
        'model': 'igrf2020',
        'name': 'Earth',
        'radius_km': 6378.1,
        'oblateness': 0.99665,  # polar/equatorial ratio
        'rotation_period_hours': 23.9345,
        'axial_tilt': 23.44,  # Earth's obliquity
        'l_shells': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                     2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                     8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                     16.0, 17.0, 18.0, 19.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0],
        'custom_field': None,
        'moons': []
    },
    'jupiter': {
        'model': 'jrm33',
        'name': 'Jupiter',
        'radius_km': 71492.0,
        'oblateness': 0.93513,
        'rotation_period_hours': 9.925,
        'axial_tilt': 3.13,  # Jupiter is nearly upright
        'l_shells': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                     2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                     8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
                     25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0,
                     65.0, 70.0, 75.0, 80.0],
        'custom_field': get_jupiter_field,
        'moons': [
            {'name': 'Io', 'orbit_radius': 5.905, 'period_hours': 42.459, 'size': 0.051}
        ]
    },
    'saturn': {
        'model': 'cassini11',
        'name': 'Saturn',
        'radius_km': 60268.0,
        'oblateness': 0.902,
        'rotation_period_hours': 10.656,
        'axial_tilt': 26.73,  # Similar to Earth
        'l_shells': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                     2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                     3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
                     8.5, 9.0, 9.5, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                     16.0, 17.0, 18.0, 19.0, 20.0, 22.0, 24.0, 26.0,
                     28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0],
        'custom_field': None,
        'moons': [
            {'name': 'Enceladus', 'orbit_radius': 3.948, 'period_hours': 32.885, 'size': 0.0084}
        ]
    },
    'uranus': {
        'model': 'gsfcq3',
        'name': 'Uranus',
        'radius_km': 25559.0,
        'oblateness': 0.97707,
        'rotation_period_hours': 17.24,
         'axial_tilt': 82.23,  # Uranus is tilted on its side! (IAU convention)
        'magnetic_tilt': 59.0,  # Magnetic axis tilted 59° from rotation axis
        'flip_north': True,  # GSFCQ3 model has opposite polarity convention
        'l_shells': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                     2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                     8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
                     22.0, 24.0, 26.0, 28.0, 30.0, 35.0, 40.0, 45.0, 50.0],
        'max_trace_r': 200,  # Wide area to capture full field lines
        'custom_field': None,
        'moons': [
            {'name': 'Miranda', 'orbit_radius': 5.08, 'period_hours': 33.92, 'size': 0.0184},
            {'name': 'Ariel', 'orbit_radius': 7.48, 'period_hours': 60.49, 'size': 0.0453}
        ]
    },
    'neptune': {
        'model': 'gsfco8',
        'name': 'Neptune',
        'radius_km': 24764.0,
        'oblateness': 0.98292,
        'rotation_period_hours': 16.11,
        'axial_tilt': 28.32,  # Similar to Earth/Saturn
        'magnetic_tilt': 47.0,  # Magnetic axis tilted 47° from rotation axis
        'l_shells': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                     2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                     8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
                     22.0, 24.0, 26.0, 28.0, 30.0, 35.0, 40.0, 45.0, 50.0],
        'max_trace_r': 200,  # Wide area to capture full field lines
        'custom_field': None,
        'moons': []
    }
}

def trace_field_line_bidirectional(start_pos, get_field_func, ds=0.02, max_steps=20000, max_r=150):
    """
    Trace field line in both directions from starting point.
    Coordinate system: x,y in equatorial plane, z along rotation axis.
    For Three.js: we swap y<->z (y becomes up).
    
    Stops cleanly at r=1.0 (planet surface) to avoid lines poking through.
    """
    def trace_one_direction(start, direction_sign):
        points = []
        pos = np.array(start, dtype=float)
        prev_r = np.linalg.norm(pos)
        
        for _ in range(max_steps):
            r = np.linalg.norm(pos)
            
            # Stop if we've crossed the surface (going inward)
            if r < 1.0:
                # Interpolate back to surface
                if len(points) > 0 and prev_r >= 1.0:
                    # Linear interpolation to find surface crossing
                    prev_pos = np.array(points[-1])
                    t = (1.0 - prev_r) / (r - prev_r)  # fraction along step
                    surface_pos = prev_pos + t * (pos - prev_pos)
                    surface_pos = surface_pos / np.linalg.norm(surface_pos)  # Normalize to exactly r=1
                    points.append(surface_pos.tolist())
                break
            
            if r > max_r:  # Too far - use configurable max
                break
            
            points.append(pos.tolist())
            prev_r = r
            
            try:
                B = get_field_func(pos[0], pos[1], pos[2])
            except:
                break
            
            B_mag = np.linalg.norm(B)
            if B_mag < 1e-12:
                break
            
            B_unit = B / B_mag
            pos = pos + direction_sign * B_unit * ds
        
        return points
    
    # Trace both directions
    forward = trace_one_direction(start_pos, +1)
    backward = trace_one_direction(start_pos, -1)
    
    # Combine: backward (reversed) + forward
    backward.reverse()
    return backward + forward

def generate_field_lines_for_planet(planet_key):
    """Generate field lines for a single planet."""
    config = PLANETS[planet_key]
    model = config['model']
    l_shells = config['l_shells']
    custom_field = config['custom_field']
    
    # Set max_r - use custom value if specified, otherwise 1.5x max L-shell
    max_r = config.get('max_trace_r', l_shells[-1] * 1.5)
    
    print(f"\n{'='*60}")
    print(f"Generating field lines for {config['name']}")
    print(f"Model: {model}")
    print(f"L-shells: {len(l_shells)} ({l_shells[0]} to {l_shells[-1]})")
    print(f"Max tracing radius: {max_r:.0f} R")
    print(f"{'='*60}")
    
    # Field function
    if custom_field:
        get_B = custom_field
    else:
        get_B = lambda x, y, z: get_field(model, x, y, z)
    
    all_l_shells = []
    num_longitudes = 24  # Field lines per L-shell for visualization
    
    # Generate regular L-shells with 24 longitudes for visualization
    for L in l_shells:
        print(f"  L = {L}...", end=' ', flush=True)
        lines = []
        
        for i in range(num_longitudes):
            phi = 2 * np.pi * i / num_longitudes
            # Start in equatorial plane at distance L
            start_x = L * np.cos(phi)
            start_y = L * np.sin(phi)
            start_z = 0.0
            
            points = trace_field_line_bidirectional(
                [start_x, start_y, start_z],
                get_B,
                ds=0.02 if L < 5 else 0.05,
                max_steps=20000,
                max_r=max_r
            )
            
            if len(points) > 10:
                # Convert to Three.js coords: swap y and z
                threejs_points = [[p[0], p[2], p[1]] for p in points]
                threejs_points = [[round(c, 3) for c in p] for p in threejs_points]
                lines.append(threejs_points)
        
        print(f"{len(lines)} lines")
        all_l_shells.append({
            'L': L,
            'lines': lines
        })
    
    # Generate moon flux tubes: 720 longitudes × multiple radial samples
    moon_flux_tubes = []
    for moon in config.get('moons', []):
        moon_name = moon['name']
        orbit_r = moon['orbit_radius']
        
        # Sample radial range around moon orbit (±0.1 Rp at 0.05 Rp resolution)
        radial_tolerance = 0.1
        radial_step = 0.05
        radial_samples = np.arange(orbit_r - radial_tolerance, orbit_r + radial_tolerance + 0.001, radial_step)
        radial_samples = [round(r, 3) for r in radial_samples if r > 1.0]  # Filter out values inside planet
        
        print(f"  Moon {moon_name}: flux tubes at L = {radial_samples} (720 lons each)")
        
        flux_tube_data = {
            'moon_name': moon_name,
            'orbit_radius': orbit_r,
            'radial_samples': []
        }
        
        for L in radial_samples:
            print(f"    L = {L}...", end=' ', flush=True)
            lines = []
            
            for i in range(720):  # 0.5 degree steps
                phi = 2 * np.pi * i / 720
                start_x = L * np.cos(phi)
                start_y = L * np.sin(phi)
                start_z = 0.0
                
                points = trace_field_line_bidirectional(
                    [start_x, start_y, start_z],
                    get_B,
                    ds=0.02 if L < 5 else 0.05,
                    max_steps=20000,
                    max_r=max_r
                )
                
                if len(points) > 10:
                    threejs_points = [[p[0], p[2], p[1]] for p in points]
                    threejs_points = [[round(c, 3) for c in p] for p in threejs_points]
                    lines.append(threejs_points)
            
            print(f"{len(lines)} lines")
            flux_tube_data['radial_samples'].append({
                'L': L,
                'lines': lines
            })
        
        moon_flux_tubes.append(flux_tube_data)
    
    # Save to JSON
    output = {
        'planet': config['name'],
        'model': model,
        'radius_km': config['radius_km'],
        'oblateness': config['oblateness'],
        'rotation_period_hours': config['rotation_period_hours'],
        'axial_tilt': config.get('axial_tilt', 0.0),
        'flip_north': config.get('flip_north', False),
        'moons': config['moons'],
        'l_shells': all_l_shells,
        'moon_flux_tubes': moon_flux_tubes  # Separate high-res flux tube data
    }
    
    # Add magnetic tilt if present
    if 'magnetic_tilt' in config:
        output['magnetic_tilt'] = config['magnetic_tilt']
    
    filename = f"field_lines_{planet_key}.json"
    with open(filename, 'w') as f:
        json.dump(output, f)
    
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"  Saved: {filename} ({size_mb:.2f} MB)")
    
    return filename

def main():
    """Generate field lines for all planets."""
    os.chdir('/Users/jamesodonoghue/planetary-mag-fields')
    
    # Earth and Jupiter already done with new format
    planets_to_generate = ['saturn', 'uranus', 'neptune']
    
    for planet in planets_to_generate:
        generate_field_lines_for_planet(planet)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)

if __name__ == '__main__':
    main()
