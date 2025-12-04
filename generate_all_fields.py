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
        'max_trace_r': 60,  # Extend further to capture full extent
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
        'max_trace_r': 150,  # Extended to capture full extent
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
        'max_trace_r': 90,  # Extended to capture full extent
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
                     7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0, 13.0, 14.0,
                     15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        'max_trace_r': 500,  # Extended to capture full field lines extending far from planet
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
                     7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0, 13.0, 14.0,
                     15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        'max_trace_r': 500,  # Extended to capture full field lines extending far from planet
        'custom_field': None,
        'moons': []
    }
}

def rotate_to_magnetic_frame(pos, magnetic_tilt_deg, rotation_axis_idx=1):
    """
    Rotate position from rotation frame to magnetic frame.
    magnetic_tilt_deg: angle between rotation and magnetic axes (degrees)
    rotation_axis_idx: which axis is the rotation axis (0=x, 1=y, 2=z)
    """
    # For Uranus/Neptune, magnetic axis is tilted in the y-z plane
    # rotation axis is z, so we rotate around x-axis (index 0)
    tilt_rad = np.radians(magnetic_tilt_deg)
    cos_t = np.cos(tilt_rad)
    sin_t = np.sin(tilt_rad)
    
    x, y, z = pos
    # Rotation around x-axis: y and z change
    y_rot = y * cos_t - z * sin_t
    z_rot = y * sin_t + z * cos_t
    return np.array([x, y_rot, z_rot])

def rotate_to_rotation_frame(pos, magnetic_tilt_deg):
    """Rotate position from magnetic frame back to rotation frame."""
    tilt_rad = np.radians(magnetic_tilt_deg)
    cos_t = np.cos(tilt_rad)
    sin_t = np.sin(tilt_rad)
    
    x, y, z = pos
    # Inverse rotation around x-axis
    y_rot = y * cos_t + z * sin_t
    z_rot = -y * sin_t + z * cos_t
    return np.array([x, y_rot, z_rot])

def trace_field_line_bidirectional(start_pos, get_field_func, ds=0.02, max_steps=20000, max_r=150):
    """
    Trace field line in both directions from starting point.
    Coordinate system: x,y in equatorial plane, z along rotation axis.
    For Three.js: we swap y<->z (y becomes up).
    
    Stops cleanly at r=1.0 (planet surface) to avoid lines poking through.
    Uses adaptive step size for better tracing in weak field regions.
    """
    def trace_one_direction(start, direction_sign):
        points = []
        pos = np.array(start, dtype=float)
        prev_r = np.linalg.norm(pos)
        
        for step_num in range(max_steps):
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
            # Much more lenient threshold for weak fields - keep tracing!
            # Field strength falls off as ~1/r^3, so can be very weak far from planet
            if B_mag < 1e-20:
                break
            
            B_unit = B / B_mag
            
            # Adaptive step size: larger steps when far from planet (r > 10)
            # This helps trace through weak field regions faster
            adaptive_ds = ds * (1.0 + max(0, (r - 10) / 20))  # Increases smoothly for r > 10
            
            pos = pos + direction_sign * B_unit * adaptive_ds
        
        return points
    
    # Trace both directions
    forward = trace_one_direction(start_pos, +1)
    backward = trace_one_direction(start_pos, -1)
    
    # Combine: backward (reversed) + forward
    backward.reverse()
    return backward + forward

def generate_grid_field_lines(get_B, max_r=500, r_min=1.1, r_max=10.0, step=0.1):
    """
    Generate field lines from a 3D grid of starting points.
    For Uranus/Neptune where the magnetic field is highly tilted,
    we sample points at X,Y,Z = ±r_min to ±r_max in steps.
    Returns a dict mapping radial distance shells to field lines.
    """
    print(f"  Generating grid field lines from r={r_min} to r={r_max} (step {step})")
    
    # Create grid values: negative to positive
    grid_vals = np.arange(r_min, r_max + step/2, step)
    grid_vals = np.concatenate([-grid_vals[::-1], grid_vals])  # e.g., [-10, ..., -1.1, 1.1, ..., 10]
    
    # Dictionary to store field lines by their starting radial distance
    field_lines_by_r = {}
    total_lines = 0
    
    # Sample on all 6 faces of a cube at each radial distance
    radial_shells = np.arange(r_min, r_max + step/2, step)
    
    for r_shell in radial_shells:
        r_key = round(r_shell, 1)
        lines = []
        
        # Generate starting points on the surface of a cube at this radial distance
        # Use 24 evenly spaced points per shell (like regular L-shells)
        num_points = 24
        
        for i in range(num_points):
            # Distribute points on sphere surface using Fibonacci spiral
            phi = 2 * np.pi * i / num_points
            # Place points in a ring around z-axis at this radius
            theta = np.pi / 2  # equatorial plane
            
            start_x = r_shell * np.cos(phi) * np.sin(theta)
            start_y = r_shell * np.sin(phi) * np.sin(theta)
            start_z = r_shell * np.cos(theta)  # = 0 for equator
            
            points = trace_field_line_bidirectional(
                [start_x, start_y, start_z],
                get_B,
                ds=0.02 if r_shell < 5 else 0.05,
                max_steps=50000,
                max_r=max_r
            )
            
            if len(points) > 10:
                # Convert to Three.js coords: swap y and z, take every 2nd point to reduce size
                threejs_points = [[p[0], p[2], p[1]] for p in points[::2]]
                threejs_points = [[round(c, 2) for c in p] for p in threejs_points]
                lines.append(threejs_points)
        
        if lines:
            field_lines_by_r[r_key] = {
                'L': r_key,
                'lines': lines
            }
            total_lines += len(lines)
            print(f"    r = {r_key}: {len(lines)} lines")
    
    print(f"  Total: {total_lines} field lines across {len(field_lines_by_r)} radial shells")
    return field_lines_by_r

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
    
    # Get magnetic tilt if present (for Uranus/Neptune)
    magnetic_tilt = config.get('magnetic_tilt', 0.0)
    
    # For Uranus and Neptune, use grid-based approach due to highly tilted fields
    if planet_key in ['uranus', 'neptune']:
        print(f"  Using grid-based L-shell generation for {config['name']}")
        # Generate field lines from grid points at r = 1.1 to 10 in steps of 0.1
        grid_shells = generate_grid_field_lines(get_B, max_r=500, r_min=1.1, r_max=10.0, step=0.1)
        # Convert to list format
        all_l_shells = list(grid_shells.values())
    else:
        # Generate regular L-shells with 24 longitudes for visualization
        for L in l_shells:
            print(f"  L = {L}...", end=' ', flush=True)
            lines = []
            
            for i in range(num_longitudes):
                phi = 2 * np.pi * i / num_longitudes
                # Start in MAGNETIC equatorial plane at distance L
                start_x = L * np.cos(phi)
                start_y = L * np.sin(phi)
                start_z = 0.0
                
                # If field has magnetic tilt, rotate from magnetic frame to rotation frame for display
                if magnetic_tilt > 0:
                    start_pos = rotate_to_rotation_frame(np.array([start_x, start_y, start_z]), magnetic_tilt)
                else:
                    start_pos = np.array([start_x, start_y, start_z])
                
                points = trace_field_line_bidirectional(
                    start_pos.tolist(),
                    get_B,
                    ds=0.02 if L < 5 else 0.05,
                    max_steps=20000,
                    max_r=max_r
                )
                
                if len(points) > 10:
                    # Convert to Three.js coords: swap y and z, take every 2nd point to reduce size
                    threejs_points = [[p[0], p[2], p[1]] for p in points[::2]]
                    threejs_points = [[round(c, 2) for c in p] for p in threejs_points]  # 2 decimal places to reduce file size
                    lines.append(threejs_points)
            
            print(f"{len(lines)} lines")
            all_l_shells.append({
                'L': L,
                'lines': lines
            })
    
    # Generate moon flux tubes: 720 longitudes (0.5° steps) at exact orbital radius
    # Moon orbits are nearly circular (e < 0.005), so we only need 1 radial sample
    moon_flux_tubes = []
    for moon in config.get('moons', []):
        moon_name = moon['name']
        orbit_r = moon['orbit_radius']
        
        print(f"  Moon {moon_name}: flux tubes at L = {orbit_r:.3f} (720 lons, 0.5° steps)")
        
        flux_tube_data = {
            'moon_name': moon_name,
            'orbit_radius': orbit_r,
            'radial_samples': []
        }
        
        print(f"    L = {orbit_r:.3f}...", end=' ', flush=True)
        lines = []
        
        for i in range(720):  # 0.5 degree steps for smooth coverage
            phi = 2 * np.pi * i / 720
            # MOON FLUX TUBES: Start in ROTATION equatorial plane where moons actually orbit
            # Use -sin to match Three.js moon position: z = -sin(angle)
            start_x = orbit_r * np.cos(phi)
            start_y = -orbit_r * np.sin(phi)  # Negative to match viewer's moon.position.z = -sin(angle)
            start_z = 0.0  # z=0 in ROTATION frame (where moons orbit)
            
            start_pos = np.array([start_x, start_y, start_z])
            
            # Use very small step size and many steps to ensure we always reach the surface
            # Adaptive stepping will increase step size in weak field regions automatically
            points = trace_field_line_bidirectional(
                start_pos.tolist(),
                get_B,
                ds=0.005,  # Very small initial step for accuracy near moon
                max_steps=500000,  # MANY steps to ensure we reach surface even in weak fields
                max_r=500  # 500 planetary radii to ensure full coverage
            )
            
            if len(points) > 10:
                # Take every 2nd point to reduce file size
                threejs_points = [[p[0], p[2], p[1]] for p in points[::2]]
                threejs_points = [[round(c, 2) for c in p] for p in threejs_points]  # 2 decimal places to reduce file size
                lines.append(threejs_points)
            else:
                # If tracing failed, add empty placeholder to maintain indexing
                lines.append(None)
        
        # Count valid lines and check endpoints
        valid_count = sum(1 for l in lines if l is not None)
        disconnected_count = 0
        for l in lines:
            if l is not None:
                # Check if endpoints are near surface (r ≈ 1.0)
                start_r = np.sqrt(l[0][0]**2 + l[0][1]**2 + l[0][2]**2)
                end_r = np.sqrt(l[-1][0]**2 + l[-1][1]**2 + l[-1][2]**2)
                if abs(start_r - 1.0) > 0.05 or abs(end_r - 1.0) > 0.05:
                    disconnected_count += 1
        
        print(f"{valid_count} valid lines, {disconnected_count} not reaching surface")
        flux_tube_data['radial_samples'].append({
            'L': orbit_r,
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
    
    # Regenerate all planets with new flux tube format
    planets_to_generate = ['earth', 'jupiter', 'saturn', 'uranus', 'neptune']
    
    for planet in planets_to_generate:
        generate_field_lines_for_planet(planet)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)

if __name__ == '__main__':
    main()
