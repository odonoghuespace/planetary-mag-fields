#!/usr/bin/env python3
"""
Simplified Planetary Magnetic Field Model using libinternalfield
Works for Jupiter and Saturn

Usage:
    from planetary_magfield_simple import get_field, list_models
    
    # Saturn with Dougherty 2018 model (cassini11)
    Bx, By, Bz = get_field('cassini11', x=10, y=0, z=0)
    
    # Jupiter with JRM33 model
    Bx, By, Bz = get_field('jrm33', x=10, y=0, z=0)
    
    # List available models
    list_models()
"""

import ctypes
import numpy as np
from pathlib import Path
import platform


class MagneticFieldLibrary:
    """Wrapper for libinternalfield library"""
    
    _instance = None
    _lib = None
    
    # Available models by planet
    MODELS = {
        'jupiter': {
            'jrm33': 'JRM33 (Connerney et al., 2022) - degree 30, order 13',
            'jrm09': 'JRM09 (Connerney et al., 2018) - degree 20, order 10',
            'isaac': 'ISaAC (Hess et al., 2017) - degree 10, order 10',
            'vipal': 'VIPAL (Hess et al., 2011) - degree 5, order 5',
            'vip4': 'VIP4 (Connerney et al., 1998) - degree 4, order 4',
            'vit4': 'VIT4 (Connerney et al., 1998) - degree 4, order 4',
            'o6': 'O6 (Connerney 2007) - degree 3, order 3',
            'o4': 'O4 (Connerney 2007) - degree 3, order 3',
            'gsfc15evs': 'GSFC 15EVS - degree 15, order 15',
            'gsfc13ev': 'GSFC 13EV - degree 13, order 13',
            'gsfc15ev': 'GSFC 15EV - degree 15, order 15',
        },
        'saturn': {
            'cassini11': 'Cassini 11 / Dougherty 2018 - degree 12, order 11',
            'cassini5': 'Cassini 5 (Cao et al., 2012) - degree 5, order 5',
            'cassini3': 'Cassini 3 (Cao et al., 2011) - degree 3, order 3',
            'burton2009': 'Burton et al., 2009 - degree 3, order 3',
            'p11as': 'Pioneer 11 A&S - degree 3, order 0',
            'p1184': 'Pioneer 11 1984 - degree 3, order 0',
            'soi': 'SOI (Dougherty et al., 2007) - degree 3, order 3',
            'spv': 'SPV (Davis and Smith 1990) - degree 3, order 2',
            'v1': 'Voyager 1 - degree 3, order 0',
            'v2': 'Voyager 2 - degree 3, order 0',
            'z3': 'Z3 (Connerney 1993) - degree 3, order 0',
        }
    }
    
    # Planet radii in km
    PLANET_RADII = {
        'jupiter': 71492,  # 1 RJ
        'saturn': 60268    # 1 RS
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize and load the library"""
        if self._lib is not None:
            return
        
        # Find library
        library_path = self._find_library()
        
        # Load library
        self._lib = ctypes.CDLL(library_path)
        
        # Define function signatures
        # modelFieldPtr getModelFieldPtr(const char *Model)
        self._lib.getModelFieldPtr.argtypes = [ctypes.c_char_p]
        self._lib.getModelFieldPtr.restype = ctypes.c_void_p
        
        print(f"Loaded libinternalfield from: {library_path}")
    
    def _find_library(self):
        """Find the library file"""
        system = platform.system()
        
        search_paths = []
        
        if system == 'Darwin':  # macOS
            lib_name = 'libinternalfield.dylib'
        elif system == 'Linux':
            lib_name = 'libinternalfield.so'
        elif system == 'Windows':
            lib_name = 'libinternalfield.dll'
        else:
            raise RuntimeError(f"Unsupported platform: {system}")
        
        # Search locations
        search_paths = [
            f'/usr/local/lib/{lib_name}',
            f'/usr/lib/{lib_name}',
            f'./libinternalfield/lib/{lib_name}',
            f'../libinternalfield/lib/{lib_name}',
            str(Path.home() / f'jupitermag_temp/libinternalfield/lib/{lib_name}'),
            f'./{lib_name}',
        ]
        
        for path in search_paths:
            if Path(path).exists():
                return path
        
        raise FileNotFoundError(
            f"Could not find {lib_name}. Please install libinternalfield:\n"
            "  git clone https://github.com/mattkjames7/libinternalfield.git\n"
            "  cd libinternalfield\n"
            "  make"
        )
    
    def get_field_function(self, model):
        """
        Get a field calculation function for the specified model
        
        Parameters:
        -----------
        model : str
            Model name (e.g., 'cassini11', 'jrm33')
        
        Returns:
        --------
        function : callable
            Function that takes (x, y, z) and returns (Bx, By, Bz)
        """
        # Get the function pointer
        model_bytes = model.lower().encode('utf-8')
        func_ptr = self._lib.getModelFieldPtr(model_bytes)
        
        if func_ptr is None or func_ptr == 0:
            available = []
            for planet_models in self.MODELS.values():
                available.extend(planet_models.keys())
            raise ValueError(
                f"Model '{model}' not found. Available models:\n" + 
                "\n".join(f"  - {m}" for m in sorted(available))
            )
        
        # Define the function signature for the returned function pointer
        # void (*modelFieldPtr)(double x, double y, double z, double *Bx, double *By, double *Bz)
        FIELD_FUNC = ctypes.CFUNCTYPE(
            None,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)
        )
        
        field_func = FIELD_FUNC(func_ptr)
        
        def calculate_field(x, y, z):
            """Calculate field at position(s)"""
            # Convert to numpy arrays
            x = np.atleast_1d(np.asarray(x, dtype=np.float64))
            y = np.atleast_1d(np.asarray(y, dtype=np.float64))
            z = np.atleast_1d(np.asarray(z, dtype=np.float64))
            
            # Ensure same length
            n = max(len(x), len(y), len(z))
            if len(x) == 1:
                x = np.full(n, x[0])
            if len(y) == 1:
                y = np.full(n, y[0])
            if len(z) == 1:
                z = np.full(n, z[0])
            
            if not (len(x) == len(y) == len(z)):
                raise ValueError("x, y, z must have same length")
            
            # Calculate field for each point
            Bx = np.zeros(n, dtype=np.float64)
            By = np.zeros(n, dtype=np.float64)
            Bz = np.zeros(n, dtype=np.float64)
            
            for i in range(n):
                bx_val = ctypes.c_double()
                by_val = ctypes.c_double()
                bz_val = ctypes.c_double()
                
                field_func(
                    float(x[i]), float(y[i]), float(z[i]),
                    ctypes.byref(bx_val),
                    ctypes.byref(by_val),
                    ctypes.byref(bz_val)
                )
                
                Bx[i] = bx_val.value
                By[i] = by_val.value
                Bz[i] = bz_val.value
            
            # Return scalar if input was scalar
            if n == 1:
                return Bx[0], By[0], Bz[0]
            else:
                return Bx, By, Bz
        
        return calculate_field


# Global library instance
_lib_instance = None


def get_library():
    """Get or create library instance"""
    global _lib_instance
    if _lib_instance is None:
        _lib_instance = MagneticFieldLibrary()
    return _lib_instance


def get_field(model, x, y, z, coords='cartesian'):
    """
    Calculate magnetic field for a given model
    
    Parameters:
    -----------
    model : str
        Model name (e.g., 'cassini11' for Saturn Dougherty 2018, 'jrm33' for Jupiter)
    x, y, z : float or array-like
        Position in planetary radii
        If coords='cartesian': Cartesian coordinates (x, y, z)
        If coords='spherical': (r, theta, phi) where theta is colatitude in degrees
    coords : str
        'cartesian' or 'spherical'
    
    Returns:
    --------
    Bx, By, Bz : float or array
        Magnetic field components in nT
        If coords='cartesian': (Bx, By, Bz)
        If coords='spherical': (Br, Btheta, Bphi)
    
    Examples:
    ---------
    >>> # Saturn Dougherty 2018 model at 10 RS
    >>> Bx, By, Bz = get_field('cassini11', x=10, y=0, z=0)
    >>> 
    >>> # Jupiter JRM33 model at 5 RJ
    >>> Bx, By, Bz = get_field('jrm33', x=5, y=0, z=0)
    >>> 
    >>> # Spherical coordinates
    >>> Br, Bt, Bp = get_field('cassini11', r=10, theta=45, phi=0, coords='spherical')
    """
    lib = get_library()
    field_func = lib.get_field_function(model)
    
    # Convert spherical to Cartesian if needed
    if coords.lower() == 'spherical':
        # x=r, y=theta, z=phi
        r, theta, phi = x, y, z
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        
        x_cart = r * np.sin(theta_rad) * np.cos(phi_rad)
        y_cart = r * np.sin(theta_rad) * np.sin(phi_rad)
        z_cart = r * np.cos(theta_rad)
        
        Bx, By, Bz = field_func(x_cart, y_cart, z_cart)
        
        # Convert field back to spherical
        sin_theta = np.sin(theta_rad)
        cos_theta = np.cos(theta_rad)
        sin_phi = np.sin(phi_rad)
        cos_phi = np.cos(phi_rad)
        
        Br = (Bx * sin_theta * cos_phi + 
              By * sin_theta * sin_phi + 
              Bz * cos_theta)
        
        Btheta = (Bx * cos_theta * cos_phi + 
                  By * cos_theta * sin_phi - 
                  Bz * sin_theta)
        
        Bphi = (-Bx * sin_phi + By * cos_phi)
        
        return Br, Btheta, Bphi
    else:
        return field_func(x, y, z)


def get_field_magnitude(model, x, y, z, coords='cartesian'):
    """
    Calculate magnetic field magnitude
    
    Parameters: Same as get_field()
    
    Returns:
    --------
    B : float or array
        Magnetic field magnitude in nT
    """
    Bx, By, Bz = get_field(model, x, y, z, coords=coords)
    return np.sqrt(Bx**2 + By**2 + Bz**2)


def list_models(planet=None):
    """
    List available magnetic field models
    
    Parameters:
    -----------
    planet : str, optional
        If specified, only show models for this planet ('jupiter' or 'saturn')
    """
    lib = get_library()
    
    if planet is None:
        print("\nAvailable Magnetic Field Models:")
        print("=" * 70)
        for planet_name, models in lib.MODELS.items():
            print(f"\n{planet_name.upper()}:")
            print(f"  Planet radius: {lib.PLANET_RADII[planet_name]} km")
            print("-" * 70)
            for model_name, description in models.items():
                print(f"  {model_name:<15} - {description}")
    else:
        planet = planet.lower()
        if planet not in lib.MODELS:
            raise ValueError(f"Unknown planet '{planet}'. Use 'jupiter' or 'saturn'")
        
        models = lib.MODELS[planet]
        print(f"\n{planet.upper()} Magnetic Field Models:")
        print(f"Planet radius: {lib.PLANET_RADII[planet]} km")
        print("=" * 70)
        for model_name, description in models.items():
            print(f"  {model_name:<15} - {description}")


def compare_models(planet, models=None, position=(10, 0, 0)):
    """
    Compare different models at a specific position
    
    Parameters:
    -----------
    planet : str
        'jupiter' or 'saturn'
    models : list, optional
        List of model names. If None, compares all models for the planet
    position : tuple
        (x, y, z) in planet radii
    """
    lib = get_library()
    planet = planet.lower()
    
    if planet not in lib.MODELS:
        raise ValueError(f"Unknown planet '{planet}'")
    
    if models is None:
        models = list(lib.MODELS[planet].keys())
    
    x, y, z = position
    
    print(f"\n{planet.upper()} Magnetic Field Comparison")
    print(f"Position: ({x:.1f}, {y:.1f}, {z:.1f}) planetary radii")
    print("=" * 75)
    print(f"{'Model':<15} {'Bx (nT)':<12} {'By (nT)':<12} {'Bz (nT)':<12} {'|B| (nT)':<12}")
    print("-" * 75)
    
    for model in models:
        try:
            Bx, By, Bz = get_field(model, x, y, z)
            B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
            print(f"{model:<15} {Bx:>11.3f} {By:>11.3f} {Bz:>11.3f} {B_mag:>11.3f}")
        except Exception as e:
            print(f"{model:<15} ERROR: {e}")
    
    print("=" * 75)


def save_field_line_data(model, planet, r_range=(1, 30), n_points=100, filename=None, phi=0, line=0):
    """
    Save magnetic field data along a line to CSV
    
    Parameters:
    -----------
    model : str
        Model name
    planet : str
        Planet name ('jupiter' or 'saturn')
    r_range : tuple
        (r_min, r_max) in planet radii
    n_points : int
        Number of points
    filename : str, optional
        Output filename. If None, auto-generates based on model name
    phi : float
        Phi angle identifier for this line (degrees)
    line : int
        Line number identifier
    """
    import csv
    
    if filename is None:
        filename = f"{planet}_{model}_field_line.csv"
    
    # Create radial line along x-axis (in equatorial plane at specified phi angle)
    r = np.linspace(r_range[0], r_range[1], n_points)
    
    # Convert to Cartesian coordinates
    # For equatorial plane: theta = 90 degrees (colatitude)
    theta = 90.0  # equatorial plane
    phi_rad = np.deg2rad(phi)
    theta_rad = np.deg2rad(theta)
    
    x = r * np.sin(theta_rad) * np.cos(phi_rad)
    y = r * np.sin(theta_rad) * np.sin(phi_rad)
    z = r * np.cos(theta_rad)
    
    # Save to CSV in the format: phi, line, x, y, z
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['phi', 'line', 'x', 'y', 'z'])
        for i in range(n_points):
            writer.writerow([phi, line, x[i], y[i], z[i]])
    
    print(f"Saved field data to: {filename}")
    return filename


def plot_field_comparison(save_dir='.'):
    """
    Create and save plots comparing magnetic field models
    
    Parameters:
    -----------
    save_dir : str
        Directory to save plots
    """
    import matplotlib.pyplot as plt
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # ========== Plot 1: Saturn Models Comparison ==========
    print("\nCreating Saturn field comparison plot...")
    
    saturn_models = ['cassini11', 'cassini5', 'cassini3', 'burton2009', 'soi']
    x = np.linspace(1, 25, 100)
    y = np.zeros(100)
    z = np.zeros(100)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    for model in saturn_models:
        try:
            Bx, By, Bz = get_field(model, x, y, z)
            B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
            
            ax1.plot(x, B_mag, label=model, linewidth=2)
            ax2.plot(x, Bz, label=model, linewidth=2)
        except Exception as e:
            print(f"Error with {model}: {e}")
    
    ax1.set_xlabel('Distance (Saturn Radii)', fontsize=12)
    ax1.set_ylabel('|B| (nT)', fontsize=12)
    ax1.set_title('Saturn Magnetic Field Magnitude - Model Comparison\n(Equatorial Plane, x-axis)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.set_xlabel('Distance (Saturn Radii)', fontsize=12)
    ax2.set_ylabel('Bz (nT)', fontsize=12)
    ax2.set_title('Saturn Bz Component', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    saturn_file = os.path.join(save_dir, 'saturn_field_comparison.png')
    plt.savefig(saturn_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {saturn_file}")
    plt.close()
    
    # ========== Plot 2: Jupiter Models Comparison ==========
    print("Creating Jupiter field comparison plot...")
    
    jupiter_models = ['jrm33', 'jrm09', 'isaac', 'vip4', 'o6']
    x = np.linspace(1, 30, 100)
    y = np.zeros(100)
    z = np.zeros(100)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    for model in jupiter_models:
        try:
            Bx, By, Bz = get_field(model, x, y, z)
            B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
            
            ax1.plot(x, B_mag, label=model, linewidth=2)
            ax2.plot(x, Bz, label=model, linewidth=2)
        except Exception as e:
            print(f"Error with {model}: {e}")
    
    ax1.set_xlabel('Distance (Jupiter Radii)', fontsize=12)
    ax1.set_ylabel('|B| (nT)', fontsize=12)
    ax1.set_title('Jupiter Magnetic Field Magnitude - Model Comparison\n(Equatorial Plane, x-axis)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.set_xlabel('Distance (Jupiter Radii)', fontsize=12)
    ax2.set_ylabel('Bz (nT)', fontsize=12)
    ax2.set_title('Jupiter Bz Component', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    jupiter_file = os.path.join(save_dir, 'jupiter_field_comparison.png')
    plt.savefig(jupiter_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {jupiter_file}")
    plt.close()
    
    # ========== Plot 3: Saturn Cassini11 Detail ==========
    print("Creating Saturn cassini11 detailed plot...")
    
    x = np.linspace(1, 30, 200)
    y = np.zeros(200)
    z = np.zeros(200)
    
    Bx, By, Bz = get_field('cassini11', x, y, z)
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1.plot(x, B_mag, 'b-', linewidth=2)
    ax1.set_xlabel('Distance (Saturn Radii)', fontsize=11)
    ax1.set_ylabel('|B| (nT)', fontsize=11)
    ax1.set_title('Total Field Magnitude', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.plot(x, Bx, 'r-', linewidth=2, label='Bx')
    ax2.plot(x, By, 'g-', linewidth=2, label='By')
    ax2.plot(x, Bz, 'b-', linewidth=2, label='Bz')
    ax2.set_xlabel('Distance (Saturn Radii)', fontsize=11)
    ax2.set_ylabel('Field Component (nT)', fontsize=11)
    ax2.set_title('Field Components', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(x, np.abs(Bx), 'r-', linewidth=2, label='|Bx|')
    ax3.plot(x, np.abs(By), 'g-', linewidth=2, label='|By|')
    ax3.plot(x, np.abs(Bz), 'b-', linewidth=2, label='|Bz|')
    ax3.set_xlabel('Distance (Saturn Radii)', fontsize=11)
    ax3.set_ylabel('|Field Component| (nT)', fontsize=11)
    ax3.set_title('Absolute Field Components (log scale)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Calculate field at various distances for table
    distances = np.array([2, 5, 10, 15, 20, 25, 30])
    fields_at_d = []
    for d in distances:
        Bx_d, By_d, Bz_d = get_field('cassini11', d, 0, 0)
        B_d = np.sqrt(Bx_d**2 + By_d**2 + Bz_d**2)
        fields_at_d.append(B_d)
    
    ax4.plot(distances, fields_at_d, 'o-', markersize=8, linewidth=2, color='purple')
    ax4.set_xlabel('Distance (Saturn Radii)', fontsize=11)
    ax4.set_ylabel('|B| (nT)', fontsize=11)
    ax4.set_title('Field at Key Distances', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    for i, (d, b) in enumerate(zip(distances, fields_at_d)):
        ax4.annotate(f'{b:.1f} nT', (d, b), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    fig.suptitle('Saturn Magnetic Field - Cassini11 (Dougherty 2018)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    saturn_detail_file = os.path.join(save_dir, 'saturn_cassini11_detailed.png')
    plt.savefig(saturn_detail_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {saturn_detail_file}")
    plt.close()
    
    # ========== Plot 4: Jupiter JRM33 Detail ==========
    print("Creating Jupiter jrm33 detailed plot...")
    
    x = np.linspace(1, 30, 200)
    y = np.zeros(200)
    z = np.zeros(200)
    
    Bx, By, Bz = get_field('jrm33', x, y, z)
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1.plot(x, B_mag, 'b-', linewidth=2)
    ax1.set_xlabel('Distance (Jupiter Radii)', fontsize=11)
    ax1.set_ylabel('|B| (nT)', fontsize=11)
    ax1.set_title('Total Field Magnitude', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.plot(x, Bx, 'r-', linewidth=2, label='Bx')
    ax2.plot(x, By, 'g-', linewidth=2, label='By')
    ax2.plot(x, Bz, 'b-', linewidth=2, label='Bz')
    ax2.set_xlabel('Distance (Jupiter Radii)', fontsize=11)
    ax2.set_ylabel('Field Component (nT)', fontsize=11)
    ax2.set_title('Field Components', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(x, np.abs(Bx), 'r-', linewidth=2, label='|Bx|')
    ax3.plot(x, np.abs(By), 'g-', linewidth=2, label='|By|')
    ax3.plot(x, np.abs(Bz), 'b-', linewidth=2, label='|Bz|')
    ax3.set_xlabel('Distance (Jupiter Radii)', fontsize=11)
    ax3.set_ylabel('|Field Component| (nT)', fontsize=11)
    ax3.set_title('Absolute Field Components (log scale)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Calculate field at various distances for table
    distances = np.array([2, 5, 10, 15, 20, 25, 30])
    fields_at_d = []
    for d in distances:
        Bx_d, By_d, Bz_d = get_field('jrm33', d, 0, 0)
        B_d = np.sqrt(Bx_d**2 + By_d**2 + Bz_d**2)
        fields_at_d.append(B_d)
    
    ax4.plot(distances, fields_at_d, 'o-', markersize=8, linewidth=2, color='orange')
    ax4.set_xlabel('Distance (Jupiter Radii)', fontsize=11)
    ax4.set_ylabel('|B| (nT)', fontsize=11)
    ax4.set_title('Field at Key Distances', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    for i, (d, b) in enumerate(zip(distances, fields_at_d)):
        ax4.annotate(f'{b:.1f} nT', (d, b), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    fig.suptitle('Jupiter Magnetic Field - JRM33 (Connerney et al., 2022)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    jupiter_detail_file = os.path.join(save_dir, 'jupiter_jrm33_detailed.png')
    plt.savefig(jupiter_detail_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {jupiter_detail_file}")
    plt.close()
    
    print("\nAll plots saved successfully!")


if __name__ == '__main__':
    import os
    
    print("\n" + "="*70)
    print("Planetary Magnetic Field Calculator")
    print("="*70)
    
    # Get current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\nWorking directory: {script_dir}")
    
    # List available models
    list_models()
    
    # Example 1: Saturn Dougherty 2018 (cassini11)
    print("\n\n" + "="*70)
    print("Example 1: Saturn - Cassini11 (Dougherty 2018)")
    print("="*70)
    
    model = 'cassini11'
    x, y, z = 10, 0, 0  # 10 Saturn radii from center
    
    Bx, By, Bz = get_field(model, x, y, z)
    B_mag = get_field_magnitude(model, x, y, z)
    
    print(f"\nPosition: ({x}, {y}, {z}) RS")
    print(f"  Bx = {Bx:.3f} nT")
    print(f"  By = {By:.3f} nT")
    print(f"  Bz = {Bz:.3f} nT")
    print(f"  |B| = {B_mag:.3f} nT")
    
    # Example 2: Jupiter JRM33
    print("\n\n" + "="*70)
    print("Example 2: Jupiter - JRM33 (Latest model)")
    print("="*70)
    
    model = 'jrm33'
    x, y, z = 10, 0, 0  # 10 Jupiter radii from center
    
    Bx, By, Bz = get_field(model, x, y, z)
    B_mag = get_field_magnitude(model, x, y, z)
    
    print(f"\nPosition: ({x}, {y}, {z}) RJ")
    print(f"  Bx = {Bx:.3f} nT")
    print(f"  By = {By:.3f} nT")
    print(f"  Bz = {Bz:.3f} nT")
    print(f"  |B| = {B_mag:.3f} nT")
    
    # Example 3: Spherical coordinates
    print("\n\n" + "="*70)
    print("Example 3: Using Spherical Coordinates")
    print("="*70)
    
    r, theta, phi = 5, 45, 0  # 5 RS, 45° colatitude, 0° longitude
    Br, Btheta, Bphi = get_field('cassini11', r, theta, phi, coords='spherical')
    
    print(f"\nSaturn field at (r={r} RS, θ={theta}°, φ={phi}°):")
    print(f"  Br = {Br:.3f} nT")
    print(f"  Bθ = {Btheta:.3f} nT")
    print(f"  Bφ = {Bphi:.3f} nT")
    
    # Example 4: Vector of positions
    print("\n\n" + "="*70)
    print("Example 4: Multiple Positions (Vector)")
    print("="*70)
    
    x = np.array([5, 10, 15, 20])
    y = np.zeros(4)
    z = np.zeros(4)
    
    Bx, By, Bz = get_field('cassini11', x, y, z)
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    
    print(f"\nSaturn cassini11 model along x-axis:")
    print(f"{'Distance (RS)':<15} {'Bx (nT)':<12} {'|B| (nT)':<12}")
    print("-" * 40)
    for i in range(len(x)):
        print(f"{x[i]:<15.1f} {Bx[i]:<12.3f} {B_mag[i]:<12.3f}")
    
    # Example 5: Compare Saturn models
    print("\n")
    compare_models('saturn', position=(10, 0, 0))
    
    # Example 6: Compare Jupiter models
    print("\n")
    compare_models('jupiter', models=['jrm33', 'jrm09', 'vip4', 'o6'], position=(10, 0, 0))
    
    # Save CSV files for both planets
    print("\n\n" + "="*70)
    print("Saving CSV field line data...")
    print("="*70)
    
    saturn_csv = save_field_line_data('cassini11', 'saturn', r_range=(1, 30), n_points=200,
                                      filename=os.path.join(script_dir, 'saturn_cassini11_field_line.csv'),
                                      phi=0, line=0)
    
    jupiter_csv = save_field_line_data('jrm33', 'jupiter', r_range=(1, 30), n_points=200,
                                       filename=os.path.join(script_dir, 'jupiter_jrm33_field_line.csv'),
                                       phi=0, line=0)
    
    # Create and save plots
    print("\n\n" + "="*70)
    print("Creating and saving plots...")
    print("="*70)
    
    plot_field_comparison(save_dir=script_dir)
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
    print(f"\nOutput files saved in: {script_dir}")
    print("\nGenerated files:")
    print(f"  - saturn_cassini11_field_line.csv")
    print(f"  - jupiter_jrm33_field_line.csv")
    print(f"  - saturn_field_comparison.png")
    print(f"  - jupiter_field_comparison.png")
    print(f"  - saturn_cassini11_detailed.png")
    print(f"  - jupiter_jrm33_detailed.png")
    print("="*70)
