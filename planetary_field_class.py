#!/usr/bin/env python3
"""
Unified Planetary Magnetic Field Model
Supports Jupiter and Saturn using libinternalfield library

Installation:
    pip install numpy matplotlib scipy
    
    # Build libinternalfield:
    git clone https://github.com/mattkjames7/libinternalfield.git
    cd libinternalfield
    make
    sudo make install  # Optional: system-wide installation

Usage:
    from planetary_magfield import PlanetaryMagField
    
    # Jupiter with JRM33 model
    jupiter = PlanetaryMagField('jupiter', model='jrm33')
    Bx, By, Bz = jupiter.field(x=10, y=5, z=2)
    
    # Saturn with Dougherty 2018 model
    saturn = PlanetaryMagField('saturn', model='cassini11')
    Bx, By, Bz = saturn.field(x=10, y=5, z=2)
"""

import ctypes
import numpy as np
from pathlib import Path
import platform


class PlanetaryMagField:
    """
    Unified planetary magnetic field model for Jupiter and Saturn
    
    Supported Models:
    ----------------
    Jupiter:
        - jrm33 (default): JRM33 model (Connerney et al., 2022) - degree 30
        - jrm09: JRM09 model (Connerney et al., 2018) - degree 20
        - isaac: ISaAC model (Hess et al., 2017) - degree 10
        - vipal: VIPAL model (Hess et al., 2011) - degree 5
        - vip4: VIP4 model (Connerney 2007) - degree 4
        - vit4: VIT4 model (Connerney 2007) - degree 4
        - o6: O6 model (Connerney 2007) - degree 3
    
    Saturn:
        - cassini11 (default): Dougherty 2018 model - degree 12
        - cassini5: Cassini 5 model (Cao et al., 2012) - degree 5
        - cassini3: Cassini 3 model (Cao et al., 2011) - degree 3
        - burton2009: Burton et al., 2009 - degree 3
        - soi: Saturn Orbit Insertion (Dougherty et al., 2007) - degree 3
        - spv: Davis and Smith 1990 - degree 3
    """
    
    # Default models for each planet
    DEFAULT_MODELS = {
        'jupiter': 'jrm33',
        'saturn': 'cassini11'
    }
    
    # Planet radii in km
    PLANET_RADII = {
        'jupiter': 71492,  # 1 RJ
        'saturn': 60268    # 1 RS
    }
    
    def __init__(self, planet='jupiter', model=None, library_path=None):
        """
        Initialize planetary magnetic field model
        
        Parameters:
        -----------
        planet : str
            Planet name: 'jupiter' or 'saturn'
        model : str, optional
            Model name (see class docstring for options)
            If None, uses default model for planet
        library_path : str, optional
            Path to libinternalfield library
            If None, searches standard locations
        """
        self.planet = planet.lower()
        if self.planet not in ['jupiter', 'saturn']:
            raise ValueError(f"Planet must be 'jupiter' or 'saturn', got '{planet}'")
        
        # Set model (use default if not specified)
        if model is None:
            self.model = self.DEFAULT_MODELS[self.planet]
        else:
            self.model = model.lower()
        
        # Get planet radius
        self.R_planet = self.PLANET_RADII[self.planet]
        
        # Load library
        self._load_library(library_path)
        
        # Set model configuration
        self._configure_model()
        
        print(f"Initialized {self.planet.capitalize()} magnetic field model: {self.model}")
        print(f"Planet radius: {self.R_planet} km")
    
    def _load_library(self, library_path=None):
        """Load the libinternalfield shared library"""
        if library_path is None:
            # Try to find library in standard locations
            system = platform.system()
            
            search_paths = []
            
            if system == 'Darwin':  # macOS
                search_paths = [
                    '/usr/local/lib/libinternalfield.dylib',
                    '/usr/lib/libinternalfield.dylib',
                    './libinternalfield/lib/libinternalfield.dylib',
                    '../libinternalfield/lib/libinternalfield.dylib',
                    str(Path.home() / 'jupitermag_temp/libinternalfield/lib/libinternalfield.dylib'),
                    './libinternalfield.dylib',
                    '../libinternalfield/libinternalfield.dylib'
                ]
            elif system == 'Linux':
                search_paths = [
                    '/usr/local/lib/libinternalfield.so',
                    '/usr/lib/libinternalfield.so',
                    './libinternalfield/lib/libinternalfield.so',
                    '../libinternalfield/lib/libinternalfield.so',
                    './libinternalfield.so',
                    '../libinternalfield/libinternalfield.so'
                ]
            elif system == 'Windows':
                search_paths = [
                    'C:/Windows/System32/libinternalfield.dll',
                    './libinternalfield/lib/libinternalfield.dll',
                    '../libinternalfield/lib/libinternalfield.dll',
                    './libinternalfield.dll',
                    '../libinternalfield/libinternalfield.dll'
                ]
            
            # Find first existing path
            for path in search_paths:
                if Path(path).exists():
                    library_path = path
                    break
            
            if library_path is None:
                raise FileNotFoundError(
                    "Could not find libinternalfield library. "
                    "Please install it or provide library_path parameter.\n"
                    "Install with:\n"
                    "  git clone https://github.com/mattkjames7/libinternalfield.git\n"
                    "  cd libinternalfield\n"
                    "  make\n"
                    "  sudo make install"
                )
        
        # Load library
        self.lib = ctypes.CDLL(library_path)
        
        # Define function signatures
        self._setup_function_signatures()
        
        print(f"Loaded library: {library_path}")
    
    def _setup_function_signatures(self):
        """Define C function signatures for ctypes"""
        # void InternalField(int n, double *p0, double *p1, double *p2,
        #                    double *B0, double *B1, double *B2);
        self.lib.InternalField.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)
        ]
        self.lib.InternalField.restype = None
        
        # void SetInternalCFG(char *Model, bool CartIn, bool CartOut);
        self.lib.SetInternalCFG.argtypes = [
            ctypes.c_char_p,
            ctypes.c_bool,
            ctypes.c_bool
        ]
        self.lib.SetInternalCFG.restype = None
        
        # void GetInternalCFG(char *Model, bool *CartIn, bool *CartOut);
        self.lib.GetInternalCFG.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_bool),
            ctypes.POINTER(ctypes.c_bool)
        ]
        self.lib.GetInternalCFG.restype = None
    
    def _configure_model(self):
        """Configure the internal field model"""
        # Set model name and coordinate systems
        # Use Cartesian input and output
        model_bytes = self.model.encode('utf-8')
        self.lib.SetInternalCFG(model_bytes, True, True)
        
        # Verify configuration
        model_buffer = ctypes.create_string_buffer(100)
        cart_in = ctypes.c_bool()
        cart_out = ctypes.c_bool()
        self.lib.GetInternalCFG(model_buffer, ctypes.byref(cart_in), ctypes.byref(cart_out))
        
        configured_model = model_buffer.value.decode('utf-8')
        if configured_model != self.model:
            raise RuntimeError(f"Model configuration failed. Expected '{self.model}', got '{configured_model}'")
    
    def field(self, x, y, z, coords='cartesian', units='nT'):
        """
        Calculate magnetic field at given position(s)
        
        Parameters:
        -----------
        x, y, z : float or array-like
            Position coordinates
            If coords='cartesian': Cartesian coordinates in planet radii
            If coords='spherical': (r, theta, phi) where:
                r = radial distance in planet radii
                theta = colatitude in degrees (0 at north pole)
                phi = longitude in degrees
        coords : str
            'cartesian' or 'spherical'
        units : str
            'nT' (nanoTesla) or 'G' (Gauss)
            
        Returns:
        --------
        Bx, By, Bz : float or array
            Magnetic field components
            If coords='cartesian': (Bx, By, Bz) in nT or Gauss
            If coords='spherical': (Br, Btheta, Bphi) in nT or Gauss
        """
        # Convert to numpy arrays
        x = np.atleast_1d(x).astype(np.float64)
        y = np.atleast_1d(y).astype(np.float64)
        z = np.atleast_1d(z).astype(np.float64)
        
        # Ensure same shape
        n = max(len(x), len(y), len(z))
        if len(x) == 1:
            x = np.full(n, x[0])
        if len(y) == 1:
            y = np.full(n, y[0])
        if len(z) == 1:
            z = np.full(n, z[0])
        
        if not (len(x) == len(y) == len(z)):
            raise ValueError("x, y, z must have same length")
        
        # Convert spherical to cartesian if needed
        if coords.lower() == 'spherical':
            x_cart, y_cart, z_cart = self._spherical_to_cartesian(x, y, z)
        else:
            x_cart, y_cart, z_cart = x, y, z
        
        # Prepare output arrays
        Bx = np.zeros(n, dtype=np.float64)
        By = np.zeros(n, dtype=np.float64)
        Bz = np.zeros(n, dtype=np.float64)
        
        # Call C library
        self.lib.InternalField(
            n,
            x_cart.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_cart.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            z_cart.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Bx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            By.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Bz.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        
        # Convert back to spherical if needed
        if coords.lower() == 'spherical':
            Br, Btheta, Bphi = self._cartesian_to_spherical_field(
                x, y, z, Bx, By, Bz
            )
            Bx, By, Bz = Br, Btheta, Bphi
        
        # Convert units if needed
        if units.lower() == 'g':
            # 1 G = 10^5 nT
            Bx /= 1e5
            By /= 1e5
            Bz /= 1e5
        
        # Return scalars if input was scalar
        if n == 1:
            return Bx[0], By[0], Bz[0]
        else:
            return Bx, By, Bz
    
    def _spherical_to_cartesian(self, r, theta, phi):
        """Convert spherical to Cartesian coordinates"""
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        
        x = r * np.sin(theta_rad) * np.cos(phi_rad)
        y = r * np.sin(theta_rad) * np.sin(phi_rad)
        z = r * np.cos(theta_rad)
        
        return x, y, z
    
    def _cartesian_to_spherical_field(self, r, theta, phi, Bx, By, Bz):
        """Convert Cartesian field components to spherical"""
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        
        sin_theta = np.sin(theta_rad)
        cos_theta = np.cos(theta_rad)
        sin_phi = np.sin(phi_rad)
        cos_phi = np.cos(phi_rad)
        
        # Transformation matrix
        Br = (Bx * sin_theta * cos_phi + 
              By * sin_theta * sin_phi + 
              Bz * cos_theta)
        
        Btheta = (Bx * cos_theta * cos_phi + 
                  By * cos_theta * sin_phi - 
                  Bz * sin_theta)
        
        Bphi = (-Bx * sin_phi + By * cos_phi)
        
        return Br, Btheta, Bphi
    
    def field_magnitude(self, x, y, z, coords='cartesian'):
        """
        Calculate magnetic field magnitude |B|
        
        Parameters: Same as field()
        
        Returns:
        --------
        B : float or array
            Magnetic field magnitude in nT
        """
        Bx, By, Bz = self.field(x, y, z, coords=coords)
        return np.sqrt(Bx**2 + By**2 + Bz**2)
    
    def plot_equatorial_field(self, r_range=None, n_points=100, component='magnitude'):
        """
        Plot magnetic field in equatorial plane
        
        Parameters:
        -----------
        r_range : tuple, optional
            (r_min, r_max) in planet radii
            Default: (1, 20) for Jupiter, (1, 15) for Saturn
        n_points : int
            Number of points in each direction
        component : str
            'magnitude', 'Bx', 'By', 'Bz', or 'Br'
        """
        import matplotlib.pyplot as plt
        
        if r_range is None:
            r_range = (1, 20) if self.planet == 'jupiter' else (1, 15)
        
        # Create grid
        x = np.linspace(-r_range[1], r_range[1], n_points)
        y = np.linspace(-r_range[1], r_range[1], n_points)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)  # Equatorial plane
        
        # Calculate field
        Bx, By, Bz = self.field(X.ravel(), Y.ravel(), Z.ravel())
        Bx = Bx.reshape(X.shape)
        By = By.reshape(X.shape)
        Bz = Bz.reshape(X.shape)
        
        # Choose component to plot
        if component.lower() == 'magnitude':
            B = np.sqrt(Bx**2 + By**2 + Bz**2)
            title = f'{self.planet.capitalize()} Magnetic Field Magnitude (Equatorial Plane)'
            label = '|B| (nT)'
        elif component.lower() == 'bx':
            B = Bx
            title = f'{self.planet.capitalize()} Bx Component (Equatorial Plane)'
            label = 'Bx (nT)'
        elif component.lower() == 'by':
            B = By
            title = f'{self.planet.capitalize()} By Component (Equatorial Plane)'
            label = 'By (nT)'
        elif component.lower() == 'bz':
            B = Bz
            title = f'{self.planet.capitalize()} Bz Component (Equatorial Plane)'
            label = 'Bz (nT)'
        elif component.lower() == 'br':
            # Radial component
            r = np.sqrt(X**2 + Y**2)
            Br = (Bx * X + By * Y) / (r + 1e-10)
            B = Br
            title = f'{self.planet.capitalize()} Br Component (Equatorial Plane)'
            label = 'Br (nT)'
        
        # Mask inside planet
        r_grid = np.sqrt(X**2 + Y**2)
        B[r_grid < 1] = np.nan
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 9))
        
        # Use log scale for magnitude
        if component.lower() == 'magnitude':
            im = ax.pcolormesh(X, Y, B, shading='auto', cmap='viridis', 
                              norm=plt.matplotlib.colors.LogNorm(vmin=B[~np.isnan(B)].min(), 
                                                                  vmax=B[~np.isnan(B)].max()))
        else:
            im = ax.pcolormesh(X, Y, B, shading='auto', cmap='RdBu_r')
        
        # Draw planet
        circle = plt.Circle((0, 0), 1, color='gray', fill=True, alpha=0.5)
        ax.add_patch(circle)
        
        ax.set_xlabel(f'X (R_{self.planet[0].upper()})')
        ax.set_ylabel(f'Y (R_{self.planet[0].upper()})')
        ax.set_title(f'{title}\nModel: {self.model}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(im, ax=ax, label=label)
        
        plt.tight_layout()
        return fig, ax


def compare_models(planet='jupiter', models=None, position=(10, 0, 0)):
    """
    Compare different magnetic field models for a planet
    
    Parameters:
    -----------
    planet : str
        'jupiter' or 'saturn'
    models : list, optional
        List of model names to compare
        If None, uses common models for the planet
    position : tuple
        (x, y, z) position in planet radii
    """
    if models is None:
        if planet == 'jupiter':
            models = ['jrm33', 'jrm09', 'isaac', 'vip4', 'o6']
        else:
            models = ['cassini11', 'cassini5', 'cassini3', 'burton2009']
    
    print(f"\n{planet.capitalize()} Magnetic Field Model Comparison")
    print(f"Position: x={position[0]}, y={position[1]}, z={position[2]} R_{planet[0].upper()}")
    print("=" * 70)
    print(f"{'Model':<15} {'Bx (nT)':<12} {'By (nT)':<12} {'Bz (nT)':<12} {'|B| (nT)':<12}")
    print("-" * 70)
    
    for model_name in models:
        try:
            mag_model = PlanetaryMagField(planet, model=model_name)
            Bx, By, Bz = mag_model.field(position[0], position[1], position[2])
            B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
            print(f"{model_name:<15} {Bx:>11.2f} {By:>11.2f} {Bz:>11.2f} {B_mag:>11.2f}")
        except Exception as e:
            print(f"{model_name:<15} ERROR: {e}")
    
    print("=" * 70)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    print("\n" + "="*70)
    print("Planetary Magnetic Field Model - Demo")
    print("="*70)
    
    # Example 1: Jupiter field
    print("\n1. Jupiter Magnetic Field (JRM33 model)")
    print("-" * 70)
    jupiter = PlanetaryMagField('jupiter', model='jrm33')
    
    # Calculate field at a point
    x, y, z = 10, 0, 0  # 10 RJ from center
    Bx, By, Bz = jupiter.field(x, y, z)
    B_mag = jupiter.field_magnitude(x, y, z)
    
    print(f"Field at (x={x}, y={y}, z={z}) RJ:")
    print(f"  Bx = {Bx:.2f} nT")
    print(f"  By = {By:.2f} nT")
    print(f"  Bz = {Bz:.2f} nT")
    print(f"  |B| = {B_mag:.2f} nT")
    
    # Example 2: Saturn field (Dougherty 2018)
    print("\n2. Saturn Magnetic Field (Cassini11 / Dougherty 2018 model)")
    print("-" * 70)
    saturn = PlanetaryMagField('saturn', model='cassini11')
    
    x, y, z = 10, 0, 0  # 10 RS from center
    Bx, By, Bz = saturn.field(x, y, z)
    B_mag = saturn.field_magnitude(x, y, z)
    
    print(f"Field at (x={x}, y={y}, z={z}) RS:")
    print(f"  Bx = {Bx:.2f} nT")
    print(f"  By = {By:.2f} nT")
    print(f"  Bz = {Bz:.2f} nT")
    print(f"  |B| = {B_mag:.2f} nT")
    
    # Example 3: Spherical coordinates
    print("\n3. Using Spherical Coordinates")
    print("-" * 70)
    r, theta, phi = 5, 45, 0  # 5 RJ, 45° colatitude, 0° longitude
    Br, Btheta, Bphi = jupiter.field(r, theta, phi, coords='spherical')
    
    print(f"Jupiter field at (r={r}, θ={theta}°, φ={phi}°):")
    print(f"  Br = {Br:.2f} nT")
    print(f"  Bθ = {Btheta:.2f} nT")
    print(f"  Bφ = {Bphi:.2f} nT")
    
    # Example 4: Compare models
    print("\n4. Model Comparison")
    compare_models('jupiter', position=(10, 0, 0))
    compare_models('saturn', position=(10, 0, 0))
    
    # Example 5: Plot equatorial field
    print("\n5. Plotting equatorial magnetic field...")
    print("-" * 70)
    
    fig1, ax1 = jupiter.plot_equatorial_field(r_range=(1, 15), component='magnitude')
    fig2, ax2 = saturn.plot_equatorial_field(r_range=(1, 12), component='magnitude')
    
    plt.show()
    
    print("\nDemo complete!")
