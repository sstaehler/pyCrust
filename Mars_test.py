#!/usr/bin/env python3
"""
pyCrust_Mars

Create a crustal thickness map of Mars from gravity and topography.

This script generates two different crustal thickness maps. The first assumes
that the density of both the crust and mantle are constant, whereas the second
includes the effect of different densities on either side of the dichotomy
boundary. The average crustal thickness is iterated in order to obtain
a specified minimum crustal thickness.
"""
import numpy as np

import pyshtools

import pyMoho
from Hydrostatic import HydrostaticShapeLith
from Hydrostatic import HydrostaticShape

# ==== MAIN FUNCTION ====


def main():

    gravfile = 'Data/gmm3_120_sha.tab'
    topofile = 'Data/MarsTopo719.shape'
    densityfile = 'Data/dichotomy_359.sh'

    model_name = ['dwcold', 'dwhot_modif', 'dwhot_stairs', 'DWTh2Ref1',
                  'DWTh2Ref2', 'eh70cold', '"eh70hot', 'Gudkova']
    spec = 'Data/Mars-reference-interior-models/model_'
    interior_file = [spec + name for name in model_name]

    lmax_calc = 90
    lmax = lmax_calc * 4

    potcoefs, lmaxp, header = pyshtools.shio.shread(gravfile, header=True,
                                                    lmax=lmax)
    potential = pyshtools.SHCoeffs.from_array(potcoefs)
    potential.r_ref = float(header[0]) * 1.e3
    potential.gm = float(header[1]) * 1.e9
    potential.mass = potential.gm / float(pyshtools.constant.grav_constant)

    print('Gravity file = {:s}'.format(gravfile))
    print('Lmax of potential coefficients = {:d}'.format(lmaxp))
    print('Reference radius (km) = {:f}'.format(potential.r_ref / 1.e3))
    print('GM = {:e}'.format(potential.gm))
    print('Mass = {:e}'.format(potential.mass))

    model = 3
    lmax_hydro = 15
    omega = float(pyshtools.constant.omega_mars)
    print('Omega = {:e}'.format(omega))

    # --- read 1D reference interior model ---

    print('=== Reading model {:s} ==='.format(model_name[model]))

    with open(interior_file[model], 'r') as f:
        lines = f.readlines()
        print(lines[0].strip())
        data = lines[1].split()
        if float(data[2]) != 1:
            raise RuntimeError('Program not capable of reading polynomial ' +
                               'files')
        num = int(lines[2].split()[0])
        crust_index = int(lines[2].split()[3])
        mantle_index = int(lines[2].split()[3])
        radius = np.zeros(num)
        rho = np.zeros(num)
        for i in range(0, num):
            data = lines[i+3].split()
            radius[i] = float(data[0])
            rho[i] = float(data[1])

        r0_model = radius[num-1]
        print('Surface radius of model (km) = {:f}'.format(r0_model / 1.e3))

        n = num - 1
        rho[n] = 0.  # the density above the surface is zero
        rho_mantle = rho[crust_index-1]
        print('Mantle density (kg/m3) = {:f}'.format(rho_mantle))

    # --- Compute gravity contribution from hydrostatic density interfaces ---

    if True:
        for i in range(1, 8):
            # compute values for a planet that is completely fluid
            fn = True
            nmax = i

            hlm_fluid, clm_fluid, mass_model = \
                HydrostaticShape(radius, rho, omega, potential.gm,
                                 potential.r_ref, nmax=nmax,
                                 finiteamplitude=fn)

            print('--- Hydrostatic relief of surface ---')
            print('nmax = {:d}'.format(nmax))
            print('h20 = {:e}\n'.format(hlm_fluid[n].coeffs[0, 2, 0]) +
                  'h40 = {:e}'.format(hlm_fluid[n].coeffs[0, 4, 0]))


# ==== EXECUTE SCRIPT ====


if __name__ == "__main__":
    main()
