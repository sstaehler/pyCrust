#!/usr/bin/env python3
"""
pyCrust-Moon

Create a crustal thickness map of the Moon from gravity and topography.

This script generates two different crustal thickness maps. The first assumes
that the density of both the crust and mantle are constant, whereas the second
includes the effect of lateral variations in crustal density. This script can
be used to reproduce the results presented in Wieczorek et al. (2013).


Wieczorek, M. A., G. A. Neumann, F. Nimmo, W. S. Kiefer, G. J. Taylor,
    H. J. Melosh, R. J. Phillips, S. C. Solomon, J. C. Andrews-Hanna,
    S. W. Asmar, A. S. Konopliv, F. G. Lemoine, D. E. Smith, M. M. Watkins,
    J. G. Williams, M. T. Zuber (2013), The crust of the Moon as seen by GRAIL,
    Science, 339, 671-675, doi:10.1126/science.1231530, 2013.
"""
import numpy as np

import pyshtools

import pyMoho

# ==== MAIN FUNCTION ====


def main():

    lmax = 900
    lmax_calc = 600

    gravfile = 'Data/JGGRAIL_900C11A_SHA.TAB'
    topofile = 'Data/LOLA1500p.sh'
    densityfile = 'Data/density_no_mare_n3000_f3050_719.sh'

    potcoefs, lmaxp, header = pyshtools.shio.shread(gravfile, header=True,
                                                    lmax=lmax)
    pot = pyshtools.SHCoeffs.from_array(potcoefs)
    pot.r_ref = float(header[0]) * 1.e3
    pot.gm = float(header[1]) * 1.e9
    pot.mass = pot.gm / float(pyshtools.constant.grav_constant)

    print('Gravity file = {:s}'.format(gravfile))
    print('Lmax of potential coefficients = {:d}'.format(lmaxp))
    print('Reference radius (m) = {:e}'.format(pot.r_ref))
    print('GM = {:e}\n'.format(pot.gm))

    topo = pyshtools.SHCoeffs.from_file(topofile, lmax=lmax)
    topo.r0 = topo.coeffs[0, 0, 0]

    print('Topography file = {:s}'.format(topofile))
    print('Lmax of topography coefficients = {:d}'.format(topo.lmax))
    print('Reference radius (m) = {:e}\n'.format(topo.r0))

    density = pyshtools.SHCoeffs.from_file(densityfile, lmax=lmax)
    rho_c = density.coeffs[0, 0, 0]

    print('Average grain density of crust (kg/m3) = {:e}'.format(rho_c))
    print('Lmax of density coefficients = {:d}\n'.format(density.lmax))

    a_12_14_lat = -3.3450
    a_12_14_long = -20.450

    # These parameters correspond to model 1 of Wieczorek et al. (2013).
    thickave = 34.e3
    porosity = 0.12
    rho_m = 3220.0
    filter = 1
    half = 80

    # Constant density model

    rho_c = rho_c * (1. - 0.12)  # assumed constant density

    moho = pyMoho.pyMoho(pot, topo, lmax, rho_c, rho_m, thickave,
                         filter_type=filter, half=half, lmax_calc=lmax_calc,
                         quiet=False)

    thick_grid = (topo.pad(lmax) - moho.pad(lmax)).expand(grid='DH2')
    thick_grid.plot(show=False, fname='Thick-Moon-1.png')
    moho.plot_spectrum(show=False, fname='Moho-spectrum-Moon-1.png')

    print('Crustal thickness at Apollo 12/14 landing sites (km) = {:e}'
          .format((topo.pad(lmax) - moho.pad(lmax))
                  .expand(lat=a_12_14_lat, lon=a_12_14_long) / 1.e3))

    # Model with variable density

    moho = pyMoho.pyMohoRho(pot, topo, density, porosity, lmax, rho_m,
                            thickave, filter_type=filter, half=half,
                            lmax_calc=lmax_calc, quiet=False)

    thick_grid = (topo-moho.pad(topo.lmax)).expand(grid='DH2')
    thick_grid.plot(show=False, fname='Thick-Moon-2.png')
    moho.plot_spectrum(show=False, fname='Moho-spectrum-Moon-2.png')

    print('Crustal thickness at Apollo 12/14 landing sites (km) = {:e}'.format(
        (topo-moho.pad(topo.lmax)).expand(lat=a_12_14_lat,
                                          lon=a_12_14_long)/1.e3))


# ==== EXECUTE SCRIPT ====
if __name__ == "__main__":
    main()
