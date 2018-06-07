#!/usr/bin/env python3

# coding: utf-8

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
import sys
import pyMoho
from Hydrostatic import HydrostaticShapeLith
from multiprocessing import Pool

def calc_crust(fnam='MVD/model4mvdSAN_AK100.dat',
               t0=1.e3,       # minimum crustal thickness
               d_lith=150.e3  # Lithosphere thickness
               ):
    # Load input file
    gravfile = 'Data/gmm3_120_sha.tab'
    topofile = 'Data/MarsTopo719.shape'
    densityfile = 'Data/dichotomy_359.sh'

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
    print('GM = {:e}\n'.format(potential.gm))

    topo = pyshtools.SHCoeffs.from_file(topofile, lmax=lmax)
    topo.r0 = topo.coeffs[0, 0, 0]

    print('Topography file = {:s}'.format(topofile))
    print('Lmax of topography coefficients = {:d}'.format(topo.lmax))
    print('Reference radius (km) = {:f}\n'.format(topo.r0 / 1.e3))

    density = pyshtools.SHCoeffs.from_file(densityfile, lmax=lmax)

    print('Lmax of density coefficients = {:d}\n'.format(density.lmax))

    lat_insight = 4.43
    lon_insight = 135.84

    filter = 1
    half = 50
    nmax = 7
    lmax_hydro = 15
    t0_sigma = 5.  # maximum difference between minimum crustal thickness
    omega = float(pyshtools.constant.omega_mars)

    # --- read 1D reference interior model ---

    print('=== Reading model {:s} ==='.format(fnam))

    with open(fnam, 'r') as f:
        lines = f.readlines()

    ncomments = 4  # Remove initial four lines in AxiSEM files
    nlines = len(lines)
    nlayer = nlines - ncomments
    radius = np.zeros(nlayer)
    rho = np.zeros(nlayer)
    lines = lines[::-1]
    crust_index = nlayer - 6
    for i in range(0, nlayer):
        data = lines[i].split()
        radius[i] = float(data[0])
        rho[i] = float(data[1])
        # Q = float(data[4])
        # if (Q in (500, 600)) and Q_last not in (500, 600):
        #     crust_index = i
        # Q_last = Q

    # Calculate crustal density
    mass_crust = 0
    for i in range(crust_index, nlayer-1):
        vol_layer = (radius[i+1] - radius[i]) * 1e3 * 4 * np.pi
        mass_crust += rho[i] * vol_layer
    vol_crust = (radius[-1] - radius[crust_index]) * 1e3 * 4 * np.pi
    rho_c = mass_crust / vol_crust
    print('Crustal density: = {:8.1f}'.format(rho_c))

    r0_model = radius[nlayer-1]
    print('Surface radius of model (km) = {:8.1f}'.format(r0_model / 1.e3))

    # Find layer at bottom of lithosphere
    for i in range(0, nlayer):
        if radius[i] <= (r0_model - d_lith) and \
                radius[i+1] > (r0_model - d_lith):
            if radius[i] == (r0_model - d_lith):
                i_lith = i
            elif (r0_model - d_lith) - radius[i] <= \
                    radius[i+1] - (r0_model - d_lith):
                i_lith = i
            else:
                i_lith = i + 1
            break

    n = nlayer - 1
    rho[n] = 0.  # the density above the surface is zero
    rho_mantle = rho[crust_index-1]
    print('Mantle density (kg/m3) = {:8.1f}'.format(rho_mantle))

    print('Assumed depth of lithosphere (km) = {:6.1f}'.format(d_lith / 1.e3))
    print('Actual depth of lithosphere in discretized model (km) = {:6.1f}'
          .format((r0_model - radius[i_lith]) / 1.e3))

    thickave = r0_model - radius[crust_index]  # initial guess of average crustal thickness
    print('Crustal thickness (km) = {:5.1f}'.format((r0_model -
                                                     radius[crust_index]) / 1e3))
    print('Moho layer: {:d}'.format(crust_index))

    # --- Compute gravity contribution from hydrostatic density interfaces ---

    hlm, clm_hydro, mass_model = HydrostaticShapeLith(radius, rho, i_lith,
                                                      potential, omega,
                                                      lmax_hydro, finiteamplitude=False)

    print('Total mass of model (kg) = {:e}'.format(mass_model))
    print('% of J2 arising from beneath lithosphere = {:f}'
          .format(clm_hydro.coeffs[0, 2, 0]/potential.coeffs[0, 2, 0] * 100.))

    potential.coeffs[:, :lmax_hydro+1, :lmax_hydro+1] -= clm_hydro.coeffs[:, :lmax_hydro+1, :lmax_hydro+1]

    # --- Constant density model ---
    print('-- Constant density model --\nrho_c = {:f}'.format(rho_c))

    tmin = 1.e9

    while abs(tmin - t0) > t0_sigma:
        # iterate to fit assumed minimum crustal thickness

        moho = pyMoho.pyMoho(potential, topo, lmax, rho_c, rho_mantle,
                             thickave, filter_type=filter, half=half,
                             lmax_calc=lmax_calc, nmax=nmax, quiet=True)

        thick_grid = (topo.pad(lmax) - moho.pad(lmax)).expand(grid='DH2')
        print('Average crustal thickness (km) = {:6.2f}'.format(thickave / 1.e3))
        print('Crustal thickness at InSight landing sites (km) = {:6.2f}'
              .format((topo.pad(lmax) - moho.pad(lmax))
                      .expand(lat=lat_insight, lon=lon_insight) / 1.e3))
        tmin = thick_grid.data.min()
        tmax = thick_grid.data.max()
        print('Minimum thickness (km) = {:6.2f}'.format(tmin / 1.e3))
        print('Maximum thickness (km) = {:6.2f}'.format(tmax / 1.e3))
        thickave += t0 - tmin

    #thick_grid.plot(show=False, fname='Thick-Mars_%s-1.png' % model_name[model])
    thick_grid.plot(show=False, fname='Thick-Mars-1.png')

    #moho.plot_spectrum(show=False, fname='Moho-spectrum-Mars-1.png')
    #return r0_model - radius[crust_index], thickave, tmin, tmax, rho_mantle
    with open('thickness.txt', 'a') as fid:
        fid.write('%d, %f, %f, %f, %f, %f\n' %
                  (ifile, crust_av_in, crust_av_out,
                   tmin, tmax, rho))


# ==== EXECUTE SCRIPT ====
if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        calc_crust(fnam=filename, d_lith=float(sys.argv[2])*1e3)
    else:
        files = []
        for ifile in np.arange(1, 400):
            files.append('MVD/model4mvdSAN_AK%d00.dat' % ifile)

        with Pool(4) as p:
            p.map(calc_crust, files)
        # with open('thickness.txt', 'w') as fid:

        #     for ifile in np.arange(1, 400):
        #         fnam = 'MVD/model4mvdSAN_AK%d00.dat' % ifile
        #         try:
        #             crust_av_in, crust_av_out, tmin, tmax, rho \
        #                     = calc_crust(fnam=fnam, d_lith=200e3)
        #         except(ValueError):
        #             print('Model %d did not converge' % ifile)
        #         else:
        #             fid.write('%d, %f, %f, %f, %f, %f\n' %
        #                       (ifile, crust_av_in, crust_av_out,
        #                        tmin, tmax, rho))
