#!/usr/bin/env python

from __future__ import division

import numpy as np
import corner #https://github.com/dfm/corner.py
import argparse
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description="Corner triangle based LIGO CBC-PE comparison script.")

parser.add_argument('-p', '--posterior', type=str, nargs='+', help='list of posterior files (ASCII table with one-line header).')

parser.add_argument('-n', '--names', type=str, nargs='*', help='list of header names (parameter) to plot.')

parser.add_argument('-t', '--truths', type=float, nargs='*', help='list of (true) values to plot, in the same order as the header names.',
                    default=None)

parser.add_argument('-c', '--color', type=str, nargs='*', help='list of colors to use.',
                    default = ['blue','red', 'green','yellow','purple'])

parser.add_argument('-o', '--output', type=str, nargs='?', help='output file.',
                    default="triangle_plot.png")

args = parser.parse_args()

non_params = ["signalmodelflag", "psdscaleflag", "h1_optimal_snr", "l1_optimal_snr", "v1_optimal_snr", "deltaf", "sky_frame", "lal_pnorder", "lal_amporder", "lal_approximant", "alpha", "t0", "logpost", "cycle", "logprior", "logl", "loglh1", "logll1", "loglv1", "timestamp", "snrh1", "snrl1", "snrv1", "snr", "time_mean", "time_maxl", "flow", "f_ref", "chain", "v1_end_time", "v1h1_delay", "l1v1_delay", "l1h1_delay", "l1_end_time", "v1l1_delay", "costheta_jn", "snr", "post", "cycle", "h1v1_delay", "logprior", "snrv1", "h1l1_delay", "prior", "h1_end_time", "snrh1", "costilt1", "costilt2", "f_lower", "cosiota", "costhetas", "thetas", "beta", "cosbeta", "m1", "m2", "iota", "logw", "deltaloglh1", "deltalogll1", "deltaloglv1", "mtotal", "spcal_active", "deltalogl", "spcal_npts"]

#===============================================================================
# Function used to generate plot labels.
# Taken from the bayespputils module in PyLAL
# (https://www.lsc-group.phys.uwm.edu/daswg/projects/pylal.html)
#===============================================================================
def plot_label(param):
  """
  A lookup table for plot labels.
  """
  m1_names = ['mass1', 'm1']
  m2_names = ['mass2', 'm2']
  mc_names = ['mc','mchirp','chirpmass']
  eta_names = ['eta','massratio','sym_massratio']
  q_names = ['q','asym_massratio']
  iota_names = ['iota','incl','inclination']
  dist_names = ['dist','distance']
  ra_names = ['rightascension','ra']
  dec_names = ['declination','dec']
  phase_names = ['phi_orb', 'phi', 'phase', 'phi0']

  labels={
      'm1':r'$m_1\,(\mathrm{M}_\odot)$',
      'm2':r'$m_2\,(\mathrm{M}_\odot)$',
      'mc':r'$\mathcal{M}\,(\mathrm{M}_\odot)$',
      'eta':r'$\eta$',
      'q':r'$q$',
      'mtotal':r'$M_\mathrm{total}\,(\mathrm{M}_\odot)$',
      'm1_source':r'$m_{1}^\mathrm{source}\,(\mathrm{M}_\odot)$',
      'm2_source':r'$m_{2}^\mathrm{source}\,(\mathrm{M}_\odot)$',
      'mtotal_source':r'$M_\mathrm{total}^\mathrm{source}\,(\mathrm{M}_\odot)$',
      'mc_source':r'$\mathcal{M}^\mathrm{source}\,(\mathrm{M}_\odot)$',
      'redshift':r'$z$',
      'mf':r'$M_\mathrm{final}\,(\mathrm{M}_\odot)$',
      'mf_source':r'$M_\mathrm{final}^\mathrm{source}\,(\mathrm{M}_\odot)$',
      'af':r'$a_\mathrm{final}$',
      'spin1':r'$S_1$',
      'spin2':r'$S_2$',
      'a1':r'$a_1$',
      'a2':r'$a_2$',
      'a1z':r'$a_{1z}$',
      'a2z':r'$a_{2z}$',
      'theta1':r'$\theta_1\,(\mathrm{rad})$',
      'theta2':r'$\theta_2\,(\mathrm{rad})$',
      'phi1':r'$\phi_1\,(\mathrm{rad})$',
      'phi2':r'$\phi_2\,(\mathrm{rad})$',
      'chi_eff':r'$\chi_\mathrm{eff}$',
      'chi_tot':r'$\chi_\mathrm{total}$',
      'chi_p':r'$\chi_\mathrm{P}$',
      'tilt1':r'$t_1\,(\mathrm{rad})$',
      'tilt2':r'$t_2\,(\mathrm{rad})$',
      'costilt1':r'$\mathrm{cos}(t_1)$',
      'costilt2':r'$\mathrm{cos}(t_2)$',
      'iota':r'$\iota\,(\mathrm{rad})$',
      'cosiota':r'$\mathrm{cos}(\iota)$',
      'time':r'$t_\mathrm{c}\,(\mathrm{s})$',
      'time_mean':r'$<t>\,(\mathrm{s})$',
      'dist':r'$d_\mathrm{L}\,(\mathrm{Mpc})$',
      'ra':r'$\alpha$',
      'dec':r'$\delta$',
      'phase':r'$\phi\,(\mathrm{rad})$',
      'psi':r'$\psi\,(\mathrm{rad})$',
      'theta_jn':r'$\theta_\mathrm{JN}\,(\mathrm{rad})$',
      'costheta_jn':r'$\mathrm{cos}(\theta_\mathrm{JN})$',
      'beta':r'$\beta\,(\mathrm{rad})$',
      'cosbeta':r'$\mathrm{cos}(\beta)$',
      'phi_jl':r'$\phi_\mathrm{JL}\,(\mathrm{rad})$',
      'phi12':r'$\phi_\mathrm{12}\,(\mathrm{rad})$',
      'logl':r'$\mathrm{log}(\mathcal{L})$',
      'h1_end_time':r'$t_\mathrm{H}$',
      'l1_end_time':r'$t_\mathrm{L}$',
      'v1_end_time':r'$t_\mathrm{V}$',
      'h1l1_delay':r'$\Delta t_\mathrm{HL}$',
      'h1v1_delay':r'$\Delta t_\mathrm{HV}$',
      'l1v1_delay':r'$\Delta t_\mathrm{LV}$',
      'lambdat' : r'$\tilde{\Lambda}$',
      'dlambdat': r'$\delta \tilde{\Lambda}$',
      'lambda1' : r'$\lambda_1$',
      'lambda2': r'$\lambda_2$',
      'lam_tilde' : r'$\tilde{\Lambda}$',
      'dlam_tilde': r'$\delta \tilde{\Lambda}$',
      'calamp_h1' : r'$\delta A_{H1}$',
      'calamp_l1' : r'$\delta A_{L1}$',
      'calpha_h1' : r'$\delta \phi_{H1}$',
      'calpha_l1' : r'$\delta \phi_{L1}$',
      'polar_eccentricity':r'$\epsilon_{polar}$',
      'polar_angle':r'$\alpha_{polar}$',
      'alpha':r'$\alpha_{polar}$'
    }

  # Handle cases where multiple names have been used
  if param in m1_names:
    param = 'm1'
  elif param in m2_names:
    param = 'm2'
  elif param in mc_names:
    param = 'mc'
  elif param in eta_names:
    param = 'eta'
  elif param in q_names:
    param = 'q'
  elif param in iota_names:
    param = 'iota'
  elif param in dist_names:
    param = 'dist'
  elif param in ra_names:
    param = 'ra'
  elif param in dec_names:
    param = 'dec'
  elif param in phase_names:
    param = 'phase'

  try:
    label = labels[param]
  except KeyError:
    # Use simple string if no formated label is available for param
    label = param

  return label

pos=[]
for posfile in args.posterior:
    print "Reading file "+posfile
    pos.append(np.genfromtxt(posfile, names=True))

if args.names:
    params = args.names
else:
    params = [col for col in pos[0].dtype.names if col not in non_params]
    for p in pos[1:]:
        newparams = [col for col in p.dtype.names if col not in non_params]
        params=list(set(params).intersection(newparams))

print "Plotting "+str(params)
legends=[]
fig=None

for p,c,l in zip(pos,args.color,args.posterior):
    X = p[params].view(float).reshape(-1, len(params))
    fig = corner.corner(X, bins=30, labels=[plot_label(param) for param in params],plot_datapoints=False,color=c,fig=fig,quantiles=[0.05,0.95],hist_kwargs={"normed":True},normed=True,truths=args.truths)#,extents=extents)
    label=os.path.basename(l).replace('posterior_samples','').strip('_').replace('.dat','')
    legends.append(lines.Line2D([0,0], [1,0], color=c, label=label))

plt.legend(handles=legends,loc='lower right', bbox_to_anchor=(1, len(params)-0.5))

fig.savefig(args.output)
