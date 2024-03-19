

import os
import sys
import numpy as np
#import jax
#from jax import numpy as jnp
#from jax import jit
#import optax
import itertools
#from tqdm.auto import trange
from astropy.table import Table, vstack
from astropy.io import fits
from pkg_resources import resource_filename
from matplotlib import pyplot as plt
from pypeit.spectrographs.util import load_spectrograph
from pypeit.core.wavecal import templates
from pypeit.core.wavecal import wvutils
from pypeit.core import arc
from pypeit.core.fitting import robust_fit
from pypeit.core import coadd
from pypeit.core import fitting
from pypeit.core.wavecal import autoid, waveio, wv_fitting
from pypeit.core.wavecal.wvutils import  get_xcorr_arc, xcorr_shift
from pypeit import utils
from pypeit import msgs
from pypeit import wavecalib
from astropy import table
from scipy import interpolate
from IPython import embed
from astropy import constants as const

c_kms = const.c.to('km/s').value

#@jit
#def poly_transform_y2(theta, x1, wave2, y2, wave2_min, wave2_max):
#
#    wave1 = wave2_min + (wave2_max - wave2_min)*jnp.polyval(theta, x1)
#    y2_corr = jnp.interp(wave1, wave2, y2)
#    return wave1, y2_corr

#@jit
#def zerolag_poly_corr(theta, x1, y1, wave2, y2,  wave2_min, wave2_max):
#
#     """
#     Utility function which is run by the differential evolution
#     optimizer in scipy. These is the fucntion we optimize.  It is the
#     zero lag cross-correlation coefficient of spectrum with a shift and
#     stretch applied.
#
#     Parameters
#     ----------
#     theta (float `numpy.ndarray`_):
#         Function parameters to optmize over. theta[0] = shift, theta[1] = stretch
#     y1 (float `numpy.ndarray`_):  shape = (nspec,)
#         First spectrum which acts as the refrence
#     y2 (float `numpy.ndarray`_):  shape = (nspec,)
#         Second spectrum which will be transformed by a shift and stretch to match y1
#
#     Returns
#     -------
#     corr_norm: float
#         Negative of the zero lag cross-correlation coefficient (since we
#         are miniziming with scipy.optimize). scipy.optimize will thus
#         determine the shift,stretch that maximize the cross-correlation.
#
#     """
#
#
#     wave1, y2_corr = poly_transform_y2(theta, x1, wave2, y2, wave2_min, wave2_max)
#     # Zero lag correlation
#     corr_zero = jnp.sum(y1*y2_corr)
#     corr_denom = jnp.sqrt(jnp.sum(y1*y1)*jnp.sum(y2_corr*y2_corr))
#     corr_norm = corr_zero/corr_denom
#     return -corr_norm
#
#
# def xcorr_poly(wave1_in, inspec1, wave2, inspec2, smooth=1.0, percent_ceil=80.0, use_raw_arc=False,
#                sigdetect = 10.0, fwhm = 4.0, debug=False, seed = 42):
#
#     """ Determine the shift and stretch of inspec2 relative to inspec1.  This routine computes an initial
#     guess for the shift via maximimizing the cross-correlation. It then performs a two parameter search for the shift and stretch
#     by optimizing the zero lag cross-correlation between the inspec1 and the transformed inspec2 (shifted and stretched via
#     wvutils.shift_and_stretch()) in a narrow window about the initial estimated shift. The convention for the shift is that
#     positive shift means inspec2 is shifted to the right (higher pixel values) relative to inspec1. The convention for the stretch is
#     that it is float near unity that increases the size of the inspec2 relative to the original size (which is the size of inspec1)
#
#     Parameters
#     ----------
#     inspec1 : ndarray
#         Reference spectrum
#     inspec2 : ndarray
#         Spectrum for which the shift and stretch are computed such that it will match inspec1
#     cc_thresh: float, default = -1.0
#         A number in the range [-1.0,1.0] which is the threshold on the
#         initial cross-correlation coefficient for the shift/stretch.  If
#         the value of the initial cross-correlation is < cc_thresh the
#         code will just exit and return this value and the best shift.
#         This is desirable behavior since the shif/stretch optimization
#         is slow and this allows one to test how correlated the spectra
#         are before attempting it, since there is little value in that
#         expensive computation for spectra with little overlap. The
#         default cc_thresh =-1.0 means shift/stretch is always attempted
#         since the cross correlation coeficcient cannot be less than
#         -1.0.
#     smooth: float, default
#         Gaussian smoothing in pixels applied to both spectra for the computations. Default is 5.0
#     percent_ceil: float, default=90.0
#         Apply a ceiling to the input spectra at the percent_ceil
#         percentile level of the distribution of peak amplitudes.  This
#         prevents extremely strong lines from completely dominating the
#         cross-correlation, which can causes the cross-correlation to
#         have spurious noise spikes that are not the real maximum.
#     use_raw_arc: bool, default = False
#         If this parameter is True the raw arc will be used rather than the continuum subtracted arc
#     shift_mnmx: tuple of floats, default = (-0.05,0.05)
#         Range to search for the shift in the optimization about the
#         initial cross-correlation based estimate of the shift.  The
#         optimization will search the window (shift_cc +
#         nspec*shift_mnmx[0],shift_cc + nspec*shift_mnmx[1]) where nspec
#         is the number of pixels in the spectrum
#     stretch_mnmx: tuple of floats, default = (0.97,1.03)
#         Range to search for the stretch in the optimization. The code
#         may not work well if this range is significantly expanded
#         because the linear approximation used to transform the arc
#         starts to break down.
#     seed: int or np.random.RandomState, optional, default = None
#         Seed for scipy.optimize.differential_evolution optimizer. If not
#         specified, the calculation will not be repeatable
#     toler (float):
#         Tolerance for differential evolution optimizaiton.
#     debug = False
#        Show plots to the screen useful for debugging.
#
#     Returns
#     -------
#     success: int
#         A flag indicating the exist status.  Values are:
#
#           - success = 1, shift and stretch performed via sucessful
#             optimization
#           - success = 0, shift and stretch optimization failed
#           - success = -1, initial x-correlation is below cc_thresh (see
#             above), so shift/stretch optimization was not attempted
#
#     shift: float
#         the optimal shift which was determined.  If cc_thresh is set,
#         and the initial cross-correlation is < cc_thresh,  then this
#         will be just the cross-correlation shift
#     stretch: float
#         the optimal stretch which was determined.  If cc_thresh is set,
#         and the initial cross-correlation is < cc_thresh,  then this
#         will be just be 1.0
#     cross_corr: float
#         the value of the cross-correlation coefficient at the optimal
#         shift and stretch. This is a number between zero and unity,
#         which unity indicating a perfect match between the two spectra.
#         If cc_thresh is set, and the initial cross-correlation is <
#         cc_thresh, this will be just the initial cross-correlation
#     shift_init:
#         The initial shift determined by maximizing the cross-correlation
#         coefficient without allowing for a stretch.  If cc_thresh is
#         set, and the initial cross-correlation is < cc_thresh, this will
#         be just the shift from the initial cross-correlation
#     cross_corr_init:
#         The maximum of the initial cross-correlation coefficient
#         determined without allowing for a stretch.  If cc_thresh is set,
#         and the initial cross-correlation is < cc_thresh, this will be
#         just the initial cross-correlation
#
#     """
#
#     nspec1 = inspec1.size
#     nspec2 = inspec2.size
#     x1 = jnp.arange(nspec1, dtype=float)/float(nspec1-1)
#     x2 = jnp.arange(nspec2, dtype=float)/float(nspec2-1)
#
#     wave2_min, wave2_max = wave2.min(), wave2.max()
#
#     y1 = jnp.fmax(jnp.array(smooth_ceil_cont(inspec1,smooth,percent_ceil=percent_ceil,use_raw_arc=use_raw_arc, sigdetect = sigdetect, fwhm = fwhm)), -10.0)
#     y2 = jnp.fmax(jnp.array(smooth_ceil_cont(inspec2,smooth,percent_ceil=percent_ceil,use_raw_arc=use_raw_arc, sigdetect = sigdetect, fwhm = fwhm)), -10.0)
#
#
#     #start_learning_rate = 5e-4
#     start_learning_rate = 1e-3
#     optimizer = optax.adam(start_learning_rate)
#
#
#     # Initialize parameters of the model + optimizer.
#     norder = 3
#
#     # Random central wavelength noise
#     wave1_min, wave1_max = wave1_in.min(), wave1_in.max()
#     key = jax.random.PRNGKey(423)
#     dwave1 = (wave1_max- wave1_min)/nspec1
#     dwave = 10.0*dwave1*jax.random.normal(key, shape=(2,))
#     wave1_min_guess, wave1_max_guess = wave1_min + dwave[0], wave1_max + dwave[1]
#     wave1_guess = wave1_min_guess + x1*(wave1_max_guess - wave1_min_guess)
#     params = jnp.polyfit(x1, (wave1_guess - wave2_min)/(wave2_max -wave2_min), norder)
#     params_true = jnp.polyfit(x1, (wave1_in - wave2_min)/(wave2_max -wave2_min), norder)
#     corr_true = -zerolag_poly_corr(params_true, x1, y1, wave2, y2, wave2_min, wave2_max)
#     #params = jnp.polyfit(x2, wave2, norder)
#     #params = jax.random.uniform(key, minval=0.0, maxval=0.1, shape=(norder+1,))
#
#     # Vanilla optimizaton
#     #opt_state = optimizer.init(params)
#
#     nsteps =2000
#     # Exponential decay of the learning rate.
#     scheduler = optax.exponential_decay(init_value=start_learning_rate,transition_steps=nsteps, decay_rate=0.99)
#
#     # Combining gradient transforms using `optax.chain`.
#     gradient_transform = optax.chain(
#         optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
#         optax.scale_by_adam(),  # Use the updates from adam.
#         optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
#         # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
#         optax.scale(-1.0)
#     )
#     opt_state = gradient_transform.init(params)
#
#     iterator = trange(nsteps, leave=True)
#     losses = np.zeros(nsteps)
#
#     best_loss = np.inf  # Models are only saved if they reduce the loss
#     params_out = params.copy()
#     for i in iterator:
#         #losses[i] = zerolag_poly_corr(params, x1, y1, wave2, y2, wave2_min, wave2_max)
#         losses[i], grads = jax.value_and_grad(zerolag_poly_corr, argnums=0, has_aux=False)(params, x1, y1, wave2, y2, wave2_min, wave2_max)
#         if losses[i] < best_loss:
#             params_out = params.copy()
#             best_loss = losses[i]
#         #grads = jax.grad(zerolag_poly_corr)(params, x1, y1, wave2, y2, wave2_min, wave2_max)
#         iterator.set_description(
#             'Loss: {:f}'.format(losses[i]) + ", Grads: {:5.3f},{:5.3f},{:5.3f},{:5.3f}, Best Loss: {:5.3f}".format(*grads, best_loss) + ", iter no: " + str(i))
#         updates, opt_state = gradient_transform.update(grads, opt_state)
#         #updates, opt_state = optimizer.update(grads, opt_state)
#         params = optax.apply_updates(params, updates)
#
#     if debug:
#         wave1, y2_trans = poly_transform_y2(params_out, x1, wave2, y2, wave2_min, wave2_max)
#         plt.figure(figsize=(14, 6))
#         plt.plot(wave1, y1, 'k-', drawstyle='steps', label='inspec1, input spectrum')
#         plt.plot(wave1, y2_trans, 'r-', drawstyle='steps', label='inspec2, polynomial transformed archive')
#         plt.title('corr = {:5.3f}, corr_true={:5.3f}'.format(-best_loss, corr_true))
#         plt.legend()
#         plt.show()
#


def get_variable_dlam_wavegrid(lam_min, lam_max, wave_grid_fit, dwave_fit):

    lam_out = [lam_min]
    while lam_out[-1] < lam_max:
        lam_next = lam_out[-1] + np.interp(lam_out[-1], wave_grid_fit, dwave_fit)
        lam_out.append(lam_next)

    return np.array(lam_out, dtype=float)


def empty_design_table(nrows, norders, n_final=4):
    """
    Construct an empty arxiv table.

    Args:
        nrows (:obj:`int`):
            Number of rows in the table.
        norders (:obj:`int`):
            Number of orders in the table.
        n_final (:obj:`int`, optional):
            Number of final coefficients in the wavelength solution.

    Returns:
        `astropy.table.Table`_: Instance of the empty arxiv table.
    """
    return Table([np.zeros(nrows, dtype="<U30"),
                        np.zeros(nrows, dtype=int),
                        np.zeros(nrows, dtype=int),
                        np.zeros(nrows, dtype=int),
                        np.zeros(nrows, dtype=int),
                        np.zeros(nrows, dtype=int),
                        np.zeros(nrows, dtype=int),
                        np.zeros(nrows, dtype="<U5"),
                        np.zeros(nrows, dtype=float),
                        np.zeros(nrows, dtype=float),
                        np.zeros(nrows, dtype=int),
                        np.zeros((nrows, norders), dtype=int),
                        np.zeros((nrows, norders), dtype=bool),
                        np.zeros((nrows, norders), dtype=bool),
                        np.full((nrows, norders), -1e10),
                        np.full((nrows, norders), -1e10),
                        np.zeros((nrows, norders), dtype="<U5"),
                        np.zeros((nrows, norders), dtype=int),
                        np.zeros((nrows, norders), dtype=int),
                        np.zeros((nrows, norders)),
                        np.zeros((nrows, norders, n_final + 1)),],
                        names=('filename', 'nsolns', 'nsolns_good', 'bluest_order', 'bluest_good_order',
                               'reddest_order', 'reddest_good_order', 'xdisp_file',
                               'ech_angle_file', 'xd_angle_file', 'det_file',
                               'order', 'populated', 'populated_and_good', 'ech_angle', 'xd_angle',
                               'xdisp', 'det', 'binspec','lambda_cen', 'coeff'))


def ingest_xidl_archive(outfile, n_final=4, func='legendre'):
    """
    Read the XIDL archive file and write a pypeit format archive file.

    Args:
    -----
    outfile: str
        Name of the output file
    """

    # Read template file
    #templ_table_file = os.path.join(resource_filename('pypeit', 'data'), 'arc_lines', 'hires', 'hires_templ_xidl.dat')
    templ_table_file = '/Users/yaqi/Desktop/pypeit_HIRES_orig/hires_orig_templ_xidl.dat'
    tbl = Table.read(templ_table_file, format='ascii')
    nrows = len(tbl)
    msgs.info(f'nrows = {nrows}')

    order_min = tbl['IOrder'].min()
    order_max = tbl['EOrder'].max() #This is 1 shy of 118
    msgs.info(f'order_min = {order_min}')
    msgs.info(f'order_max = {order_max}')

    order_vec = np.arange(order_min, order_max + 1, 1)
    norders = order_vec.size
    msgs.info(f'order_vec = {order_vec}')
    msgs.info(f'norders = {norders}')


    # xmin, xmax for wavelength vs pixel fits
    fmin, fmax = 0.0, 1.0
    params=Table([[order_min],[order_max],[norders],[n_final],[func],[fmin],[fmax]]
                ,names=('order_min','order_max','norders','n_final','func','xmin','xmax'))

    table_xidl = empty_design_table(nrows, norders, n_final=n_final)
    msgs.info('table_xidl constructed!')

    msgs.info(f'###### start the loop of rows with nrows = {nrows}')
    for irow in np.arange(nrows):
        #this_order_vec_raw, this_wave, this_arc = templates.xidl_esihires(os.path.join(os.getenv('HIRES_CALIBS'), 'ARCS', tbl[irow]['Name']), specbin=tbl[irow]['Rbin'])
        this_order_vec_raw, this_wave, this_arc = templates.xidl_esihires(os.path.join('/Users/yaqi/Desktop/pypeit_HIRES_orig/MAKEE', tbl[irow]['Name']), specbin=tbl[irow]['Rbin'])
        msgs.info(f'row {irow}, orders\t'+str(this_order_vec_raw))
        if irow == 0:
            nspec = this_wave.shape[1]
            xnspecmin1 = float(nspec - 1)
            xvec = np.arange(nspec)/xnspecmin1
            params['nspec'] = nspec
            table_xidl['wave'] = np.zeros((nrows, norders, nspec))
            table_xidl['arcspec'] = np.zeros((nrows, norders, nspec))
        else:
            assert this_wave.shape[1] == nspec

        # Good is what is labeled as good in the Table, we also store everything in the file
        igood = (this_order_vec_raw >= tbl[irow]['IOrder']) & (this_order_vec_raw <= tbl[irow]['EOrder'])
        nsolns_good = np.sum(igood)
        this_order_vec = this_order_vec_raw[igood]
        indx = this_order_vec_raw - order_min
        indx_good = this_order_vec - order_min
        nsolns = this_order_vec_raw.size
        # Information for the file is stored for convenience, although this is redundant with the arrays below
        table_xidl['filename'][irow] = tbl[irow]['Name']
        table_xidl['nsolns'][irow] = nsolns
        table_xidl['nsolns_good'][irow] = nsolns_good
        table_xidl['bluest_order'][irow] = this_order_vec_raw[-1]
        table_xidl['bluest_good_order'][irow] = this_order_vec[-1]
        table_xidl['reddest_order'][irow] = this_order_vec_raw[0]
        table_xidl['reddest_good_order'][irow] = this_order_vec[0]
        table_xidl['xdisp_file'][irow] = tbl[irow]['XDISP']
        table_xidl['ech_angle_file'][irow] = tbl[irow]['ECH']
        table_xidl['xd_angle_file'][irow] = tbl[irow]['XDAng']
        table_xidl['det_file'][irow] = tbl[irow]['Chip']
        # Arrays (nfile, norders)
        table_xidl['order'][irow, indx] = this_order_vec_raw
        table_xidl['populated'][irow, indx] = True
        table_xidl['populated_and_good'][irow, indx_good] = True
        table_xidl['ech_angle'][irow, indx] = tbl[irow]['ECH']
        table_xidl['xd_angle'][irow, indx] = tbl[irow]['XDAng']
        table_xidl['xdisp'][irow, indx] = tbl[irow]['XDISP']
        table_xidl['det'][irow, indx] = tbl[irow]['Chip']
        table_xidl['binspec'][irow, indx] = tbl[irow]['Rbin']
        table_xidl['lambda_cen'][irow, indx] = np.median(this_wave, axis=1)
        table_xidl['wave'][irow, indx, :] = this_wave
        table_xidl['arcspec'][irow, indx, :] = this_arc
        # Fit the wavelengths
        this_coeff_array = np.zeros((nsolns_good, n_final + 1))
        for ii, iwave in enumerate(this_wave[igood, :]):
            pypeitFit = fitting.robust_fit(xvec, iwave, n_final, function=func, maxiter=10,
                                           lower=1e10, upper=1e10, maxrej=0, sticky=True,
                                           minx=fmin, maxx=fmax, weights=None)
            this_coeff_array[ii, :] = pypeitFit.fitc
        table_xidl['coeff'][irow, indx_good, :] = this_coeff_array

    # Write out to multi-extension fits
    print(f'Writing HIRES xidl wv_calib archive to file: {outfile}')
    hdu_param = fits.BinTableHDU(params.as_array())
    hdu_table = fits.BinTableHDU(table_xidl.as_array())

    hdulist = fits.HDUList()
    hdulist.append(hdu_param)
    hdulist.append(hdu_table)
    hdulist.writeto(outfile, overwrite=True)


    return


def append_pypeit_archive(outfile, xidl_arxiv_file):
    """
    Append the pypeit format archived templates to the XIDL format archive table
    Args:
        outfile (str):
            Name of the output file
        xidl_arxiv_file (str):
            File containing the XIDL templates archive

    """

    # Load the XIDL format archive file
    xidl_params = Table.read(xidl_arxiv_file, hdu=1)[0]
    xidl_tab = Table.read(xidl_arxiv_file, hdu=2)

    # load the list of pypeit templates
    ptempl_table_file = os.path.join(resource_filename('pypeit', 'data'), 'arc_lines',
        'hires', 'hires_templ_pypeit.dat')
    ptbl = Table.read(ptempl_table_file, format='ascii')
    p_nrows = len(ptbl)

    # recomputing the final order min, max, order_vec, norders
    order_min = min(ptbl['IOrder'].min(), xidl_params['order_min'])
    order_max = max(ptbl['EOrder'].max(), xidl_params['order_max'])

    order_vec = np.arange(order_min, order_max + 1, 1)
    norders = order_vec.size

    # updata parms
    xidl_params['order_min'] = order_min
    xidl_params['order_max'] = order_max
    xidl_params['norders'] = norders

    # create new final table
    tot_nrows = len(xidl_tab) + p_nrows
    final_table = empty_design_table(tot_nrows, norders, n_final=xidl_params['n_final'])
    final_table['wave'] = np.zeros((tot_nrows, norders, xidl_params['nspec']))
    final_table['arcspec'] = np.zeros((tot_nrows, norders, xidl_params['nspec']))

    # copy the xidl table to the final table (we need to do this, instead of just merging 2 tables,
    # because the number of the orders increased with the pypeit templates)
    for xirow in np.arange(len(xidl_tab)):
        # get the right index
        this_xorder_vec_raw = xidl_tab['order'][xirow][xidl_tab['populated'][xirow]]
        this_xorder_vec = xidl_tab['order'][xirow][xidl_tab['populated_and_good'][xirow]]
        this_xindx = this_xorder_vec_raw - order_min
        this_xindx_good = this_xorder_vec - order_min
        # fill the table
        final_table['filename'][xirow] = xidl_tab['filename'][xirow]
        final_table['nsolns'][xirow] = xidl_tab['nsolns'][xirow]
        final_table['nsolns_good'][xirow] = xidl_tab['nsolns_good'][xirow]
        final_table['bluest_order'][xirow] = xidl_tab['bluest_order'][xirow]
        final_table['bluest_good_order'][xirow] = xidl_tab['bluest_good_order'][xirow]
        final_table['reddest_order'][xirow] = xidl_tab['reddest_order'][xirow]
        final_table['reddest_good_order'][xirow] = xidl_tab['reddest_good_order'][xirow]
        final_table['xdisp_file'][xirow] = xidl_tab['xdisp_file'][xirow]
        final_table['ech_angle_file'][xirow] = xidl_tab['ech_angle_file'][xirow]
        final_table['xd_angle_file'][xirow] = xidl_tab['xd_angle_file'][xirow]
        final_table['det_file'][xirow] = xidl_tab['det_file'][xirow]
        # Arrays (nfile, norders)
        final_table['order'][xirow, this_xindx] = this_xorder_vec_raw
        final_table['populated'][xirow, this_xindx] = True
        final_table['populated_and_good'][xirow, this_xindx_good] = True
        final_table['ech_angle'][xirow, this_xindx] = xidl_tab['ech_angle'][xirow][xidl_tab['populated'][xirow]]
        final_table['xd_angle'][xirow, this_xindx] = xidl_tab['xd_angle'][xirow][xidl_tab['populated'][xirow]]
        final_table['xdisp'][xirow, this_xindx] = xidl_tab['xdisp'][xirow][xidl_tab['populated'][xirow]]
        final_table['det'][xirow, this_xindx] = xidl_tab['det'][xirow][xidl_tab['populated'][xirow]]
        final_table['binspec'][xirow, this_xindx] = xidl_tab['binspec'][xirow][xidl_tab['populated'][xirow]]
        final_table['lambda_cen'][xirow, this_xindx] = xidl_tab['lambda_cen'][xirow][xidl_tab['populated'][xirow]]
        final_table['wave'][xirow, this_xindx, :] = xidl_tab['wave'][xirow][xidl_tab['populated'][xirow]]
        final_table['arcspec'][xirow, this_xindx, :] = xidl_tab['arcspec'][xirow][xidl_tab['populated'][xirow]]
        final_table['coeff'][xirow, this_xindx_good, :] = xidl_tab['coeff'][xirow][xidl_tab['populated_and_good'][xirow]]

    # Now deal with the pypeit format
    # load the pypeit templates (they are WaveCalib files)
    msgs.info(f"start loop of p_nrows = {p_nrows}")
    for irow in np.arange(p_nrows):
        ifinal_row = irow + len(xidl_tab)
        #templ_file = os.path.join(os.getenv('PYPEIT_DEV'), 'dev_algorithms', 'hires_wvcalib', 'templates_from_pypeit', ptbl[irow]['Name'])
        templ_file = os.path.join('/Users/yaqi/Desktop/', 'pypeit_HIRES_orig', 'templates_from_pypeit', ptbl[irow]['Name'])
        waveCalib = wavecalib.WaveCalib.from_file(templ_file, chk_version=False)
        # this is the order vector available for this wavecalib file
        this_order_vec_raw = np.arange(ptbl[irow]['IOrder'], ptbl[irow]['EOrder'] + 1, 1)
        # select the orders that we want to use
        if ptbl[irow]['Spatids'] == 'all':
            igood = np.ones(waveCalib.spat_ids.size, dtype=bool)
            if this_order_vec_raw.size != waveCalib.spat_ids.size:
                msgs.error('the number of order determined using IOrder and EOrder does not match the number of '
                           'orders in the WaveCalib file')
        elif isinstance(ptbl[irow]['Spatids'], (list, tuple, np.integer)):
            spat_ids = np.atleast_1d(ptbl[irow]['Spatids'])
            igood = np.isin(waveCalib.spat_ids, spat_ids)
        else:
            msgs.error('Unrecognized format for Spatids')
        this_order_vec = this_order_vec_raw[igood]
        this_arc = np.array([arc.resize_spec(ww, xidl_params['nspec']) for ww in waveCalib.arc_spectra.T])
        this_wave = np.array([arc.resize_spec(wvfit.wave_soln, xidl_params['nspec']) for wvfit in waveCalib.wv_fits])
        nsolns = this_order_vec_raw.size
        nsolns_good = np.sum(igood)
        indx = this_order_vec_raw - order_min
        indx_good = this_order_vec - order_min
        # Information for the file is stored for convenience, although this is redundant with the arrays below
        final_table['filename'][ifinal_row] = ptbl[irow]['Name']
        final_table['nsolns'][ifinal_row] = nsolns
        final_table['nsolns_good'][ifinal_row] = nsolns_good
        final_table['bluest_order'][ifinal_row] = this_order_vec_raw[-1]
        final_table['bluest_good_order'][ifinal_row] = this_order_vec[-1]
        final_table['reddest_order'][ifinal_row] = this_order_vec_raw[0]
        final_table['reddest_good_order'][ifinal_row] = this_order_vec[0]
        final_table['xdisp_file'][ifinal_row] = ptbl[irow]['XDISP']
        final_table['ech_angle_file'][ifinal_row] = ptbl[irow]['ECH']
        final_table['xd_angle_file'][ifinal_row] = ptbl[irow]['XDAng']
        final_table['det_file'][ifinal_row] = ptbl[irow]['Chip']
        # Arrays (nfile, norders)
        final_table['order'][ifinal_row, indx] = this_order_vec_raw
        final_table['populated'][ifinal_row, indx] = True
        final_table['populated_and_good'][ifinal_row, indx_good] = True
        final_table['ech_angle'][ifinal_row, indx] = ptbl[irow]['ECH']
        final_table['xd_angle'][ifinal_row, indx] = ptbl[irow]['XDAng']
        final_table['xdisp'][ifinal_row, indx] = ptbl[irow]['XDISP']
        final_table['det'][ifinal_row, indx] = ptbl[irow]['Chip']
        final_table['binspec'][ifinal_row, indx] = ptbl[irow]['Rbin']
        final_table['lambda_cen'][ifinal_row, indx] = np.median(this_wave, axis=1)
        final_table['wave'][ifinal_row, indx, :] = this_wave
        final_table['arcspec'][ifinal_row, indx, :] = this_arc
        # Fit the wavelengths
        xnspecmin1 = float(xidl_params['nspec'] - 1)
        xvec = np.arange(xidl_params['nspec']) / xnspecmin1
        this_coeff_array = np.zeros((nsolns_good, xidl_params['n_final'] + 1))
        for ii, iwave in enumerate(this_wave[igood, :]):
            pypeitFit = fitting.robust_fit(xvec, iwave, xidl_params['n_final'], function=xidl_params['func'], maxiter=10,
                                           lower=1e10, upper=1e10, maxrej=0, sticky=True,
                                           minx=xidl_params['xmin'], maxx=xidl_params['xmax'], weights=None)
            this_coeff_array[ii, :] = pypeitFit.fitc
        final_table['coeff'][ifinal_row, indx_good, :] = this_coeff_array

    # Write out to multi-extension fits
    print(f'Writing HIRES xidl+pypeit wv_calib archive to file: {outfile}')
    hdu_param = fits.BinTableHDU(xidl_params.table.as_array())
    hdu_table = fits.BinTableHDU(final_table.as_array())

    hdulist = fits.HDUList()
    hdulist.append(hdu_param)
    hdulist.append(hdu_table)
    hdulist.writeto(outfile, overwrite=True)


def fit_wvcalib_vs_angles(arxiv_file, outfile, func='legendre',
                         ech_nmax = 3, ech_coeff_fit_order_min=1, ech_coeff_fit_order_max=2,
                         xd_reddest_fit_polyorder=2, sigrej=3.0, maxrej=1, debug=False):
    """
    Fit the coefficients of the wavelength solution vs. the ECH angle. Also fit the bluest_order as a function of XD angle
    for each XDISP

    Args:
        arxiv_file (str):
             File containing the XIDL archive of the HIRES calibration
        outfile (str):
            File to write the output to.
        func (str):
            Function to fit the coefficients.
        ech_nmax (int):
            Polynomial coefficients from nmax to n_final + 1 will be fit with the lower order coeff_fit_order_min
        ech_coeff_fit_order_min (int):
            Polynomial order to fit the set of coefficients from nmax to n_final+1. These coefficients have
            weaker trends with ech_angle so we fit them with lower order.
        ech_coeff_fit_order_max (int):
            Polynomial order to fit the set of coefficients from 0 to nmax. These coefficients show stronger
            trends with ech_angle so we fit them with higher order.
        sigrej (float):
             Rejection threshold for the coefficient fits.
        maxrej (int):
             Maximum number of rejections to allow for each iteration of the coefficient fits rejection.
        debug (bool):
             If True, show plots illustring the fits

    Returns:
        None
    """


    arxiv_params = Table.read(arxiv_file, hdu=1)[0]
    arxiv = Table.read(arxiv_file, hdu=2)

    ech_angle_fit_params, ech_angle_fit_coeffs = fit_coeffs_vs_ech_angle(
        arxiv_params, arxiv, func=func, nmax = ech_nmax, coeff_fit_order_min=ech_coeff_fit_order_min,
        coeff_fit_order_max=ech_coeff_fit_order_max, sigrej=sigrej, maxrej=maxrej, debug=debug)

    xd_angle_fit_params, xd_angle_fit_coeffs = fit_reddest_vs_xd_angle(
        arxiv, polyorder=xd_reddest_fit_polyorder, func=func, sigrej=sigrej, maxrej=maxrej, debug=debug)

    fit_params = table.hstack((ech_angle_fit_params, xd_angle_fit_params))

    hdulist = fits.HDUList()
    hdulist.append(fits.BinTableHDU(fit_params.as_array()))  # hdu = 1
    hdulist.append(fits.ImageHDU(np.array(ech_angle_fit_coeffs)))  # hdu = 2
    hdulist.append(fits.ImageHDU(np.array(xd_angle_fit_coeffs)))  # hdu = 3
    hdulist.writeto(outfile, overwrite=True)


def fit_coeffs_vs_ech_angle(arxiv_params, arxiv, func='legendre', nmax = 3, coeff_fit_order_min=1, coeff_fit_order_max=2,
                            sigrej=3.0, maxrej=1, debug=False):
    """
    Fit the coefficients of the wavelength solution vs. the ECH angle. Called by fit_coeffs_vs_angles

    Args:
        arxiv_file (str):
             File containing the XIDL archive of the HIRES calibration
        outfile (str):
            File to write the output to.
        func (str):
            Function to fit the coefficients.
        nmax (int):
            Polynomial coefficients from nmax to n_final + 1 will be fit with the lower order coeff_fit_order_min
        coeff_fit_order_min (int):
            Polynomial order to fit the set of coefficients from nmax to n_final+1. These coefficients have
            weaker trends with ech_angle so we fit them with lower order.
        coeff_fit_order_max (int):
            Polynomial order to fit the set of coefficients from 0 to nmax. These coefficients show stronger
            trends with ech_angle so we fit them with higher order.
        sigrej (float):
             Rejection threshold for the coefficient fits.
        maxrej (int):
             Maximum number of rejections to allow for each iteration of the coefficient fits rejection.
        debug (bool):
             If True, show plots illustring the fits

    Returns:
        ech_angle_fit_params (astropy.table.Table):
            Table containing the fit parameters.
        ech_angle_fit_coeffs (numpy.ndarray):
            Array containing the fit coefficients.
    """

    order_min, order_max = arxiv_params['order_min'], arxiv_params['order_max']
    order_vec = np.arange(order_min, order_max + 1, 1)
    norders = arxiv_params['norders'] # Total number of orders in the arxiv
    n_final = arxiv_params['n_final'] # order of wavelength solution fits
    ech_angles = arxiv["ech_angle"][arxiv["populated_and_good"]]
    msgs.info(f"order_min = {order_min},\t order_max = {order_max}")
    msgs.info(f"order_vec = {order_vec}")
    msgs.info(f"norders = {norders}")
    msgs.info(f"n_final = {n_final}")
    msgs.info(f"ech_angle = {ech_angles}")

    # Determine the min,max params for the fits using all the echelle angles in the arxiv
    ech_min, ech_max = ech_angles.min(), ech_angles.max()
    ech_vec = ech_min + (ech_max - ech_min) * np.arange(100) / 99
    msgs.info(f"ech_min = {ech_min},\t ech_max = {ech_max}")
    msgs.info(f"ech_vec = {ech_vec}")

    # Assign orders for each coefficient that we are fitting
    if nmax > n_final + 1:
        msgs.error(f'nmax={nmax} cannot be greater than n_final+1={n_final+1}. Reduce nmax')
    # This vector holds the polynomial order used to fit each coefficient
    coeff_fit_order_vec = np.full(n_final+1, coeff_fit_order_min)
    msgs.info(f"coeff_fit_order_vec = {coeff_fit_order_vec}")
    # DP: the fits look better if we remove the following line
    # coeff_fit_order_vec[0:nmax] = coeff_fit_order_max
    ## TODO This needs to be modified to be lower order for cases where there are very few fits in the arxiv. Right now
    # we fit the first 0:nmax coeffs always with a coeff_fit_order_max orderp polynomial.

    ech_angle_fit_params=Table([
        [ech_min],[ech_max],[norders], [order_min], [order_max], [n_final],[coeff_fit_order_vec], [func],[arxiv_params['func']],
        [arxiv_params['xmin']], [arxiv_params['xmax']]],
        names=('ech_xmin','ech_xmax','norders','order_min', 'order_max', 'ech_n_final','ech_coeff_fit_order', 'ech_func',
               'wave_func',
               'wave_xmin', 'wave_xmax'))

    ech_angle_fit_coeffs = np.zeros((norders, n_final + 1, coeff_fit_order_max + 1))

    for iord, this_order in enumerate(order_vec):
        populated = arxiv['populated_and_good'][:, iord] # & (arxiv['xdisp'][:, iord] == xdisp)
        nsolns_this_order = np.sum(populated)
        msgs.info(f"iord = {iord}, this_order = {this_order}, nsolns_this_order = {nsolns_this_order}")
        if nsolns_this_order > 0:
            ech_angle_this_order = arxiv['ech_angle'][:, iord][populated]
            coeff_this_order = arxiv['coeff'][:, iord, :][populated, :]
            msgs.info(f"ech_angle_this_order = {ech_angle_this_order}, coeff_this_order = {coeff_this_order}")
            #xd_angle_this_order = arxiv['xd_angle'][:, iord][populated]
            #lambda_cen_this_order = arxiv['lambda_cen'][:, iord][populated]
            for ic in range(coeff_this_order.shape[1]):
                msgs.info(f"working for iord = {iord}, this_order = {this_order}, nsolns_this_order = {nsolns_this_order}, ic = {ic} =============================================")
                msgs.info(f"xarray = ech_angle_this_order = {ech_angle_this_order}")
                msgs.info(f"yarray = coeff_this_order = {coeff_this_order[:, ic]}")
                msgs.info(f"order = coeff_fit_order_vec[ic] = {coeff_fit_order_vec[ic]}")
                msgs.info(f"function = {func}")
                msgs.info(f"minx = {ech_min}")
                msgs.info(f"maxx = {ech_max}")
                msgs.info(f"lower = upper = {sigrej}")
                msgs.info(f"maxrej = {maxrej}")
                pypeitFit = fitting.robust_fit(ech_angle_this_order, coeff_this_order[:, ic], coeff_fit_order_vec[ic], function=func,
                                               minx=ech_min, maxx=ech_max, maxiter=25,
                                               lower=sigrej, upper=sigrej, maxrej=maxrej, sticky=True, use_mad=True,
                                               weights=None)
                ech_angle_fit_coeffs[iord, ic, 0:coeff_fit_order_vec[ic]+1] = pypeitFit.fitc
                if debug:
                    this_fit = fitting.evaluate_fit(pypeitFit.fitc, func, ech_vec, minx=ech_min, maxx=ech_max)
                    plt.plot(ech_vec, this_fit, color='blue', label='fit')
                    fit_gpm = pypeitFit.bool_gpm
                    plt.plot(ech_angle_this_order[fit_gpm], coeff_this_order[fit_gpm, ic], marker='o', markersize=7.0, mfc='black',
                             mec='black', fillstyle='full', linestyle='None', zorder=5, label='used by fit')
                    plt.plot(ech_angle_this_order[np.logical_not(fit_gpm)], coeff_this_order[np.logical_not(fit_gpm), ic],
                             marker='s', markersize=9.0, mfc='red', mec='red', fillstyle='full', linestyle='None',
                             zorder=7, label='rejected')
                    plt.legend()
                    plt.title(
                        f'order={this_order}, cc_ii={ic}, nkept={np.sum(fit_gpm)}, nrej={np.sum(np.logical_not(fit_gpm))}')
                    plt.xlabel('ech_angle')
                    plt.ylabel('coeff')
                    plt.ylim(this_fit.min() - 0.05 * np.abs(this_fit.min()),
                             this_fit.max() + 0.05 * np.abs(this_fit.max()))
                    plt.show()


    return ech_angle_fit_params, ech_angle_fit_coeffs


def fit_reddest_vs_xd_angle(arxiv, func='legendre', polyorder = 2, sigrej=3.0, maxrej=1, debug=False):

    # Fit the reddest order on detector 3. We use this as the reference since the bluest order seems
    # to have a more noisy trend
    xd_angles = arxiv['xd_angle_file']
    # Determine the min,max params for the fits using all the XD angles in the arxiv
    xd_min, xd_max = xd_angles.min(), xd_angles.max()
    xd_vec = xd_min + (xd_max - xd_min) * np.arange(100) / 99

    xd_angle_fit_params=Table([[xd_min],[xd_max], [['UV', 'RED', 'RED97']], [polyorder], [func]]
                ,names=('xd_xmin','xd_xmax','xdisp_vec', 'xd_polyorder', 'xd_func'))

    # First dimension is UV or RED, second dimension is the set of polynomial coefficients
    xd_angle_fit_coeffs = np.zeros((3, polyorder + 1))

    #for idisp, xdisp in enumerate(['UV', 'RED']):
    for idisp, xdisp in enumerate(['UV','RED','RED97']):


        msgs.info(f"xdisp = {xdisp}")
#        indx = (arxiv['det_file'] == 3) & (arxiv['xdisp_file'] == xdisp)
        indx = (arxiv['det_file'] == 1) & (arxiv['xdisp_file'] == xdisp)
        if len(np.where(indx)[0])==0:
            continue
        xd_angles_this_disp = xd_angles[indx]
        reddest_order_this_disp = arxiv['reddest_order'][indx].astype(float)
        msgs.info(f"{indx}")
        pypeitFit = fitting.robust_fit(xd_angles_this_disp, reddest_order_this_disp, polyorder,
                                       function=func, minx=xd_min, maxx=xd_max, maxiter=25,
                                       lower=sigrej, upper=sigrej, maxrej=maxrej, sticky=True, use_mad=True,
                                       weights=None)
        xd_angle_fit_coeffs[idisp, :] = pypeitFit.fitc
        debug = True
        if debug:
            this_fit = fitting.evaluate_fit(pypeitFit.fitc, func, xd_vec, minx=xd_min, maxx=xd_max)
            plt.plot(xd_vec, this_fit, color='green', label='fit')
            fit_gpm = pypeitFit.bool_gpm
            plt.plot(xd_angles_this_disp[fit_gpm], reddest_order_this_disp[fit_gpm], marker='o', markersize=7.0,
                     mfc='black', mec='black', fillstyle='full', linestyle='None', zorder=5, label='used by fit')
            plt.plot(xd_angles_this_disp[np.logical_not(fit_gpm)],reddest_order_this_disp[np.logical_not(fit_gpm)],
                     marker='s', markersize=9.0, mfc='red', mec='red', fillstyle='full', linestyle='None',
                     zorder=7, label='rejected')
            plt.legend()
            plt.title(f'XDISP={xdisp}, nkept={np.sum(fit_gpm)}, nrej={np.sum(np.logical_not(fit_gpm))}')
            plt.xlabel('xd_angle')
            plt.ylabel('reddest_order')
            plt.ylim(this_fit.min()-3, this_fit.max() +3)
            plt.show()

    return xd_angle_fit_params, xd_angle_fit_coeffs





def echelle_composite_arcspec(arxiv_file, outfile, show_individual_solns=False, do_total=False, show_orders=False, debug=False):

    color_tuple = ('green', 'cyan', 'magenta', 'blue', 'darkorange', 'yellow', 'dodgerblue', 'purple',
                   'lightgreen', 'cornflowerblue')
    colors = itertools.cycle(color_tuple)

    arxiv_params = Table.read(arxiv_file, hdu=1)[0]
    arxiv = Table.read(arxiv_file, hdu=2)
    norders = arxiv_params['norders']
    order_vec = np.arange(arxiv_params['order_min'], arxiv_params['order_max'] + 1, 1)
    msgs.info(f"order_vec = {order_vec}")


    # First loop over the orders to determine wavelength coverage and sampling of each order
    wave_grid_min = np.zeros(norders)
    wave_grid_max = np.zeros(norders)
    dwave_pix = np.zeros(norders)
    dloglam_pix = np.zeros(norders)
    nspec_per_order = np.zeros(norders, dtype=int)
    for iord, this_order in enumerate(order_vec):
        msgs.info(f"iord = {iord},  this_order = {this_order}")
        # TODO Should we make XDISP specific composites? I don't think so
        populated = arxiv['populated_and_good'][:, iord]
        nsolns_this_order = np.sum(populated)
        if nsolns_this_order > 0:
            this_wave = arxiv['wave'][:, iord, :][populated, :]
            this_arc = arxiv['arcspec'][:, iord, :][populated, :]
            this_gpm = (this_wave > 0.0) & (this_arc != 0.0)
            this_dwave = np.zeros_like(this_wave)
            for ii in range(nsolns_this_order):
                this_dwave[ii, :] = wvutils.get_delta_wave(this_wave[ii, :], this_gpm[ii, :])

            wave_grid_min[iord], wave_grid_max[iord] = this_wave.min(), this_wave.max()
            dwave_pix[iord] = np.median(this_dwave.min(axis=1, where=this_dwave!=0, initial=10))
            dloglam_pix[iord] = np.median((this_dwave/this_wave/np.log(10.0)).min(axis=1, where=this_dwave!=0, initial=10))
        else:
            msgs.error(f'No arc solutions contribute to order={iord}. There must be a bug')
    msgs.info("end the loop of enumerate(order_vec)")

    # Use the smallest value of dloglam across all orders for the spectral grid spacing
    dloglam_pix_final = dloglam_pix.min()
    dv_pix_final = np.log(10.0)*c_kms*dloglam_pix_final
    msgs.info(f"dloglam_pix_final = {dloglam_pix_final}")
    msgs.info(f"dv_pix_final = {dv_pix_final}")
    msgs.info(f"wave_grid_min.min = {wave_grid_min.min()}")
    msgs.info(f"wave_grid_max.max = {wave_grid_max.max()}")


    # Use the same wavelength grid for all orders
    wave_total_composite, wave_total_composite_mid, dsamp = wvutils.wavegrid(
        wave_grid_min.min(), wave_grid_max.max(), dloglam_pix_final, log10=True)

    nspec_composite = wave_total_composite.size
    ind_min =  np.zeros(norders, dtype=int)
    ind_max =  np.zeros(norders, dtype=int)
    nvec = np.arange(nspec_composite)
    msgs.info(f"nspec_composite = {nspec_composite}")

    # Determine the size of the output array by looping over all orders and finding the maximum grid we need to store
    for iord, this_order in enumerate(order_vec):
        indx = (wave_total_composite >= wave_grid_min[iord]) & (wave_total_composite <= wave_grid_max[iord])
        nspec_per_order[iord] = np.sum((wave_total_composite >= wave_grid_min[iord]) & (wave_total_composite <= wave_grid_max[iord]))
        ind_min[iord] = nvec[indx].min()
        ind_max[iord] = nvec[indx].max()
        msgs.info(f"iord = {iord},  this_order = {this_order}, ind_min[iord] = {ind_min[iord]}, ind_max[iord] = {ind_max[iord]}")

    # Allocate output arrays for composite arc
    nspec_max = nspec_per_order.max()
    wave_composite = np.zeros((nspec_max, norders))
    arc_composite = np.zeros((nspec_max, norders))
    gpm_composite = np.zeros((nspec_max, norders), dtype=bool)

    sn_smooth_npix = 1  # Should not matter since we use uniform weights
    for iord, this_order in enumerate(order_vec):
        msgs.info(f"iord = {iord},  this_order = {this_order}")
        populated = arxiv['populated_and_good'][:, iord]
        nsolns_this_order = np.sum(populated)
        if nsolns_this_order > 0:
            this_wave = arxiv['wave'][:, iord, :][populated, :]
            this_arc = arxiv['arcspec'][:, iord, :][populated, :]
            this_gpm = (this_wave > 0.0) & (this_arc != 0.0)
            this_wave_composite = wave_total_composite[ind_min[iord]:ind_max[iord]]
            this_nspec = this_wave_composite.size
            arc_interp_iord = np.zeros((this_nspec, nsolns_this_order))
            gpm_arc_iord = np.zeros((this_nspec, nsolns_this_order), dtype=bool)
            # Interpolate our arcs onto the new grid
            for ii in range(nsolns_this_order):
                this_arc_interp = interpolate.interp1d(
                    this_wave[ii, this_gpm[ii, :]], this_arc[ii, this_gpm[ii,:]], kind='cubic',
                    bounds_error=False, fill_value=-1e10)(this_wave_composite)
                arc_interp_iord[:, ii] = this_arc_interp
                gpm_arc_iord[:, ii] = arc_interp_iord[:, ii] > -1e9
                if show_individual_solns:
                    # plt.plot(iwave[in_gpm], this_arc[ii, in_gpm], color=next(colors), alpha=0.7)
                    plt.plot(this_wave_composite[gpm_arc_iord[ii, :]],this_arc_interp[gpm_arc_iord[ii, :]],
                             color=next(colors), alpha=0.7)
            if show_individual_solns:
                plt.title(f'Order={this_order}')
                plt.show()

            wave_grid_in = np.repeat(this_wave_composite[:, np.newaxis], nsolns_this_order, axis=1)
            ivar_arc_iord = utils.inverse(np.abs(arc_interp_iord) + 10.0)
            # wave_grid_mid, wave_grid_stack, arcspec_stack, _, arcspec_gpm, = coadd.combspec(
            #     utils.array_to_explist(wave_grid_in), utils.array_to_explist(arc_interp_iord),
            #     utils.array_to_explist(ivar_arc_iord), utils.array_to_explist(gpm_arc_iord), sn_smooth_npix,
            #     wave_method='user_input', wave_grid_input=this_wave_composite,
            #     ref_percentile=70.0, maxiter_scale=5, sigrej_scale=3.0, scale_method='median',
            #     sn_min_polyscale=2.0, sn_min_medscale=0.5, const_weights=True, maxiter_reject=5, sn_clip=30.0,
            #     lower=5.0, upper=5.0,
            #     debug=debug, debug_scale=debug, show_scale=debug, show=show_orders, verbose=True)
            wave_grid_mid, wave_grid_stack, arcspec_stack, _, arcspec_gpm, = coadd.combspec(
                utils.array_to_explist(wave_grid_in), utils.array_to_explist(arc_interp_iord),
                utils.array_to_explist(ivar_arc_iord), utils.array_to_explist(gpm_arc_iord), sn_smooth_npix,
                wave_method='user_input', wave_grid_input=this_wave_composite,
                ref_percentile=70.0, maxiter_scale=5, sigrej_scale=3.0, scale_method='median',
                sn_min_polyscale=2.0, sn_min_medscale=0.5, weight_method='constant', maxiter_reject=5, sn_clip=30.0,
                lower=5.0, upper=5.0,
                debug=debug, debug_scale=debug, show_scale=debug, show=show_orders, verbose=True)
            #ind_mid_min[iord] = np.argmin(np.abs(wave_grid_mid.min() - wave_total_composite_mid))
            #ind_mid_max[iord] = np.argmin(np.abs(wave_grid_mid.max() - wave_total_composite_mid))
            wave_composite[0:wave_grid_mid.size, iord] = wave_grid_mid
            arc_composite[0:wave_grid_mid.size, iord] = arcspec_stack
            gpm_composite[0:wave_grid_mid.size, iord] = arcspec_gpm

    # Now generate a final composite arc combining all the orders. Experimental. Not sure we need this.
    if do_total:
        show_total=False
        ivar_composite = utils.inverse(np.abs(arc_composite) + 10.0)
        # TODO this will crash since it is not taking lists.
        wave_grid_mid, wave_grid_stack, arcspec_stack, _, arcspec_gpm = coadd.combspec(
            wave_composite, arc_composite, ivar_composite, gpm_composite, sn_smooth_npix,
            wave_method='user_input', wave_grid_input=wave_total_composite, ref_percentile=70.0, maxiter_scale=5, sigrej_scale=3.0, scale_method='median',
            sn_min_polyscale=2.0, sn_min_medscale=0.5, const_weights=True, maxiter_reject=5, sn_clip=30.0,
            lower=5.0, upper=5.0,
            debug=debug, debug_scale=debug, show_scale=debug, show=show_total, verbose=True)


    params=Table([[os.path.basename(arxiv_file)], [arxiv_params['order_min']],[arxiv_params['order_max']],[norders],
                  [wave_composite[gpm_composite > 0.0].min()], [wave_composite[gpm_composite > 0.0].max()],
                  [dloglam_pix_final], [dv_pix_final]],
                  names=('arxiv_file','order_min', 'order_max', 'norders','wave_min','wave_max','dloglam','dv'))


    print(f'Writing HIRES xidl+pypeit composite_arc archive to file: {outfile}')
    hdulist = fits.HDUList()
    hdulist.append(fits.BinTableHDU(params.as_array()))  # hdu = 1
    hdulist.append(fits.ImageHDU(np.array(wave_composite)))  # hdu = 2
    hdulist.append(fits.ImageHDU(np.array(arc_composite)))  # hdu = 3
    hdulist.append(fits.ImageHDU(np.array(gpm_composite.astype(float))))  # hdu = 3
    hdulist.writeto(outfile, overwrite=True)








#xidl_arxiv_file = os.path.join(os.getenv('PYPEIT_DEV'), 'dev_algorithms', 'hires_wvcalib', 'hires_wvcalib_xidl.fits')
xidl_arxiv_file = '/Users/yaqi/Desktop/pypeit_HIRES_orig/hires_wvcalib_xidl.fits'
# Create the astropy table form of the xidl save file arxiv
if not os.path.isfile(xidl_arxiv_file):
    msgs.info("######################\tingest_xidl_archive")
    ingest_xidl_archive(xidl_arxiv_file)



# # append the pypeit templates to the xidl archive
# #arxiv_file = os.path.join(os.getenv('PYPEIT_DEV'), 'dev_algorithms', 'hires_wvcalib', 'hires_wvcalib.fits')
# arxiv_file = '/Users/yaqi/Desktop/pypeit_HIRES_orig/hires_wvcalib.fits'
# if not os.path.isfile(arxiv_file):
#     msgs.info("######################\tappend_pypeit_archive")
#     append_pypeit_archive(arxiv_file, xidl_arxiv_file)






# sys.exit(-1)
# Perform fits to the coefficients vs ech angle
# TODO see if pca works better here
debug=False
# wvcalib_angle_fit_file = os.path.join(os.getenv('PYPEIT_DEV'), 'dev_algorithms', 'hires_wvcalib', 'wvcalib_angle_fits.fits')
wvcalib_angle_fit_file = '/Users/yaqi/Desktop/pypeit_HIRES_orig/wvcalib_angle_fits.fits'
if not os.path.isfile(wvcalib_angle_fit_file):
    msgs.info("######################\tfit_wvcalib_vs_angles")
    # fit_wvcalib_vs_angles(arxiv_file, wvcalib_angle_fit_file, func='legendre',
    #                   ech_nmax = 3, ech_coeff_fit_order_min=1, ech_coeff_fit_order_max=2,
    #                   xd_reddest_fit_polyorder=2, sigrej=3.0, maxrej=1, debug=debug)
    fit_wvcalib_vs_angles(xidl_arxiv_file, wvcalib_angle_fit_file, func='legendre',
                      ech_nmax = 3, ech_coeff_fit_order_min=1, ech_coeff_fit_order_max=2,
                      xd_reddest_fit_polyorder=2, sigrej=3.0, maxrej=1, debug=debug)

# Compute a composite arc from the solution arxiv
#composite_arcfile = os.path.join(os.getenv('PYPEIT_DEV'), 'dev_algorithms', 'hires_wvcalib', 'HIRES_composite_arc.fits')
composite_arcfile = '/Users/yaqi/Desktop/pypeit_HIRES_orig/HIRES_composite_arc.fits'

if not os.path.isfile(composite_arcfile):
    msgs.info("######################\techelle_composite_arcspec")
    # echelle_composite_arcspec(arxiv_file, composite_arcfile, show_orders=debug)
    echelle_composite_arcspec(xidl_arxiv_file, composite_arcfile, show_orders=debug)

sys.exit(-1)




















use_unknowns = True
line_lists_all = waveio.load_line_lists(['ThAr'])
line_lists = line_lists_all[np.where(line_lists_all['ion'] != 'UNKNWN')]
unknwns = line_lists_all[np.where(line_lists_all['ion'] == 'UNKNWN')]
tot_line_list = table.vstack([line_lists, unknwns]) if use_unknowns else line_lists
spectrograph = load_spectrograph('keck_hires')
par = spectrograph.default_pypeit_par()['calibrations']['wavelengths']
n_final = 4
# xmin, xmax for wavelength vs pixel fits
fmin, fmax = 0.0, 1.0
color_tuple = ('green', 'cyan', 'magenta', 'blue', 'darkorange', 'yellow', 'dodgerblue', 'purple',
               'lightgreen', 'cornflowerblue')
colors = itertools.cycle(color_tuple)
#
#
# # Read template file
# templ_table_file = os.path.join(
#     resource_filename('pypeit', 'data'), 'arc_lines',
#     'hires', 'hires_templ_xidl.dat')
# tbl = Table.read(templ_table_file, format='ascii')
# nrows = len(tbl)
#
# order_min = tbl['IOrder'].min()
# order_max = 118
#
# order_vec = np.arange(order_min, order_max +1, 1)
# norders = order_vec.size
#
# # Subset of orders in every file. Populated indicates whether a given order is populated
# lambda_cen = np.zeros((norders, nrows))
# ech_angle = np.zeros((norders, nrows))
# populated = np.zeros((norders, nrows), dtype=bool)
# XDISP_is_red = np.zeros((norders, nrows), dtype=bool)
# binspec = np.zeros((norders, nrows), dtype=int)
# det = np.zeros((norders, nrows), dtype=int)
# xd_angle = np.zeros((norders, nrows))
# coeff = np.zeros((norders, nrows, n_final+1))
# bluest_order = np.zeros(nrows, dtype=int)
# xd_angle_file = np.zeros(nrows)
# ech_angle_file = np.zeros(nrows)
# det_file = np.zeros(nrows)
# XDISP_is_red_file = np.zeros(nrows, dtype=bool)
#
# for irow in np.arange(nrows):
#     this_order_vec_raw, this_wave, this_arc = templates.xidl_hires(
#         os.path.join(os.getenv('HIRES_CALIBS'), 'ARCS', tbl[irow]['Name']), specbin=tbl[irow]['Rbin'])
#     if irow == 0:
#         nspec = this_wave.shape[1]
#         xnspecmin1 = float(nspec - 1)
#         xvec = np.arange(nspec)/xnspecmin1
#         wave = np.zeros((norders, nrows, nspec))
#         arcspec = np.zeros((norders, nrows, nspec))
#         table_xidl = Table([np.zeros((nrows, norders), dtype="<U30"),
#                             np.zeros((nrows, norders)),
#                             np.zeros((nrows, norders)),
#                             np.zeros((nrows, norders), dtype=int),
#                             np.zeros((nrows,), dtype=int),
#                             np.zeros((nrows, norders), dtype="<U3"),
#                             np.zeros((nrows, norders), dtype=int),
#                             np.zeros((nrows, norders), dtype=int),
#                             np.zeros((nrows, norders), dtype=bool),
#                             np.zeros((nrows, norders, n_final+1)),
#                             np.zeros((nrows, norders, nspec)),
#                             np.zeros((nrows, norders, nspec)),],
#                             names = ('filename', 'ech_angle', 'xd_angle', 'order', 'bluest_order',
#                                      'xdisp', 'det', 'binspec', 'populated', 'coeff', 'wave', 'arcspec'))
#     else:
#         assert this_wave.shape[1] == nspec
#     # Restrict to what is labeled as good in the Table
#     igood = (this_order_vec_raw >= tbl[irow]['IOrder']) & (this_order_vec_raw <= tbl[irow]['EOrder'])
#     nsolns = np.sum(igood)
#     this_order_vec = this_order_vec_raw[igood]
#     indx = this_order_vec - order_min
#     populated[indx, irow] = True
#     ech_angle[indx, irow] = tbl[irow]['ECH']
#     xd_angle[indx, irow] = tbl[irow]['XDAng']
#     XDISP_is_red[indx, irow] = tbl[irow]['XDISP'] == 'RED'
#     binspec[indx, irow] =  tbl[irow]['Rbin']
#     det[indx, irow] =  tbl[irow]['Chip']
#
#     wave[indx, irow, :] = this_wave[igood, :]
#     arcspec[indx, irow, :] = this_arc[igood, :]
#     lambda_cen[indx, irow] = np.median(this_wave[igood, :], axis=1)
#     # Fit the wavelengths
#     coeff_array = np.zeros((nsolns, n_final +1))
#     for ii, iwave in enumerate(this_wave[igood, :]):
#         pypeitFit = fitting.robust_fit(xvec, iwave, n_final, function=par['func'], maxiter=10,
#                                        lower=1e10, upper=1e10, maxrej=0, sticky=True,
#                                        minx=fmin, maxx=fmax, weights=None)
#         coeff_array[ii, :] = pypeitFit.fitc
#     coeff[indx, irow, :] = coeff_array
#     # file specific
#     bluest_order[irow] = this_order_vec[-1]
#     ech_angle_file[irow] = tbl[irow]['ECH']
#     xd_angle_file[irow] = tbl[irow]['XDAng']
#     det_file[irow] = tbl[irow]['Chip']
#     XDISP_is_red_file[irow] = tbl[irow]['XDISP'] == 'RED'

#all_dlam = []
#all_lam = []
#all_orders = []


pad_factor = 0.10
# xvec_pad = np.arange(-int(np.round(pad_factor*nspec)), int(np.round((1.0 + pad_factor)*nspec)))/xnspecmin1
#
# # Plot the polynomial coefficients versus echelle angle order by order
# debug_all=False
# show_wv_grid=False
# ncoeff_fit_order = 2
# coeff_vs_order = np.zeros((norders, n_final + 1, ncoeff_fit_order+1))
# ech_min, ech_max = ech_angle_file.min(), ech_angle_file.max()
# ech_vec = ech_min + (ech_max-ech_min)*np.arange(100)/99
# func = 'legendre'
# debug_fits=False
# for iord, this_order in enumerate(order_vec):
#     if np.any(populated[iord, :]):
#         nsolns = np.sum( populated[iord, :])
#         this_ech = ech_angle[iord, populated[iord, :]]
#         this_xd_angle = xd_angle[iord, populated[iord, :]]
#         this_lambda_cen = lambda_cen[iord, populated[iord, :]]
#         this_coeff = coeff[iord, populated[iord, :], :]
#         for ic in range(n_final + 1):
#             pypeitFit = fitting.robust_fit(this_ech, this_coeff[:, ic], ncoeff_fit_order, function=func,
#                                            minx=ech_min, maxx=ech_max, maxiter=25,
#                                            lower=3.0, upper=3.0, maxrej=2, sticky=True,use_mad=True, weights=None)
#             coeff_vs_order[iord, ic, :] = pypeitFit.fitc
#             if debug_fits:
#                 this_fit = fitting.evaluate_fit(pypeitFit.fitc, func, ech_vec, minx=ech_min, maxx=ech_max)
#                 plt.plot(ech_vec, this_fit, color='blue', label='fit')
#                 fit_gpm = pypeitFit.bool_gpm
#                 plt.plot(this_ech[fit_gpm], this_coeff[fit_gpm, ic], marker='o', markersize=7.0, mfc='black',
#                          mec='black', fillstyle='full', linestyle='None', zorder=5, label='used by fit')
#                 plt.plot(this_ech[np.logical_not(fit_gpm)], this_coeff[np.logical_not(fit_gpm), ic],
#                          marker='s', markersize=9.0, mfc='red', mec='red',fillstyle='full', linestyle='None', zorder=7,label='rejected')
#                 plt.legend()
#                 plt.title(f'order={this_order}, cc_ii={ic}, nkept={np.sum(fit_gpm)}, nrej={np.sum(np.logical_not(fit_gpm))}')
#                 plt.xlabel('ech_angle')
#                 plt.ylabel('coeff')
#                 plt.ylim(this_fit.min() - 0.05*np.abs(this_fit.min()), this_fit.max() + 0.05*np.abs(this_fit.max()))
#                 plt.show()
#         #for ii in range(n_final+1):
#         #    plt.plot(this_xd_angle, this_coeff[:, ii], 'k.', label=f'order={iorder}, cc_ii={ii}')
#         #    plt.legend()
#         #    plt.xlabel('xd_angle')
#         #    plt.ylabel('coeff')
#         #    plt.show()


# Plot lam vs dlam/lam for each order

#for iord, this_order in enumerate(order_vec):
for iord in np.arange(norders)[::-1]:
    this_order = order_vec[iord]
    if np.any(populated[iord, :]):
        nsolns = np.sum(populated[iord, :])
        this_ech = ech_angle[iord, populated[iord, :]]
        this_xd_angle = xd_angle[iord, populated[iord, :]]
        this_lambda_cen = lambda_cen[iord, populated[iord, :]]
        this_wave = wave[iord, populated[iord, :], :]
        this_arc = arcspec[iord, populated[iord, :], :]
        #this_coeff = coeff[iord, populated[iord, :], :]

        this_dwave = np.zeros_like(this_wave)
        for ii, iwave in enumerate(this_wave):
            this_dwave[ii, :] = wvutils.get_delta_wave(iwave, (iwave > 0.0))

        # Now try a fit. TODO any wavelength grid will work here
        med_dlam = np.median(this_dwave[this_wave > 1.0])
        fit = robust_fit(this_wave.flatten(), this_dwave.flatten(), 3, maxiter=25, maxdev = 0.10*med_dlam, groupbadpix=True)
        wave_grid_fit, wave_grid_fit_mid, dsamp = wvutils.get_wave_grid(this_wave.T,wave_method='log10')
        dwave_fit = fit.eval(wave_grid_fit)
        gpm = fit.bool_gpm.copy()
        gpm.resize(this_wave.shape)

        if show_wv_grid:
            for ii, iwave in enumerate(this_wave):
                this_color=next(colors)
                this_gpm = gpm[ii, :]
                plt.plot(iwave[this_gpm], this_dwave[ii, this_gpm], marker='o', markersize=1.0, mfc=this_color,
                fillstyle='full',  linestyle='None', zorder=1)
                plt.plot(iwave[np.logical_not(this_gpm)], this_dwave[ii, np.logical_not(this_gpm)], marker='o',
                     markersize=2.0, mfc='red', fillstyle='full', zorder=3, linestyle='None')

            plt.plot(wave_grid_fit, dwave_fit, color='black', label='fit', zorder=10)
            plt.title(f'order={this_order}', fontsize=14)
            plt.legend()
            plt.show()

        lam_min, lam_max = this_wave[gpm].min(), this_wave[gpm].max()
        wave_grid = get_variable_dlam_wavegrid(lam_min, lam_max, wave_grid_fit, dwave_fit)
        nspec_tmpl = wave_grid.shape[0]
        # TESTING
        #dwave_chk = wvutils.get_delta_wave(wave_grid, (wave_grid > 0.0))
        #plt.plot(wave_grid, dwave_chk, color='red', label='our new grid')
        #plt.plot(wave_grid_fit, dwave_fit, color='black', label='fit')
        #plt.title('dwave compared to our fit')
        #plt.legend()
        #plt.show()
        tmpl_iord = np.zeros((nsolns, nspec_tmpl))
        gpm_tmpl = np.zeros((nsolns, nspec_tmpl), dtype=bool)
        # Interpolate our arcs onto the new grid
        for ii, iwave in enumerate(this_wave):
            in_gpm = this_arc[ii, :] != 0.0
            tmpl_iord[ii, :] = interpolate.interp1d(iwave[in_gpm], this_arc[ii, in_gpm], kind='cubic', bounds_error=False, fill_value=-1e10)(wave_grid)
            gpm_tmpl[ii, :] = tmpl_iord[ii, :] > -1e9
            if show_wv_grid:
                #plt.plot(iwave[in_gpm], this_arc[ii, in_gpm], color=next(colors), alpha=0.7)
                plt.plot(wave_grid[gpm_tmpl[ii, :]], tmpl_iord[ii, gpm_tmpl[ii, :]], color=next(colors), alpha=0.7)
                plt.show()

        sn_smooth_npix = 1 # Should not matter since we use uniform weights
        wave_grid_in = np.repeat(wave_grid[:, np.newaxis], nsolns, axis=1)
        ivar_tmpl_iord = utils.inverse(np.abs(tmpl_iord) + 10.0)
        wave_grid_mid, wave_grid_stack, arcspec_tmpl, _, arcspec_tmpl_gpm = coadd.combspec(
            wave_grid_in, tmpl_iord.T, ivar_tmpl_iord.T, gpm_tmpl.T, sn_smooth_npix,
            wave_method='iref',  ref_percentile=70.0, maxiter_scale=5, sigrej_scale=3.0, scale_method='median',
            sn_min_polyscale=2.0, sn_min_medscale=0.5, const_weights=True, maxiter_reject=5, sn_clip=30.0, lower=5.0, upper=5.0,
            debug=debug_all, debug_scale=debug_all, show_scale=debug_all, show=True, verbose=True)

        all_patt_dict = {}
        detections = {}
        wv_calib = {}

        all_patt_dict_pad = {}
        detections_pad = {}
        wv_calib_pad = {}

        for slit in range(nsolns):
            print('Working on soln={:d}'.format(slit))
            # Trim the template to the relevant range. Hack this for now
            #itmpl = (wave_grid_mid >= 0.999*iwave.min()) & (wave_grid_mid <= 1.001*iwave.max())
            coeff_predict = np.zeros(n_final + 1)
            for ic in range(n_final+1):
                coeff_predict[ic] = fitting.evaluate_fit(coeff_vs_order[iord, ic, :], func, this_ech[slit], minx=ech_min, maxx=ech_max)

            wave_predict = fitting.evaluate_fit(coeff_predict, par['func'], xvec, minx=fmin, maxx=fmax)
            wave_predict_pad = fitting.evaluate_fit(coeff_predict, par['func'], xvec_pad, minx=fmin, maxx=fmax)
            wave_true = this_wave[slit, :]
            # Substitute wave_true here as a test
            arcspec_templ_predict_pad =  interpolate.interp1d(wave_grid_stack[arcspec_tmpl_gpm], arcspec_tmpl[arcspec_tmpl_gpm],
                                                          kind='cubic', bounds_error=False, fill_value=0.0)(wave_predict_pad)
            arcspec_templ_predict =  interpolate.interp1d(wave_grid_stack[arcspec_tmpl_gpm], arcspec_tmpl[arcspec_tmpl_gpm],
                                                          kind='cubic', bounds_error=False, fill_value=0.0)(wave_predict)
            #arcspec_tmpl_trim = arcspec_tmpl[itmpl]
            #wave_grid_mid_trim = wave_grid_mid[itmpl]
            #arc_in_pad = np.zeros_like(arcspec_tmpl_trim)
            #in_gpm = this_arc[slit, :] != 0.0
            #npix = np.sum(in_gpm)
            #arc_in_pad[:npix] = this_arc[slit, in_gpm]
            #xcorr_poly(this_wave[slit, in_gpm], this_arc[slit, in_gpm], wave_grid_mid, arcspec_tmpl, smooth=1.0, percent_ceil=50.0, use_raw_arc=False,
            #           sigdetect=10.0, fwhm=4.0, debug=True, seed=42)

            # WITH PADDING
            #arc_pad = np.zeros_like(xvec_pad)
            #arc_pad[0:nspec] = this_arc[slit, :]
            #detections_pad[str(slit)], spec_cont_sub_pad, all_patt_dict_pad[str(slit)] = autoid.reidentify(
            #    arc_pad, arcspec_templ_predict_pad, wave_predict_pad,  tot_line_list, par['nreid_min'],
            #    cc_thresh=par['cc_thresh'], match_toler=par['match_toler'], cc_local_thresh=par['cc_local_thresh'],
            #    nlocal_cc=par['nlocal_cc'], nonlinear_counts=1e10,
            #    sigdetect=par['sigdetect'], fwhm=par['fwhm'], debug_peaks=True, debug_xcorr=True, debug_reid=True)

            # WITHOUT PADDING
            detections[str(slit)], spec_cont_sub, all_patt_dict[str(slit)] = autoid.reidentify(
                this_arc[slit, :], arcspec_templ_predict, wave_predict,  tot_line_list, par['nreid_min'],
                cc_thresh=par['cc_thresh'], match_toler=par['match_toler'], cc_local_thresh=par['cc_local_thresh'],
                nlocal_cc=par['nlocal_cc'], nonlinear_counts=1e10,
                sigdetect=par['sigdetect'], fwhm=par['fwhm'], debug_peaks=True, debug_xcorr=True, debug_reid=True)

            # Check if an acceptable reidentification solution was found
            if not all_patt_dict[str(slit)]['acceptable']:
                wv_calib[str(slit)] = None
                continue

            final_fit = wv_fitting.fit_slit(spec_cont_sub, all_patt_dict[str(slit)], detections[str(slit)],
                                            tot_line_list, match_toler=par['match_toler'],func=par['func'], n_first=par['n_first'],
                                            sigrej_first=par['sigrej_first'], n_final=n_final,sigrej_final=par['sigrej_final'])

            #autoid.arc_fit_qa(final_fit, title='Silt: {}'.format(str(slit)))


#        dlam = []
#        lam = []
#        for iwave in this_wave:
#            dlam += list(wvutils.get_delta_wave(iwave, np.ones_like(iwave,dtype=bool)))
#            lam += iwave.tolist()
#            #xlam += ((np.array(iwave) - np.array(iwave).min())/(np.array(iwave).max() - np.array(iwave).min())).tolist()
#        plt.plot(lam, 3.0e5*np.array(dlam)/np.array(lam), '.', label=f'order={iorder}')
#        plt.legend()
#        plt.show()

 #       all_dlam += dlam
 #       all_lam += lam
 #       all_orders += [iorder]*len(lam)


#plt.plot(all_lam, 3.0e5*np.array(all_dlam)/np.array(all_lam), '.')
#plt.legend()
#plt.show()

# Plot the central wavelength vs echelle angle order by order
for iord, iorder in enumerate(order_vec):
    if np.any(populated[iord, :]):
        this_ech = ech_angle[iord, populated[iord, :]]
        this_xd_angle = xd_angle[iord, populated[iord, :]]
        this_lambda_cen = lambda_cen[iord, populated[iord, :]]
        plt.plot(this_ech, this_lambda_cen, 'k.', label=f'order={iorder}')
        plt.legend()
        plt.show()


for xdisp in ['UV', 'RED']:
    for idet in [1,2,3]:
        iord = (XDISP_is_red_file == (xdisp == 'RED')) & (det_file == idet)
        plt.plot(xd_angle_file[iord], bluest_order[iord], 'k.', label=f'XDISP={xdisp}, det={idet}')
        plt.legend()
        plt.show()





