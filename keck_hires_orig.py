"""
Module for Keck/HIRES

.. include:: ../include/links.rst
"""
import os

from IPython import embed



import numpy as np
from scipy.io import readsav

from astropy.table import Table
from astropy.time import Time

from pypeit import msgs
from pypeit import telescopes
from pypeit import io
from pypeit.core import parse
from pypeit.core import framematch
from pypeit.spectrographs import spectrograph
from pypeit.images import detector_container
from pypeit.par import pypeitpar
from pypeit.images.mosaic import Mosaic
from pypeit.core.mosaic import build_image_mosaic_transform


class KECKHIRESORIGSpectrograph(spectrograph.Spectrograph):
    """
    Child to handle KECK/HIRES specific code.

    This spectrograph is not yet supported.
    """

    ndet = 1
    name = 'keck_hires_orig'
    telescope = telescopes.KeckTelescopePar()
    camera = 'HIRES'
    url = 'https://www2.keck.hawaii.edu/inst/hires/'
    header_name = 'HIRES'
    url = 'https://www2.keck.hawaii.edu/inst/hires/'
    pypeline = 'Echelle'
    ech_fixed_format = False
    supported = False
    # TODO before support = True
    # 1. Implement flat fielding
    # 2. Test on several different setups
    # 3. Implement PCA extrapolation into the blue 


    @classmethod
    def default_pypeit_par(cls):
        """
        Return the default parameters to use for this instrument.

        Returns:
            :class:`~pypeit.par.pypeitpar.PypeItPar`: Parameters required by
            all of PypeIt methods.
        """
        par = super().default_pypeit_par()

        par['rdx']['detnum'] = [(1)]

        # Adjustments to parameters for Keck HIRES
        turn_on = dict(use_biasimage=False, use_overscan=True, overscan_method='median',
                       use_darkimage=False, use_illumflat=False, use_pixelflat=False,
                       use_specillum=False)
        par.reset_all_processimages_par(**turn_on)
        par['calibrations']['traceframe']['process']['overscan_method'] = 'median'

        # Right now we are using the overscan and not biases becuase the
        # standards are read with a different read mode and we don't yet have
        # the option to use different sets of biases for different standards,
        # or use the overscan for standards but not for science frames
        # TODO testing
        par['scienceframe']['process']['use_biasimage'] = False
        par['scienceframe']['process']['use_illumflat'] = False
        par['scienceframe']['process']['use_pixelflat'] = False
        par['calibrations']['standardframe']['process']['use_illumflat'] = False
        par['calibrations']['standardframe']['process']['use_pixelflat'] = False
        # par['scienceframe']['useframe'] ='overscan'

        par['calibrations']['slitedges']['edge_thresh'] = 8.0
        par['calibrations']['slitedges']['fit_order'] = 8
        par['calibrations']['slitedges']['max_shift_adj'] = 0.5
        par['calibrations']['slitedges']['trace_thresh'] = 10.
        par['calibrations']['slitedges']['left_right_pca'] = True
        par['calibrations']['slitedges']['length_range'] = 0.3
        par['calibrations']['slitedges']['max_nudge'] = 0.
        par['calibrations']['slitedges']['overlap'] = True
        par['calibrations']['slitedges']['dlength_range'] = 0.25

        # These are the defaults
        par['calibrations']['tilts']['tracethresh'] = 15
        par['calibrations']['tilts']['spat_order'] = 3
        par['calibrations']['tilts']['spec_order'] = 5  # [5, 5, 5] + 12*[7] # + [5]

        # 1D wavelength solution
        par['calibrations']['wavelengths']['lamps'] = ['ThAr']
        # This is for 1x1 binning. TODO GET BINNING SORTED OUT!!
        par['calibrations']['wavelengths']['rms_threshold'] = 0.50
        par['calibrations']['wavelengths']['sigdetect'] = 5.0
        par['calibrations']['wavelengths']['n_final'] = 4 #[3] + 13 * [4] + [3]
        # This is for 1x1 binning. Needs to be divided by binning for binned data!!
        par['calibrations']['wavelengths']['fwhm'] = 8.0
        # Reidentification parameters
        par['calibrations']['wavelengths']['method'] = 'echelle'
        # TODO: the arxived solution is for 1x1 binning. It needs to be
        # generalized for different binning!
        #par['calibrations']['wavelengths']['reid_arxiv'] = 'vlt_xshooter_vis1x1.fits'
        par['calibrations']['wavelengths']['cc_thresh'] = 0.50
        par['calibrations']['wavelengths']['cc_local_thresh'] = 0.50
#        par['calibrations']['wavelengths']['ech_fix_format'] = True
        # Echelle parameters
        par['calibrations']['wavelengths']['echelle'] = True
        par['calibrations']['wavelengths']['ech_nspec_coeff'] = 4
        par['calibrations']['wavelengths']['ech_norder_coeff'] = 4
        par['calibrations']['wavelengths']['ech_sigrej'] = 3.0
        par['calibrations']['wavelengths']['ech_separate_2d'] = True

        # Flats
        par['calibrations']['flatfield']['tweak_slits_thresh'] = 0.90
        par['calibrations']['flatfield']['tweak_slits_maxfrac'] = 0.10

        # Extraction
        par['reduce']['skysub']['bspline_spacing'] = 0.6
        par['reduce']['skysub']['global_sky_std'] = False
        # local sky subtraction operates on entire slit
        par['reduce']['extraction']['model_full_slit'] = True
        # Mask 3 edges pixels since the slit is short, insted of default (5,5)
        par['reduce']['findobj']['find_trim_edge'] = [3, 3]
        # Continnum order for determining thresholds

        # Sensitivity function parameters
        par['sensfunc']['algorithm'] = 'IR'
        par['sensfunc']['polyorder'] = 5 #[9, 11, 11, 9, 9, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7]
        par['sensfunc']['IR']['telgridfile'] = 'TelFit_MaunaKea_3100_26100_R20000.fits'

        # Coadding
        par['coadd1d']['wave_method'] = 'log10'

        return par


    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        That is, this associates the PypeIt-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        self.meta = {}
        # Required (core)
        self.meta['ra'] = dict(ext=0, card='RA', required_ftypes=['science', 'standard'])
        self.meta['dec'] = dict(ext=0, card='DEC', required_ftypes=['science', 'standard'])
        self.meta['target'] = dict(ext=0, card='OBJECT')
        self.meta['decker'] = dict(ext=0, card='DECKNAME')
        self.meta['binning'] = dict(card=None, compound=True)
        self.meta['mjd'] = dict(ext=0, card='MJD')
        # This may depend on the old/new detector
        self.meta['exptime'] = dict(ext=0, card='ELAPTIME')
        self.meta['airmass'] = dict(ext=0, card='AIRMASS')
        #self.meta['dispname'] = dict(ext=0, card='ECHNAME')
        # Extras for config and frametyping
        self.meta['hatch'] = dict(ext=0, card='HATOPEN')
        self.meta['dispname'] = dict(ext=0, card='XDISPERS')
        self.meta['filter1'] = dict(ext=0, card='FIL1NAME')
        self.meta['echangle'] = dict(ext=0, card='ECHANGL', rtol=1e-3)
        self.meta['xdangle'] = dict(ext=0, card='XDANGL', rtol=1e-2)
#        self.meta['idname'] = dict(ext=0, card='IMAGETYP')
        # NOTE: This is the native keyword.  IMAGETYP is from KOA.
        self.meta['idname'] = dict(ext=0, card='OBSTYPE')
        self.meta['frameno'] = dict(ext=0, card='FRAMENO')
        self.meta['instrument'] = dict(ext=0, card='INSTRUME')



    def compound_meta(self, headarr, meta_key):
        """
        Methods to generate metadata requiring interpretation of the header
        data, instead of simply reading the value of a header card.

        Args:
            headarr (:obj:`list`):
                List of `astropy.io.fits.Header`_ objects.
            meta_key (:obj:`str`):
                Metadata keyword to construct.

        Returns:
            object: Metadata value read from the header(s).
        """
        if meta_key == 'binning':
            # TODO JFH Is this correct or should it be flipped?
            binspatial, binspec = parse.parse_binning(headarr[0]['BINNING'])
            binning = parse.binning2string(binspec, binspatial)
            return binning
        else:
            msgs.error("Not ready for this compound meta")



    def configuration_keys(self):
        """
        Return the metadata keys that define a unique instrument
        configuration.

        This list is used by :class:`~pypeit.metadata.PypeItMetaData` to
        identify the unique configurations among the list of frames read
        for a given reduction.

        Returns:
            :obj:`list`: List of keywords of data pulled from file headers
            and used to constuct the :class:`~pypeit.metadata.PypeItMetaData`
            object.
        """
        return ['filter1', 'echangle', 'xdangle', 'binning']



    def raw_header_cards(self):
        """
        Return additional raw header cards to be propagated in
        downstream output files for configuration identification.

        The list of raw data FITS keywords should be those used to populate
        the :meth:`~pypeit.spectrographs.spectrograph.Spectrograph.configuration_keys`
        or are used in :meth:`~pypeit.spectrographs.spectrograph.Spectrograph.config_specific_par`
        for a particular spectrograph, if different from the name of the
        PypeIt metadata keyword.

        This list is used by :meth:`~pypeit.spectrographs.spectrograph.Spectrograph.subheader_for_spec`
        to include additional FITS keywords in downstream output files.

        Returns:
            :obj:`list`: List of keywords from the raw data files that should
            be propagated in output files.
        """
        return ['FIL1NAME', 'ECHANGL', 'XDANGL']



    def pypeit_file_keys(self):
        """
        Define the list of keys to be output into a standard PypeIt file.

        Returns:
            :obj:`list`: The list of keywords in the relevant
            :class:`~pypeit.metadata.PypeItMetaData` instance to print to the
            :ref:`pypeit_file`.
        """
        return super().pypeit_file_keys() + ['frameno']



    def check_frame_type(self, ftype, fitstbl, exprng=None):
        """
        Check for frames of the provided type.

        Args:
            ftype (:obj:`str`):
                Type of frame to check. Must be a valid frame type; see
                frame-type :ref:`frame_type_defs`.
            fitstbl (`astropy.table.Table`_):
                The table with the metadata for one or more frames to check.
            exprng (:obj:`list`, optional):
                Range in the allowed exposure time for a frame of type
                ``ftype``. See
                :func:`pypeit.core.framematch.check_frame_exptime`.

        Returns:
            `numpy.ndarray`_: Boolean array with the flags selecting the
            exposures in ``fitstbl`` that are ``ftype`` type frames.
        """
        good_exp = framematch.check_frame_exptime(fitstbl['exptime'], exprng)
        # TODO: Allow for 'sky' frame type, for now include sky in
        # 'science' category
        if ftype == 'science':
            return good_exp & (fitstbl['idname'] == 'Object')
        if ftype == 'standard':
            return good_exp & (fitstbl['idname'] == 'Object')
        if ftype == 'bias':
            return good_exp & (fitstbl['idname'] == 'Bias')
        if ftype == 'dark':
            return good_exp & (fitstbl['idname'] == 'Dark')
        if ftype in ['pixelflat', 'trace']:
            # Flats and trace frames are typed together
            return good_exp & (fitstbl['idname'] == 'IntFlat')
        if ftype in ['arc', 'tilt']:
            # Arc and tilt frames are typed together
            return good_exp & (fitstbl['idname'] == 'Line')

        msgs.warn('Cannot determine if frames are of type {0}.'.format(ftype))
        return np.zeros(len(fitstbl), dtype=bool)


    def get_detector_par(self, det, hdu=None):
        """
        Return metadata for the selected detector.

        Args:
            det (:obj:`int`):
                1-indexed detector number.
            hdu (`astropy.io.fits.HDUList`_, optional):
                The open fits file with the raw image of interest.  If not
                provided, frame-dependent parameters are set to a default.

        Returns:
            :class:`~pypeit.images.detector_container.DetectorContainer`:
            Object with the detector metadata.
        """
        # Binning
        binning = '1,1' if hdu is None else self.get_meta_value(self.get_headarr(hdu), 'binning')

        # Detector 1
        detector_dict = dict(
            binning         = binning,
            det             = 1,
            dataext         = 0,
            specaxis        = 0,
            specflip        = False,
            spatflip        = False,
            platescale      = 0.135,
            darkcurr        = 0.0,  # e-/pixel/hour
            saturation      = 65535.,
            nonlinear       = 0.7, # Website says 0.6, but we'll push it a bit
            mincounts       = -1e10,
            numamplifiers   = 1,
            ronoise         = np.atleast_1d([2.8]),
            )

        # Set gain 
        # https://www2.keck.hawaii.edu/inst/hires/instrument_specifications.html
        if hdu is None or hdu[0].header['CCDGAIN'].strip() == 'F':
            detector_dict['gain'] = np.atleast_1d([1.9])
        elif hdu[0].header['CCDGAIN'].strip() == 'T':
            detector_dict['gain'] = np.atleast_1d([0.78])
        else:
            msgs.error("Bad CCDGAIN mode for HIRES")
            
        return detector_container.DetectorContainer(**detector_dict)



    def get_rawimage(self, raw_file, det, spectrim=20):
        """ Read the image
        """
        # Check for file; allow for extra .gz, etc. suffix
        if not os.path.isfile(raw_file):
            msgs.error(f'{raw_file} not found!')
        hdu = io.fits_open(raw_file)
        head0 = hdu[0].header

        # Number of AMPS
        namp = head0['NUMAMPS']

        # Get post, pre-pix values
        prepix = head0['PREPIX']
        postpix = head0['POSTPIX']
        preline = head0['PRELINE']
        postline = head0['POSTLINE']

        # Grab the data
        full_image = hdu[0].data.astype(float)
        rawdatasec_img = np.zeros_like(full_image, dtype=int)
        oscansec_img = np.zeros_like(full_image, dtype=int)

        # 
        nspat = int(head0['WINDOW'].split(',')[3]) // namp
        for amp in range(namp):
            col0 = prepix*2 + nspat*amp
            # Data
            rawdatasec_img[:, col0:col0+nspat] = amp+1
            # Overscan
            o0 = prepix*2 + nspat*namp + postpix*amp
            oscansec_img[:, o0:o0+postpix] = amp+1

        #embed(header='435 of keck_esi.py')

        return self.get_detector_par(1, hdu=hdu), \
                full_image, hdu, head0['ELAPTIME'], rawdatasec_img, oscansec_img


    def order_platescale(self, order_vec, binning=None):
        """
        Return the platescale for each echelle order.

        This routine is only defined for echelle spectrographs, and it is
        undefined in the base class.

        Args:
            order_vec (`numpy.ndarray`_):
                The vector providing the order numbers.
            binning (:obj:`str`, optional):
                The string defining the spectral and spatial binning.

        Returns:
            `numpy.ndarray`_: An array with the platescale for each order
            provided by ``order``.
        """
        det = self.get_detector_par(1)
        binspectral, binspatial = parse.parse_binning(binning)

        # Assume no significant variation (which is likely true)
        return np.ones_like(order_vec)*det.platescale*binspatial






