import os
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

def download_data(starname,mission,quarter_number,cadence,verbose=True):
    from lightkurve.search import _search_products

    degrees = 21/3600 #size of TESS pixels in degrees on the sky
    
    if mission=='TESS':
        if (cadence=='long') or (cadence=='30 minute') or (cadence=='10 minute'):
                ffi_or_tpf='FFI'
        if (cadence=='short') or (cadence=='2 minute') or (cadence=='20 second') or (cadence=='fast'):
            ffi_or_tpf='Target Pixel'
    if mission=='Kepler':
        if (cadence=='long') or (cadence=='30 minute') or (cadence=='10 minute'):
            ffi_or_tpf='FFI'
        if (cadence=='short') or (cadence=='2 minute') or (cadence=='20 second') or (cadence=='fast'):
            ffi_or_tpf='Target Pixel'        
    
    if mission=='TESS':
        try:
            search_string=_search_products(starname, radius=degrees,\
                                           filetype=ffi_or_tpf, \
                                           cadence=cadence,\
                                           mission=mission,sector=quarter_number)
        except SearchError as e:
            print('No ',cadence,' cadence ',mission,' data for ',starname,' in Sector ',quarter_number,'!')
            return None
    if mission=='Kepler':
        search_string = lk.search_targetpixelfile(starname,author=mission, \
                              quarter=quarter_number,\
                              exptime=cadence)
        search_string=search_string[search_string.author==mission]
    #
    # #NEW
    if verbose==True:
        if mission=='TESS':        
            print('Number of data products for ',starname,' in ',mission,' with \"',cadence,'\" cadence and in Sector \"',quarter_number,'\":',len(search_string.exptime.value))
        if mission=='Kepler':        
            print('Number of data products for ',starname,' in ',mission,' with \"',cadence,'\" cadence and in Quarter \"',quarter_number,'\":',len(search_string.exptime.value))        
    # # NEW
    #
    # filter search result by cadence input argument
    if (cadence=='30 minute') or (cadence=='long'):
        mask = np.where((search_string.exptime.value>600) & (search_string.exptime.value<=1800) )[0]
        search_string_filtered = search_string[mask]
    if (cadence=='10 minute'):
        mask = np.where((search_string.exptime.value>120) & (search_string.exptime.value<=600) )[0]
        search_string_filtered = search_string[mask]
    if (cadence=='2 minute') or (cadence=='short'):
        mask = np.where((search_string.exptime.value>20) & (search_string.exptime.value<=120) )[0]
        search_string_filtered = search_string[mask]
    if (cadence=='20 second') or (cadence=='fast'):
        mask = np.where((search_string.exptime.value>0) & (search_string.exptime.value<=20) )[0]
        search_string_filtered = search_string[mask]
        
    #make sure there are no duplicate quarters/sectors
    u, indices = np.unique(search_string_filtered.mission, return_index=True)
    search_string_filtered = search_string_filtered[indices]
    # # NEW    
    return search_string_filtered

def catalog_info(TIC_ID):
    """Takes TIC_ID, returns stellar information from online catalog using Vizier.
    Input Parameters
    ----------
    TIC_ID : int
        TESS Input Catalog ID number to search for and analyze.
        
    Returns
    -------
        A list of select stellar parameters.
        * (a,b): quadratic limb darkening coefficients cross-matched from 
        limb darkening Claret et al. (2012) tables based on Effective Temperature
        and Surface Gravity of target star (according to the TIC).
        * mass: Stellar Mass (from the TIC) in units of solar masses.
        * mass_min, mass_max: Uncertainty of Stellar Mass (from the TIC) in units of solar masses.
        * radius: Stellar Radius (from the TIC) in units of solar radii.
        * radius_min, radius_max: Uncertainty of Stellar Radius (from the TIC) in units of solar radii.
          
    """
    import requests
    import numpy
    from os import path
    import time as clock
    try:
        from astroquery.mast import Catalogs
    except:
        raise ImportError("Package astroquery required but failed to import")
    #
    #
    #
    result = Catalogs.query_criteria(catalog="Tic", ID=TIC_ID).as_array()
    Teff = float(result['Teff'].data)
    logg = float(result['logg'].data)
    radius = float(result['rad'].data)
    radius_max = float(result['e_rad'].data)
    radius_min = float(result['e_rad'].data)
    mass = float(result['mass'].data)
    mass_max = float(result['e_mass'].data)
    mass_min = float(result['e_mass'].data)
    
    return (mass, mass_min, mass_max, radius, radius_min, radius_max)

def SMA_AU_from_Period_to_stellar(Period,R_star,M_star):
    """
    This function will calculate the Semi-Major Axis (SMA)
    using Kepler's third law.
    
    Input Parameters
    ----------
    Period : float
        Orbital period in days
    R_star : float
        Stellar Radius in solar radii
    M_star : float
        Stellar Mass in solar masses
    Returns
    -------
        * SMA
            Semi-Major Axis in solar units
        * SMA_cm
            Semi-Major Axis in units of centimeters        
    """
    #assumes circular orbit
    #using Kepler's third law, calculate SMA
    #solar units
    import astropy.units as u
    from astropy import constants as const
    RS = u.R_sun.to(u.cm) # in cm
    MS = u.M_sun.to(u.g) # in grams
    #
    G = const.G.cgs.value #cm^3 per g per s^2
    #
    R = R_star*RS
    M = M_star*MS
    P=Period*60.0*24.0*60.0 #in seconds
    #
    #SMA
    SMA_cm = ((G*M*(P**2))/(4*(np.pi**2)))**(1/3)
    #
    #note R_star is already in solar units so we need to convert to cm using
    # solar radius as a constant
    Stellar_Radius = R #now in cm
    #
    SMA = SMA_cm / Stellar_Radius #now unitless (cm / cm)
    return SMA, SMA_cm

def Tdur(Period, R_star,M_star, R_planet_RE):
    """
    This function will calculate the transit duration
    time based on Kepler's third law. 
    Input Parameters
    ----------
    Period : float
        Orbital period in days
    R_star : float
        Stellar Radius in Solar radii
    M_star : float
        Stellar Mass in Solar masses
    R_planet_RE: float
        Planet radius in Earth Radii

    Returns
    -------
        * Tdur : float
            Estimated transit duration time in units of days
    """
    from astropy import units as u

    RE = u.R_earth.to(u.cm) # in cm
    RS = u.R_sun.to(u.cm) # in cm
    A = Period/np.pi #in days
    #
    SMA_cm = SMA_AU_from_Period_to_stellar(Period,R_star,M_star)[1]
    #
    B =(R_star*RS +R_planet_RE*RE)/ SMA_cm # unitless (cm/cm)
    #
    T_dur = A*np.arcsin(B) #in days
    return T_dur    

def smoothing_function(ID,input_LC,window_size_in_days=None,verbose=True,filter_type='biweight'):
    """   
    This function will smooth the input light curve. If no window size is
    provided, a window size corresponding to 3 times the transit duration
    of an Earth-sized planet eclipsing this target star will be used.
    Input Parameters
    ----------
    ID : int
        TESS Input Catalog identification number
    input_LC : pandas dataframe
        Input NEMESIS light curve to be smoothed
    window_size_in_days : float or None
        Desired smoothing window in units of days. Default is set to None.
    verbose : boolean
        This flag is used to print the window size used for smoothing.
    Returns
     -------
    * newlc : Lightkurve LightCurve object
        This Detrended Light Curve contains the follow column names:
        * time
        * flux #this is detrended flux
        * flux_err #this is detrended flux error
    * trend : Lightkurve LightCurve object
        This trend Light Curve contains the follow column names:
        * time
        * flux #this is smoothed lightcurve trend line
        * flux_err #this is detrended flux error (same as input_LC.flux_err)
    """
    from wotan import flatten
    import numpy as np
    import lightkurve as lk
    #
    #
    #read in LC data
    time = input_LC.time.value
    flux_raw = input_LC.flux.value
    flux_error = input_LC.flux_err.value
    #
    if window_size_in_days==None:
        LCDur=(np.nanmax(time) - np.nanmin(time))
        maxP = LCDur/2 #longest period for 2 transits in a light curve (~14 days for TESS single sector LCs)
        R_planet_RE = 1
        M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
        window_size_in_days = 5*Tdur(maxP, R_star,M_star, R_planet_RE)
    if verbose==True:
        print('window size (days): ',window_size_in_days)
    flatten_lc, trend_lc = flatten(time, flux_raw, window_length=window_size_in_days, \
                                   return_trend=True, method=filter_type,robust=True)
    T=time
    F=flatten_lc
    FE=flux_error
    #checking for NaNs
    nanmask = np.where(np.isfinite(F)==True)[0]
    T = T[nanmask]
    F = F[nanmask]
    FE =FE[nanmask]
    F_raw = flux_raw[nanmask]
    trend_lc=trend_lc[nanmask]
    newlc = lk.LightCurve(time=T,flux=F,flux_err=FE)
    trend = lk.LightCurve(time=time[nanmask],flux=trend_lc,flux_err=flux_error[nanmask])
    return newlc, trend


def extract_TESS_photometry_and_smooth(starname,author,nsigma,
                                       save_directory, mask_threshold=None,
                                       Sector=None, window_size_in_days=None,
                                       verbose=True,
                                       filter_type='biweight',max_sector=26):
    
    #Step 0: collect TESS images
    search_result=download_data(starname,mission='TESS',
                                quarter_number=Sector,cadence='2 minute')
    
    N_data_products=len(search_result)
    while N_data_products<1:
        # see which LC has the most data
        search_result_20s=download_data(starname,mission='TESS',
                                        quarter_number=Sector,
                                        cadence='20 second',verbose=False)
        search_result_10m=download_data(starname,mission='TESS',
                                        quarter_number=Sector,
                                        cadence='10 minute',verbose=False)
        search_result_30m=download_data(starname,mission='TESS',
                                        quarter_number=Sector,
                                        cadence='30 minute',verbose=False)
        #
        results=[len(search_result_20s), len(search_result_10m),len(search_result_30m)]
        most_data_index = np.argmax(results)
        if most_data_index==0:
            print('most data for 20s cadence')
            most_data = search_result_20s
        if most_data_index==1:
            print('most data for 10m cadence')
            most_data = search_result_10m
        if most_data_index==2:
            print('most data for 30m cadence')        
            most_data = search_result_30m
        N_data_products+=len(most_data)

        search_result = most_data
        break        
        
    def get_primary_mission_sectors(search_string,last_updated_sector=26):
        #find what available TESS sectors are able to be processed by our code:
        mask = np.where(np.array(list(map(int, [x[-2:] for x in search_string.mission])))<int(last_updated_sector))[0]
        imask = np.where(np.array(list(map(int, [x[-2:] for x in search_string.mission])))>int(last_updated_sector))[0]
        print('Our code is currently coded to handle TESS Sectors 1-26. ', 
              np.unique(search_string[imask].mission),
              'light curves are not yet available.')
        print(' ')
        return search_string[mask]

    search_result = get_primary_mission_sectors(search_result)
    print('processing',len(search_result),'data products:')
    Sectors=np.unique(search_result.mission)
    print(np.unique(search_result.mission))
    print(' ')
    Sectors_str=[]
    for ss in Sectors:
        Sectors_str=np.append(Sectors_str,str(int(ss[-2:])))    
    Sectors_str=','.join(Sectors_str)
    
    tpfs = search_result.download_all(bitmask_quality='hardest')
    #Step 1: extract and smooth TESS data
    for t in range(len(tpfs)):
        tpf = tpfs[t]
        #Step 2: Perform aperture photometry
        if mask_threshold is None:
            pixel_mask = tpf.pipeline_mask
            background_mask = ~tpf.pipeline_mask
        if mask_threshold is not None:
            pixel_mask = tpf.create_threshold_mask(threshold=mask_threshold)
            background_mask = ~tpf.create_threshold_mask(threshold=mask_threshold)

        quality_mask = tpf.quality==0
        lc = tpf.to_lightcurve(aperture_mask=pixel_mask)
        bkg = tpf.to_lightcurve(aperture_mask=background_mask)
        lc = lc[quality_mask]
        bkg = bkg[quality_mask]

        #Step 3: Perform Background Subtraction and Normalization
        bkg_subtracted_flux=lc.flux.value - bkg.flux.value

        #create new "LightCurve" object
        bkg_subtracted_lc = lk.LightCurve(time=lc.time.value,
                                          flux=bkg_subtracted_flux,
                                          flux_err = lc.flux_err.value)

        # normalize the background subtracted light curve
        normalized_bkg_subtracted_lc =  bkg_subtracted_lc.normalize()

        outlier_removed_normalized_bkg_subtracted_lc = normalized_bkg_subtracted_lc.remove_outliers(sigma_upper=nsigma)
        
        # rename
        input_lc = outlier_removed_normalized_bkg_subtracted_lc

        # Rename output light curve from Step 1:
        input_lc = outlier_removed_normalized_bkg_subtracted_lc

        # Step 2: Smoothing light curve before transit searching (NEW STEP)
        ID=int(starname[4:]) #assuming it begins with 'TIC '
        smoothed_lc,trend_lc = smoothing_function(ID,input_lc,
                                                  window_size_in_days=window_size_in_days,
                                                  verbose=verbose,filter_type=filter_type)
        #remove outliers again after smoothing
        #smoothed_lc.remove_outliers(sigma_upper=nsigma,sigma_lower=2*nsigma)
#         smoothed_lc.scatter()
    #     plt.title('Sector '+str(tpf.sector))
    #     plt.show()
        if t==0:
            normalized_bkg_subtracted_lcs=normalized_bkg_subtracted_lc
            outlier_removed_normalized_bkg_subtracted_lcs=outlier_removed_normalized_bkg_subtracted_lc
            output_lc = smoothed_lc   
            trend_lcs=trend_lc
        print(t)
        if t>0: 
            normalized_bkg_subtracted_lcs= normalized_bkg_subtracted_lcs.append(normalized_bkg_subtracted_lc)
            outlier_removed_normalized_bkg_subtracted_lcs= outlier_removed_normalized_bkg_subtracted_lcs.append(outlier_removed_normalized_bkg_subtracted_lc)
            output_lc = output_lc.append(smoothed_lc)
            trend_lcs = trend_lcs.append(trend_lc)
            
    print('photom len check#1:',len(trend_lcs),len(output_lc),
          len(outlier_removed_normalized_bkg_subtracted_lcs),len(normalized_bkg_subtracted_lcs))
    # Step 4: Visualize the light curve
#     fig=plt.figure(figsize=(10,5))
#     ax1=fig.add_subplot(221)
#     ax2=fig.add_subplot(222)
    
#     tpf.plot(aperture_mask=pixel_mask,mask_color='red',
#              ax=ax1,show_colorbar=True)
#     tpf.plot(aperture_mask=background_mask,mask_color='pink',
#              ax=ax1,show_colorbar=False)
    
#     if len(Sectors_str)>1:
#         ax2.set_title(starname+' in Sectors '+str(Sectors_str))
#     else:
#         ax2.set_title(starname+' in Sector '+str(Sectors_str))
    
#     ax2.plot(normalized_bkg_subtracted_lcs.time.value,
#              normalized_bkg_subtracted_lcs.flux.value,
#            marker='.',color='red',linestyle='none')
    
#     ax2.plot(outlier_removed_normalized_bkg_subtracted_lcs.time.value,
#              outlier_removed_normalized_bkg_subtracted_lcs.flux.value,
#            marker='.',color='black',linestyle='none')    
  
#     ax2.set_xlabel('Time [Days]')
#     ax2.set_ylabel('Normalized Relative Flux')
#     fig.tight_layout(pad=1)
#     plt.savefig(save_directory+starname+'_Sector_'+str(Sectors_str)+'_lightcurve.png',
#                 bbox_inches='tight')
#     plt.show()            
        
    return output_lc,Sectors_str,tpfs[0],pixel_mask,background_mask,normalized_bkg_subtracted_lcs,outlier_removed_normalized_bkg_subtracted_lcs,trend_lcs


def phasefold_version2(time,flux,flux_err,T0,Period):
    ''' 
    Version 2 is calculating the number of orbital cycles in 
    our light curve but is also shifted so that the phase value
    of 0 is at the T0 reference time.
    Input:
        time: array of timestamps
        flux: array of flux values
        flux_err: array of flux uncertainty values
        T0: float, reference time
        Period: float, orbital period
    Output:
        phase: array of phase values
        flux: array of flux values
        flux_err: array of flux uncertainty values
    '''    
    phase = (time - T0 + 0.5*Period) % Period - 0.5*Period
    ind = np.argsort(phase,axis=0)
    
    return phase[ind], flux[ind], flux_err[ind]

def fullphasefold(time,T0,period,flux,offset):
    """
    This function will phase-fold the input light curve (time, flux)
    using a Mid-transit time and orbital period.
    
    Input Parameters
    ----------
    time: array
        An array of timestamps from TESS observations.        
    TO : float
        The Mid-transit time of a periodic event.
    period : float
        An orbital period of a periodic event.
    flux : array
        An array of flux values from TESS observations.
    offset : int or float
        A value used to offset the phase by a fraction of an orbit.
        
    Returns
    -------
        * phase : array
            An array of Orbital phase of the phase-folded light curve.
        * flux : array
            An array of flux values from TESS observations of the 
            phase-folded light curve.
    """
    phase= (time - T0+ offset*period) / period - np.floor((time - T0+ offset*period) / period)
    ind=np.argsort(phase, axis=0)
    return phase[ind],flux[ind]
        
# grid functions for transit searching
# constants from transitleastsquares
#
# astrophysical constants
import astropy.units as u
from astropy import constants as const
R_sun = u.R_sun.to(u.cm) # in cm
M_sun = u.M_sun.to(u.g) # in grams
G = const.G.cgs.value #cm^3 per g per s^2
R_earth = u.R_earth.to(u.cm) # in cm
R_jup = u.R_jupiter.to(u.cm) # in cm
SECONDS_PER_DAY = u.day.to(u.second)


# For the duration grid
FRACTIONAL_TRANSIT_DURATION_MAX = 0.12
M_STAR_MIN = 0.1
M_STAR_MAX = 1.0
R_STAR_MIN = 0.13
R_STAR_MAX = 3.5
DURATION_GRID_STEP = 1.05
OVERSAMPLING_FACTOR = 5
N_TRANSITS_MIN = 3
MINIMUM_PERIOD_GRID_SIZE = 100

def T14(
    R_s, M_s, P, upper_limit=FRACTIONAL_TRANSIT_DURATION_MAX, small=False
):
    """Input:  Stellar radius and mass; planetary period
               Units: Solar radius and mass; days
       Output: Maximum planetary transit duration T_14max
               Unit: Fraction of period P"""
    import numpy
    
    pi = numpy.pi
    P = P * SECONDS_PER_DAY
    R_s = R_sun * R_s
    M_s = M_sun * M_s

    if small:  # small planet assumption
        T14max = R_s * ((4 * P) / (pi * G * M_s)) ** (1 / 3)
    else:  # planet size 2 R_jup
        T14max = (R_s + 2 * R_jup) * (
            (4 * P) / (pi * G * M_s)
        ) ** (1 / 3)

    result = T14max / P
    if result > upper_limit:
        result = upper_limit
    return result


def duration_grid(periods, shortest, log_step=DURATION_GRID_STEP):
    import numpy    
    duration_max = T14(
        R_s=R_STAR_MAX,
        M_s=M_STAR_MAX,
        P=numpy.min(periods),
        small=False  # large planet for long transit duration
    )
    duration_min = T14(
        R_s=R_STAR_MIN,
        M_s=M_STAR_MIN,
        P=numpy.max(periods),
        small=True  # small planet for short transit duration
    )

    durations = [duration_min]
    current_depth = duration_min
    while current_depth * log_step < duration_max:
        current_depth = current_depth * log_step
        durations.append(current_depth)
    durations.append(duration_max)  # Append endpoint. Not perfectly spaced.
    return durations
def find_min_trial_duration(input_LC,durations,periods):
    # this function takes the duration_grid and period_grid
    # provided by TLS' functions (which are based on stellar params)
    #
    def Special_Bin_func(time,flux,error,binsize):  #<-- doesn't allow empty bins to exist
        import math
        good = np.where(np.isfinite(time)) #finds values that are not +/- inf, NaN
        timefit = time[good]
        fluxfit = flux[good]
        errfit  = error[good]
        timemax = np.max(timefit)
        timemin = np.min(timefit)
        npoints = len(timefit)
        nbins   = int(math.ceil((timemax - timemin)/binsize)) #binsize in days
        bintime = np.full((nbins,), np.nan) #fills array with NaNs to be filled with values
        binflux = np.full((nbins,), np.nan)
        binerr  = np.full((nbins,), np.nan)
        for i in range(0,nbins-1):
            tobin = np.where( (timefit >= (timemin + i*binsize)) & (timefit < (timemin + (i+1)*binsize)) )[0]
            if len(tobin) != 0:         
                # inverse variance weighted means
                binflux[i] = ((fluxfit[tobin]/(errfit[tobin]**2.0)).sum()) / ((1.0/errfit[tobin]**2.0).sum())
                bintime[i] = ((timefit[tobin]/(errfit[tobin]**2.0)).sum()) / ((1.0/errfit[tobin]**2.0).sum())
                binerr[i]  = 1.0 / (np.sqrt( (1.0/errfit[tobin]**2.0)).sum() )
        #
        return bintime, binflux, binerr
    #
    def calc_phase_coverage(input_LC,trial_duration,trial_period):
        t,f,e = input_LC.time.value, input_LC.flux.value, input_LC.flux_err.value
        T0 = np.nanmin(t)
        pf,ff,fe = phasefold_version2(t,f,e,T0,trial_period)
        binpf,binff,binfe = Special_Bin_func(pf,ff,e,binsize=trial_duration)
        ntotal=len(binpf)
        fin_bin = len(np.where(np.isfinite(binpf))[0])
        phase_coverage = fin_bin / ntotal
        return phase_coverage
    #
    # calculate phase coverage for input lightcurve
    phase_coverage = np.empty(np.size(durations))
    #do min P only
    for c in range(len(phase_coverage)):
        pc = calc_phase_coverage(input_LC,\
                                 trial_duration=durations[c],\
                                 trial_period=np.min(periods))
        phase_coverage[c] = pc
        
    q=0.25
    q_ind = np.where(phase_coverage==np.quantile(phase_coverage,q,interpolation='nearest'))[0][0]
    min_trial_duration = durations[q_ind]

    if min_trial_duration < (np.nanmedian(np.diff( input_LC.time.value ))/2 ):
        #print('min duration < half cadence!')
        min_trial_duration = (np.nanmedian(np.diff( input_LC.time.value )))/2

    durations=np.array(durations)
    new_durations = durations[durations>=min_trial_duration]
    new_durations = new_durations.tolist()
    
    return new_durations


def period_grid(
    R_star,
    M_star,
    time_span,
    period_min=0,
    period_max=float("inf"),
    oversampling_factor=OVERSAMPLING_FACTOR,
    n_transits_min=N_TRANSITS_MIN,
):
    """Returns array of optimal sampling periods for transit search in light curves
       Following Ofir (2014, A&A, 561, A138)"""
    import numpy
    pi = numpy.pi

    if R_star < 0.01:
        text = (
            "Warning: R_star was set to 0.01 for period_grid (was unphysical: "
            + str(R_star)
            + ")"
        )
        warnings.warn(text)
        R_star = 0.1

    if R_star > 10000:
        text = (
            "Warning: R_star was set to 10000 for period_grid (was unphysical: "
            + str(R_star)
            + ")"
        )
        warnings.warn(text)
        R_star = 10000

    if M_star < 0.01:
        text = (
            "Warning: M_star was set to 0.01 for period_grid (was unphysical: "
            + str(M_star)
            + ")"
        )
        warnings.warn(text)
        M_star = 0.01

    if M_star > 1000:
        text = (
            "Warning: M_star was set to 1000 for period_grid (was unphysical: "
            + str(M_star)
            + ")"
        )
        warnings.warn(text)
        M_star = 1000

    R_star = R_star * R_sun
    M_star = M_star * M_sun
    time_span = time_span * SECONDS_PER_DAY  # seconds

    # boundary conditions
    f_min = n_transits_min / time_span
    f_max = 1.0 / (2 * pi) * np.sqrt(G * M_star / (3 * R_star) ** 3)

    # optimal frequency sampling, Equations (5), (6), (7)
    A = (
        (2 * pi) ** (2.0 / 3)
        / pi
        * R_star
        / (G * M_star) ** (1.0 / 3)
        / (time_span * oversampling_factor)
    )
    C = f_min ** (1.0 / 3) - A / 3.0
    N_opt = (f_max ** (1.0 / 3) - f_min ** (1.0 / 3) + A / 3) * 3 / A

    X = numpy.arange(N_opt) + 1
    f_x = (A / 3 * X + C) ** 3
    P_x = 1 / f_x

    # Cut to given (optional) selection of periods
    periods = P_x / SECONDS_PER_DAY
    selected_index = numpy.where(
        numpy.logical_and(periods > period_min, periods <= period_max)
    )

    number_of_periods = numpy.size(periods[selected_index])

    if number_of_periods > 10 ** 6:
        text = (
            "period_grid generates a very large grid ("
            + str(number_of_periods)
            + "). Recommend to check physical plausibility for stellar mass, radius, and time series duration."
        )
        warnings.warn(text)

    if number_of_periods < MINIMUM_PERIOD_GRID_SIZE:
        if time_span < 5 * SECONDS_PER_DAY:
            time_span = 5 * SECONDS_PER_DAY
        warnings.warn(
            "period_grid defaults to R_star=1 and M_star=1 as given density yielded grid with too few values"
        )
        return period_grid(
            R_star=1, M_star=1, time_span=time_span / SECONDS_PER_DAY
        )
    else:
        return periods[selected_index]  # periods in [days]

def radius_from_mass(M_star):
    if M_star<=1:
        R_star = M_star**0.8
    if M_star>1:
        R_star = M_star**0.57
    return R_star
def mass_from_radius(R_star):
    if R_star<=1:
        M_star = R_star**(1/0.8)
    if R_star>1:
        M_star = R_star**(1/0.57)        
    return M_star    
    
def BLS_function(starname, input_lc,min_period,max_period,R_star,M_star,
                 oversampling_factor=9,duration_grid_step=1.05):
    '''
    This function is designed to conduct blind 
    transit searches using the BLS algorithm.
    
    Inputs:
        starname: 
            Name of the star, string
        input_lc: 
            lightkurve object of the input lightcurve, lk.LightCurve
        min_period: 
            minimum period for the blind BLS transit search, floats
        max_period: 
            maximum period for the blind BLS transit search, floats
        R_star:
            Stellar Radius of target star in Solar units, float
        M_star:
            Stellar Mass of target star in Solar units, float            
        duration_grid_step: the number of periods in our period grid, integer       
    
    Outputs:
       - planet_period: The best-fit orbital period from the 
                        BLS transit search, float
       - planet_t0: The best-fit transit time from the 
                        BLS transit search, float
       - planet_dur: The best-fit transit duration from the 
                        BLS transit search, float
       - figure: a two panel figure showing the BLS power spectrum
                 and the phase-folded light curve on the BLS result, 
                 PNG file
    '''
    
    #step 1: creating period grid for transit searches
#     period_grid = np.linspace(min_period,max_period,N_periods)
#     duration_grid=np.linspace(min_dur  ,max_dur,N_durations)
    #
    LC_time_span = (np.max(input_lc.time.value)-np.min(input_lc.time.value)) #time span of light curve
    #
    periods = period_grid(R_star=R_star, M_star=M_star, time_span=LC_time_span,\
                              period_min=min_period, period_max=max_period,
                              oversampling_factor=oversampling_factor)
    #
    durations= duration_grid(periods,shortest=None,log_step=duration_grid_step)
    durations = find_min_trial_duration(input_lc,durations,periods)
    
    #step 2: conducting the blind BLS transit search
    bls = input_lc.to_periodogram(method='bls',
                                  period=periods,
                                  duration=durations)#,
                                  #frequency_factor=frequency_factor)
        
#     # calculating BLS Period, T0 and Duration
    planet_period = bls.period_at_max_power
    planet_t0 = bls.transit_time_at_max_power
    planet_dur = bls.duration_at_max_power        
    
    #step 3: plotting BLS power spectra and calculating
    #        BLS Period, T0 and Duration
    
    #plotting power spectram AKA periodogram
#     fig=plt.figure(figsize=(10,5))
#     ax0=fig.add_subplot(311)
#     ax1=fig.add_subplot(312)
#     ax2=fig.add_subplot(313)
    
#     ax0.scatter(input_lc.time.value,input_lc.flux.value,s=2,color='black')
#     ax0.set_xlabel('Time [BTDJ]')
#     ax0.set_ylabel('Normalized Relative Flux')
    
#     bls.plot(color='black',lw=2,ax=ax1)
#     ax1.set_title(starname+' BLS Power Spectrum')
   
    

#     ax1.axvline(x=planet_period.value,color='red',lw=3,alpha=0.5,zorder=-10)
#     print('BLS period is ',planet_period)
#     print('BLS reference time is',planet_t0)
#     print('BLS duration is',planet_dur)
#     print('')
    
    
#     # step 4: phasefolding on BLS results
    
#     phase, flux, flux_err = phasefold_version2(input_lc.time.value,input_lc.flux.value,input_lc.flux_err.value,
#                                                planet_t0.value, planet_period.value)
    
#     ax2.plot(phase*24,flux,'k.',markersize=3) #note phase2 x 24 makes phase in units of hours, not days like above
#     ax2.set_xlabel('Orbital Phase [Hours since T0]\nzoomed in +/- 3 transit durations')
#     ax2.set_ylabel('Normalized Relative Flux')
#     ax2.set_xlim(-3.5*planet_dur.value*24,3.5*planet_dur.value*24)
#     ax2.axvline(x=0)
#     fig.tight_layout(pad=1)
#     fig.savefig(starname+'_BLS_result.png',bbox_inches='tight')
#     plt.show()
    
    
    return bls,planet_period.value, planet_t0.value, planet_dur.value


def plot_results(starname,tpf,sectors,pixel_mask,background_mask,
                 normalized_bkg_subtracted_lcs,outlier_removed_normalized_bkg_subtracted_lcs,
                 input_lc,trend_lcs,
                 bls,planet_period, planet_t0, planet_dur):
    
    import matplotlib.gridspec as gridspec
    fig = plt.figure(constrained_layout=True,figsize=(10,10))
    gs = fig.add_gridspec(5, 2)
    ax_im = fig.add_subplot(gs[0, 0])
    ax_lc = fig.add_subplot(gs[0, 1])
    ax_lc2 = fig.add_subplot(gs[1,:])
    ax1 = fig.add_subplot(gs[2,:])
    ax2 = fig.add_subplot(gs[3,:])

    
    tpf.plot(aperture_mask=pixel_mask,mask_color='red',
             ax=ax_im,show_colorbar=False,title=None)
    tpf.plot(aperture_mask=background_mask,mask_color='pink',
             ax=ax_im,show_colorbar=False,title=None)    
    ax_im.title.set_visible(not ax_im.title.get_visible())
    
    
    ax_lc.set_title(starname+' in Sectors '+str(sectors))
    
    ax_lc.plot(normalized_bkg_subtracted_lcs.time.value,
             normalized_bkg_subtracted_lcs.flux.value,
           marker='.',color='red',linestyle='none')
    
    ax_lc.plot(outlier_removed_normalized_bkg_subtracted_lcs.time.value,
             outlier_removed_normalized_bkg_subtracted_lcs.flux.value,
           marker='.',color='black',linestyle='none')    
    ax_lc.plot(trend_lcs.time.value,trend_lcs.flux.value,'r-')
  
    ax_lc.set_xlabel('Time [Days]')
    ax_lc.set_ylabel('Normalized\nRelative Flux')
    
    ax_lc2.scatter(input_lc.time.value,input_lc.flux.value,s=2,color='black')
    ax_lc2.set_xlabel('Time [BTDJ]')
    ax_lc2.set_ylabel('Normalized\nRelative Flux')
    
    #bls.plot(color='black',lw=2,ax=ax1)
    ax1.plot(bls.period,
             (bls.power-np.nanmean(bls.power))/np.nanstd(bls.power),
             color='black',lw=2)
    ax1.set_title(starname+' BLS Power Spectrum')
    ax1.set_xlabel('Period [days]')
    ax1.set_ylabel('Standardized\nBLS Power')
   
    
    # calculating BLS Period, T0 and Duration
    planet_period = bls.period_at_max_power
    planet_t0 = bls.transit_time_at_max_power
    planet_dur = bls.duration_at_max_power
    ax1.axvline(x=planet_period.value,color='red',linestyle='-',lw=3,alpha=0.5,zorder=-10)
    for i in range(2,15):
        if planet_period.value*i<np.nanmax(input_lc.time.value):
            ax1.axvline(x=planet_period.value*i,color='red',linestyle='--',lw=2,alpha=0.5,zorder=-10)
        if planet_period.value/i>np.nanmin(input_lc.time.value):
            ax1.axvline(x=planet_period.value/i,color='red',linestyle='--',lw=2,alpha=0.5,zorder=-10)
    print('BLS period is ',planet_period)
    print('BLS reference time is',planet_t0)
    print('BLS duration is',planet_dur)
    print('')
    
    
    # step 4: phasefolding on BLS results
    
    phase, flux, flux_err = phasefold_version2(input_lc.time.value,input_lc.flux.value,input_lc.flux_err.value,
                                               planet_t0.value, planet_period.value)
    
    ax2.plot(phase*24,flux,'k.',markersize=3) #note phase2 x 24 makes phase in units of hours, not days like above
    ax2.set_xlabel('Orbital Phase [Hours since T0]\nzoomed in +/- 3 transit durations')
    ax2.set_ylabel('Normalized\nRelative Flux')
    ax2.set_xlim(-3.5*planet_dur.value*24,3.5*planet_dur.value*24)
    ax2.axvline(x=0)
    fig.tight_layout(pad=1)
    fig.savefig(starname+'_BLS_result.png',bbox_inches='tight')
    plt.show()
    

def pipeline(starname, author, Sector, mask_threshold, nsigma, save_directory,
             window_size_in_days,filter_type,
             min_period,max_period, oversampling_factor,
             duration_grid_step,verbose):
    '''
    This function is used to extract single sector TESS light curves. 
    Currently, this function will only grab the first set of TESS 
    observations from the observation tables (search_result object). 
    After photometry extraction, this code will then conduct blind 
    transit searches using the BLS algorithm.
    
    Inputs
    ------------------------------------------------------
        starname: 
            Name of the star. Ex: 'TIC 12345678', 'Proxima Centauri'. Object type: string, str
        author: 
            'Source of the TESS data. Ex: SPOC,TESS-SPOC, QLP'. Object type: string, str 
        Sector:
            desired sector of TESS observations. Type: NoneType, int, list, array, or tuple
        nsigma: 
            The number of standard deviations above and below 
            the median of our light curves to remove data from. type: float    
        save_directory: 
            location on computer where figures are saved, type: str
        mask_threshold: 
            Input value for aperture selection. type: float or NoneType.
        window_size_in_days: 
            Input value for smoothing. type: float or NoneType
        filter_type: 
            string, 'biweight' by default. See Wotan documentation for other options.
        min_period: 
            minimum period for the blind BLS transit search, floats
        max_period: 
            maximum period for the blind BLS transit search, floats
        N_periods: 
            the number of periods in our period grid, integer
        min_dur: 
            minimum duration for the blind BLS transit search, floats
        max_dur: 
            maximum duration for the blind BLS transit search, floats
        N_durations: 
            the number of periods in our period grid, integer        
        frequency_factor: the frequency spacing in between periods 
                          of the power spectrum, integer
      verbose: 
          boolean, used for printing debugging checks
    
    Outputs
    ------------------------------------------------------
        outlier_removed_normalized_bkg_subtracted_lc: a lightkurve object containing
        extracted TESS photometry that is background subtracted, outlier removed and
        then normalized.
        
       - planet_period: The best-fit orbital period from the 
                        BLS transit search, float
       - planet_t0: The best-fit transit time from the 
                        BLS transit search, float
       - planet_dur: The best-fit transit duration from the 
                        BLS transit search, float
       - figure: a two panel figure showing the BLS power spectrum
                 and the phase-folded light curve on the BLS result, 
                 PNG file
    '''
    #
    # Step 0: Get stellar parameters:
    ID=int(starname[4:]) #assuming it begins with 'TIC '
    M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    # check stellar mass and radius:
    if (np.isnan(M_star)==True) & (np.isnan(R_star)==False):
        M_star = mass_from_radius(R_star)
    if (np.isnan(R_star)==True) & (np.isnan(M_star)==False):
        R_star = radius_from_mass(M_star)      
    #
    # Step 1: create lightcurve using lightkurve and smooth data before transit searching
    step_1_results = extract_TESS_photometry_and_smooth(starname=starname,
                                                       author=author,nsigma=nsigma,
                                                       mask_threshold=mask_threshold,
                                                       save_directory=save_directory,
                                                     Sector=Sector, window_size_in_days=window_size_in_days,
                                                     verbose=verbose,filter_type=filter_type)
    #
    smoothed_lc,Sectors,tpf0,pixel_mask,background_mask,normalized_bkg_subtracted_lcs,outlier_removed_normalized_bkg_subtracted_lcs,trend_lcs = step_1_results
    #
    #remove outliers again after smoothing
    smoothed_lc=smoothed_lc.remove_outliers(sigma_upper=nsigma,sigma_lower=5*nsigma)
    
    
# #     outlier_removed_normalized_bkg_subtracted_lc = extract_TESS_photometry(starname=starname,
# #                                                        author=author,nsigma=nsigma,
# #                                                        mask_threshold=mask_threshold,nsigma=nsigma,
# #                                                        save_directory=save_directory)
#     # Rename output light curve from Step 1:
#     input_lc = outlier_removed_normalized_bkg_subtracted_lc
    
#     # Step 2: Smoothing light curve before transit searching (NEW STEP)
#     ID=int(starname[4:]) #assuming it begins with 'TIC '
#     smoothed_lc,trend_lc = smoothing_function(ID,input_lc,
#                                               window_size_in_days=window_size_in_days,
#                                               verbose=verbose,filter_type=filter_type)
    
    # Step 3: search lightcurve for transits with BLS
    bls, planet_period, planet_t0, planet_dur = BLS_function(starname, smoothed_lc,
                                                        min_period,max_period,R_star,M_star,
                                                        oversampling_factor=oversampling_factor,
                                                        duration_grid_step=duration_grid_step)
    #old code
#     planet_period, planet_t0, planet_dur = BLS_function(starname=starname, input_lc=smoothed_lc,
#                                                         min_period=min_period, max_period=max_period, 
#                                                         N_periods=N_periods,
#                                                         min_dur=min_dur, max_dur=max_dur,N_durations=N_durations,
#                                                         frequency_factor=frequency_factor)
    
    output_lc = smoothed_lc
    
    
    plot_results(starname,tpf0,Sectors,pixel_mask,background_mask,
                 normalized_bkg_subtracted_lcs,outlier_removed_normalized_bkg_subtracted_lcs,
                 output_lc,trend_lcs,
                 bls,planet_period, planet_t0, planet_dur)
    
    
    return output_lc, planet_period, planet_t0, planet_dur    