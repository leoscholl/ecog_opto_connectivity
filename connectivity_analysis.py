import aopy
from aopy.data import db
from aopy.analysis import accllr
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import h5py
import traceback
from scipy.stats import zscore
import datetime
from tqdm.auto import tqdm
from IPython.display import display, Markdown
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib import colors
import multiprocessing as mp

from aopy.visualization import annotate_spatial_map_channels, place_Opto32_subplots, plot_angles
from aopy.data.bmi3d import tabulate_ts_data
from aopy.preproc.bmi3d import get_laser_trial_times
from aopy.preproc.quality import detect_bad_trials
from aopy.analysis.connectivity import get_acq_ch_near_stimulation_site, calc_connectivity_map_coh, prepare_erp
from aopy.analysis import calc_itpc, calc_tfr_mean
from aopy.analysis.latency import detect_itpc_response
from aopy.visualization import overlay_sulci_on_spatial_map, plot_xy_scalebar, plot_tf_map_grid
from aopy.data.bmi3d import tabulate_ts_data
from aopy.visualization import annotate_spatial_map_channels, place_Opto32_subplots, plot_angles
from aopy.data.bmi3d import tabulate_ts_data
from aopy.preproc.bmi3d import get_laser_trial_times
from aopy.preproc.quality import detect_bad_trials
from aopy.analysis.connectivity import get_acq_ch_near_stimulation_site, calc_connectivity_map_coh, prepare_erp
from aopy.analysis import calc_itpc
from aopy.analysis.latency import detect_itpc_response
from aopy.analysis import calc_stat_over_dist_from_pos, calc_stat_over_angle_from_pos
from aopy.analysis import calc_fdrc_ranktest, calc_tfr_mean_fdrc_ranktest, calc_tfr_mean
from aopy.visualization import plot_annotated_spatial_drive_map_stim, plot_annotated_stim_drive_data
from aopy.visualization import overlay_sulci_on_spatial_map, plot_xy_scalebar, plot_tf_map_grid, plot_spatial_drive_maps
from aopy.analysis import calc_spatial_data_correlation, calc_spatial_tf_data_correlation
from aopy.utils import scale_data_by_p_value

from aopy import data as aodata
from aopy import utils
from aopy import preproc
from aopy import precondition
from aopy import analysis
from aopy import visualization
 

def load_df(postproc_dir, subject, task, version):
    filename = f'postproc_{subject}_{task}_stim_{version}.pkl'
    filepath = os.path.join(postproc_dir, subject, filename)
    return pd.read_pickle(filepath)

def load_stim_erp(df, time_before, time_after, channels=None, preproc_dir='/data/preprocessed'):
    if len(df) == 0:
        raise ValueError("No trials to load")
    
    df = df.reset_index(drop=True)
    erp, samplerate = tabulate_ts_data(preproc_dir, df['subject'], df['te_id'], df['date'], 
                                       df['trial_time'], time_before, time_after, channels=channels)
    time = np.arange(len(erp))/samplerate
    erp = analysis.subtract_erp_baseline(erp, time, 0, time_before)
    
    return erp, samplerate

def calc_max_erp(df, time_before=0.1, time_after=0.1):
    _, acq_ch, _ = aopy.data.load_chmap()

    erp, samplerate = load_stim_erp(df, time_before, time_after, channels=acq_ch-1)
    max_erp = aopy.analysis.get_max_erp(erp, time_before, time_after, samplerate, trial_average=True)

    return max_erp

def make_equal_sizes(df, seed=None):
    # Count condition sizes
    counts = df['condition'].value_counts()
    min_count = counts.min()
    print("Before:", ", ".join(f"{c} trials for {cond}" for cond, c in counts.items()))
    
    # Sample equally from each condition
    df_balanced = (
        df.groupby('condition', group_keys=False)
          .apply(lambda x: x.sample(min_count, random_state=seed))
          .reset_index(drop=True)
    )
    
    # Re-check
    counts_after = df_balanced['condition'].value_counts()
    print("After:", ", ".join(f"{c} trials for {cond}" for cond, c in counts_after.items()))
    
    return df_balanced, min_count

def test_equal_sizes(df):
    plt.figure()
    plt.subplot(2,1,1)
    plt.scatter(df['date'], df['condition'])

    df_sub, min_count = make_equal_sizes(df)
    plt.subplot(2,1,2)
    plt.scatter(df_sub['date'], df_sub['condition'])
    
from aopy.analysis import compare_conditions_bootstrap_spatial_corr

def state_wrapper(df, labels, n_trials=200, n_bootstraps=50, n_shuffle=50, debug=False,
        statistic=None, parallel=False):
    '''
    wrapper around aopy.analysis.compare_conditions_bootstrap_spatial_corr for two conditions
    '''
    elec_pos, acq_ch, _ = aopy.data.load_chmap()
    (observed_dists, conditions, observed_corr, observed_dprime, shuff_dists_dist, shuff_corr_dist, 
     shuff_dprime_dist) = compare_conditions_bootstrap_spatial_corr(
        df, elec_pos, labels, n_trials=n_trials, n_bootstraps=n_bootstraps, n_shuffle=n_shuffle, 
        statistics=statistic, rng=None, parallel=parallel
    )
    mean1 = statistic(df.iloc[labels == conditions[0]].reset_index(drop=True))
    mean2 = statistic(df.iloc[labels == conditions[1]].reset_index(drop=True))
    dist1 = observed_dists[0]
    dist2 = observed_dists[1]
    
    return mean1, mean2, dist1, dist2, observed_corr, observed_dprime, shuff_dprime_dist
    
from aopy.data.bmi3d import tabulate_ts_data
def filter_stim_trials(df, gain_range=None, power_range=None, min_isi=None, debug=True):
    
    # Check ISI
    isi = df['trial_time'].to_numpy()[1:]-df['trial_time'].to_numpy()[:len(df)-1]
    isi[isi < 0] = np.nan
    isi = np.append(isi, 1)
    df['isi'] = isi

    correct_isi = np.ones(len(df), dtype='bool')
    if min_isi is not None:
        correct_isi = df['isi'] > min_isi
    
    # Filter stimulation parameters
    trial_gain = df['trial_gain']
    trial_power = df['trial_power']
    if debug:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.hist(trial_gain)
        plt.xlabel('laser gain')
        plt.subplot(1,3,2)
        plt.hist(trial_power)
        plt.xlabel('laser power (mW)')
        plt.subplot(1,3,3)
        plt.hist(1000*isi, 50)
        plt.xlabel('isi (ms)')
        plt.xlim(0,1000)
        
    # Filter gain and power
    correct_gain = np.ones(len(df), dtype='bool')
    if gain_range is not None:
        correct_gain = (trial_gain >= gain_range[0]) & (trial_gain < gain_range[1])
    correct_power = np.ones(len(df), dtype='bool')
    if power_range is not None:
        correct_power = (trial_power >= power_range[0]) & (trial_power < power_range[1])
    
    return correct_isi & correct_gain & correct_power
    
    
def get_bad_stim_trials(df, time_before=0.25, time_after=0, sd_thr=5, ch_frac=0.1, 
                        channels=None, debug=False, 
                        preproc_dir='/data/preprocessed'):
    if channels is None:
        _, acq_ch, _ = aopy.data.load_chmap()
        channels = acq_ch - 1
    
    # Load data
    erp, samplerate = tabulate_ts_data(preproc_dir, df['subject'], df['te_id'], 
                                       df['date'], df['trial_time'], 
                                       time_before, time_after, channels=channels)
    time = np.arange(len(erp))/samplerate
    
    # Filter trials with no change in voltage across all channels
    no_data_trials = np.all(np.sum(np.diff(erp, axis=0), axis=0) == 0, axis=0)

    # Filter outlier RMS over the whole trial
    rms = aopy.analysis.calc_rms(erp)
    rms = np.expand_dims(rms, 0)
    trial_outliers = preproc.quality.detect_bad_trials(rms, sd_thr=sd_thr, 
                                                       ch_frac=ch_frac, debug=debug)
        
    # TODO: remove stim response outliers
    # e.g. outliers in max_erp
    
    return trial_outliers | no_data_trials

def calc_slic_map(df, stim_site=None, time_before=0.25, time_after=0.25,
                  taper_len=0.06, parallel=True, debug=False, verbose=True, 
                  diff=True, return_angle=False):
    '''
    if stim_site is None then find in the df
    taper_len is the window_len
    full window is (-taper_len, taper_len) but returns window[1]-window[0] as the SLIC
    '''
    _, acq_ch, _ = aopy.data.load_chmap()
    
    bad_trials = get_bad_stim_trials(df, channels=acq_ch-1, debug=debug)
    if verbose:
        print(f"removed {np.sum(bad_trials)} bad trials")
    erp, samplerate = load_stim_erp(df[~bad_trials].reset_index(drop=True), 
                                    time_before, time_after, channels=acq_ch-1)
    
    if stim_site is None:
        stim_site = np.unique(df['stimulation_site']).astype(int)[0]
    ch_near_stim = get_acq_ch_near_stimulation_site(stim_site)
    stim_ch = np.where(np.isin(acq_ch, ch_near_stim))[0]
    
    freqs, time, coh_all, angle_all = calc_connectivity_map_coh(
        erp, samplerate, time_before, time_after, stim_ch, 
        window=(-taper_len,taper_len), n=taper_len, step=taper_len, 
        parallel=parallel, verbose=verbose
    )
    
    # Take the difference between post- and pre- stimulation
    if return_angle and diff:
        return freqs, time[1:], coh_all[:,[1],:]-coh_all[:,[0],:], angle_all[:,[1],:]
    elif return_angle:
        return freqs, time, coh_all, angle_all
    elif diff:
        return freqs, time[1:], coh_all[:,[1],:]-coh_all[:,[0],:]
    else:
        return freqs, time, coh_all
    

def calc_null_maps(df_null, n_reps, n_trials, mapping_fn, parallel=False, **kwargs):
    '''
    '''
    null_trial_idx = 0
    pool = False
    if parallel:
        pool = mp.Pool(mp.cpu_count()//2)
    null_data = []
    for n in tqdm(range(n_reps)):

        df_null_sub = df_null.iloc[null_trial_idx:null_trial_idx+n_trials].reset_index()
        null_trial_idx += n_trials

        null_data.append(mapping_fn(df_null_sub, pool, **kwargs))
        
    if parallel:
        pool.close()
    return null_data


def calc_null_slic_maps(df_null, n_reps, stim_site=None, n_trials=None, like=None,
                       time_before=0.25, time_after=0.25, taper_len=0.06, diff=True):
    '''
    wrapper for slic. specify n_trials and stim_site or provide a dataframe in 'like'
    '''
    if stim_site is None:
        stim_site = np.unique(like['stimulation_site']).astype(int)[0]
    if n_trials is None:
        n_trials = len(like)

    mapping_fn = lambda df, pool: calc_slic_map(df, stim_site=stim_site, time_before=time_before, 
                                                time_after=time_after, taper_len=taper_len, 
                                                parallel=pool, diff=diff)
    maps = calc_null_maps(df_null, n_reps, n_trials, mapping_fn, parallel=True)
    
    freqs_, time_, maps = zip(*maps)
    return freqs_[0], time_[0], maps


def extract_band_across_maps(freqs, time, coh_maps, band=(12,150), window=(-np.inf, np.inf)):
    
    coh_band = [calc_tfr_mean(freqs, time, coh_map, band=band, window=window)
                for coh_map in coh_maps]
    return coh_band

# to be deleted if not needed
# def extract_coh_band_days(long_data, band=(12,150), window=(-np.inf, np.inf)):
    
#     coh_band_sites = []
#     null_band_sites = []
#     diff_sites = []
#     p_sites = []
#     gc_band_sites = []
#     for idx in range(len(long_data['sites'])):
#         coh_band_days = []
#         null_band_days = []
#         diff_days = []
#         p_days = []
#         gc_band_days = []
#         for d in range(len(long_data['coh_map'][idx])):
#             null_band, coh_band, diff, p = calc_tfr_mean_statistics(long_data['freqs'], long_data['time'], 
#                                                                 long_data['coh_map'][idx][d], long_data['null_maps'][idx],
#                                                                 band=band, window=window)
#             gc_band = calc_tfr_mean(long_data['freqs'], long_data['time'][1:], long_data['gc_map'][idx][d],
#                                 band=band, window=window)
            
#             coh_band_days.append(coh_band)
#             null_band_days.append(null_band)
#             diff_days.append(diff)
#             p_days.append(p)
#             gc_band_days.append(gc_band)
        
#         coh_band_sites.append(coh_band_days)
#         null_band_sites.append(null_band_days)
#         diff_sites.append(diff_days)
#         p_sites.append(p_days)
#         gc_band_sites.append(gc_band_days)
        
#     return coh_band_sites, null_band_sites, diff_sites, p_sites, gc_band_sites

def calc_rolling_coh_trials(df, win_size, step=None, time_before=0.25, time_after=0.25, **kwargs):
    stim_sites = np.unique(df['stimulation_site'])    
    if len(stim_sites) > 1:
        print(stim_sites)
        raise ValueError("Too many stim sites in recordings!")
    stim_site = int(stim_sites[0])
    print(f"Stim site: {stim_site}")

    pool = mp.Pool(mp.cpu_count()//2)
    
    if step is None:
        step = int(win_size / 2)
    
    ntr = len(df)
    nwin = 1 + int(np.floor((ntr - win_size) / int(step))) # no partial windows
    
    coh_trials = []
    for w_idx in tqdm(range(nwin)):
        
        df_sub = df[w_idx * step:w_idx * step+win_size].reset_index()
        freqs, time, coh_all = calc_slic_map(df_sub, parallel=pool, **kwargs)
        coh_trials.append(coh_all)
        
    pool.close()
        
    return freqs, time, coh_trials

#########
# erp
#######
def calc_opto_resp(df, stim_site=None, time_before=0.25, time_after=0.25, debug=False):
    elec_pos, acq_ch, _ = aopy.data.load_chmap()
    
    if stim_site is None:
        stim_site = np.unique(df['stimulation_site']).astype(int)[0]

    bad_trials = get_bad_stim_trials(df, channels=acq_ch-1, debug=debug)
    # print(f"removed {np.sum(bad_trials)} bad trials")
    erp, samplerate = load_stim_erp(df[~bad_trials].reset_index(drop=True), 
                                    time_before, time_after, channels=acq_ch-1)
            
    altcond_window = (0, time_after)
    nullcond_window = (-time_before, 0)
    altcond, nullcond = aopy.analysis.latency.prepare_erp(erp, erp, samplerate, 
                                                          time_before, time_after, 
                                                          nullcond_window, altcond_window)[:2]

    z_erp = np.std(nullcond, axis=0)
    sd_erp = (altcond - np.mean(nullcond, axis=0)) / z_erp
    max_erp = aopy.analysis.get_max_erp(sd_erp, 0, time_after, samplerate, trial_average=True)

    return max_erp

    
def calc_null_opto_resp(df_null, n_reps, stim_site=None, n_trials=None, like=None,
                        time_before=0.25, time_after=0.25):
    '''
    wrapper for slic. specify n_trials and stim_site or provide a dataframe in 'like'
    '''
    if stim_site is None:
        stim_site = np.unique(like['stimulation_site']).astype(int)[0]
    if n_trials is None:
        n_trials = len(like)

    mapping_fn = lambda df, pool: calc_opto_resp(df, stim_site=stim_site,
                                                 time_before=time_before, 
                                                 time_after=time_after)
    maps = calc_null_maps(df_null, n_reps, n_trials, mapping_fn)
    return maps

# Quantify significant connections
def count_significant_connections(p_sites, alpha=0.01):
    
    counts = []
    for idx in range(len(p_sites)):
        p = p_sites[idx]
        counts.append(np.sum(p<=alpha))

    return counts

###############################################
# latency
###############################################
def convert_latency_to_serr(stim_site, latency_itpc, max_itpc, p, alpha=0.05, latency_cutoff=0.018):
    _, acq_ch, _ = aodata.load_chmap()
    ch_near_stim = analysis.connectivity.get_acq_ch_near_stimulation_site(stim_site)
    stim_ch = np.where(np.isin(acq_ch, ch_near_stim))[0]

    connected_resp = (latency_itpc > latency_cutoff) & (p < alpha)
    norm = np.nanmean(max_itpc[stim_ch])
    conn = max_itpc/norm
    conn[~connected_resp] = 0.
    return conn

def plot_itpc_latency_map(max_itpc, latency_itpc, stimulation_site, theta, grid_size, clim=(15, 21), 
                          alpha=0.05, colorbar=True, fontsize=12, color='w', ax=None):
    if ax is None:
        ax = plt.gca()
    
    elec_pos, acq_ch, elecs = aopy.data.load_chmap(theta=theta)
    ax.set_facecolor('black')
    alpha_map, _ = aopy.visualization.calc_data_map(max_itpc, elec_pos[:,0], elec_pos[:,1], grid_size, interp_method='cubic')
    data_map, xy = aopy.visualization.calc_data_map(latency_itpc*1000, elec_pos[:,0], elec_pos[:,1], grid_size, interp_method='cubic')
    im = aopy.visualization.plot_spatial_map(data_map, xy[0], xy[1], alpha_map=alpha_map, clim=clim, cmap='spring', ax=ax)
    im.set_clim(clim)

    if colorbar:
        pcm = plt.colorbar(im, shrink=0.7, ax=ax)
    else:
        ax.axis("off")
        ax.add_artist(ax.patch)
        ax.patch.set_zorder(-1)

    aopy.visualization.annotate_spatial_map_channels(acq_ch=[stimulation_site], theta=theta,
                                                     fontsize=fontsize, color=color, drive_type='Opto32', ax=ax)

from statsmodels.stats.multitest import fdrcorrection  
from scipy.stats import chisquare
from scipy.stats import kstest, uniform
def calc_latency(df, time_before=0.25, time_after=0.25, band=[50, 200], taper_len=0.03, window_len=0.16, debug=False):
    
    subject = df['subject'][0]
    if subject == 'affi':
        theta = 90
    else:
        theta = 0
    elec_pos, acq_ch, elecs = aopy.data.load_chmap(theta=theta)
    bad_trials = get_bad_stim_trials(df, channels=acq_ch-1, debug=debug)
    erp, samplerate = load_stim_erp(df[~bad_trials].reset_index(drop=True), 
                                    time_before, time_after, channels=acq_ch-1)
    
    filt_data_im = aopy.precondition.mt_bandpass_filter(erp.reshape(len(erp),-1), band, taper_len, samplerate, complex_output=True)
    filt_data_im = np.reshape(filt_data_im, erp.shape)
    time = np.arange(len(erp))/samplerate - time_before
    filt_data_im = aopy.analysis.subtract_erp_baseline(filt_data_im, time, -taper_len-window_len, -taper_len)
    
    altcond_window = (-taper_len/2, window_len - taper_len/2)
    nullcond_window = (-taper_len - window_len, -taper_len)
    altcond_im, nullcond_im = aopy.analysis.latency.prepare_erp(filt_data_im, filt_data_im, samplerate, time_before, time_after, nullcond_window, altcond_window)[:2]
    
    # Compute max ERP in units of baseline SD
    sd_erp = (np.abs(altcond_im) - np.mean(np.abs(nullcond_im), axis=0)) / np.std(np.abs(nullcond_im), axis=0)
    max_erp = aopy.analysis.get_max_erp(sd_erp, 0, window_len, samplerate, trial_average=True)
    latency_itpc = np.argmax(calc_itpc(altcond_im), axis=0).astype(float)/samplerate
    max_itpc = np.max(calc_itpc(altcond_im), axis=0)
    null_itpc = np.max(calc_itpc(nullcond_im), axis=0)

    return max_itpc, latency_itpc


def calc_null_latency_maps(df_null, n_reps, n_trials=None, like=None, time_before=0.25, time_after=0.25, 
                           band=[50, 200], taper_len=0.03, window_len=0.16):
    '''
    wrapper for latency. specify n_trials or provide a dataframe in 'like'
    '''
    if n_trials is None:
        n_trials = len(like)

    mapping_fn = lambda df, pool: calc_latency(df, time_before=time_before, 
                                               time_after=time_after, band=band, taper_len=taper_len, 
                                               window_len=window_len)
    maps = calc_null_maps(df_null, n_reps, n_trials, mapping_fn)
    
    max_itpc, latency_itpc = zip(*maps)
    return max_itpc, latency_itpc

def calc_latency_accllr(df, time_before=0.25, time_after=0.25, 
                        band=[50, 200], taper_len=0.03, window_len=0.16, 
                        alpha=0.05, selectivity_cutoff=0.05, 
                        latency_cutoff=None, debug=False, 
                        verbose=True, parallel=True):
    subject = df['subject'][0]
    if subject == 'affi':
        theta = 90
    else:
        theta = 0
        
    if latency_cutoff is None:
        latency_cutoff = 0.003 + taper_len/2
    chamber = df['drmap_chamber'][0]
    drive_type = df['drmap_drive_type'][0]
    elec_pos, acq_ch, elecs = aopy.data.load_chmap(drive_type, theta=theta)
    stimulation_site = int(df['stimulation_site'][0])
    
    bad_trials = get_bad_stim_trials(df, channels=acq_ch-1, debug=debug)
    if verbose:
        print(f"removed {np.sum(bad_trials)} bad trials")
    erp, samplerate = load_stim_erp(df[~bad_trials].reset_index(drop=True), 
                                    time_before, time_after, channels=acq_ch-1)
    
    filt_data = np.abs(aopy.precondition.mt_bandpass_filter(erp.reshape(len(erp),-1), band, taper_len, samplerate, complex_output=True))
    filt_data = np.reshape(filt_data, erp.shape)
    time = np.arange(len(erp))/samplerate - time_before
    filt_data = aopy.analysis.subtract_erp_baseline(filt_data, time, -taper_len-window_len, -taper_len)
    
    if parallel is True:
        pool = mp.Pool(mp.cpu_count()//2)
    elif parallel:
        pool = parallel
    else:
        pool = False
    altcond_window = (-taper_len/2, window_len - taper_len/2)
    nullcond_window = (-taper_len - window_len, -taper_len)
    altcond, nullcond = aopy.analysis.latency.prepare_erp(filt_data, filt_data, samplerate, time_before, time_after, nullcond_window, altcond_window)[:2]
    st, auc, se, p = aopy.analysis.latency.calc_accllr_st(
        altcond, nullcond, 
        altcond, nullcond,
        'lfp', 1./samplerate, nlevels=None, match_selectivity=False, 
        match_ch=None, noise_sd_step=5, parallel=pool, verbose=verbose
    )
    
    # Select channels
    selectivity_ch = auc - 0.5 > selectivity_cutoff
    alpha_ch = (p < alpha)
    match_ch = alpha_ch & selectivity_ch

    # Re-run AccLLR with selectivity matching to remove amplitude bias from latency estimates
    st, _, _, _ = aopy.analysis.latency.calc_accllr_st(
        altcond.copy(), nullcond.copy(), 
        altcond.copy(), nullcond.copy(),
        'lfp', 1./samplerate, nlevels=None, match_selectivity=True, 
        match_ch=match_ch, noise_sd_step=5, parallel=pool, verbose=verbose
    )
    if parallel is True:
        pool.close()

    # Convert to connectivity
    latency = np.nanmedian(st, axis=1)

    # Convert to connectivity
    max_erp = aopy.analysis.get_max_erp(filt_data, time_before, time_after, samplerate, max_search_window=None, trial_average=True)
    serr = convert_latency_to_serr(stimulation_site, latency, max_erp, p, alpha=0.05, latency_cutoff=latency_cutoff)
    
    return serr, latency, p, max_erp, st, auc, se

###############################################
# granger prediction
###############################################
from aopy.analysis import accllr
from aopy import precondition
from aopy.analysis import calc_mt_tfr

from spectral_connectivity import Multitaper, Connectivity

# Must install spectral_connectivity from github.com/leoscholl/spectral_connectivity
# 
# OR use this commented out code
#
# from spectral_connectivity.connectivity import _complex_inner_product, _nonsorted_unique
# from spectral_connectivity.connectivity import _estimate_transfer_function, _remove_instantaneous_causality, _estimate_noise_covariance, _estimate_predictive_power
# from spectral_connectivity.minimum_phase_decomposition import (
#     minimum_phase_decomposition,
# )
# def pairwise_spectral_granger_prediction(c, pairs):
#     """The amount of power at a node in a frequency explained by (is
#     predictive of) the power at other nodes.

#     Also known as spectral granger causality.

#     References
#     ----------
#     .. [1] Geweke, J. (1982). Measurement of Linear Dependence and
#            Feedback Between Multiple Time Series. Journal of the
#            American Statistical Association 77, 304.

#     """
#     pairs = np.array(pairs)
    
#     fourier_coefficients = c.fourier_coefficients[..., np.newaxis]
#     fourier_coefficients = fourier_coefficients.astype(c._dtype)

#     # get unique indices
#     _sxu = _nonsorted_unique(pairs[:, 0])
#     _syu = _nonsorted_unique(pairs[:, 1])

#     # compute subset of connections
#     csm_shape = list(c._power.shape)
#     csm_shape += [csm_shape[-1]]
#     dtype = c._dtype
#     csm = np.zeros(csm_shape, dtype=dtype)

#     # compute forward connections
#     _out = c._expectation(
#         _complex_inner_product(
#             fourier_coefficients[..., _sxu, :],
#             fourier_coefficients[..., _syu, :],
#             dtype=c._dtype,
#         )
#     )
#     csm[..., _sxu.reshape(-1, 1), _syu.reshape(1, -1)] = _out

#     # compute backward connections
#     _out = c._expectation(
#         _complex_inner_product(
#             fourier_coefficients[..., _syu, :],
#             fourier_coefficients[..., _sxu, :],
#             dtype=c._dtype,
#         )
#     )
#     csm[..., _syu.reshape(-1, 1), _sxu.reshape(1, -1)] = _out

#     # compute diagonals
#     for x in _syu:
#         diag_out = c._expectation(
#             _complex_inner_product(
#                 fourier_coefficients[..., [x], :],
#                 fourier_coefficients[..., [x], :],
#                 dtype=c._dtype,
#             )
#         )
#         csm[..., [x], [x]] = diag_out[..., 0]
#     for x in _sxu:
#         diag_out = c._expectation(
#             _complex_inner_product(
#                 fourier_coefficients[..., [x], :],
#                 fourier_coefficients[..., [x], :],
#                 dtype=c._dtype,
#             )
#         )
#         csm[..., [x], [x]] = diag_out[..., 0]
    
#     total_power = c._power
#     n_frequencies = total_power.shape[-2]
#     non_neg_index = np.arange(0, (n_frequencies + 1) // 2)
#     total_power = np.take(total_power, indices=non_neg_index, axis=-2)

#     n_frequencies = csm.shape[-3]
#     new_shape = list(csm.shape)
#     new_shape[-3] = non_neg_index.size
#     predictive_power = np.empty(new_shape)

#     for pair_indices in pairs:
#         pair_indices = np.array(pair_indices)[:, np.newaxis]
#         try:
#             minimum_phase_factor = minimum_phase_decomposition(
#                 csm[..., pair_indices, pair_indices.T]
#             )
#             transfer_function = _estimate_transfer_function(minimum_phase_factor)[
#                 ..., non_neg_index, :, :
#             ]
#             rotated_covariance = _remove_instantaneous_causality(
#                 _estimate_noise_covariance(minimum_phase_factor)
#             )
#             predictive_power[
#                 ..., pair_indices, pair_indices.T
#             ] = _estimate_predictive_power(
#                 total_power[..., pair_indices[:, 0]],
#                 rotated_covariance,
#                 transfer_function,
#             )
#         except np.linalg.LinAlgError:
#             predictive_power[..., pair_indices, pair_indices.T] = np.nan

#     n_signals = csm.shape[-1]
#     diagonal_ind = np.diag_indices(n_signals)
#     predictive_power[..., diagonal_ind[0], diagonal_ind[1]] = np.nan

#     return predictive_power

def calc_variability(df, stim_site=None, time_before=0.25, time_after=0.25,
                              band=[50,200], taper_len=0.03, debug=False):
    elec_pos, acq_ch, _ = aopy.data.load_chmap()

    if stim_site is None:
        stim_site = np.unique(df['stimulation_site']).astype(int)[0]

    bad_trials = get_bad_stim_trials(df, channels=acq_ch-1, debug=debug)
    print(f"removed {np.sum(bad_trials)} bad trials")

    ch_near_stim = get_acq_ch_near_stimulation_site(stim_site)
    erp, samplerate = load_stim_erp(df[~bad_trials].reset_index(drop=True), 
                                    time_before, time_after, channels=ch_near_stim-1)

    filt_data = np.abs(aopy.precondition.mt_bandpass_filter(erp.reshape(len(erp),-1), band, taper_len, samplerate, complex_output=True))
    filt_data = np.reshape(filt_data, erp.shape)

    time_var = np.std(filt_data, axis=0) # std over time
    var = np.mean(time_var) # mean over trials and channels
    return var


import functools
@functools.lru_cache(maxsize=50)
def calc_fft_coeff_gc(erp_bytes, shape, dtype, samplerate, time_before, time_after, window, n, bw, verbose):
    '''
    Returns:
        freqs (nfreq):
        time (nt):
        coeff (nt, ntr, ntaper, nfreq, nch): fourier coefficients for spectral connectivity
    '''
    erp = np.frombuffer(erp_bytes, dtype=dtype).reshape(shape)

    n, p, k = precondition.convert_taper_parameters(n, bw)

    if verbose:
        print(f"using {k} tapers")
    
    if window is None:
        window = (0, n)
    nullcond_window = (-n, 0)
    data_altcond = prepare_erp(
        erp, samplerate, time_before, time_after, nullcond_window, window, 
        zscore=True, ref=True
    )
    
    # Calculate Fourier coefficients using aopy
    f, t, coeff = calc_mt_tfr(data_altcond, n, p, k, samplerate, step=n, fk=250, complex_output=True,
                             dtype='complex128', nonnegative_freqs=False)
    coeff = np.expand_dims(coeff.transpose(1,3,0,2), 2)
    return f, t+window[0], coeff

def calc_gc(f, t, coeff, stim_ch=None):
    # Test granger causality on the stim and null map data
    import os
    if ('SPECTRAL_CONNECTIVITY_ENABLE_GPU' in os.environ and 
        os.environ['SPECTRAL_CONNECTIVITY_ENABLE_GPU'] == 'true'):
        import cupy
        coeff = cupy.array(coeff)
    c = Connectivity(fourier_coefficients=coeff, frequencies=f, time=t)

    if stim_ch is not None:
        pairs = []
        for ch in range(coeff.shape[-1]):
            for sch in stim_ch:
                pairs.append((sch, ch))
        gc = c.subset_pairwise_spectral_granger_prediction(pairs)
    else:
        gc = c.pairwise_spectral_granger_prediction()
        return c.frequencies, c.time, gc.transpose(1,0,2,3) # nfreq, nt, nch, nch
    
    gc_mean = []
    for ch in range(coeff.shape[-1]):
        this_mean = []
        for sch in stim_ch:
            this_mean.append(gc[:,:,ch,sch])
        gc_mean.append(np.nanmean(this_mean, axis=0))
    gc_mean = np.array(gc_mean)

    return c.frequencies, c.time, gc_mean.transpose(2,1,0) # nfreq, nt, nch

import logging
from spectral_connectivity import minimum_phase_decomposition
def calc_connectivity_map_gc(erp, samplerate, time_before, time_after, stim_ch=None, 
                             window=None, n=0.06, bw=25, verbose=True):
    '''
    map of (n_stim_ch, nch) mean coherence
    input 0-indexed stim_ch
    
    Args:
        erp (nt, nch, ntrial)
        stim_ch (list of idx)
    
    Returns:
        freqs (nfreq)
        time (nt)
        coh (nfreq, nt, nch)
    '''
    logger = logging.getLogger(minimum_phase_decomposition.__name__)
    if verbose:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.ERROR)
        
    f, t, coeff = calc_fft_coeff_gc(erp.tobytes(), erp.shape, erp.dtype, float(samplerate), 
                                    time_before, time_after, 
                                    window, n, bw, verbose=verbose)
    return calc_gc(f, t, coeff, stim_ch=stim_ch)

def calc_gc_map(df, stim_site=None, time_before=0.25, time_after=0.25, debug=False, verbose=True):
    '''
    use granger prediction to calculate a map of connectivity
    '''
    elec_pos, acq_ch, elecs = aopy.data.load_chmap()
    
    if stim_site is None:
        stim_site = np.unique(df['stimulation_site']).astype(int)[0]

    bad_trials = get_bad_stim_trials(df, channels=acq_ch-1, debug=debug)
    if verbose:
        print(f"removed {np.sum(bad_trials)} bad trials")
    erp, samplerate = load_stim_erp(df[~bad_trials].reset_index(drop=True), 
                                    time_before, time_after, channels=acq_ch-1)

    ch_near_stim = get_acq_ch_near_stimulation_site(stim_site)
    stim_ch = np.where(np.isin(acq_ch, ch_near_stim))[0]
    
    return calc_connectivity_map_gc(erp, samplerate, time_before, time_after, stim_ch=stim_ch, verbose=verbose)

def calc_null_gc_maps(df, n_reps, stim_site=None, like=None, 
                      time_before=0.25, time_after=0.25, debug=False):
    '''
    calculate granger prediction multiple times on time-shuffled versions of the data
    to generate a null distribution of granger values
    
    Note: super inefficient, should only calculate fft once per shuffle and then compute
    all the gc predictions on those
    
    Note 2: maybe fixed this inefficiency by using a cache on the fft coefficients and a seed for the
    shuffling here.. not sure if it works yet.
    '''
    elec_pos, acq_ch, elecs = aopy.data.load_chmap()

    if like is not None:
        stim_site = np.unique(like['stimulation_site']).astype(int)[0]
    elif stim_site is None:
        stim_site = np.unique(df['stimulation_site']).astype(int)[0]
        
    bad_trials = get_bad_stim_trials(df, channels=acq_ch-1, debug=debug)
    erp, samplerate = load_stim_erp(df[~bad_trials].reset_index(drop=True), 
                                    time_before, time_after, channels=acq_ch-1)

    ch_near_stim = get_acq_ch_near_stimulation_site(stim_site)
    stim_ch = np.where(np.isin(acq_ch, ch_near_stim))[0]
    
    rng = np.random.default_rng(seed=5)

    null_gc = []
    for n in tqdm(range(n_reps)):

        shuffled_order = rng.permutation(len(erp)) # shuffled time index
        freqs, time, gc = calc_connectivity_map_gc(erp[shuffled_order], samplerate, time_before, time_after, stim_ch=stim_ch)
        null_gc.append(gc)
        
    return freqs, time, null_gc


def slic_statistic(data, stim_site, band, window, parallel, verbose):
    freqs, time, coh_all = calc_slic_map(data, stim_site=stim_site, parallel=parallel, verbose=verbose)
    if band is None or window is None:
        return coh_all
    return calc_tfr_mean(freqs, time, coh_all, band=band, window=window)

def granger_statistic(data, stim_site, band, window, verbose):
    freqs, time, gc_all = calc_gc_map(data, stim_site=stim_site, verbose=verbose)
    if band is None or window is None:
        return gc_all
    return calc_tfr_mean(freqs, time, gc_all, band=band, window=window)

def accllr_statistic(data, band, parallel, verbose):
    serr, latency, p, max_erp, st, auc, se = calc_latency_accllr(data, band=band, 
                                                                 parallel=parallel, verbose=verbose)
    return serr

from functools import partial

def get_connectivity_statistic(stim_site, method='slic', band=(12,150), window=(0,1), parallel=False, verbose=False):
    if method == 'slic':
        return partial(slic_statistic, stim_site=stim_site, band=band, 
                       window=window, parallel=parallel, verbose=verbose)
    elif method == 'granger':
        return partial(granger_statistic, stim_site=stim_site, band=band, 
                       window=window, verbose=verbose)
    elif method == 'accllr':
        return partial(accllr_statistic, band=band, parallel=parallel, verbose=verbose)
    else:
        raise ValueError(f"Unsupported method: {method}")
        
def convert_days_to_implant(days, max_gap=5):
    try:
        chg = [0 if td.days < max_gap else 1 for td in np.diff(days)]
    except:
        chg = [np.diff(days) >= max_gap]
    implant = np.insert(np.cumsum(chg), 0, 0)
    return implant

def get_open_closed_mask(preproc_dir, df, open_ratio=0.5, closed_ratio=0.8, time_before=0.5,
                        time_after=0.5, samplerate=1000):
    eye_data = aopy.data.tabulate_kinematic_data(preproc_dir, df['subject'], df['te_id'], df['date'], df['trial_time']-time_before, df['trial_time']+time_after, datatype='eye_raw', samplerate=samplerate)
    exp_data, exp_metadata = aopy.data.load_preproc_exp_data(preproc_dir, df['subject'][0],
                                                         df['te_id'][0], df['date'][0])
    data_mean = []
    eye_closed = []
    for trial in eye_data:
        mask = aopy.preproc.oculomatic.detect_noise(trial/exp_metadata['analog_voltsperbit'], samplerate, min_step=1, step_thr=3, t_closed_min=0.01)
        eye_closed.append(np.mean(np.all(mask, axis=1)))
        data_mean.append(np.mean(trial))

    eyes_closed_trials = np.array(eye_closed) > closed_ratio
    eyes_open_trials = np.array(eye_closed) < open_ratio

    assert len(eyes_closed_trials) == len(df)
    assert len(eyes_open_trials) == len(df)
    
    print(f"{np.sum(eyes_open_trials)} open, {np.sum(eyes_closed_trials)} closed ({len(eyes_closed_trials)} total)")
    
    return eyes_open_trials, eyes_closed_trials


def calc_volume(erp_maps, erp_null_maps, alpha=0.01):
    
    volume = [np.array(0) for _ in range(len(erp_maps))]
    null_volume = [np.array(0) for _ in range(len(erp_maps))]
    for idx in range(len(erp_maps)):
        max_erp_ = erp_maps[idx]
        null_max_erp_ = erp_null_maps[idx]
        volume[idx] = abs(np.mean(max_erp_))
        null_volume[idx] = [abs(np.mean(data)) for data in null_max_erp_]

    volume, p = calc_fdrc_ranktest(volume, np.array(null_volume).T, alpha=alpha)
    return volume, p