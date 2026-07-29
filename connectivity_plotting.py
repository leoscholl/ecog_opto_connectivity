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
from matplotlib import colors as mpcolors

from aopy.visualization import annotate_spatial_map_channels, place_Opto32_subplots, plot_angles
from aopy.data.bmi3d import tabulate_ts_data
from aopy.preproc.bmi3d import get_laser_trial_times
from aopy.preproc.quality import detect_bad_trials
from aopy.analysis.connectivity import get_acq_ch_near_stimulation_site, calc_connectivity_map_coh, prepare_erp
from aopy.analysis import calc_itpc, calc_fdrc_ranktest, calc_tfr_mean_fdrc_ranktest, calc_spatial_data_correlation
from aopy.analysis.latency import detect_itpc_response
from aopy.visualization import overlay_sulci_on_spatial_map, plot_xy_scalebar, plot_tf_map_grid

color_monkey_1 = '#90d0df'
color_monkey_2 = '#227fb1'
color_accllr = '#ead292'
color_accllr_text = '#dfbd5c'
color_slic = '#c2e4d7'
color_slic_text = '#7fc6aa'
color_granger = '#bab0d9'
color_granger_text = '#9d8ec9'
color_granger_stim = '#f2aade'
color_granger_stim_text = '#e873c9'
cmap_accllr = sns.blend_palette(['black', color_accllr], n_colors=100, as_cmap=True)
cmap_slic = sns.blend_palette(['black', color_slic], n_colors=100, as_cmap=True)
cmap_granger = sns.blend_palette(['black', color_granger], n_colors=100, as_cmap=True)
cmap_granger_stim = sns.blend_palette(['black', color_granger_stim], n_colors=100, as_cmap=True)
cmap_accllr_inv = sns.blend_palette(['white', color_accllr], n_colors=100, as_cmap=True)
cmap_slic_inv = sns.blend_palette(['white', color_slic], n_colors=100, as_cmap=True)
cmap_granger_inv = sns.blend_palette(['white', color_granger], n_colors=100, as_cmap=True)
cmap_granger_stim_inv = sns.blend_palette(['white', color_granger_stim], n_colors=100, as_cmap=True)

from connectivity_analysis import *

def ordinaltg(n):
    return str(n) + {1: 'st', 2: 'nd', 3: 'rd'}.get(4 if 10 <= n % 100 < 20 else n % 10, "th")
def plot_correlation_matrix(maps, days, ax=None):
    if ax is None:
        plt.figure(figsize=(2,2))
        ax = plt.gca()
        
    elec_pos, acq_ch, elecs = aopy.data.load_chmap()
    ncc, shifts = calc_spatial_data_correlation(maps, elec_pos, interp=True, grid_size=(16,16), interp_method='linear')
    ncc[np.triu_indices(ncc.shape[0], k=1)] = np.nan

    im = ax.imshow(ncc, cmap='Grays', vmin=0., vmax=1)
    clb = plt.colorbar(im, shrink=0.8, ax=ax)
    clb.set_ticks([0.0,1.0])

    # Set x and y ticks to label new implant
    implant = convert_days_to_implant(days)
    reps, chg_idx = aopy.utils.count_repetitions(implant)
    # chg_idx = np.insert(chg_idx[1:]+1, 0, 0) # TODO may need to label the next day
    ax.set_xticks(chg_idx)
    ax.set_yticks(chg_idx)
    ax.set_xticklabels([ordinaltg(n) for n in implant[chg_idx]+1])
    ax.set_yticklabels([ordinaltg(n) for n in implant[chg_idx]+1])

    sns.despine(ax=ax)

    
def animate_coh_band(freqs, time, coh_trials, samplerate, cmap, clim, grid_size, 
                     null_maps=None, band=(12,150), window=(-np.inf, np.inf), theta=0,
                    alpha=0.05):
    elec_pos, acq_ch, elecs = aopy.data.load_chmap(theta=theta)
    
    maps = []
    for coh_all in coh_trials:
        coh_band = calc_fdrc_ranktest(freqs, time, coh_all, band, window)
        if null_maps is not None:
            diff, p = calc_tfr_mean_fdrc_ranktest(freqs, time, coh_all, null_maps, band=band, window=window)
            coh_band[p > alpha] = 0.
        data_map, xy = aopy.visualization.calc_data_map(coh_band, elec_pos[:,0], elec_pos[:,1], grid_size, interp_method='cubic')
        maps.append(data_map)
    
    return aopy.visualization.animate_spatial_map(maps, elec_pos[:,0], elec_pos[:,1], samplerate, cmap, clim)

def plot_erp(erp, time_before, time_after, samplerate, subject, stim_site, theta=0):
    elec_pos, acq_ch, elecs = aopy.data.load_chmap(theta=theta)
    max_erp = aopy.analysis.get_max_erp(erp, time_before, time_after, samplerate, max_search_window=None, trial_average=True)
    max_erp_data = np.zeros(256,)
    max_erp_data[acq_ch-1] = max_erp
    im = aopy.visualization.plot_ECoG244_data_map(max_erp_data, theta=theta)
    im.set_clim(-500,500)
    plt.gca().set(xticks=[], yticks=[], xticklabels=[], yticklabels=[], xlabel='', ylabel='') 
    overlay_sulci_on_spatial_map(subject, 'lm1', 'ECoG244', theta=theta, color='k')
    aopy.visualization.annotate_spatial_map_channels(acq_ch=[stim_site], 
                                                     fontsize=12, color='k', drive_type='Opto32', theta=theta)

def plot_stim_erp_time(erp_ch, time_before, time_after, samplerate, stim_site, subject, theta=0, clim=(-5,5), latency=None, ax=None):
    if ax is None:
        ax = plt.gca()
    time = np.arange(len(erp_ch))/samplerate - time_before
    max_erp = analysis.get_max_erp(np.expand_dims(erp_ch,1), time_before, time_after, samplerate, trial_average=False)
    if latency is not None:
        sort = np.argsort(latency)
    else:
        sort = np.arange(erp_ch.shape[-1]) #np.argsort(np.mean(max_erp, axis=0))
    im = visualization.plot_image_by_time(1000*time, erp_ch[:,sort], ylabel='trials')
    im.set_clim(*clim)
    if latency is not None:
        plt.scatter(1000*latency[sort], np.arange(len(latency)), color='k', marker='.', s=0.75**2)
        plt.axvline(1000*np.nanmedian(latency), linestyle='--', linewidth=0.75, color='k')
    ax.set_xlabel('time (ms)')
    ax.set_xticks([0, 1000*time_after])
    ax.set_yticks([0, round(erp_ch.shape[1],-1)])
    ax.set_xlim(0, 1000*time_after)
    return im

    
def compare_connectivity(df, label_1, label_2, label_column='condition', 
                         time_before=0.25, time_after=0.25, theta=0):
    
    # Check stim sites are the same
    stim_sites = np.unique(df['stimulation_site'])    
    if len(stim_sites) > 1:
        print(stim_sites)
        raise ValueError("Too many stim sites in recordings!")
    stim_site = int(stim_sites[0])
    print(f"Stim site: {stim_site}")
    
    # Create equal sample sizes
    conditions, counts = np.unique(df['condition'], return_counts=True)
    print(f"{counts[0]} trials for {conditions[0]}, {counts[1]} trials for {conditions[1]}")
    min_count = np.min(counts)
    df.sort_values('condition', inplace=True)
    df_sub = pd.concat([df[:min_count], df[-min_count:]]).reset_index()
    conditions, counts = np.unique(df_sub['condition'], return_counts=True)
    print(f"{counts[0]} trials for {conditions[0]}, {counts[1]} trials for {conditions[1]}")

    # Load data
    _, acq_ch, _ = aopy.data.load_chmap()
    erp, samplerate = load_stim_erp(df_sub, time_before, time_after, acq_ch-1)
    subject = df_sub['subject'][0]
    
    # To-do: move this plotting into calc_coh_map, then use that
    
    # Plot ERPs
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plot_erp(erp[:,:,df_sub['condition'] == label_1], time_before, time_after, samplerate, 
             subject, stim_site, theta=theta) 
    plt.title(label_1)
    
    plt.subplot(1,2,2)
    plot_erp(erp[:,:,df_sub['condition'] == label_2], time_before, time_after, samplerate, 
             subject, stim_site, theta=theta) 
    plt.title(label_2)

    # ch_near_stim = get_acq_ch_near_stimulation_site(stim_site)
    # stim_ch_idx = np.where(np.isin(acq_ch, ch_near_stim))[0]
    # freqs, time, coh_all_1, _ = calc_connectivity_map_coh(erp[:,:,df_sub['condition'] == label_1], samplerate, 
    #                                                      time_before, time_after, stim_ch_idx, parallel=True)
    # freqs, time, coh_all_2, _ = calc_connectivity_map_coh(erp[:,:,df_sub['condition'] == label_2], samplerate, 
    #                                                      time_before, time_after, stim_ch_idx, parallel=True)

    freqs, time, coh_all_1 = calc_slic_map(df_sub[df_sub['condition'] == label_1].reset_index(drop=True), diff=False)
    freqs, time, coh_all_2 = calc_slic_map(df_sub[df_sub['condition'] == label_2].reset_index(drop=True), diff=False)
    
    
    return freqs, time, coh_all_1, coh_all_2

def plot_corr_matrices(corr_matrices, labels):
    n_sites = len(corr_matrices)

    fig, axes = plt.subplots(1, n_sites, figsize=(5 * n_sites, 5))  # Adjust the figure size based on number of sites

    for site in range(n_sites):
        ax = axes[site] if n_sites > 1 else axes
        im = ax.imshow(corr_matrices[site], cmap='Grays', vmin=0, vmax=1)
        ax.set_title(f'Site {labels[site]}')
        plt.colorbar(im, ax=ax, shrink=0.6)


def plot_norm_corr(corr_matrix, ax=None):
    
    if ax is None:
        ax = plt.gca()
    ax.plot(corr_matrix[1:,0])
    ax.set_ylim(0,1)

def plot_coh_band_summary(subject, theta, band=(80,150), window=(-np.inf, np.inf), 
                     alpha=0.01, diff_cutoff=0.1, scale=10, grid_size=(16,16)):
    

    subject_data = aopy.data.load_hdf_group(postproc_dir, f'{subject}_coh_all_{version}.hdf')
    _, _, stim_ch = aopy.data.load_chmap('Opto32')
    conn_sites = []
    for idx, stim_site in enumerate(stim_ch):
        
        diff, p = calc_tfr_mean_fdrc_ranktest(subject_data['freqs'], subject_data['time'], 
                                                            subject_data['coh_map'][idx], subject_data['null_maps'][idx],
                                                            band=band, window=window)
        diff[p>alpha] = 0.
        conn_sites.append(diff)
    
    return plot_stim_connectivity_summary(conn_sites, subject, theta, cutoff=diff_cutoff,
                                          scale=scale, grid_size=grid_size)


def plot_stim_connectivity_summary(conn_sites, subject, theta, scale=10, 
                                   stim_sites=None, grid_size=(16,16), colors=None, ax=None):

    # fig, (ax_from, ax_to) = plt.subplots(1, 2, figsize=(10,5))
    if ax is None:
        ax = plt.gca()
    ax.set_facecolor('#EAEAF2')
    # ax_from.set_facecolor('#EAEAF2')
    
    elec_pos, acq_ch, elecs = aopy.data.load_chmap(theta=theta)
    stim_pos, _, stim_ch = aopy.data.load_chmap('Opto32', theta=theta)

    if colors is None:
        colors = sns.color_palette('tab10', n_colors=32)
        np.random.shuffle(colors)
    
    # To map
    color_idx = 0
    to_map = []
    from_map = []
    for idx, stim_site in enumerate(stim_ch):

        if stim_sites is not None and stim_site not in stim_sites:
            from_map.append(0.0)
            continue
        
        m = scale*conn_sites[idx]
        m[m>1] = 1
        m[m<0] = 0

        from_map.append(np.max(m))
                    
        cmap = mpcolors.ListedColormap([colors[idx]])

        data_map, xy = aopy.visualization.calc_data_map(np.ones(m.shape), elec_pos[:,0], elec_pos[:,1], grid_size, interp_method='cubic')
        alpha_map, _ = aopy.visualization.calc_data_map(m, elec_pos[:,0], elec_pos[:,1], grid_size)
        im = aopy.visualization.plot_spatial_map(data_map, xy[0], xy[1], alpha_map=alpha_map, cmap=cmap, ax=ax)    

        # Mark the stimulation site in the appropriate color
        # ax_from.scatter([stim_pos[idx,0]], [stim_pos[idx,1]], color=stim_colors[idx], 
        #             s=200, alpha=1, zorder=10, edgecolor='black')

    ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[], xlabel='', ylabel='') 
    overlay_sulci_on_spatial_map(subject, 'lm1', 'ECoG244', theta=theta, color='k', ax=ax)
    ax.axis('off')

    # # From map
    # cmap = colors.ListedColormap(stim_colors)
    # data_map = aopy.visualization.get_data_map(np.arange(len(stim_ch)).astype(float), stim_pos[:,0], stim_pos[:,1])
    # alpha_map = aopy.visualization.get_data_map(from_map, stim_pos[:,0], stim_pos[:,1])
    # im = aopy.visualization.plot_spatial_map(data_map, stim_pos[:,0], stim_pos[:,1], alpha_map=alpha_map, cmap=cmap, ax=ax_from)    
    # ax_from.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[], xlabel='', ylabel='') 
    # overlay_sulci_on_spatial_map(subject, 'lm1', 'ECoG244', theta=theta, color='k', ax=ax_from)

    # return fig, ax_from, ax_to
    
    
def plot_connectivity_comparison(freqs, time, coh_all_1, coh_all_2, label_1, label_2, stimulation_site, 
                                 subject, theta=0, bands=[(12,50),(50,80),(80,150),(12,150)], 
                                 window=(0,1), null_coh=None, alpha=0.05):
    
    fig, ax = plt.subplots(len(bands), 2, figsize=(8,4*len(bands)), squeeze=False)
    
    for idx, band in enumerate(bands):
    
        if null_coh is None:
            conn = calc_fdrc_ranktest(freqs, time, coh_all_1, band, window)
        else:
            conn, p = calc_tfr_mean_fdrc_ranktest(freqs, time, coh_all_1, null_coh, band=band, window=window)
            conn[p>alpha] = 0.

        plot_stim_spatial_map(conn, stimulation_site, (16,16), 'viridis', (0,0.1), subject, theta, 
                              colorbar=False, ax=ax[idx][0])

        if null_coh is None:
            conn = calc_fdrc_ranktest(freqs, time, coh_all_2, band, window)
        else:
            conn, p = calc_tfr_mean_fdrc_ranktest(freqs, time, coh_all_2, null_coh, band=band, window=window)
            conn[p>alpha] = 0.

        plot_stim_spatial_map(conn, stimulation_site, (16,16), 'viridis', (0,0.1), subject, theta, 
                              colorbar=False, ax=ax[idx][1])
        
        ax[idx][0].set_ylabel(band)

    ax[0][0].set_title(label_1)
    ax[0][1].set_title(label_2)
    
    
def plot_rolling_comparison(freqs, time, rolling, splits, grid_size, theta, band=(12,150), window=(0,1), null_maps=None):

    ncc, shifts = calc_tf_map_similarity(freqs, time, rolling, (16,16), theta, band=band, window=(0,1), null_maps=null_maps)
    plot_corr_matrices([ncc], [date])
    
    # Set x and y ticks to label perturbed condition
    if np.shape(splits) == ():
        splits = [splits]
    colors = sns.color_palette(n_colors=len(splits))
    for split, color in zip(splits, colors):
        ax = plt.gca()
        ax.set_xticks(range(len(ncc)))
        ax.set_yticks(range(len(ncc)))
        for label, tick in zip(ax.get_yticklabels(), ax.get_yticks()):
            if (tick >= split):
                label.set_color(color)
        for label, tick in zip(ax.get_xticklabels(), ax.get_xticks()):
            if (tick >= split):
                label.set_color(color)

                
def plot_all_sites_connectivity(subject, theta, band=(80, 150), window=(0, 1), alpha=0.001, grid_size=(16,16)):
    '''
    Use opto32 layout grid to make a comprehensive set of connectivity maps
    '''
    elec_pos, acq_ch, elecs = aopy.data.load_chmap(theta=theta)
    _, _, stim_ch = aopy.data.load_chmap('Opto32', theta=theta)

    subject_data = aopy.data.load_hdf_group(postproc_dir, f'{subject}_coh_all_{version}.hdf')
    
    fig1, ax_coh = place_Opto32_subplots(theta=theta)
    plt.suptitle(f'{subject} coh', x=0., ha='left')
    fig2, ax_gc = place_Opto32_subplots(theta=theta)
    plt.suptitle(f'{subject} gc', x=0., ha='left')

    for idx, stim_site in enumerate(stim_ch):

        # Band limit
        diff, p = calc_tfr_mean_fdrc_ranktest(subject_data['freqs'], subject_data['time'], 
                                                            subject_data['coh_map'][idx], subject_data['null_maps'][idx],
                                                            band=band, window=window)
        gc_diff, gc_p = calc_tfr_mean_fdrc_ranktest(subject_data['freqs'], subject_data['time'], 
                                                              subject_data['gc_map'][idx], subject_data['gc_null_maps'][idx],
                                                              band=band, window=window)

        # Plot
        diff[p>alpha] = 0
        plot_stim_spatial_map(diff, stim_site, grid_size, 'viridis', (0, 0.1), subject, theta=theta,
                 colorbar=False, fontsize=6, color='w', ax=ax_coh[idx])

        gc_diff[gc_p>alpha] = 0
        plot_stim_spatial_map(gc_diff, stim_site, grid_size, 'viridis', (0, 0.05), subject, theta=theta,
                 colorbar=False, fontsize=6, color='k', ax=ax_gc[idx])

    return fig1, fig2




##########################################
# Latency
##########################################
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


# ---------------------------------------------------------------------------
# Source-data export
#
# Helpers for saving the numbers behind each figure to a per-figure Excel
# workbook (one worksheet per subplot / data component), e.g. for journal
# source-data files. Used by the notebooks alongside each savefig call.
# ---------------------------------------------------------------------------

def _clean_sheet_name(name):
    """Coerce an arbitrary label into a valid, unique-ish Excel sheet name."""
    name = str(name)
    for ch in '[]:*?/\\':
        name = name.replace(ch, '_')
    return name[:31]


# ---------------------------------------------------------------------------
# Source-data relabeling
#
# The notebooks build the source-data DataFrames using the terse variable
# names from the analysis code (``slic``, ``gc``, ``beignet`` ...). For the
# published source-data workbooks we want the sheet names, column headers and
# subject values to read the way they are labeled in the figures. All of that
# relabeling is centralized here so the notebook re-run path and any one-off
# rewrite of an existing workbook use exactly the same mapping.
# ---------------------------------------------------------------------------

# Subject codenames -> figure labels (also applied to any ``subject`` column).
SUBJECT_LABELS = {'beignet': 'Monkey 1', 'affi': 'Monkey 2'}

# Structural columns shared by every spatial-map / stim-map sheet.
_BASE_COLUMN_LABELS = {
    'elec': 'Electrode',
    'acq_ch': 'Acquisition channel',
    'stim_ch': 'Stimulation site',
    'x': 'x (mm)',
    'y': 'y (mm)',
}

# Example-map column labels reused across several figures (response-size /
# metric role kept in parentheses; subject codename -> figure label).
_EXAMPLE_MAP_LABELS = {
    'beignet_site11': 'Monkey 1, site 11', 'beignet_site7': 'Monkey 1, site 7',
    'affi_site6': 'Monkey 2, site 6', 'affi_site13': 'Monkey 2, site 13',
    'beignet_site15': 'Monkey 1, site 15', 'beignet_site22': 'Monkey 1, site 22',
    'affi_site14': 'Monkey 2, site 14', 'affi_site10': 'Monkey 2, site 10',
    'big_beignet_site7': 'Monkey 1, site 7 (big)',
    'little_beignet_site11': 'Monkey 1, site 11 (little)',
    'little_beignet_site7': 'Monkey 1, site 7 (little)',
    'little_affi_site6': 'Monkey 2, site 6 (little)',
    'little_affi_site13': 'Monkey 2, site 13 (little)',
    'small_beignet_site15': 'Monkey 1, site 15 (small)',
    'small_beignet_site28': 'Monkey 1, site 28 (small)',
    'small_affi_site14': 'Monkey 2, site 14 (small)',
    'small_affi_site29': 'Monkey 2, site 29 (small)',
}

_BAND_LABELS = {
    '0.5-12': '0.5–12 Hz', '12-30': '12–30 Hz', '30-80': '30–80 Hz',
    '80-120': '80–120 Hz', '120-200': '120–200 Hz',
}

# Per-figure, per-sheet column relabeling. Keys are the ORIGINAL sheet/column
# names produced by the notebooks. ``subject`` columns and the structural
# columns above are handled automatically and need not be listed here.
SOURCE_DATA_LABELS = {
    'figure2': {
        'example_erp_map': {'response_sigma': 'Response (σ)'},
        'erp_mean_timeseries': {'time_s': 'Time (s)'},
        'erp_pulsewidth_sweep': {'time_s': 'Time (s)', 'width_0.0': 'Pulse width 0.0 s',
                                 'width_0.01': 'Pulse width 0.01 s', 'width_0.02': 'Pulse width 0.02 s'},
        'single_trial_stim': {'__trials__': True},
        'single_trial_near': {'__trials__': True},
        'single_trial_far': {'__trials__': True},
        'example_response_maps': _EXAMPLE_MAP_LABELS,
        'auc_volume_beignet': {'mean_response_sigma': 'Mean response (σ)'},
        'auc_volume_affi': {'mean_response_sigma': 'Mean response (σ)'},
        'erp_example_maps': {
            'implant1_first_day2022-02-15': 'Implant 1, first day (2022-02-15)',
            'implant1_last_day2022-02-22': 'Implant 1, last day (2022-02-22)',
            'implant2_first_day2022-03-15': 'Implant 2, first day (2022-03-15)',
            'implant2_last_day2022-04-14': 'Implant 2, last day (2022-04-14)',
            'implant3_first_day2022-06-08': 'Implant 3, first day (2022-06-08)',
            'implant3_last_day2022-06-27': 'Implant 3, last day (2022-06-27)',
        },
        'day1_vs_day8_scatter': {'day1_response': 'Day 1 response (σ)',
                                 'day8_response': 'Day 8 response (σ)'},
        'erp_longitude_correlation': {'stim_site': 'Stimulation site',
                                      'implant_index': 'Implant index',
                                      'correlation_r': 'Spatial correlation (r)'},
    },
    'figure3': {
        'connections_pooled': {'stim_site': 'Stimulation site', 'latency_accllr': 'AccLLR latency (ms)',
                               'latency_itpc': 'ITPC latency (ms)', 'distance': 'Distance (mm)',
                               'slic': 'SLIC', 'angle': 'Phase difference (rad)', 'gp': 'GP'},
        'latency_example_maps_ms': _EXAMPLE_MAP_LABELS,
        'phase_diff_vs_distance': {'stim_site': 'Stimulation site', 'distance': 'Distance (mm)',
                                   'angle': 'Phase difference (rad)',
                                   'absolute_angle': 'Absolute phase difference (rad)'},
        'cutoff_sweep': {'site_idx': 'Stimulation site index', 'cutoff': 'Cutoff latency (ms)',
                         'accllr': 'AccLLR connection count'},
        'cutoff_serr_maps': {
            'beignet_site7_cutoff10ms': 'Monkey 1, site 7 (10 ms cutoff)',
            'beignet_site7_cutoff15ms': 'Monkey 1, site 7 (15 ms cutoff)',
            'beignet_site7_cutoff20ms': 'Monkey 1, site 7 (20 ms cutoff)',
            'affi_site25_cutoff10ms': 'Monkey 2, site 25 (10 ms cutoff)',
            'affi_site25_cutoff15ms': 'Monkey 2, site 25 (15 ms cutoff)',
            'affi_site25_cutoff20ms': 'Monkey 2, site 25 (20 ms cutoff)',
        },
        'accllr_example_maps': _EXAMPLE_MAP_LABELS,
    },
    'figure4': {
        'slic_phase_example_maps': {
            'main_beignet_site7_slic': 'Monkey 1, site 7 SLIC (main)',
            'main_beignet_site7_phase': 'Monkey 1, site 7 phase (main)',
            'little_beignet_site11_slic': 'Monkey 1, site 11 SLIC (little)',
            'little_beignet_site11_phase': 'Monkey 1, site 11 phase (little)',
            'little_beignet_site7_slic': 'Monkey 1, site 7 SLIC (little)',
            'little_beignet_site7_phase': 'Monkey 1, site 7 phase (little)',
            'little_affi_site6_slic': 'Monkey 2, site 6 SLIC (little)',
            'little_affi_site6_phase': 'Monkey 2, site 6 phase (little)',
            'little_affi_site13_slic': 'Monkey 2, site 13 SLIC (little)',
            'little_affi_site13_phase': 'Monkey 2, site 13 phase (little)',
            'small_beignet_site15_slic': 'Monkey 1, site 15 SLIC (small)',
            'small_beignet_site15_phase': 'Monkey 1, site 15 phase (small)',
            'small_beignet_site28_slic': 'Monkey 1, site 28 SLIC (small)',
            'small_beignet_site28_phase': 'Monkey 1, site 28 phase (small)',
            'small_affi_site14_slic': 'Monkey 2, site 14 SLIC (small)',
            'small_affi_site14_phase': 'Monkey 2, site 14 phase (small)',
            'small_affi_site29_slic': 'Monkey 2, site 29 SLIC (small)',
            'small_affi_site29_phase': 'Monkey 2, site 29 phase (small)',
        },
    },
    'figure5': {
        'counts_by_site': {'stim_site': 'Stimulation site', 'volume': 'Response volume',
                           'accllr': 'AccLLR connection count', 'slic': 'SLIC connection count',
                           'gcs': 'SEGP connection count', 'gc': 'GP connection count'},
        'example_maps_beignet_site7': {'accllr': 'AccLLR', 'slic': 'SLIC',
                                       'gc_stim': 'SEGP', 'gc': 'GP'},
        'example_maps_site12': {'slic_group1': 'SLIC (group 1)', 'slic_group2': 'SLIC (group 2)',
                                'accllr_group1': 'AccLLR (group 1)', 'accllr_group2': 'AccLLR (group 2)',
                                'gc_group1': 'GP (group 1)', 'gc_group2': 'GP (group 2)'},
        'correlation_matrix': {'row': 'Metric group'},
        'within_vs_across': {'site': 'Stimulation site',
                             'within_mean': 'Within-metric correlation, mean (SLIC–SLIC, GP–GP)',
                             'across_mean': 'Across-metric correlation, mean (SLIC–GP)',
                             'within_err': 'Within-metric correlation, SD',
                             'across_err': 'Across-metric correlation, SD'},
        'within_vs_across_gp': {'site': 'Stimulation site',
                                'within_mean': 'Within-metric correlation, mean (GP–GP, SEGP–SEGP)',
                                'across_mean': 'Across-metric correlation, mean (GP–SEGP)',
                                'within_err': 'Within-metric correlation, SD',
                                'across_err': 'Across-metric correlation, SD'},
        'correlation_vs_volume': {'site': 'Stimulation site', 'volume': 'Mean response (σ)',
                                  'slic_gps': 'SLIC–SEGP correlation', 'slic_gp': 'SLIC–GP correlation',
                                  'gps_gp': 'SEGP–GP correlation'},
    },
    'figure6': {
        'slic_beignet': dict(_BAND_LABELS, from_band='From band (Hz)'),
        'gp_beignet': dict(_BAND_LABELS, from_band='From band (Hz)'),
        'segp_beignet': dict(_BAND_LABELS, from_band='From band (Hz)'),
        'slic_affi': dict(_BAND_LABELS, from_band='From band (Hz)'),
        'gp_affi': dict(_BAND_LABELS, from_band='From band (Hz)'),
        'segp_affi': dict(_BAND_LABELS, from_band='From band (Hz)'),
        'slic_bands': dict(_BAND_LABELS),
        'gp_bands': dict(_BAND_LABELS),
        'segp_bands': dict(_BAND_LABELS),
        'distance_pooled': {'stim_site': 'Stimulation site', 'band_idx': 'Band index',
                            'distance': 'Distance (mm)', 'slic': 'SLIC', 'gp': 'GP',
                            'angle': 'Phase difference (rad)'},
        'ks_effect_size': {'band': 'Band (Hz)', 'ks_observed': 'KS statistic (observed)',
                           'null_mean': 'Null mean', 'null_p05': 'Null 5th percentile',
                           'null_p95': 'Null 95th percentile'},
    },
    'figure7': {
        'slic_example_maps': {'implant1_first': 'Implant 1, first', 'implant1_last': 'Implant 1, last',
                              'implant2_first': 'Implant 2, first', 'implant2_last': 'Implant 2, last',
                              'implant3_first': 'Implant 3, first', 'implant3_last': 'Implant 3, last'},
        'volume_vs_correlation': {'volume': 'Mean response (σ)', 'variance': 'Response variance',
                                  'mean': 'Mean connectivity (normalized)',
                                  'correlation': 'Spatial correlation (r)', 'group': 'Metric'},
        'slic_correlation_box': {'value': 'Spatial correlation (r)', 'site': 'Implant', 'cat': 'Condition'},
        'gc_correlation_box': {'value': 'Spatial correlation (r)', 'site': 'Implant', 'cat': 'Condition'},
        'state_example_erp': {'erp_open': 'Eyes open response (σ)',
                              'erp_closed': 'Eyes closed response (σ)'},
        'state_example_slic': {'slic_mean1': 'SLIC, state 1', 'slic_mean2': 'SLIC, state 2',
                               'slic_mean2_minus_mean1': 'ΔSLIC (state 2 − state 1)'},
        'state_summary_slic_hist': {'bin_center': 'SLIC bin center',
                                    'dist1_counts': 'State 1 count', 'dist2_counts': 'State 2 count'},
        'state_summary_dprime_map': {'dprime': 'd-prime'},
        'state_all_dprime': {'beignet_site11': 'Monkey 1, site 11', 'beignet_site7': 'Monkey 1, site 7',
                             'beignet_site15': 'Monkey 1, site 15', 'affi_site14': 'Monkey 2, site 14',
                             'affi_site20': 'Monkey 2, site 20'},
        'state_all_sig_counts': {'site': 'Site', 'n_significant_electrodes': 'Significant electrode count'},
    },
}

# Cell VALUES (not headers) that carry a subject codename or example-map label.
_SOURCE_DATA_VALUE_RELABEL = {
    'figure7': {'state_all_sig_counts': {'site': {
        'beignet_site11': 'Monkey 1, site 11', 'beignet_site7': 'Monkey 1, site 7',
        'beignet_site15': 'Monkey 1, site 15', 'affi_site14': 'Monkey 2, site 14',
        'affi_site20': 'Monkey 2, site 20'}}},
}


def _relabel_source_sheet(fig_name, sheet_name, df):
    """Return ``(new_sheet_name, relabeled_df)`` for a source-data worksheet.

    Renames the sheet, its columns and any subject/example-map cell values from
    the notebook's variable names to the labels used in the figures, using
    :data:`SOURCE_DATA_LABELS`. Anything not covered by the table is passed
    through unchanged.
    """
    df = df.copy()
    col_spec = dict(SOURCE_DATA_LABELS.get(fig_name, {}).get(sheet_name, {}))

    # Single-trial sheets: columns are trial indices -> "Trial 1", "Trial 2", ...
    if col_spec.pop('__trials__', False):
        df.columns = [f'Trial {i + 1}' for i in range(len(df.columns))]

    # Remap subject-codename cell values before any column is renamed.
    for col, value_map in _SOURCE_DATA_VALUE_RELABEL.get(fig_name, {}).get(sheet_name, {}).items():
        if col in df.columns:
            df[col] = df[col].map(lambda v: value_map.get(v, v))
    if 'subject' in df.columns:
        df['subject'] = df['subject'].map(lambda v: SUBJECT_LABELS.get(v, v))

    renames = {'subject': 'Subject'}
    renames.update({k: v for k, v in _BASE_COLUMN_LABELS.items() if k in df.columns})
    renames.update(col_spec)
    df.rename(columns=renames, inplace=True)

    new_name = str(sheet_name)
    for code, label in SUBJECT_LABELS.items():
        new_name = new_name.replace(code, label.replace(' ', '').lower())
    return new_name, df


def _source_data_filename(fig_name):
    """``'figure2'`` / ``'Fig2'`` / ``'2'`` -> ``'Fig2_data.xlsx'``."""
    m = re.search(r'(\d+)', str(fig_name))
    return f'Fig{m.group(1)}_data.xlsx' if m else f'{fig_name}_data.xlsx'


def _normalize_fig_name(fig_name):
    """Map any figure identifier to the ``'figureN'`` key used in the tables."""
    m = re.search(r'(\d+)', str(fig_name))
    return f'figure{m.group(1)}' if m else str(fig_name)


def relabel_source_workbook(path, fig_name=None):
    """Rewrite an existing source-data workbook in place with figure labels.

    Applies :func:`_relabel_source_sheet` to every worksheet, preserving order.
    ``fig_name`` defaults to the figure number parsed from the file name.
    """
    if fig_name is None:
        fig_name = os.path.basename(path)
    fig_name = _normalize_fig_name(fig_name)
    sheets = pd.read_excel(path, sheet_name=None)
    with pd.ExcelWriter(path, engine='openpyxl', mode='w') as writer:
        for name, df in sheets.items():
            new_name, df2 = _relabel_source_sheet(fig_name, name, df)
            df2.to_excel(writer, sheet_name=_clean_sheet_name(new_name), index=False)
    print(f"[relabel] {path}: {len(sheets)} sheet(s) relabeled")


def save_source_data(fig_name, sheets, fig_dir='./figures', new_file=False):
    """Write the data behind a figure to ``<fig_dir>/Fig<N>_data.xlsx``.

    Args:
        fig_name (str): figure identifier, e.g. ``'figure2'`` (written to
            ``Fig2_data.xlsx``).
        sheets (dict): mapping of ``{sheet_name: pandas.DataFrame}``; one
            worksheet is written per entry.
        fig_dir (str, optional): output directory. Default ``'./figures'``.
        new_file (bool, optional): if True (or the file does not yet exist),
            start a fresh workbook. Use this on the FIRST write of a given
            figure. Later writes append/replace sheets in place, so a figure
            can be assembled across several cells and notebooks. Default False.

    Sheet names, column headers and subject values are relabeled to the way
    they appear in the figures (see :func:`_relabel_source_sheet`) before the
    workbook is written.

    A DataFrame is written with its index only when the index is named.
    """
    path = os.path.join(fig_dir, _source_data_filename(fig_name))
    fig_key = _normalize_fig_name(fig_name)
    if new_file or not os.path.exists(path):
        mode, extra = 'w', {}
    else:
        mode, extra = 'a', {'if_sheet_exists': 'replace'}
    written = []
    with pd.ExcelWriter(path, engine='openpyxl', mode=mode, **extra) as writer:
        for name, df in sheets.items():
            new_name, df = _relabel_source_sheet(fig_key, name, df)
            new_name = _clean_sheet_name(new_name)
            df.to_excel(writer, sheet_name=new_name,
                        index=bool(getattr(df.index, 'name', None)))
            written.append(new_name)
    print(f"[source data] {path}: wrote {len(sheets)} sheet(s): " + ", ".join(written))


def map_source_df(values, theta=0, columns=None, value_name='value'):
    """Per-electrode spatial-map values as a tidy DataFrame keyed by electrode.

    Args:
        values: a 1-D per-electrode array (one map) or a 2-D array / sequence
            of per-electrode arrays stacked along axis 0 (one value column each).
        theta (int, optional): chamber rotation used for the x/y columns.
            Electrode ordering (and therefore value alignment) is independent
            of theta. Default 0.
        columns (list, optional): names for the value columns in the 2-D case.
        value_name (str, optional): base name for the value column(s).

    Returns:
        pandas.DataFrame with columns ``elec, acq_ch, x, y`` plus one value
        column per map.
    """
    elec_pos, acq_ch, elecs = aopy.data.load_chmap(theta=theta)
    n = len(elecs)
    df = pd.DataFrame({
        'elec': np.asarray(elecs),
        'acq_ch': np.asarray(acq_ch),
        'x': np.asarray(elec_pos)[:, 0],
        'y': np.asarray(elec_pos)[:, 1],
    })
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
        names = [value_name]
    elif columns is not None:
        names = list(columns)
    else:
        names = [f'{value_name}_{i}' for i in range(arr.shape[0])]
    for name, row in zip(names, arr):
        col = np.full(n, np.nan)
        m = min(n, len(row))
        col[:m] = np.asarray(row, dtype=float)[:m]
        df[str(name)] = col
    return df


def stim_source_df(values, theta=0, columns=None, value_name='value'):
    """Per-stimulation-site (Opto32) values as a tidy DataFrame keyed by stim site.

    Same conventions as :func:`map_source_df`, but keyed by the 32 optical
    stimulation sites instead of recording electrodes.
    """
    stim_pos, _, stim_ch = aopy.data.load_chmap('Opto32', theta=theta)
    n = len(stim_ch)
    df = pd.DataFrame({
        'stim_ch': np.asarray(stim_ch),
        'x': np.asarray(stim_pos)[:, 0],
        'y': np.asarray(stim_pos)[:, 1],
    })
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
        names = [value_name]
    elif columns is not None:
        names = list(columns)
    else:
        names = [f'{value_name}_{i}' for i in range(arr.shape[0])]
    for name, row in zip(names, arr):
        col = np.full(n, np.nan)
        m = min(n, len(row))
        col[:m] = np.asarray(row, dtype=float)[:m]
        df[str(name)] = col
    return df

