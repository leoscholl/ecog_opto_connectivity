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

