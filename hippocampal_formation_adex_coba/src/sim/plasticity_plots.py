"""Plasticity visualization functions.

This module provides plotting utilities for visualizing plasticity dynamics
including short-term plasticity (STP), spike-timing dependent plasticity (STDP),
and synaptic tagging mechanisms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def plot_stp_dynamics(
    t_s: np.ndarray,
    u: np.ndarray,
    R: np.ndarray,
    pathway_name: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot short-term plasticity dynamics.
    
    Parameters
    ----------
    t_s : array
        Time points in seconds
    u : array
        Release probability over time (shape: n_synapses x n_timepoints or n_timepoints)
    R : array
        Available resources over time (shape: n_synapses x n_timepoints or n_timepoints)
    pathway_name : str
        Name of the pathway for title
    ax : Axes, optional
        Matplotlib axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    # Handle both 1D and 2D arrays
    if u.ndim == 2:
        # Plot mean and individual traces
        u_mean = u.mean(axis=0)
        R_mean = R.mean(axis=0)
        
        # Plot a few example synapses with transparency
        n_examples = min(5, u.shape[0])
        for i in range(n_examples):
            ax.plot(t_s, u[i], 'b-', alpha=0.2, linewidth=0.5)
            ax.plot(t_s, R[i], 'r-', alpha=0.2, linewidth=0.5)
        
        # Plot means
        ax.plot(t_s, u_mean, 'b-', linewidth=2, label='Release prob (u)')
        ax.plot(t_s, R_mean, 'r-', linewidth=2, label='Resources (R)')
    else:
        ax.plot(t_s, u, 'b-', linewidth=2, label='Release prob (u)')
        ax.plot(t_s, R, 'r-', linewidth=2, label='Resources (R)')
    
    ax.plot(t_s, u * R if u.ndim == 1 else u_mean * R_mean, 'g--', 
            linewidth=2, label='Effective release (u×R)')
    
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Normalized value', fontsize=10)
    ax.set_title(f'Short-term plasticity: {pathway_name}', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    
    return ax


def plot_weight_evolution(
    t_s: np.ndarray,
    weights: np.ndarray,
    pathway_name: str,
    ax: Optional[plt.Axes] = None,
    show_percentiles: bool = True,
) -> plt.Axes:
    """Plot evolution of synaptic weights.
    
    Parameters
    ----------
    t_s : array
        Time points in seconds
    weights : array
        Weights over time (shape: n_synapses x n_timepoints)
    pathway_name : str
        Name of the pathway
    ax : Axes, optional
        Matplotlib axes to plot on
    show_percentiles : bool
        Whether to show percentile bands
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    if weights.ndim == 1:
        ax.plot(t_s, weights, 'b-', linewidth=2)
    else:
        w_mean = weights.mean(axis=0)
        ax.plot(t_s, w_mean, 'b-', linewidth=2, label='Mean')
        
        if show_percentiles and weights.shape[0] > 1:
            w_10 = np.percentile(weights, 10, axis=0)
            w_90 = np.percentile(weights, 90, axis=0)
            ax.fill_between(t_s, w_10, w_90, alpha=0.3, label='10-90th percentile')
        
        # Show a few example traces
        n_examples = min(10, weights.shape[0])
        idx = np.linspace(0, weights.shape[0]-1, n_examples, dtype=int)
        for i in idx:
            ax.plot(t_s, weights[i], 'gray', alpha=0.2, linewidth=0.5)
    
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Weight (normalized)', fontsize=10)
    ax.set_title(f'Weight evolution: {pathway_name}', fontsize=11)
    if show_percentiles and weights.ndim > 1:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_weight_distribution(
    weights_initial: np.ndarray,
    weights_final: np.ndarray,
    pathway_name: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot distribution of weights before and after plasticity.
    
    Parameters
    ----------
    weights_initial : array
        Initial weights (1D array of n_synapses)
    weights_final : array
        Final weights (1D array of n_synapses)
    pathway_name : str
        Name of the pathway
    ax : Axes, optional
        Matplotlib axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    bins = np.linspace(
        min(weights_initial.min(), weights_final.min()),
        max(weights_initial.max(), weights_final.max()),
        30
    )
    
    ax.hist(weights_initial, bins=bins, alpha=0.5, label='Initial', color='blue')
    ax.hist(weights_final, bins=bins, alpha=0.5, label='Final', color='red')
    
    ax.axvline(weights_initial.mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Initial mean: {weights_initial.mean():.3f}')
    ax.axvline(weights_final.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Final mean: {weights_final.mean():.3f}')
    
    ax.set_xlabel('Weight', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'Weight distribution: {pathway_name}', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_synaptic_tagging(
    t_s: np.ndarray,
    tags: np.ndarray,
    prps: np.ndarray,
    pathway_name: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot synaptic tagging and protein synthesis dynamics.
    
    Parameters
    ----------
    t_s : array
        Time points in seconds
    tags : array
        Tag state over time (shape: n_synapses x n_timepoints or n_timepoints)
    prps : array
        PRP state over time (shape: n_synapses x n_timepoints or n_timepoints)
    pathway_name : str
        Name of the pathway
    ax : Axes, optional
        Matplotlib axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    if tags.ndim == 2:
        # Show fraction of tagged synapses
        tag_fraction = (tags > 0.5).mean(axis=0)
        prp_fraction = (prps > 0.5).mean(axis=0)
        
        ax.plot(t_s, tag_fraction, 'b-', linewidth=2, label='Fraction tagged')
        ax.plot(t_s, prp_fraction, 'r-', linewidth=2, label='Fraction with PRP')
    else:
        ax.plot(t_s, tags, 'b-', linewidth=2, label='Tag state')
        ax.plot(t_s, prps, 'r-', linewidth=2, label='PRP state')
    
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('State / Fraction', fontsize=10)
    ax.set_title(f'Synaptic tagging: {pathway_name}', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    
    return ax


def save_plasticity_overview(
    plasticity_data: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
    *,
    max_pathways: int = 6,
) -> None:
    """Create comprehensive plasticity overview figure.
    
    Parameters
    ----------
    plasticity_data : dict
        Dictionary mapping pathway names to plasticity recordings:
        {
            'pathway_name': {
                't_s': np.ndarray,              # time points
                'u': np.ndarray,                # release probability (STP)
                'R': np.ndarray,                # available resources (STP)
                'weights': np.ndarray,          # synaptic weights
                'weights_initial': np.ndarray,  # initial weights
                'tags': np.ndarray,             # tag state
                'prps': np.ndarray,             # PRP state
            }
        }
    out_path : Path
        Output file path
    max_pathways : int
        Maximum number of pathways to plot
    """
    _ensure_parent(out_path)
    
    pathways = list(plasticity_data.keys())[:max_pathways]
    n_pathways = len(pathways)
    
    if n_pathways == 0:
        print("No plasticity data to plot")
        return
    
    # Determine what to plot based on available data
    has_stp = any('u' in plasticity_data[p] for p in pathways)
    has_weights = any('weights' in plasticity_data[p] for p in pathways)
    has_tagging = any('tags' in plasticity_data[p] for p in pathways)
    
    n_cols = sum([has_stp, has_weights, has_tagging])
    if n_cols == 0:
        print("No plasticity variables found in data")
        return
    
    fig = plt.figure(figsize=(6 * n_cols, 3 * n_pathways))
    gs = GridSpec(n_pathways, n_cols, figure=fig, hspace=0.3, wspace=0.3)
    
    for i, pathway in enumerate(pathways):
        data = plasticity_data[pathway]
        t_s = data['t_s']
        col = 0
        
        # STP dynamics
        if has_stp and 'u' in data and 'R' in data:
            ax = fig.add_subplot(gs[i, col])
            plot_stp_dynamics(t_s, data['u'], data['R'], pathway, ax=ax)
            col += 1
        
        # Weight evolution
        if has_weights and 'weights' in data:
            ax = fig.add_subplot(gs[i, col])
            plot_weight_evolution(t_s, data['weights'], pathway, ax=ax)
            col += 1
        
        # Synaptic tagging
        if has_tagging and 'tags' in data and 'prps' in data:
            ax = fig.add_subplot(gs[i, col])
            plot_synaptic_tagging(t_s, data['tags'], data['prps'], pathway, ax=ax)
            col += 1
    
    fig.suptitle('Plasticity Dynamics Overview', fontsize=16, y=0.995)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Plasticity overview saved to {out_path}")


def save_weight_change_summary(
    plasticity_data: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    """Create summary figure showing weight changes across pathways.
    
    Parameters
    ----------
    plasticity_data : dict
        Dictionary mapping pathway names to plasticity recordings
    out_path : Path
        Output file path
    """
    _ensure_parent(out_path)
    
    pathways = []
    mean_changes = []
    std_changes = []
    
    for pathway, data in plasticity_data.items():
        if 'weights_initial' in data and 'weights' in data:
            w_init = data['weights_initial']
            w_final = data['weights'][:, -1] if data['weights'].ndim == 2 else data['weights'][-1]
            
            if np.isscalar(w_final):
                change = w_final - w_init
                pathways.append(pathway)
                mean_changes.append(change)
                std_changes.append(0)
            else:
                changes = w_final - w_init
                pathways.append(pathway)
                mean_changes.append(changes.mean())
                std_changes.append(changes.std())
    
    if not pathways:
        print("No weight change data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(pathways))
    colors = ['green' if m > 0 else 'red' for m in mean_changes]
    
    ax.barh(y_pos, mean_changes, xerr=std_changes, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pathways, fontsize=9)
    ax.set_xlabel('Mean weight change', fontsize=10)
    ax.set_title('Summary of weight changes by pathway', fontsize=12)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Weight change summary saved to {out_path}")
