# This file was initially done by hand and based on the Lab7
# It was then adjusted and adapted using ChatGPT-5
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy.ndimage import label

from nilearn.image import (
    load_img,
    mean_img,
    math_img,
    new_img_like,
    index_img,
    resample_to_img,
)
from nilearn.masking import compute_epi_mask
from nilearn.maskers import NiftiMasker
from nilearn.decomposition import CanICA
from nilearn.plotting import plot_stat_map
from nilearn.image import resample_to_img, smooth_img

plt.rcParams["figure.dpi"] = 120

__all__ = [
    "discover_fmri_runs",
    "build_mask",
    "extract_2d_samples",
    "choose_n_components_elbow",
    "knee_from_eigvals",
    "run_ica_canica",
    "plot_ica_overlays_axial",
    "similarity_first5_abs_corr",
    "find_project_root",
    "index_img",
]


# -------------------- Discovery --------------------


def discover_fmri_runs(paths_or_dirs: List[str]) -> List[str]:
    """Return absolute paths to NIfTI files that look like fMRI runs."""
    exts = (".nii", ".nii.gz")
    out: list[str] = []

    for p in map(Path, paths_or_dirs):
        if p.is_file() and (
            p.suffix.lower() in exts or "".join(p.suffixes).lower() in exts
        ):
            out.append(str(p.resolve()))
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and (
                    f.suffix.lower() in exts or "".join(f.suffixes).lower() in exts
                ):
                    if (
                        any(s in f.name.lower() for s in ["bold", "fmri"])
                        or "functional" in str(f.parent).lower()
                    ):
                        out.append(str(f.resolve()))
        else:
            warnings.warn(f"[discover_fmri_runs] Ignoring non-existent path: {p}")

    # De-duplicate while keeping order
    seen, uniq = set(), []
    for f in out:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


# -------------------- Mask --------------------


def build_mask(
    fmri_imgs: List[str],
    gm_pve_img: Optional[str] = None,
    gm_thresh: float = 0.25,
):
    """
    Build a binary mask. If a GM partial-volume map is provided, threshold it
    at `gm_thresh`; otherwise compute an EPI mask from the mean functional.
    """
    if gm_pve_img is not None:
        gm = load_img(gm_pve_img)
        mask = math_img("img > {}".format(gm_thresh), img=gm)
        data = (mask.get_fdata() > 0).astype(np.int16)
        return new_img_like(mask, data, copy_header=True)
    mimg = mean_img([load_img(p) for p in fmri_imgs])
    return compute_epi_mask(mimg)


# -------------------- Data extraction --------------------


def extract_2d_samples(
    fmri_imgs: List[str],
    mask_img,
    standardize: bool = True,
    detrend: bool = True,
) -> Tuple[np.ndarray, NiftiMasker]:
    """
    Vectorize masked fMRI into (n_samples × n_voxels). Returns the data matrix
    and the fitted masker used for the transform/inverse-transform.
    """
    masker = NiftiMasker(
        mask_img=mask_img,
        standardize=standardize,
        detrend=detrend,
        smoothing_fwhm=None,
    )
    masker.fit()
    X_list = [masker.transform(load_img(run)) for run in fmri_imgs]
    X = np.vstack(X_list)
    print(
        f"[extract_2d_samples] X shape = {X.shape} (samples=timepoints, features=voxels-in-mask)"
    )
    return X, masker


# -------------------- PCA elbow (Kneedle) --------------------


def knee_from_eigvals(
    eigvals: np.ndarray,
    S: float = 0.7,
    interp_method: str = "interp1d",
) -> Tuple[int, dict]:
    """
    Choose K with Kneedle on the PCA eigenvalues (scree).
    Falls back to Kneedle on cumulative variance if the scree knee is not found.
    """
    ev = np.asarray(eigvals).ravel()
    kmax = len(ev)
    xs = np.arange(1, kmax + 1)

    kl = KneeLocator(
        xs,
        ev,
        curve="convex",
        direction="decreasing",
        S=S,
        interp_method=interp_method,
        polynomial_degree=7,
    )
    k_scree = int(kl.knee) if kl.knee is not None else None

    k_cum = None
    if k_scree is None:
        var_ratio = ev / (ev.sum() + 1e-12)
        cumvar = np.cumsum(var_ratio)
        klc = KneeLocator(
            xs,
            cumvar,
            curve="concave",
            direction="increasing",
            S=S,
            interp_method=interp_method,
            polynomial_degree=7,
        )
        k_cum = int(klc.knee) if klc.knee is not None else None

    if k_scree is not None:
        k_final = k_scree
    elif k_cum is not None:
        k_final = k_cum
    else:
        k_final = max(2, min(20, kmax))

    return int(k_final), {
        "kneedle_scree": k_scree,
        "kneedle_cumvar": k_cum,
        "final": int(k_final),
    }


def choose_n_components_elbow(
    X: np.ndarray,
    max_components: Optional[int] = None,
    plot_path: Optional[str] = None,
    scree_csv_path: Optional[str] = None,
    cumvar_csv_path: Optional[str] = None,
    S: float = 0.7,
    interp_method: str = "interp1d",
    logy: bool = True,
) -> int:
    """
    Run PCA (via SVD), select K with Kneedle on the eigenvalues, and optionally
    save a scree plot (log-scaled variance per PC) and CSV artifacts.
    """
    X0 = X - X.mean(axis=0, keepdims=True)
    _, Svals, _ = np.linalg.svd(X0, full_matrices=False)
    eigvals = Svals**2

    kmax = len(eigvals) if max_components is None else min(max_components, len(eigvals))
    eigvals = eigvals[:kmax]

    k_elbow, diag = knee_from_eigvals(eigvals, S=S, interp_method=interp_method)

    if scree_csv_path:
        np.savetxt(scree_csv_path, eigvals, delimiter=",")
    if cumvar_csv_path:
        var_ratio = eigvals / (eigvals.sum() + 1e-12)
        np.savetxt(cumvar_csv_path, np.cumsum(var_ratio), delimiter=",")

    if plot_path:
        xs = np.arange(1, len(eigvals) + 1)
        var_ratio = eigvals / (eigvals.sum() + 1e-12)

        tiny = var_ratio.max() * 1e-8
        mask = var_ratio >= tiny
        xs_plot = xs[mask][: min(mask.sum(), max(80, int(k_elbow * 4)))]
        vr_plot = var_ratio[mask][: len(xs_plot)]

        fig, ax = plt.subplots(figsize=(6.8, 4.4))

        ax.plot(xs_plot, vr_plot, marker="o", markersize=3, lw=1)
        ax.set_yscale("log")
        ax.axvline(k_elbow, ls="-", alpha=0.9, label=f"kneedle (scree) = {k_elbow}")
        ax.scatter([k_elbow], [var_ratio[k_elbow - 1]], s=24, zorder=3)

        ax.set_xlabel("Number of components")
        ax.set_ylabel("Variance explained per component [log]")
        ax.set_title("PCA scree")
        ax.legend(frameon=False, fontsize=9, loc="upper right")

        # make background fully white/blank
        ax.grid(False)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        fig.tight_layout()
        fig.savefig(
            plot_path, dpi=150, facecolor="white", edgecolor="none", transparent=False
        )
        plt.close(fig)

    print(f"[choose_n_components_elbow] {diag}")
    return int(k_elbow)


# -------------------- ICA --------------------


def run_ica_canica(
    fmri_imgs: List[str],
    mask_img,
    n_components: int,
    t_r: Optional[float] = None,
    random_state: int = 0,
):
    """Fit Nilearn CanICA and return a 4D NIfTI of spatial maps."""
    canica = CanICA(
        n_components=n_components,
        mask=mask_img,
        smoothing_fwhm=30.0,
        standardize=True,
        random_state=random_state,
        t_r=t_r,
        n_jobs=-1,
    )
    print(
        f"[run_ica_canica] Fitting CanICA with n_components={n_components} on {len(fmri_imgs)} run(s)..."
    )
    canica.fit(fmri_imgs)
    return canica.components_img_


# -------------------- Plotting --------------------


def plot_ica_overlays_axial(
    components_img,
    anat_img: str,
    mask_img,  # <— NEW
    out_dir: Optional[str] = None,
    display_cuts: int = 8,
    dpi: int = 120,
    skip_existing: bool = True,
    verbose: bool = False,
    percentile: float = 98.5,  # <— NEW: use 98–99 for cleaner blobs
    min_cluster: int = 40,  # <— NEW: remove tiny specks (voxels)
    smooth_fwhm_display: Optional[float] = None,  # e.g., 3–5 mm
) -> None:
    """Save axial overlays; threshold by top-|values| inside the mask."""
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    n_components = components_img.shape[-1]
    comp0 = index_img(components_img, 0)

    anat_rs = resample_to_img(
        load_img(anat_img),
        comp0,
        interpolation="continuous",
        force_resample=True,
        copy_header=True,
    )
    mask_rs = resample_to_img(
        mask_img, comp0, interpolation="nearest", force_resample=True, copy_header=True
    )
    brain = mask_rs.get_fdata().astype(bool)

    for k in range(n_components):
        out_png = Path(out_dir) / f"ica_component_{k:02d}.png" if out_dir else None
        if skip_existing and out_png is not None and out_png.exists():
            if verbose:
                print(f"[overlays] skip {k+1}/{n_components} (exists)")
            continue

        img_k = index_img(components_img, k)
        data = img_k.get_fdata()
        vals = np.abs(data[brain])
        thr = np.percentile(vals, percentile)

        kept = np.zeros_like(data)
        inmask = np.abs(data) >= thr
        kept[inmask & brain] = data[inmask & brain]

        # optional de-speckling
        if min_cluster > 0 and label is not None:
            lab, nlab = label(np.abs(kept) > 0)
            if nlab > 0:
                sizes = np.bincount(lab.ravel())
                drop = sizes < min_cluster
                drop[0] = False
                kept[drop[lab]] = 0

        img_disp = new_img_like(img_k, kept)
        if smooth_fwhm_display:
            img_disp = smooth_img(img_disp, smooth_fwhm_display)

        display = plot_stat_map(
            img_disp,
            bg_img=anat_rs,
            display_mode="z",
            cut_coords=display_cuts,
            threshold=0,
            colorbar=False,
            title=f"ICA component {k} (top {100 - percentile:.1f}% |values| in-mask)",
        )
        if out_png is not None:
            display.savefig(str(out_png), dpi=dpi)
        display.close()

    print(
        f"[plot_ica_overlays_axial] Plotted {n_components} components."
        + (f" Saved to {out_dir}." if out_dir else "")
    )


# -------------------- Similarity --------------------


def similarity_first5_abs_corr(
    components_img,
    mask_img,
    out_csv: Optional[str] = None,
    out_png: Optional[str] = None,
) -> pd.DataFrame:
    """Compute |Pearson r| among the first 5 spatial maps and optionally save a CSV/PNG."""
    n_total = components_img.shape[-1]
    n_use = min(5, n_total)

    masker = NiftiMasker(mask_img=mask_img, standardize=False, detrend=False)
    masker.fit()
    vecs = [
        masker.transform(index_img(components_img, k)).ravel() for k in range(n_use)
    ]
    V = np.vstack(vecs)

    sim = np.abs(np.corrcoef(V))
    labels = [f"IC{k}" for k in range(n_use)]
    df = pd.DataFrame(sim, index=labels, columns=labels)

    if out_csv:
        df.to_csv(out_csv, float_format="%.6f")
    if out_png:
        plt.figure(figsize=(4.5, 4))
        im = plt.imshow(sim, vmin=0, vmax=1, interpolation="nearest")
        plt.title("Pairwise |corr| (first 5 ICs)")
        plt.xticks(range(n_use), labels, rotation=45, ha="right")
        plt.yticks(range(n_use), labels)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()

    return df
