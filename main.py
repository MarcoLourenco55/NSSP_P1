from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nibabel as nib
from nilearn.image import load_img, mean_img, math_img, new_img_like, resample_to_img
from nilearn.maskers import NiftiMasker
from nilearn.masking import compute_epi_mask

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from utils import ICA as ica


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _log(step: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {step}")


def _paths_from_project_root(project_root: Path) -> dict[str, Path]:
    """Canonical project inputs."""
    return {
        "fmri": project_root
        / "Preprocessing"
        / "Functional"
        / "subj_concat_var1_mc_s5mm.nii",
        "anat": project_root
        / "Preprocessing"
        / "Structural"
        / "Skull striping"
        / "T1w_bet.nii",
        "gm_pve": project_root
        / "Preprocessing"
        / "Structural"
        / "Segmentation"
        / "T1w_fast_segmentation_pve_1.nii",
        "task_json": project_root / "data" / "task-motor_bold.json",
    }


def _load_tr(task_json: Path) -> Optional[float]:
    """Return RepetitionTime from BIDS JSON if present."""
    if not task_json.exists():
        return None
    with open(task_json, "r") as f:
        meta = json.load(f)
    tr = meta.get("RepetitionTime", None)
    return float(tr) if tr is not None else None


def find_project_root(
    start: Path = Path.cwd(),
    required: tuple[str, str] = ("Preprocessing", "data"),
) -> Path:
    """Walk upward from `start` until a folder containing all `required` entries is found."""
    here = start.resolve()
    for _ in range(8):
        if all((here / r).exists() for r in required):
            return here
        here = here.parent
    raise FileNotFoundError(
        "Couldn't find a folder containing both 'Preprocessing' and 'data' above your current directory."
    )


# ---------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------
def _mask_in_functional_space(func_img, gm_pve_path: Path, gm_thresh: float = 0.25):
    """
    Build a binary mask in functional space.
    If GM PVE1 exists, resample to the functional grid and threshold; otherwise
    compute an EPI mask from the mean functional. Final mask is resampled with
    nearest interpolation to avoid fractional labels.
    """
    if gm_pve_path is not None and gm_pve_path.exists():
        gm_pve = load_img(str(gm_pve_path))
        gm_pve_func = resample_to_img(
            gm_pve,
            func_img,
            interpolation="continuous",
            force_resample=True,
            copy_header=True,
        )
        mask = math_img(f"img > {gm_thresh}", img=gm_pve_func)
        mask = new_img_like(mask, (mask.get_fdata() > 0).astype("int16"))
    else:
        mask = compute_epi_mask(mean_img([func_img]))

    mask = resample_to_img(
        mask, func_img, interpolation="nearest", force_resample=True, copy_header=True
    )
    return mask


# ---------------------------------------------------------------------
# Extraction + PCA with caching
# ---------------------------------------------------------------------
def _extract_masked_timeseries(
    fmri_runs: list[Path],
    mask_img,
    outdir: Path,
    t_r: Optional[float],
    force: bool,
) -> Tuple[np.ndarray, NiftiMasker]:
    """
    Vectorize masked fMRI into (n_timepoints × n_voxels). Uses a simple on-disk
    cache to avoid recomputation across runs.
    """
    cache_path = outdir / "X_masked.npy"

    masker = NiftiMasker(
        mask_img=mask_img,
        standardize=True,
        detrend=True,
        smoothing_fwhm=None,
        t_r=t_r,
        reports=False,
        memory=str(outdir / "nilearn_cache"),
        memory_level=1,
        verbose=1,
    )
    masker.fit()

    if cache_path.exists() and not force:
        _log("Loading cached masked time series (X_masked.npy)")
        X = np.load(cache_path, mmap_mode="r")
        return X, masker

    _log("Extracting masked time series")
    t0 = time.time()
    X_list = []
    for i, run in enumerate(fmri_runs, 1):
        _log(f"  → run {i}/{len(fmri_runs)}: {run}")
        X_list.append(
            masker.transform(load_img(str(run))).astype("float32", copy=False)
        )
    X = np.vstack(X_list)
    np.save(cache_path, X)
    _log(
        f"Saved masked time series → {cache_path}  ({X.shape}), {time.time() - t0:.1f}s"
    )
    return X, masker


def _choose_components_with_elbow(X: np.ndarray, outdir: Path, force: bool) -> int:
    """Select n_components via Kneedle on the scree; save plot and CSV artifacts."""
    pcaplot = outdir / "pca_knee_scree.png"
    scree_csv = outdir / "pca_scree_eigvals.csv"
    cumvar_csv = outdir / "pca_cumvar.csv"

    if pcaplot.exists() and scree_csv.exists() and not force:
        _log("Using existing PCA knee outputs (pca_scree_eigvals.csv/png)")
        try:
            eigvals = np.loadtxt(scree_csv, delimiter=",")
            k, diag = ica.knee_from_eigvals(eigvals, S=0.7, interp_method="interp1d")
            _log(f"Recovered knee from scree → n_components ≈ {k}  (diag: {diag})")
            return max(int(k), 5)
        except Exception:
            pass

    _log("Computing PCA knee on eigenvalues (Kneedle) with scree plot")
    return max(
        ica.choose_n_components_elbow(
            X,
            plot_path=str(pcaplot),
            scree_csv_path=str(scree_csv),
            cumvar_csv_path=str(cumvar_csv),
            S=0.7,
            interp_method="interp1d",
            logy=True,
        ),
        5,
    )


# ---------------------------------------------------------------------
# ICA with checkpointing
# ---------------------------------------------------------------------
def _run_or_load_ica(
    fmri_runs: list[Path],
    mask_img,
    n_components: int,
    t_r: Optional[float],
    outdir: Path,
    force: bool,
):
    """Load existing components or fit CanICA and save the result."""
    comp_path = outdir / "ica_components.nii.gz"
    if comp_path.exists() and not force:
        _log(f"Skipping ICA fit (found {comp_path})")
        return nib.load(str(comp_path))

    _log(f"Fitting CanICA (n_components={n_components}, runs={len(fmri_runs)})")
    t0 = time.time()
    components_img = ica.run_ica_canica(
        [str(p) for p in fmri_runs],
        mask_img=mask_img,
        n_components=n_components,
        t_r=t_r,
        random_state=0,
    )
    nib.save(components_img, str(comp_path))
    _log(
        f"Saved components → {comp_path}  {components_img.shape}, {time.time() - t0:.1f}s"
    )

    _log("Saving individual component volumes")
    for k in range(components_img.shape[-1]):
        out_k = outdir / f"ic_{k:02d}.nii.gz"
        if out_k.exists() and not force:
            continue
        nib.save(ica.index_img(components_img, k), str(out_k))
    return components_img


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------
def _posthoc_reports(
    components_img, anat_path: Path, mask_img, outdir: Path, force: bool
) -> None:
    """Write component overlays and a small similarity report (first 5 ICs)."""
    overlay_dir = outdir / "overlays_axial"
    _log("Generating axial overlays")
    ica.plot_ica_overlays_axial(
        components_img,
        anat_img=str(anat_path),
        out_dir=str(overlay_dir),
        display_cuts=8,
        dpi=120,
        skip_existing=True,
        verbose=True,
    )
    _log(f"Overlay PNGs in: {overlay_dir}")

    sim_csv = outdir / "similarity_first5.csv"
    sim_png = outdir / "similarity_first5.png"
    if sim_csv.exists() and sim_png.exists() and not force:
        _log("Skipping similarity (existing CSV/PNG)")
    else:
        _log("Computing similarity among first 5 components")
        ica.similarity_first5_abs_corr(
            components_img,
            mask_img=mask_img,
            out_csv=str(sim_csv),
            out_png=str(sim_png),
        )
        _log(f"Saved similarity → {sim_csv}, {sim_png}")


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------
def main() -> None:
    """End-to-end orchestration with caching and simple progress logs."""
    force = "--force" in sys.argv

    project_root = find_project_root()
    outdir = SCRIPT_DIR / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    paths = _paths_from_project_root(project_root)
    fmri_runs = [paths["fmri"]]
    anat_path = paths["anat"]
    gm_pve_path = paths["gm_pve"]
    t_r = _load_tr(paths["task_json"])

    _log(f"Project root: {project_root}")
    _log(f"fMRI runs: {len(fmri_runs)}")
    for i, p in enumerate(fmri_runs, 1):
        _log(f"  run {i}: {p}")
    _log(f"Anatomy: {anat_path}")
    _log(f"GM PVE1: {gm_pve_path}")
    _log(f"Outputs: {outdir}")
    _log(f"t_r = {t_r} (None means not used)")
    if force:
        _log("FORCE mode: existing artifacts will be recomputed")

    mask_path = outdir / "mask_func_space.nii.gz"
    if mask_path.exists() and not force:
        _log(f"Skipping mask build (found {mask_path})")
        mask_img = nib.load(str(mask_path))
    else:
        _log("Building mask in functional space")
        t0 = time.time()
        func_img0 = load_img(str(fmri_runs[0]))
        mask_img = _mask_in_functional_space(func_img0, gm_pve_path)
        nib.save(mask_img, str(mask_path))
        _log(f"Saved mask → {mask_path}  {time.time() - t0:.1f}s")

    if "func_img0" not in locals():
        func_img0 = load_img(str(fmri_runs[0]))
    X, _ = _extract_masked_timeseries(fmri_runs, mask_img, outdir, t_r, force)
    n_components = _choose_components_with_elbow(X, outdir, force)

    components_img = _run_or_load_ica(
        fmri_runs, mask_img, n_components, t_r, outdir, force
    )

    _posthoc_reports(components_img, anat_path, mask_img, outdir, force)

    _log("Done.")


if __name__ == "__main__":
    main()
