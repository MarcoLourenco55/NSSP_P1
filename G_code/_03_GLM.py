from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting, image
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_design_matrix
from nilearn.datasets import fetch_atlas_aal
from nilearn.reporting import get_clusters_table

def run_glm_analysis(
    project_root: Path,
    func_path: Path,
    t1w_path: Path,
    events_lr: Path,
    events_rl: Path,
    tr: float = 0.72,
    hrf: str = "spm",
    noise_model: str = "ar1",
    drift_model: str = "polynomial",
    drift_order: int = 3,
    fdr_rate: float = 0.05,
    cluster_size: int = 40,
    cut_coords: int = 8,
) -> None:
    """
    Run first-level GLM analysis on concatenated fMRI (LR + RL) runs.

    Parameters
    ----------
    project_root : Path
        Project root directory.
    func_path : Path
        Path to preprocessed fMRI 4D NIfTI file (concatenated LR + RL).
    t1w_path : Path
        Path to skull-stripped T1w anatomical image.
    events_lr, events_rl : Path
        Paths to the event CSV files for LR and RL runs.
    """

    # --- Output directory
    out_dir = project_root / "GLM_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[GLM] Running GLM analysis")
    print(f"[GLM] Functional: {func_path}")
    print(f"[GLM] Anatomical: {t1w_path}")
    print(f"[GLM] Events LR:  {events_lr}")
    print(f"[GLM] Events RL:  {events_rl}")
    print(f"[GLM] Output dir: {out_dir}")

    # --- Helper functions
    def load_events(p: Path) -> pd.DataFrame:
        df = pd.read_csv(p)
        if "trial_type" not in df.columns and "condition" in df.columns:
            df = df.rename(columns={"condition": "trial_type"})
        df["trial_type"] = df["trial_type"].fillna("").astype(str)
        df = df[df["trial_type"].str.strip() != ""].copy()
        df["trial_type"] = df["trial_type"].str.lower().str.replace(" ", "_")
        return df[["onset", "duration", "trial_type"]].reset_index(drop=True)

    def build_contrasts(cols):
        cols = [c.lower() for c in cols]
        n = len(cols)
        def condition(name):
            v = np.zeros(n)
            if name in cols:
                v[cols.index(name)] = 1.0
            return v
        return {
            "lh_vs_baseline": condition("lh"),
            "rh_vs_baseline": condition("rh"),
            "lf_vs_baseline": condition("lf"),
            "rf_vs_baseline": condition("rf"),
            "t_vs_baseline":  condition("t"),
            "Hand_vs_Foot":   0.5*condition("lh") + 0.5*condition("rh") - 0.5*condition("lf") - 0.5*condition("rf"),
        }

    def save_clusters_with_aal(z_map_img, stat_thr, cluster_k, out_csv):
        tbl = get_clusters_table(
            z_map_img,
            stat_threshold=float(stat_thr),
            cluster_threshold=int(cluster_k),
        )

        atlas = fetch_atlas_aal(version="SPM12")
        labels_res = image.resample_to_img(atlas["maps"], z_map_img, interpolation="nearest")
        lab = labels_res.get_fdata().astype(int)
        aff = labels_res.affine

        labels = list(atlas["labels"])
        indices = list(atlas["indices"])
        idx2name = {int(idx): name for idx, name in zip(indices, labels)}
        inv_aff = np.linalg.inv(aff)

        def mni_to_ijk(x, y, z):
            ijk = nib.affines.apply_affine(inv_aff, [x, y, z])
            return tuple(np.round(ijk).astype(int))

        names = []
        for x, y, z in zip(tbl["X"], tbl["Y"], tbl["Z"]):
            i, j, k = mni_to_ijk(float(x), float(y), float(z))
            val = int(lab[i, j, k])
            names.append(idx2name.get(val, "Unknown"))
        out = tbl.copy()
        out["AAL_at_peak"] = names
        out.to_csv(out_csv, index=False)

    # --- Load images & events
    func = nib.load(str(func_path))
    t1w  = nib.load(str(t1w_path))
    ev_lr = load_events(events_lr)
    ev_rl = load_events(events_rl)

    # --- Concatenate events (LR -> RL)
    split_time = float((ev_lr["onset"] + ev_lr["duration"]).max())
    ev_rl2 = ev_rl.copy()
    ev_rl2["onset"] = ev_rl2["onset"] + split_time
    ev = pd.concat([ev_lr, ev_rl2], ignore_index=True)

    # --- Design matrix
    frame_times = np.arange(func.shape[-1]) * tr
    dm = make_first_level_design_matrix(
        frame_times=frame_times,
        events=ev,
        hrf_model=hrf,
        drift_model=drift_model,
        drift_order=drift_order,
        high_pass=None
    )
    dm["run2_const"] = (frame_times >= split_time).astype(float)

    # --- Save design matrix
    ax = plot_design_matrix(dm)
    ax.figure.suptitle("Design (poly deg=3) + run2_const", y=1.02)
    ax.figure.savefig(out_dir / "design_matrix_concat.png", dpi=160, bbox_inches="tight")
    plt.close(ax.figure)
    dm.to_csv(out_dir / "design_matrix_concat.csv", index=False)

    # --- Fit GLM
    glm = FirstLevelModel(t_r=tr, hrf_model=hrf, noise_model=noise_model, standardize=False)
    glm = glm.fit(func, design_matrices=[dm])
    cons = build_contrasts(dm.columns)

    def run_and_plot(z_map, title_prefix, fname_prefix):
        clean_map, threshold = threshold_stats_img(
            z_map, alpha=fdr_rate, height_control='fdr', cluster_threshold=cluster_size
        )

        plotting.plot_stat_map(
            clean_map, bg_img=t1w, threshold=float(threshold),
            display_mode='z', cut_coords=cut_coords, black_bg=True,
            title=f"{title_prefix} (FDR={fdr_rate}, thr={threshold:.2f}, k>{cluster_size})"
        ).savefig(out_dir / f"{fname_prefix}_zcuts.png", dpi=160, bbox_inches="tight")

        nib.save(z_map, out_dir / f"{fname_prefix}_z.nii.gz")
        nib.save(clean_map, out_dir / f"{fname_prefix}_clean.nii.gz")
        return float(threshold), clean_map

    # --- Run all contrasts
    for name in ["lh_vs_baseline","rh_vs_baseline","lf_vs_baseline","rf_vs_baseline","t_vs_baseline"]:
        z_map = glm.compute_contrast(cons[name], output_type="z_score")
        run_and_plot(z_map, title_prefix=name, fname_prefix=name)

    # --- Hand vs Foot
    hand_vec = cons["Hand_vs_Foot"]
    np.savetxt(out_dir / "Hand_vs_Foot_contrast.txt", hand_vec)

    z_map = glm.compute_contrast(hand_vec, output_type="z_score")
    thr_hf, clean_hf = run_and_plot(z_map, "Hand_vs_Foot", "Hand_vs_Foot")

    # --- AAL overlay + cluster table
    atlas = fetch_atlas_aal(version="SPM12")
    labels_res = image.resample_to_img(atlas["maps"], clean_hf, interpolation="nearest")
    disp = plotting.plot_stat_map(
        clean_hf, bg_img=t1w, threshold=float(thr_hf),
        display_mode='z', cut_coords=cut_coords, black_bg=True,
        title=f"Hand_vs_Foot + AAL (FDR={fdr_rate}, k≥{cluster_size})"
    )
    disp.add_contours(labels_res, linewidths=0.6)
    disp.savefig(out_dir / "Hand_vs_Foot_AAL_overlay.png", dpi=160, bbox_inches="tight")
    disp.close()

    save_clusters_with_aal(z_map, thr_hf, cluster_size, out_dir / "Hand_vs_Foot_clusters_with_AAL.csv")

    print(f"[GLM] Finished successfully → {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run first-level GLM analysis")
    parser.add_argument("--project_root", type=Path, required=True, help="Project root directory")
    parser.add_argument("--func", type=Path, required=True, help="Functional NIfTI (concatenated runs)")
    parser.add_argument("--t1w", type=Path, required=True, help="Anatomical T1w (skull-stripped)")
    parser.add_argument("--events_lr", type=Path, required=True, help="Events CSV (LR run)")
    parser.add_argument("--events_rl", type=Path, required=True, help="Events CSV (RL run)")
    args = parser.parse_args()

    run_glm_analysis(
        project_root=args.project_root,
        func_path=args.func,
        t1w_path=args.t1w,
        events_lr=args.events_lr,
        events_rl=args.events_rl,
    )
