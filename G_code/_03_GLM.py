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

project_root = Path("/Users/amadeus/Documents/Neural Signal Processing/NSSPProj1")
out_dir = project_root / "GLM_outputs"
out_dir.mkdir(parents=True, exist_ok=True)

func_path = project_root / "NSSP/functional preprocessing/subj_concat_var1_mc_s6mm.nii.gz"
t1w_path  = project_root / "NSSP/T1w/T1w.nii.gz"
events_lr = project_root / "fMRI/tfMRI_MOTOR_LR/events_LR.csv"
events_rl = project_root / "fMRI/tfMRI_MOTOR_RL/events_RL.csv"

# GLM params
tr = 0.72
hrf = "spm"
noise_model = "ar1"
drift_model = "polynomial"
drift_order = 3
fdr_rate = 0.05
cluster_size = 40
cut_coords = 8

def load_events(p: Path) -> pd.DataFrame:
    """BIDS Version"""
    df = pd.read_csv(p)
    if "trial_type" not in df.columns and "condition" in df.columns:
        df = df.rename(columns={"condition": "trial_type"})
    df["trial_type"] = df["trial_type"].fillna("").astype(str)
    df = df[df["trial_type"].str.strip() != ""].copy()
    df["trial_type"] = df["trial_type"].str.lower().str.replace(" ", "_")
    return df[["onset", "duration", "trial_type"]].reset_index(drop=True)

def build_contrasts(cols):
    """contrast vector dict in 'conditions[...]' style"""
    cols = [c.lower() for c in cols]
    n = len(cols)
    def condition(name):
        v = np.zeros(n)
        if name in cols:
            v[cols.index(name)] = 1.0
        return v
    return {
        "lh_vs_baseline": condition("lh"),  # left hand
        "rh_vs_baseline": condition("rh"),  # right hand
        "lf_vs_baseline": condition("lf"),  # left foot
        "rf_vs_baseline": condition("rf"),  # right foot
        "t_vs_baseline":  condition("t"),   # tongue
        "Hand_vs_Foot":   0.5*condition("lh") + 0.5*condition("rh") - 0.5*condition("lf") - 0.5*condition("rf"),
    }

def save_clusters_with_aal(z_map_img, stat_thr, cluster_k, out_csv):
    # clusters table
    tbl = get_clusters_table(
        z_map_img,
        stat_threshold=float(stat_thr),
        cluster_threshold=int(cluster_k),
    )

    # atlas
    atlas = fetch_atlas_aal(version="SPM12")
    labels_res = image.resample_to_img(atlas["maps"], z_map_img, interpolation="nearest")
    lab = labels_res.get_fdata().astype(int)   # integer atlas indices
    aff = labels_res.affine                    # voxel↔MNI transform

    # dict
    labels = list(atlas["labels"])
    indices = list(atlas["indices"])
    idx2name = {int(idx): name for idx, name in zip(indices, labels)}
    inv_aff = np.linalg.inv(aff)

    def mni_to_ijk(x, y, z):
        ijk = nib.affines.apply_affine(inv_aff, [x, y, z])
        return tuple(np.round(ijk).astype(int))

    # match
    names = []
    for x, y, z in zip(tbl["X"], tbl["Y"], tbl["Z"]):
        i, j, k = mni_to_ijk(float(x), float(y), float(z))
        val = int(lab[i, j, k])
        names.append(idx2name[val])
    out = tbl.copy()
    out["AAL_at_peak"] = names
    out.to_csv(out_csv, index=False)

def main():
    func = nib.load(str(func_path))
    t1w  = nib.load(str(t1w_path))

    # concat order: LR -> RL
    ev_lr = load_events(events_lr)
    ev_rl = load_events(events_rl)
    split_time = float((ev_lr["onset"] + ev_lr["duration"]).max())
    ev_rl2 = ev_rl.copy(); ev_rl2["onset"] = ev_rl2["onset"] + split_time
    ev = pd.concat([ev_lr, ev_rl2], ignore_index=True)

    # design matrix
    frame_times = np.arange(func.shape[-1]) * tr
    dm = make_first_level_design_matrix(
        frame_times=frame_times,
        events=ev,
        hrf_model=hrf,
        drift_model=drift_model,
        drift_order=drift_order,
        high_pass=None
    )
    # improve performance
    dm["run2_const"] = (frame_times >= split_time).astype(float)

    # save DM
    ax = plot_design_matrix(dm)
    ax.figure.suptitle("Design (poly deg=3) + run2_const", y=1.02)
    ax.figure.savefig(out_dir / "design_matrix_concat.png", dpi=160, bbox_inches="tight")
    plt.close(ax.figure)
    dm.to_csv(out_dir / "design_matrix_concat.csv", index=False)

    # GLM fit
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
            title='{} (fdr={}, threshold={}), clusters > {} voxels'.format(
                title_prefix, fdr_rate, float(threshold), cluster_size
            )
        ).savefig(out_dir / f"{fname_prefix}_zcuts.png", dpi=160, bbox_inches="tight")

        # save maps
        nib.save(z_map, out_dir / f"{fname_prefix}_z.nii.gz")
        nib.save(clean_map, out_dir / f"{fname_prefix}_fdr{int(fdr_rate*100):02d}_k{cluster_size}.nii.gz")

        return float(threshold), clean_map

    # per-task vs baseline
    for name in ["lh_vs_baseline","rh_vs_baseline","lf_vs_baseline","rf_vs_baseline","t_vs_baseline"]:
        z_map = glm.compute_contrast(cons[name], output_type="z_score")
        run_and_plot(z_map, title_prefix=name, fname_prefix=name)

    # Hand vs Foot
    hand_vec = cons["Hand_vs_Foot"]
    with open(out_dir / "Hand_vs_Foot_contrast.txt", "w") as f:
        f.write("# columns:\n" + ", ".join([str(c) for c in dm.columns]) + "\n")
        f.write("# vector (Hand_vs_Foot):\n" + ", ".join([f"{x:.3f}" for x in hand_vec.tolist()]) + "\n")

    z_map = glm.compute_contrast(hand_vec, output_type="z_score")
    thr_hf, clean_hf = run_and_plot(z_map, title_prefix="Hand_vs_Foot", fname_prefix="Hand_vs_Foot")

    # AAL overlay
    atlas = fetch_atlas_aal(version="SPM12")
    labels_res = image.resample_to_img(atlas["maps"], clean_hf, interpolation="nearest")
    disp = plotting.plot_stat_map(
        clean_hf, bg_img=t1w, threshold=float(thr_hf),
        display_mode='z', cut_coords=cut_coords, black_bg=True,
        title="Hand_vs_Foot + AAL (FDR={}, k≥{})".format(fdr_rate, cluster_size)
    )
    disp.add_contours(labels_res, linewidths=0.6)
    disp.savefig(out_dir / "Hand_vs_Foot_AAL_overlay_zcuts.png", dpi=160, bbox_inches="tight")
    disp.close()

    # AAL label
    save_clusters_with_aal(z_map, thr_hf, cluster_size, out_dir / "Hand_vs_Foot_clusters_with_AAL.csv")
    # for neg record
    foot_vec = -1 * hand_vec
    with open(out_dir / "Foot_vs_Hand_contrast.txt", "w") as f:
        f.write("# columns:\n" + ", ".join([str(c) for c in dm.columns]) + "\n")
        f.write("# vector (Foot_vs_Hand):\n" + ", ".join([f"{x:.3f}" for x in foot_vec.tolist()]) + "\n")
    z_map_fh = glm.compute_contrast(foot_vec, output_type="z_score")
    thr_fh, clean_fh = run_and_plot(
        z_map_fh, 
        title_prefix="Foot_vs_Hand (Negative of H-F)", 
        fname_prefix="Foot_vs_Hand"
    )
    save_clusters_with_aal(
        z_map_fh, 
        thr_fh, 
        cluster_size, 
        out_dir / "Foot_vs_Hand_clusters_with_AAL.csv"
    )

if __name__ == "__main__":
    main()
