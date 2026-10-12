import numpy as np
import os, logging, argparse, json, shutil
from pathlib import Path

from pymol import cmd
from Bio.PDB import PDBParser, NeighborSearch
from Bio.SVDSuperimposer import SVDSuperimposer


if "NUMEXPR_MAX_THREADS" not in os.environ:
    os.environ["NUMEXPR_MAX_THREADS"] = "16"

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='evaluate docking result')
    parser.add_argument('--native_dir', type=str, required=True,
                        help="native data directory")
    parser.add_argument('--model_dir', type=str, required=True,
                        help="docking result directory")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="metrics output directory")
    parser.add_argument('--mode', type=str, required=True,
                        help="docking mode")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    return args

def summary_json(json_file: Path, result_dir: Path):
    import os
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    def plot_rmsd_boxplot(df_melted, title, save_path):
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df_melted,
            x="Metric",
            y="Value",
            hue="Metric",
            palette="Set2",
            width=0.6,
            linewidth=1.5,
            fliersize=3,
            legend=False
        )
        plt.title(title, fontsize=14)
        plt.ylabel("RMSD Value", fontsize=12)
        plt.xlabel("Metric", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    metric_cols = ["rmsd", "ligand_rmsd", "interface_rmsd", "ring_srmsd"]

    def compute_summary(df):
        for c in metric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        stats = df[metric_cols].describe().T
        stats["median"] = df[metric_cols].fillna(np.inf).median()
        stats = stats.reset_index().rename(columns={"index": "Metric"})
        cols = ["Metric"] + [col for col in stats.columns if col != "Metric"]
        return stats[cols]

    with json_file.open("r") as fin:
        data = json.load(fin)
    df = pd.DataFrame(data)
    df["pdb_id"] = df["name"].str[:4].str.upper()
    df_melt = df.melt(id_vars="pdb_id", value_vars=metric_cols, var_name="Metric", value_name="Value")
    df_melt_plot = df_melt.dropna(subset=["Value"])
    if len(df_melt_plot) > 0:
        plot_rmsd_boxplot(
            df_melt_plot, "Box Plot of RMSD Metrics", f"{str(result_dir)}/rmsd_box.png"
        )
    summary = compute_summary(df)
    summary_csv_path = f"{result_dir}/rmsd_stats.csv"
    summary.to_csv(summary_csv_path, index=False)
    return summary


def align_and_compute_all_metrics(
    target_pdb,
    result_pdb,
    interface_distance=8.0,
):
    """
    Compute:
      1) Global RMSD (for all matched atoms)
      2) Ligand RMSD
      3) Interface RMSD
      4) Ligand self-aligned RMSD (Ring-SRMSD example)
    """

    parser = PDBParser(QUIET=True)
    structure_ref = parser.get_structure("ref", target_pdb)[0]
    structure_mob = parser.get_structure("mob", result_pdb)[0]

    def get_chains(atom_dict):
        chains = []
        for k in atom_dict:
            if k[0] not in chains:
                chains.append(k[0])
        return chains


    def is_hydrogen(atom):
        return atom.element == 'H' or atom.name.startswith('H')

    def get_atom_key(atom):
        """
        Return a unique key for each atom: (chain_id, residue_id, atom_name).
        Only use residue.id[1] as residue index for simplicity.
        """
        residue = atom.get_parent()
        chain = residue.get_parent()
        # residue.id belike (het_flag, seq_id, insertion_code)
        return (chain.id, residue.id[1], atom.name)

    def build_atom_dict(structure, skip_h=True):
        """
        Build a dictionary from atom key -> atom object.
        By default, hydrogen atoms are skipped.
        """
        atom_dict = {}
        for chain in structure:
            for residue in chain:
                for atom in residue:
                    if skip_h and is_hydrogen(atom):
                        continue
                    key = get_atom_key(atom)
                    atom_dict[key] = atom
        return atom_dict

    def extract_matched_coords(ref_dict, mob_dict, chain_filter=None):
        """
        Extract matched coordinates based on common atom keys.
        Optionally filter by chain ID(s).
        Returns: (ref_coords, mob_coords, matched_keys)
        """
        matched_keys = []
        for k in ref_dict.keys():
            if k in mob_dict:
                if chain_filter is not None:
                    if k[0] not in chain_filter:
                        continue
                matched_keys.append(k)
        matched_keys.sort()

        ref_coords = []
        mob_coords = []
        for k in matched_keys:
            ref_coords.append(ref_dict[k].coord)
            mob_coords.append(mob_dict[k].coord)

        return np.array(ref_coords), np.array(mob_coords), matched_keys

    def compute_rmsd(coords1, coords2):
        if len(coords1) == 0:
            return None
        diff = coords1 - coords2
        return np.sqrt(np.mean(np.sum(diff * diff, axis=1)))

    def superimpose_and_transform(mob_atoms, ref_coords, mob_coords):
        sup = SVDSuperimposer()
        sup.set(ref_coords, mob_coords)
        sup.run()
        rot, tran = sup.get_rotran()

        for atom in mob_atoms:
            atom.transform(rot, tran)
        return sup.get_rms()

    ref_atom_dict = build_atom_dict(structure_ref)
    mob_atom_dict = build_atom_dict(structure_mob)
    all_mob_atoms = list(structure_mob.get_atoms())
    chains_id = get_chains(ref_atom_dict)
    ligand_chain_id = chains_id[-1]


    ref_align_coords, mob_align_coords, align_keys = extract_matched_coords(
        ref_atom_dict,
        mob_atom_dict,
        chain_filter=chains_id[:-1],
    )
    if len(align_keys) == 0:
        raise ValueError(f"can't find any non-H atom")

    superimpose_and_transform(
        mob_atoms=all_mob_atoms,
        ref_coords=ref_align_coords,
        mob_coords=mob_align_coords,
    )

    # 1.rmsd
    mob_atom_dict_after = build_atom_dict(structure_mob)
    ref_all_coords, mob_all_coords, all_keys = extract_matched_coords(
        ref_atom_dict, mob_atom_dict_after, chain_filter=None
    )
    rmsd_all = compute_rmsd(ref_all_coords, mob_all_coords)

    # 2.ligand_rmsd
    ref_lig_coords, mob_lig_coords, lig_keys = extract_matched_coords(
        ref_atom_dict,
        mob_atom_dict_after,
        chain_filter=[ligand_chain_id],
    )
    rmsd_ligand = compute_rmsd(ref_lig_coords, mob_lig_coords)

    # 3.interface_rmsd
    ref_align_atoms = []
    for k, a in ref_atom_dict.items():
        if k[0] in chains_id[:-1]:  # chain_id == align_chain_id
            ref_align_atoms.append(a)

    search = NeighborSearch(ref_align_atoms)

    interface_keys = set()
    for k in lig_keys:
        atom_ref = ref_atom_dict[k]
        interface_keys.add(k)

        neighbors = search.search(atom_ref.coord, interface_distance)
        for nb in neighbors:
            nb_key = get_atom_key(nb)
            interface_keys.add(nb_key)

    interface_keys_sorted = sorted(interface_keys)
    ref_int_coords = []
    mob_int_coords = []
    for k in interface_keys_sorted:
        if k in mob_atom_dict_after:
            ref_int_coords.append(ref_atom_dict[k].coord)
            mob_int_coords.append(mob_atom_dict_after[k].coord)
    ref_int_coords = np.array(ref_int_coords)
    mob_int_coords = np.array(mob_int_coords)
    rmsd_interface = compute_rmsd(ref_int_coords, mob_int_coords)

    # 4.Ring-SRMSD
    if len(lig_keys) == 0:
        rmsd_ligand_self = None
    else:
        sup = SVDSuperimposer()
        sup.set(ref_lig_coords, mob_lig_coords)
        sup.run()
        rot_lig, tran_lig = sup.get_rotran()

        mob_lig_coords_aligned = np.dot(mob_lig_coords, rot_lig) + tran_lig
        rmsd_ligand_self = compute_rmsd(ref_lig_coords, mob_lig_coords_aligned)

    return {
        "rmsd": rmsd_all,
        "ligand_rmsd": rmsd_ligand,
        "interface_rmsd": rmsd_interface,
        "ring_srmsd": rmsd_ligand_self
    }

def sanitize_value(v):
    """Convert any value to float, string or None to make it safe for pandas."""
    if isinstance(v, np.ndarray):
        # If it's a numpy array with one element, take the scalar
        if v.size == 1:
            return float(v[0])
        else:
            return float(np.mean(v))
    elif isinstance(v, (float, int)):
        return v
    elif v is None:
        return None
    else:
        try:
            return float(v)
        except:
            return str(v)

def sanitize_item(item):
    clean = {}
    # Put "name" first
    if "name" in item:
        clean["name"] = str(item["name"])  # ensure it's a string

    # Process other keys
    for k, v in item.items():
        if k == "name":
            continue  # already handled
        if isinstance(v, np.ndarray):
            clean[k] = float(v[0]) if v.size == 1 else [float(x) for x in v]
        elif isinstance(v, (np.float32, np.float64)):
            clean[k] = float(v)
        elif isinstance(v, (float, int)) or v is None:
            clean[k] = v
        else:
            try:
                clean[k] = float(v)
            except:
                clean[k] = str(v)
    return clean

def convert_cif_to_pdb(cif_file, pdb_file):
    obj_name = "obj_" + os.path.splitext(os.path.basename(cif_file))[0]
    try:
        cmd.load(cif_file, obj_name)
        cmd.save(pdb_file, obj_name)
        cmd.delete(obj_name)
    except Exception as e:
        print(f"Error in {obj_name}: {e}")

def _null_metrics_row(name: str):
    return {
        "name": name,
        "rmsd": None,
        "ligand_rmsd": None,
        "interface_rmsd": None,
        "ring_srmsd": None,
    }


def _resolve_pred_file(model_dir: Path, mode: str, sample: str, seed: int):
    """
    Return path to prediction PDB if present, else None.
    `sample` is the 4-char id used elsewhere in this script.
    """
    if mode == "vina":
        file_name = None
        for tmp_file_name in os.listdir(model_dir):
            if tmp_file_name.startswith(str(sample).upper()):
                file_name = tmp_file_name
                break
        if file_name is None:
            return None
        pred_file = model_dir / file_name / "result_complex.pdb"
        return pred_file if pred_file.exists() else None

    if mode == "af3":
        file_name = str(sample).lower()
        work_fold = model_dir / file_name
        if not work_fold.exists():
            return None
        pred_file_cif = work_fold / f"{file_name}_model.cif"
        pred_file = work_fold / f"{file_name}_model.pdb"
        if not pred_file.exists():
            convert_cif_to_pdb(str(pred_file_cif), str(pred_file))
        return pred_file if pred_file.exists() else None

    if mode == "protenix":
        file_name = str(sample)[:4].upper()
        work_fold = model_dir / file_name
        if not work_fold.exists():
            return None
        pred_file_cif = work_fold / f"seed_{seed}/predictions/{file_name}_sample_0.cif"
        pred_file = work_fold / f"{file_name}_sample_0.pdb"
        if not pred_file.exists():
            convert_cif_to_pdb(str(pred_file_cif), str(pred_file))
        return pred_file if pred_file.exists() else None

    if mode == "boltz2":
        file_name = str(sample)[:4].upper()
        work_fold = model_dir / file_name
        if not work_fold.exists():
            return None
        pred_file = (
            work_fold
            / f"boltz_results_{file_name}"
            / "predictions"
            / f"{file_name}"
            / f"{file_name}_model_0_reordered.pdb"
        )
        return pred_file if pred_file.exists() else None

    if mode == "chai1":
        file_name = str(sample).upper()
        work_fold = model_dir / file_name
        if not work_fold.exists():
            return None
        pred_file = work_fold / f"{file_name}.pdb"
        return pred_file if pred_file.exists() else None

    if mode == "rosetta":
        file_name = str(sample)[:4].upper()
        pred_file = model_dir / f"{file_name}.pdb"
        return pred_file if pred_file.exists() else None

    return None


def evaluate(model_dir, native_dir, mode, output_dir="metrics", seed: int = 42):
    assert (
        mode in ["vina", "af3", "protenix", "boltz2", "chai1", "rosetta"]
        ), "mode is not defined, only 'vina', 'af3', 'protenix', 'boltz2', 'chai1' and 'rosetta' is admitted"

    logger = logging.getLogger(__name__)

    logger.info(f"computing {mode} metrics")
    model_dir = Path(model_dir)
    native_dir = Path(native_dir)
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    results = []
    n_total = 0
    n_present = 0

    for sample_path in sorted(native_dir.iterdir()):
        n_total += 1
        sample_name = sample_path.name
        gt_file = native_dir / sample_name
        sample = sample_name[:4]

        pred_file = _resolve_pred_file(model_dir, mode, sample, seed)
        if pred_file is None:
            logger.warning(f"prediction missing for {sample} (mode={mode})")
            results.append(_null_metrics_row(sample))
            continue

        try:
            metrics = align_and_compute_all_metrics(
                target_pdb=str(gt_file),
                result_pdb=str(pred_file),
                interface_distance=4.0
            )
            metrics["name"] = sample
            results.append(metrics)
            n_present += 1
        except Exception as e:
            logger.error(f"compute metrics encounter error in {sample}: {e}")
            results.append(_null_metrics_row(sample))


    cleaned_results = [sanitize_item(item) for item in results]
    metrics_file = output_dir / "metrics.json"
    with metrics_file.open("w") as f:
        json.dump(cleaned_results, f, indent=4)
    summary = summary_json(metrics_file, output_dir)
    logger.info(f"========================== {mode} metrics ==========================")
    logger.info(
        f"predictions present (successful compute): {n_present}/{n_total}"
    )
    print(f"predictions present (successful compute): {n_present}/{n_total}")
    logger.info(summary)
    logger.info(f"Saved result to {str(output_dir)} dir")


    median_values = summary['median'].tolist()
    median_str = ' '.join(f"{x:.6f}" for x in median_values)
    print(median_str)


def compute_ratio(dir_path):
    import json
    from pathlib import Path
    import os

    json_path = os.path.join(dir_path, "metrics.json")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    def _ligand_rmsd_for_threshold(item):
        v = item.get("ligand_rmsd")
        if v is None:
            return float("inf")
        return v

    lt2 = sum(1 for item in data if _ligand_rmsd_for_threshold(item) < 2)
    lt4 = sum(1 for item in data if _ligand_rmsd_for_threshold(item) < 4)

    print(f"ligand_rmsd < 2  : {lt2}/{total} = {lt2/total:.2%}")
    print(f"ligand_rmsd < 4  : {lt4}/{total} = {lt4/total:.2%}")


if __name__ == "__main__":
    args = parse_args()
    model_dir = Path(args.model_dir)
    native_dir = Path(args.native_dir)
    output_dir = Path(args.output_dir)
    mode = args.mode

    output_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=f"{str(output_dir)}/metrics.log",
        filemode="w"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    evaluate(
        model_dir=model_dir,
        native_dir=native_dir,
        mode=mode,
        output_dir=output_dir,
        seed=args.seed,
    )

    compute_ratio(dir_path=args.output_dir)
