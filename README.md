# ProtGlycanDock Code Repository

This repository contains the code needed to reproduce the workflow with finetuning, inference, and metric computation of ProtGlycanDock.

The expected workflow is:

1. Install the environment.
2. Put the provided data and checkpoints under `release_data/`.
3. Run one of the `finetune_*.sh` scripts.
4. Run inference with `run_inference.py` or `inference.sh`.
5. Compute metrics with `metrics/compute_metrics.py`.



## Repository Layout

- `protenix/`, `runner/`, `configs/`: model, training, inference, and configuration code.
- `finetune_*.sh`: fine-tuning entry points for each ablation setting.
- `inference.sh`: single-JSON inference wrapper.
- `run_inference.py`: multi-JSON inference launcher with GPU scheduling.
- `metrics/compute_metrics.py`: RMSD, ligand RMSD, interface RMSD, and ring self-aligned RMSD evaluation.
- `metrics/generate_gt_complex_pdb.py`: optional helper for building metric-ready native complex PDB files from packaged structures.
- `scripts/gen_ccd_cache.py`: optional CCD cache generation helper.



## Environment

Python 3.11 and CUDA-capable PyTorch are recommended. Install with:

```bash
cd /path/to/ProtGlycanDock
pip install -r requirements.txt
pip install -e .
```

For the optional CUDA kernels, set `CUTLASS_PATH` before training or inference if you use DeepSpeed Evoformer attention. The provided scripts use `triattention` and `cuequivariance` by default.



## Data And Checkpoints

The scripts assume the data bundle is unpacked at:

```text
ProtGlycanDock/
  release_data/
    checkpoint/
      protenix_base_default_v0.5.0.pt
      protenix_base_constraint_v0.5.0.pt
    components.cif	(or components.v20240608.cif)
    components.cif.rdkit_mol.pkl	(or components.v20240608.cif.rdkit_mol.pkl)
    clusters-by-entity-40.txt
    split_dict.pkl
    mono.pkl
    mmcif/
    	<sample_id>.pdb
    indices/
    	glyco_train.csv
    	glyco_valid.csv
    	glyco_protein_holdout_test.csv
    	glyco_glycan_holdout_test.csv
    mmcif_bioassembly/
			<sample_id>.pkl.gz
    json/	
      <sample_id>.json
    native_complex_pdb/
      <sample_id>.pdb
```

These data comes from three sources:

1. Provided in offical websites: ``checkpoint`` ``components.v20240608.cif`` ``components.v20240608.cif.rdkit_mol.pkl``  ``clusters-by-entity-40.txt``
2. Provided in this code repo: ``mono.pkl``
3. Provided in our [full dataset repo](https://huggingface.co/datasets/ProtGlycanDock/ProtGlycanDock): ``mmcif``  ``json``
4. Can be constructed based on these data: ``indices`` ``mmcif_bioassembly ``  ``split_dict.pkl``



If your data is elsewhere, set:

```bash
export PROTENIX_DATA_ROOT_DIR=/path/to/release_data
```

Fine-tuning reads training and validation datasets from `configs/configs_data.py`. The bundled `finetune_*.sh` scripts use `glyco_train_all_on` and `glyco_valid_all_on` (see each script’s `--data.train_sets` / `--data.test_sets`).



## Finetuning

Run a finetuning variant from the repository root. For example:

```bash
bash finetune_default_glycan_chi_0.01.sh
```

The bundled `finetune_*.sh` scripts are minimal entry points: adjust `export PROTENIX_DATA_ROOT_DIR`, `checkpoint_path`, `--base_dir`, `torchrun --nproc_per_node`, and the `runner/train.py` arguments at the top of the script you run (they are not wired to `NPROC_PER_NODE` / `MAX_STEPS`-style environment variables unless you extend them).

Fine-tuning variants:

| Script                                   | Base checkpoint                      | Glycan feature | Chi loss              |
| ---------------------------------------- | ------------------------------------ | -------------- | --------------------- |
| `finetune_default.sh`                    | `protenix_base_default_v0.5.0.pt`    | no             | no                    |
| `finetune_default_glycan.sh`             | `protenix_base_default_v0.5.0.pt`    | yes            | no                    |
| `finetune_default_chi_0.01.sh`           | `protenix_base_default_v0.5.0.pt`    | no             | yes, `alpha_chi=0.01` |
| `finetune_default_glycan_chi_0.01.sh`    | `protenix_base_default_v0.5.0.pt`    | yes            | yes, `alpha_chi=0.01` |
| `finetune_constraint.sh`                 | `protenix_base_constraint_v0.5.0.pt` | no             | no                    |
| `finetune_constraint_glycan.sh`          | `protenix_base_constraint_v0.5.0.pt` | yes            | no                    |
| `finetune_constraint_chi_0.01.sh`        | `protenix_base_constraint_v0.5.0.pt` | no             | yes, `alpha_chi=0.01` |
| `finetune_constraint_glycan_chi_0.01.sh` | `protenix_base_constraint_v0.5.0.pt` | yes            | yes, `alpha_chi=0.01` |

Key fields to edit in each script (and in `configs/configs_data.py` if you change dataset keys):

- Data root and checkpoints: `PROTENIX_DATA_ROOT_DIR`, `checkpoint_path`, and paths under `release_data/`.
- Logging destination: `--base_dir` (defaults in the scripts point under `./outputs/finetune/...`).
- Training scale: `--max_steps`, `--warmup_steps`, `--lr`, `--diffusion_batch_size`, `--train_crop_size`, `torchrun --nproc_per_node`.
- Train/validation splits: `--data.train_sets` / `--data.test_sets` (defaults use `glyco_train_all_on` and `glyco_valid_all_on`).
- `runner/train.py` accepts extra CLI overrides after the script name, for example `bash finetune_default.sh --max_steps 1000` if you add `"$@"` to the script (not enabled by default in the minimal wrappers).

Each run writes checkpoints under:

```text
outputs/finetune/<run_name>/<run_name>_<timestamp>/checkpoints/
```

Use the checkpoint directory containing `<model_name>.pt` for inference. If your saved checkpoint has a different filename, copy or symlink it to the expected model name.



## Inference

### Single JSON

```bash
MODEL_NAME=protenix_base_default_v0.5.0 CHECKPOINT_DIR=release_data/checkpoint DUMP_DIR=outputs/inference/default TYPE=default bash inference.sh release_data/json/<sample_id>.json
```

For a glycan-feature checkpoint, include `glycan` in `TYPE`:

```bash
MODEL_NAME=protenix_base_default_v0.5.0 CHECKPOINT_DIR=/path/to/glycan_checkpoint_dir DUMP_DIR=outputs/inference/default_glycan_chi_0.01 TYPE=default_glycan_chi_0.01 bash inference.sh release_data/json/<sample_id>.json
```

Key optional settings for `inference.sh`:

- `MODEL_NAME`: model config and checkpoint filename stem.
- `CHECKPOINT_DIR`: directory containing `<MODEL_NAME>.pt`.
- `DUMP_DIR`: output directory.
- `TYPE`: run label. If it contains `glycan`, `--model.glycan_feature true` is enabled.
- `SEEDS`: comma-separated inference seeds, default `42`.
- `N_SAMPLE`, `N_STEP`, `N_CYCLE`: diffusion samples, diffusion steps, and recycling cycles.
- `USE_MSA`: `true` or `false`.
- `DTYPE`: `bf16`, `fp32`, or `fp16`.
- `LOAD_STRICT`: checkpoint loading strictness, default `true`.



### Batch JSON Directory

Use `run_inference.py` to process all JSON files in a directory and schedule work across GPUs:

```bash
python run_inference.py   --json_dir release_data/json   --split_file release_data/split_dict.pkl   --split_name test   --gpus 0,1,2,3   --model_name protenix_base_default_v0.5.0   --type default   --checkpoint_dir release_data/checkpoint   --dump_dir outputs/inference/default   --seeds 42
```

If `--split_file` is omitted, every `*.json` in `--json_dir` is processed. The split file can be a JSON or pickle mapping from sample ID to split name.



## Evaluation

Compute metrics for Protenix outputs with:

```bash
python metrics/compute_metrics.py   --native_dir release_data/native_complex_pdb   --model_dir outputs/inference/default   --output_dir outputs/metrics/default   --mode protenix   --seed 42
```

Outputs:

- `metrics.json`: per-sample metrics, consisting all four metrics we reported.
- `rmsd_stats.csv`: summary statistics.
- `rmsd_box.png`: RMSD box plot if plottable values exist.
- `metrics.log`: evaluation log.

The metric script expects Protenix predictions at:

```text
<model_dir>/<SAMPLE_ID>/seed_<seed>/predictions/<SAMPLE_ID>_sample_0.cif
```

This is the layout produced by `inference.sh` and `run_inference.py`.

If your native structures are still in packaged structure directories, first build metric-ready PDB files:

```bash
python metrics/generate_gt_complex_pdb.py   --structures-dir release_data/structures   --output-dir release_data/native_complex_pdb
```



## Reproduction Checklist

A reviewer can run the full default workflow as:

```bash
cd /path/to/ProtGlycanDock
pip install -r requirements.txt
pip install -e .

bash finetune_default_glycan_chi_0.01.sh

python run_inference.py   --json_dir release_data/json   --split_file release_data/split_dict.pkl   --split_name test   --gpus 0,1,2,3   --model_name protenix_base_default_v0.5.0   --type default_glycan_chi_0.01   --checkpoint_dir outputs/finetune/default_glycan_chi_0.01/<run_dir>/checkpoints   --dump_dir outputs/inference/default_glycan_chi_0.01   --seeds 42

python metrics/compute_metrics.py   --native_dir release_data/native_complex_pdb   --model_dir outputs/inference/default_glycan_chi_0.01   --output_dir outputs/metrics/default_glycan_chi_0.01   --mode protenix   --seed 42
```

Replace `<run_dir>` with the timestamped run directory created by fine-tuning.