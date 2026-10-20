# ProtGlycanDock: Towards Accurate Protein-Glycan Docking with Tailored Dataset, Benchmark and Model

This is the official codebase of the paper **ProtGlycanDock: Towards Accurate Protein-Glycan Docking with Tailored Dataset, Benchmark and Model**, accepted by **NeurIPS 2026 Evaluations and Datasets Track** ([OpenReview link](https://openreview.net/forum?id=90gsNmRJos), [arXiv link](https://arxiv.org/abs/2610.26009)).

[Minghao Xu](https://chrisallenming.github.io/), [Jiaze Song](https://kasawa1234.github.io/), Jinyu Wang, [Wentao Zhang](https://zwt233.github.io/)

## Overview

We present **ProtGlycanDock**, a tailored dataset and benchmark for protein–glycan docking under **protein and glycan generalization settings**, together with **customized models** that incorporate glycan conformational knowledge through **glycan topological features** and a **glycan energy loss**.

## Dataset

ProtGlycanDock is a curated protein–glycan docking dataset derived from publicly available PDB structures. It provides receptor, glycan, and complex structural files, mmCIF files, and Protenix-style model-input JSON files for training and evaluation. The full dataset is available on [Hugging Face](https://huggingface.co/datasets/ProtGlycanDock/ProtGlycanDock).

The dataset contains **1,781 protein–glycan complexes** across four subsets:


| Subset               | Split label           | Cases | Purpose                                                         |
| -------------------- | --------------------- | ----- | --------------------------------------------------------------- |
| Train                | `train`               | 1,374 | Training set used for model fitting                             |
| Validation           | `valid`               | 125   | Validation set for model selection                              |
| Protein-holdout test | `test`                | 148   | Protein generalization: evaluation on held-out protein clusters |
| Glycan-holdout test  | `glycan_holdout_test` | 134   | Glycan generalization: evaluation on held-out glycans           |


The [released files](https://huggingface.co/datasets/ProtGlycanDock/ProtGlycanDock/tree/main) are organized as follows:

```text
data/
  structures/{PDB_ID}/
    receptor/
      receptor_source.pdb
      receptor_target.pdb
    glycan/
      glycan_raw.cif
      glycan_raw.smi
    complex/
      complex_raw.cif
      complex_gt.cif
  mmcif/{PDB_ID}.cif
  json/
    base/{PDB_ID}.json
    pocket/{PDB_ID}.json
metadata/
  dataset_file_manifest.csv
  splits/
    train.csv
    valid.csv
    protein_holdout_test.csv
    glycan_holdout_test.csv
```

The split manifests (`metadata/splits/{split}.csv`) record sample identifiers, artifact types, relative file paths, encoding formats, and SHA-256 checksums. The protein-holdout subset uses the split label `test` and the manifest filename `protein_holdout_test.csv`. See the [dataset card](https://huggingface.co/datasets/ProtGlycanDock/ProtGlycanDock/blob/main/README.md) for further details.

MSA directories are not redistributed. JSON inputs may contain package-relative MSA placeholders under `data/msa/mmcif_msa_inference/`; prepare the corresponding MSA files separately for runs that use MSA features.

The released dataset layout differs from this code's `release_data/` layout. See [Data and Checkpoints](#data-and-checkpoints) below for the model's required files and directories. The dataset is distributed under CC BY 4.0, as specified in its dataset card.

## Installation

This codebase is based on PyTorch and Protenix. Python 3.11 and CUDA-enabled PyTorch are recommended. Install the dependencies with:

```bash
cd /path/to/ProtGlycanDock
pip install -r requirements.txt
pip install -e .
```

For the optional CUDA kernels, set `CUTLASS_PATH` before training or inference if you use DeepSpeed Evoformer attention. The provided scripts use `triattention` and `cuequivariance` by default.

## Data and Checkpoints

The scripts assume the data bundle is unpacked at:

```text
ProtGlycanDock/
  release_data/
    checkpoint/
      protenix_base_default_v0.5.0.pt
      protenix_base_constraint_v0.5.0.pt
    components.cif                 (or components.v20240608.cif)
    components.cif.rdkit_mol.pkl   (or components.v20240608.cif.rdkit_mol.pkl)
    clusters-by-entity-40.txt
    split_dict.pkl
    mono.pkl
    mmcif/
      <sample_id>.cif
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

The required files come from four sources:

1. Available from official websites: `checkpoint`, `components.v20240608.cif`, `components.v20240608.cif.rdkit_mol.pkl`, and `clusters-by-entity-40.txt`.
2. Included in this code repository: `mono.pkl`.
3. Available in our [full dataset repository](https://huggingface.co/datasets/ProtGlycanDock/ProtGlycanDock): `mmcif` and `json`.
4. Generated from these data: `indices`, `mmcif_bioassembly`, and `split_dict.pkl`.

If your data is elsewhere, set:

```bash
export PROTENIX_DATA_ROOT_DIR=/path/to/release_data
```

Fine-tuning reads training and validation datasets from `configs/configs_data.py`. The bundled `finetune_*.sh` scripts use `glyco_train_all_on` and `glyco_valid_all_on` (see each script’s `--data.train_sets` / `--data.test_sets`).

## Finetuning

Run a fine-tuning variant from the repository root. For example:

```
bash finetune_default_glycan_chi_0.01.sh
```

The bundled `finetune_*.sh` scripts are minimal entry points: adjust `export PROTENIX_DATA_ROOT_DIR`, `checkpoint_path`, `--base_dir`, `torchrun --nproc_per_node`, and the `runner/train.py` arguments at the top of the script you run (they are not wired to `NPROC_PER_NODE` / `MAX_STEPS`-style environment variables unless you extend them).

Fine-tuning variants:


| Script                                   | Base checkpoint                      | Glycan topological features | Glycan energy loss    |
| ---------------------------------------- | ------------------------------------ | --------------------------- | --------------------- |
| `finetune_default.sh`                    | `protenix_base_default_v0.5.0.pt`    | no                          | no                    |
| `finetune_default_glycan.sh`             | `protenix_base_default_v0.5.0.pt`    | yes                         | no                    |
| `finetune_default_chi_0.01.sh`           | `protenix_base_default_v0.5.0.pt`    | no                          | yes, `alpha_chi=0.01` |
| `finetune_default_glycan_chi_0.01.sh`    | `protenix_base_default_v0.5.0.pt`    | yes                         | yes, `alpha_chi=0.01` |
| `finetune_constraint.sh`                 | `protenix_base_constraint_v0.5.0.pt` | no                          | no                    |
| `finetune_constraint_glycan.sh`          | `protenix_base_constraint_v0.5.0.pt` | yes                         | no                    |
| `finetune_constraint_chi_0.01.sh`        | `protenix_base_constraint_v0.5.0.pt` | no                          | yes, `alpha_chi=0.01` |
| `finetune_constraint_glycan_chi_0.01.sh` | `protenix_base_constraint_v0.5.0.pt` | yes                         | yes, `alpha_chi=0.01` |


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
MODEL_NAME=protenix_base_default_v0.5.0 \
CHECKPOINT_DIR=release_data/checkpoint \
DUMP_DIR=outputs/inference/default \
TYPE=default \
bash inference.sh release_data/json/<sample_id>.json
```

For a checkpoint trained with glycan topological features, include `glycan` in `TYPE`:

```bash
MODEL_NAME=protenix_base_default_v0.5.0 \
CHECKPOINT_DIR=/path/to/glycan_checkpoint_dir \
DUMP_DIR=outputs/inference/default_glycan_chi_0.01 \
TYPE=default_glycan_chi_0.01 \
bash inference.sh release_data/json/<sample_id>.json
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
python run_inference.py \
    --json_dir release_data/json \
    --split_file release_data/split_dict.pkl \
    --split_name test \
    --gpus 0,1,2,3 \
    --model_name protenix_base_default_v0.5.0 \
    --type default \
    --checkpoint_dir release_data/checkpoint \
    --dump_dir outputs/inference/default \
    --seeds 42
```

If `--split_file` is omitted, every `*.json` in `--json_dir` is processed. The split file can be a JSON or pickle mapping from sample ID to split name.

## Evaluation

Both evaluation scripts require PyMOL (`pymol`); metric summaries and plots also require Seaborn (`seaborn`). Install these additional dependencies in the environment used for evaluation. See the [PyMOL installation instructions](https://pymol.org/conda/) for PyMOL, and run `pip install seaborn==0.13.2` for Seaborn.

Compute metrics for Protenix outputs with:

```bash
python metrics/compute_metrics.py \
    --native_dir release_data/native_complex_pdb \
    --model_dir outputs/inference/default \
    --output_dir outputs/metrics/default \
    --mode protenix \
    --seed 42
```

Outputs:

- `metrics.json`: per-sample results for the four evaluation metrics reported in the paper.
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
python metrics/generate_gt_complex_pdb.py \
    --structures-dir release_data/structures \
    --output-dir release_data/native_complex_pdb
```

## License

This codebase is released under the Apache License 2.0, as described in [LICENSE](LICENSE). Upstream copyright and attribution notices are retained in the license file and relevant source files.

The dataset is released separately under CC BY 4.0; see the [dataset card](https://huggingface.co/datasets/ProtGlycanDock/ProtGlycanDock/blob/main/README.md).

## Acknowledgements

This work is supported by Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (JYB2025XDXM113), Beijing Municipal Natural Science Foundation (Grant No. L2603008), National Natural Science Foundation of China (92470121, 62402016), National Key RD Program of China (2024YFA1014003), Zhongguancun Academy (C20250204, C20250602),  Beijing Major Science and Technology Project (Z251100008125043, Z251100008425023), and High-performance Computing Platform of Peking University.

## Citation

If you find this dataset, benchmark, or codebase helpful in your research, please cite the following paper.

```bibtex
@misc{xu2026protglycandockaccurateproteinglycandocking,
      title={ProtGlycanDock: Towards Accurate Protein-Glycan Docking with Tailored Dataset, Benchmark and Model}, 
      author={Minghao Xu and Jiaze Song and Jinyu Wang and Wentao Zhang},
      year={2026},
      eprint={2610.26009},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.26009}, 
}
```
