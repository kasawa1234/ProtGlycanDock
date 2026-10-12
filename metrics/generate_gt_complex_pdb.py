#!/usr/bin/env python3
"""Build metric-ready complex PDB files from packaged structure directories.

Generic mmCIF-to-PDB conversion often packs glycans into a single residue and
breaks atom-matching metrics. This script mirrors a robust pattern: convert the
glycan CIF with PyMOL (adding hydrogens), then concatenate ATOM/HETATM lines
from the target receptor with glycan lines on a new trailing chain letter.

This step is separate from the structure packaging pipeline (which ends at the
``data/structures`` tree). Run it after structures are available.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

from pymol import cmd
from Bio.PDB import PDBParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def convert_cif_to_pdb(cif_file: str, pdb_file: str) -> None:
    obj_name = "glycan_obj"
    cmd.delete("all")
    cmd.load(cif_file, obj_name)
    cmd.h_add(obj_name)
    cmd.save(pdb_file, obj_name)
    cmd.delete(obj_name)


def get_chains(pdb_file: str) -> list:
    parser = PDBParser(QUIET=True)
    model = parser.get_structure("tmp", pdb_file)[0]
    return [chain.id for chain in model]


def merge_protein_ligand(receptor_pdb: str, ligand_pdb: str, output_pdb: str) -> None:
    chain_ids = get_chains(receptor_pdb)
    ligand_chain = chr(ord(chain_ids[-1]) + 1)

    lines = []
    with open(receptor_pdb) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("ATOM"):
                lines.append(line)
    with open(ligand_pdb) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("ATOM") or line.startswith("HETATM"):
                line = line[:21] + ligand_chain + line[22:]
                lines.append(line)

    with open(output_pdb, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures-dir", type=Path, required=True, help="Root with one subdirectory per PDB ID.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for ``{pdb_id}.pdb`` outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    folders = sorted(p.name for p in args.structures_dir.iterdir() if p.is_dir())
    logger.info("Found %s samples in %s", len(folders), args.structures_dir)

    success, fail = 0, 0
    for folder in folders:
        folder_path = args.structures_dir / folder
        pdb_id = folder[:4].upper()

        receptor_pdb = folder_path / "receptor" / "receptor_target.pdb"
        glycan_cif = folder_path / "glycan" / "glycan_raw.cif"
        output_pdb = args.output_dir / f"{pdb_id.lower()}.pdb"

        if not receptor_pdb.exists() or not glycan_cif.exists():
            logger.warning("Skipping %s: missing input files", folder)
            fail += 1
            continue

        tmp_glycan_pdb = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                tmp_glycan_pdb = tmp.name

            convert_cif_to_pdb(str(glycan_cif), tmp_glycan_pdb)
            merge_protein_ligand(str(receptor_pdb), tmp_glycan_pdb, str(output_pdb))

            os.unlink(tmp_glycan_pdb)
            success += 1
            logger.info("[%s/%s] %s -> %s", success + fail, len(folders), folder, output_pdb.name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed %s: %s", folder, exc)
            fail += 1
            if tmp_glycan_pdb and os.path.exists(tmp_glycan_pdb):
                os.unlink(tmp_glycan_pdb)

    logger.info("Done. success=%s, fail=%s, total=%s", success, fail, len(folders))


if __name__ == "__main__":
    main()
