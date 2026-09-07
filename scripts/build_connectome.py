"""Compile the FlyWire FAFB v783 connectome into a simulation-ready sparse brain.

Reads the Codex static exports in data/flywire_783/ and writes a single .npz
holding the CSR adjacency plus aligned per-neuron metadata.

Sign convention follows Shiu et al. 2024 (Nature): acetylcholine is excitatory,
GABA and glutamate are inhibitory (Drosophila glutamate acts on GluCl channels).
The monoamines (dopamine, serotonin, octopamine) carry no fast ionotropic sign;
they are split into a separate modulatory channel rather than being forced into
the excitatory/inhibitory matrix.
"""

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "flywire_783"
OUT = ROOT / "data" / "brain_783.npz"

FAST_SIGN = {"ACH": 1.0, "GABA": -1.0, "GLUT": -1.0}
MONOAMINES = ("DA", "SER", "OCT")


def load_metadata() -> pd.DataFrame:
    cls = pd.read_csv(RAW / "classification.csv.gz", dtype={"root_id": np.int64})
    neu = pd.read_csv(
        RAW / "neurons.csv.gz",
        dtype={"root_id": np.int64},
        usecols=["root_id", "group", "nt_type", "nt_type_score"],
    )
    types = pd.read_csv(
        RAW / "consolidated_cell_types.csv.gz",
        dtype={"root_id": np.int64},
        usecols=["root_id", "primary_type"],
    )
    stats = pd.read_csv(RAW / "cell_stats.csv.gz", dtype={"root_id": np.int64})

    # Soma/centroid positions, needed for anatomical maps such as the
    # ellipsoid-body compass and the optic-lobe retinotopy.
    coord = pd.read_csv(RAW / "coordinates.csv.gz", dtype={"root_id": np.int64},
                        usecols=["root_id", "position"]).drop_duplicates("root_id")
    xyz = np.array([np.fromstring(s.strip("[]"), sep=" ") for s in coord["position"]])
    coord = pd.DataFrame({"root_id": coord["root_id"].to_numpy(),
                          "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})

    meta = (
        cls.merge(neu, on="root_id", how="left")
        .merge(types, on="root_id", how="left")
        .merge(stats, on="root_id", how="left")
        .merge(coord, on="root_id", how="left")
    )
    # Deterministic order so indices are stable across rebuilds.
    meta = meta.sort_values("root_id", kind="stable").reset_index(drop=True)
    meta["idx"] = np.arange(len(meta), dtype=np.int32)
    return meta


def load_connections(monoamine_mode: str) -> pd.DataFrame:
    con = pd.read_csv(
        RAW / "connections.csv.gz",
        dtype={
            "pre_root_id": np.int64,
            "post_root_id": np.int64,
            "syn_count": np.int32,
            "nt_type": "string",
            "neuropil": "string",
        },
    )
    con["nt_type"] = con["nt_type"].fillna("UNK")
    return con


def build(monoamine_mode: str, min_syn: int) -> None:
    meta = load_metadata()
    con = load_connections(monoamine_mode)

    idx = pd.Series(meta["idx"].to_numpy(), index=meta["root_id"].to_numpy())
    con = con[con["syn_count"] >= min_syn]
    pre = con["pre_root_id"].map(idx)
    post = con["post_root_id"].map(idx)
    keep = pre.notna() & post.notna()
    dropped = int((~keep).sum())
    con, pre, post = con[keep], pre[keep].astype(np.int32), post[keep].astype(np.int32)

    nt = con["nt_type"].to_numpy()
    syn = con["syn_count"].to_numpy(dtype=np.float32)

    fast_sign = np.zeros(len(con), dtype=np.float32)
    for name, s in FAST_SIGN.items():
        fast_sign[nt == name] = s
    is_mono = np.isin(nt, MONOAMINES)
    if monoamine_mode == "excitatory":
        fast_sign[is_mono] = 1.0
    elif monoamine_mode == "zero":
        fast_sign[is_mono] = 0.0
    # "modulatory" (default): monoamines stay out of the fast matrix entirely.

    n = len(meta)
    fast = sparse.csr_matrix(
        (syn * fast_sign, (pre.to_numpy(), post.to_numpy())), shape=(n, n), dtype=np.float32
    )
    fast.sum_duplicates()

    mod = sparse.csr_matrix(
        (syn[is_mono], (pre.to_numpy()[is_mono], post.to_numpy()[is_mono])),
        shape=(n, n),
        dtype=np.float32,
    )
    mod.sum_duplicates()

    # Per-edge modulator identity, kept alongside so a reward channel can be
    # driven by dopamine specifically rather than by "any monoamine".
    mod_kind = np.zeros(n, dtype=np.int8)
    src_nt = meta["nt_type"].fillna("UNK").to_numpy()
    for k, name in enumerate(MONOAMINES, start=1):
        mod_kind[src_nt == name] = k

    np.savez_compressed(
        OUT,
        fast_data=fast.data,
        fast_indices=fast.indices,
        fast_indptr=fast.indptr,
        mod_data=mod.data,
        mod_indices=mod.indices,
        mod_indptr=mod.indptr,
        shape=np.array([n, n], dtype=np.int64),
        root_id=meta["root_id"].to_numpy(),
        mod_kind=mod_kind,
        monoamine_mode=np.array(monoamine_mode),
        min_syn=np.array(min_syn),
    )
    meta.to_parquet(ROOT / "data" / "neurons_783.parquet", index=False)

    exc = int((fast.data > 0).sum())
    inh = int((fast.data < 0).sum())
    print(f"neurons          : {n:,}")
    print(f"edges (fast)     : {fast.nnz:,}  (+{exc:,} exc / -{inh:,} inh)")
    print(f"edges (modulator): {mod.nnz:,}   mode={monoamine_mode}")
    print(f"synapses total   : {int(syn.sum()):,}   min_syn={min_syn}")
    print(f"dropped edges    : {dropped:,} (endpoint not in classification)")
    print(f"wrote            : {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--monoamine-mode", default="modulatory",
                   choices=["modulatory", "excitatory", "zero"])
    p.add_argument("--min-syn", type=int, default=1,
                   help="per-neuropil row threshold; the Codex export is unthresholded")
    a = p.parse_args()
    build(a.monoamine_mode, a.min_syn)
