import os
import json
import numpy as np
import h5py


from dda_st import compute_st_multiple
from dda_ct import compute_ct_multiple
from dda_de import compute_dynamical_ergodicity, run_full_de_analysis

# input directory of seizures
DATA_DIR = 'test_10'

# validate input directory
if not os.path.isdir(DATA_DIR):
    print("Error: Input directory not found.")
    exit(1)

# gather all .mat files in the directory
fn_list = sorted(
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.endswith('.mat')
)
print(f"found {len(fn_list)} files in {DATA_DIR}")

# output directory for per-file results
OUT_DIR = 'DE_test_10'
os.makedirs(OUT_DIR, exist_ok=True)

WL = 512
WS = 256
TAU = [3, 4] # 3-10?
dm = 4
order = 3


def _to_jsonable(obj):
    """Convert numpy scalars/arrays so the stats dict is JSON serializable."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


results = {}

# loop over all files
for FN_DATA in fn_list:
    print(f"processing {FN_DATA}")

    # The FieldTrip struct stores the signal in trimmed.trial{1,1} as
    # [channels x samples]; h5py reads it transposed as [samples x channels].
    with h5py.File(FN_DATA, 'r') as f:
        trial_ref = f['data/trial'][0, 0]
        Y = np.array(f[trial_ref])  # (samples, channels) = (time, channels)

        # 10-20 channel names from data.label (cell array of char arrays)
        label_refs = f['data/label'][:].flatten()
        channel_names = [
            ''.join(chr(c) for c in np.array(f[r]).flatten())
            for r in label_refs
        ]

        # sampling rate from data.fsample (Hz)
        fs = float(np.array(f['data/fsample']).flatten()[0])

    # ST-dda
    #ST = compute_st_multiple(Y, TAU, dm, order, WL, WS, False)
    #ST_result = ST

    # CT-dda
    #CT = compute_ct_multiple(Y, TAU, dm, order, WL, WS)
    #CT_result = CT[0]
    #channel_pairs = CT[1]

    # DE-dda
    E, dda_stats = run_full_de_analysis(
        FN_DATA, Y, TAU, dm, order, WL, WS, False,
        sampling_rate=fs, channel_names=channel_names,
    )

    results[FN_DATA] = (E, dda_stats)

    # save this file's results under its own name
    name = os.path.splitext(os.path.basename(FN_DATA))[0]
    np.save(os.path.join(OUT_DIR, f"{name}_E.npy"), E)
    with open(os.path.join(OUT_DIR, f"{name}_stats.json"), 'w') as f:
        json.dump(_to_jsonable(dda_stats), f, indent=2)
    print(f"  saved {name}_E.npy and {name}_stats.json")
