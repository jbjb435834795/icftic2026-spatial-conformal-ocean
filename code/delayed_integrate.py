# Integrate delayed-feedback online/ACI results into the merged eval files.
# - backs up original (instant-feedback) per-window arrays to
#   eval/instantfb_backup_{split}.npz
# - overwrites {fam}_{online,aci}_L{lv}_cover/_width in perwindow npz and
#   {fam}_{online,aci}_L{lv} in spatial npz with the delayed arrays
# - replaces online/aci entries in cp_results_{split}.json (instant values
#   preserved under online_instantfb / aci_instantfb)
# Usage: python delayed_integrate.py final_2021
import json
import sys
from pathlib import Path

import numpy as np

CAMP = Path(__file__).resolve().parent.parent
EVAL = CAMP / "eval"
FAMS = ["convlstm", "fno2d", "direct_resnet", "persistence", "climatology"]
LEVELS = [0.9, 0.95, 0.8]


def integrate(split):
    res_path = EVAL / f"cp_results_{split}.json"
    res = json.loads(res_path.read_text())
    sp = dict(np.load(EVAL / f"spatial_{split}.npz"))
    pw = dict(np.load(EVAL / f"perwindow_{split}.npz"))
    backup = {}

    for fam in FAMS:
        pj = EVAL / f"delayed_partial_{fam}_{split}.json"
        pa = EVAL / f"delayed_partial_{fam}_{split}_arrays.npz"
        if not pj.exists():
            print("MISSING", pj.name)
            continue
        d = json.loads(pj.read_text())
        z = np.load(pa)
        for k in z.files:
            if k.endswith("_winklerpw"):
                continue  # kept only in the delayed partial npz for CI use
            tgt = pw if ("_cover" in k or "_width" in k) else sp
            if k in tgt:
                backup.setdefault(k, tgt[k])  # preserve instant-feedback values
            tgt[k] = z[k]
        # json entries
        for lv, methods in d["levels"].items():
            fam_levels = res["results"][fam]["levels"][lv]
            for m in ["online", "aci"]:
                if m in fam_levels:
                    fam_levels[f"{m}_instantfb"] = fam_levels[m]
                fam_levels[m] = methods[m]
        print(f"integrated {fam} (delayed online/aci)")

    res["config"]["online_feedback"] = (
        "delayed (observation-aware): the lead-L error of window t enters the "
        "history only when processing window t+L; instant-feedback values are "
        "preserved under *_instantfb")
    res_path.write_text(json.dumps(res))
    np.savez_compressed(EVAL / f"spatial_{split}.npz", **sp)
    np.savez_compressed(EVAL / f"perwindow_{split}.npz", **pw)
    np.savez_compressed(EVAL / f"instantfb_backup_{split}.npz", **backup)
    print(f"INTEGRATED {split}: spatial={len(sp)} perwin={len(pw)} "
          f"backup={len(backup)}")


if __name__ == "__main__":
    integrate(sys.argv[1] if len(sys.argv) > 1 else "final_2021")
