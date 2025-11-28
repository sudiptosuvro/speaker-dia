#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from typing import List, Dict, Any
import numpy as np
from pyannote.metrics.diarization import (
    DiarizationErrorRate,
    JaccardErrorRate,
)

from config import cfg
from utils import rttm_to_annotation


# In[2]:


def compute_metrics_for_split(split: str,
                              names: List[str],
                              sys_dir: str):

    der_metric = DiarizationErrorRate()
    jer_metric = JaccardErrorRate()

    der_vals, jer_vals = [], []

    for name in names:
        ref_rttm = os.path.join(
            cfg.dataset_root, split, "rttm", f"{name}.rttm"
        )
        hyp_rttm = os.path.join(sys_dir, f"{name}.rttm")

        ref_ann = rttm_to_annotation(ref_rttm)
        hyp_ann = rttm_to_annotation(hyp_rttm)

        der = der_metric(ref_ann, hyp_ann)
        jer = jer_metric(ref_ann, hyp_ann)

        der_vals.append(der)
        jer_vals.append(jer)

        print(f"[{split}] {name}  DER={der:.4f},  JER={jer:.4f}")

    results = {
        "DER": float(np.mean(der_vals)),
        "JER": float(np.mean(jer_vals)),
    }

    print(
        f"\n[{split} summary] "
        f"DER={results['DER']:.4f}, JER={results['JER']:.4f}"
    )

    return results

def compute_metrics_for_split_with_files(
    split: str,
    names: List[str],
    sys_dir: str,
) -> Dict[str, Any]:
    der_metric = DiarizationErrorRate()
    jer_metric = JaccardErrorRate()

    records = []

    for name in names:
        ref_rttm = os.path.join(cfg.dataset_root, split, "rttm", f"{name}.rttm")
        hyp_rttm = os.path.join(sys_dir, f"{name}.rttm")

        ref_ann = rttm_to_annotation(ref_rttm)
        hyp_ann = rttm_to_annotation(hyp_rttm)

        der = float(der_metric(ref_ann, hyp_ann))
        jer = float(jer_metric(ref_ann, hyp_ann))
        row = {"file": name, "split": split, "DER": der, "JER": jer}

        records.append(row)

    # summary
    summary = {
        "DER": float(np.mean([r["DER"] for r in records])),
        "JER": float(np.mean([r["JER"] for r in records])),
    }
    if records and "CDER" in records[0]:
        summary["CDER"] = float(np.mean([r["CDER"] for r in records]))

    return {"summary": summary, "records": records}


# In[3]:


get_ipython().system('jupyter nbconvert --to script metrics.ipynb')

