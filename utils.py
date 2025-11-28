#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from pyannote.core import Annotation, Segment
from typing import List


# In[2]:


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# In[3]:


def read_list(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    return [l for l in lines if l]


# In[4]:


def rttm_to_annotation(rttm_path: str) -> Annotation:
    ann = Annotation()

    with open(rttm_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 9:
                continue
            if parts[0] != "SPEAKER":
                continue

            start = float(parts[3])
            dur   = float(parts[4])
            spk   = parts[7]

            seg = Segment(start, start + dur)
            ann[seg, spk] = 1

    return ann


# In[5]:


def save_annotation_to_rttm(annotation, uri: str, out_path: str):
    
    if hasattr(annotation, "speaker_diarization"):
        annotation = annotation.speaker_diarization

    with open(out_path, "w") as f:
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            onset = segment.start
            dur   = segment.duration
            f.write(
                f"SPEAKER {uri} 1 {onset:.3f} {dur:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
            )


# In[6]:


def segments_to_rttm(
    segments,
    uri: str,
    out_path: str,
    merge_tol: float = 0.05,
):

    segments = sorted(segments, key=lambda x: x[0])
    merged = []
    for start, end, spk in segments:
        if not merged:
            merged.append([start, end, spk])
            continue

        last_start, last_end, last_spk = merged[-1]
        if spk == last_spk and start <= last_end + merge_tol:
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end, spk])

    with open(out_path, "w") as f:
        for start, end, spk in merged:
            dur = end - start
            f.write(
                f"SPEAKER {uri} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n"
            )


# In[7]:


def merge_segments_by_speaker(segments, labels, tol=1e-3):

    assert len(segments) == len(labels)
    if not segments:
        return []

    merged = []
    cur_start, cur_end = segments[0]
    cur_spk = labels[0]

    for (s, e), spk in zip(segments[1:], labels[1:]):
        if spk == cur_spk and abs(s - cur_end) <= tol:
            cur_end = e
        else:
            merged.append((cur_start, cur_end, cur_spk))
            cur_start, cur_end, cur_spk = s, e, spk

    merged.append((cur_start, cur_end, cur_spk))
    return merged


# In[8]:


get_ipython().system('jupyter nbconvert --to script utils.ipynb')

