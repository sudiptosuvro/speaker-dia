#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import soundfile as sf
import librosa

from sklearn.cluster import AgglomerativeClustering
from hmmlearn import hmm

from speechbrain.inference.classifiers import EncoderClassifier

from config import cfg
from utils import ensure_dir, segments_to_rttm


# In[7]:


def load_audio_mono(path: str, sr: int) -> torch.Tensor:

    wav, s = librosa.load(path, sr=sr, mono=True)
    wav = torch.from_numpy(wav).float()
    return wav


# In[8]:


def run_vbx(embs, max_speakers=8):

    X = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    clustering = AgglomerativeClustering(
        n_clusters=max_speakers,
        affinity='cosine',
        linkage='average'
    )

    raw_labels = clustering.fit_predict(X)
    N = max_speakers
    trans = np.full((N, N), 0.02 / (N - 1))
    for i in range(N):
        trans[i, i] = 0.98

    model = hmm.GaussianHMM(n_components=N, covariance_type="diag", n_iter=50)
    model.startprob_ = np.full(N, 1.0 / N)
    model.transmat_ = trans
    model.means_ = np.random.randn(N, X.shape[1])
    model.covars_ = np.ones((N, X.shape[1]))

    smoothed_labels = model.predict(X)

    return smoothed_labels.tolist()


# In[ ]:


def build_ecapa_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )
    return model


# In[ ]:


def extract_ecapa_embeddings_for_file(
    model,
    wav_path: str,
    sr: int,
    win_sec: float,
    hop_sec: float,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:

    device = next(model.parameters()).device
    wav = load_audio_mono(wav_path, sr)  # (T,)
    num_samples = wav.shape[0]

    win_samp = int(win_sec * sr)
    hop_samp = int(hop_sec * sr)

    segments: List[Tuple[float, float]] = []
    embs: List[np.ndarray] = []

    i = 0
    while i + win_samp <= num_samples:
        start = i
        end   = i + win_samp

        seg = wav[start:end].unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_batch(seg)
        emb = emb.squeeze().cpu().numpy()

        t_start = start / sr
        t_end   = end   / sr

        segments.append((t_start, t_end))
        embs.append(emb)

        i += hop_samp

    if not embs:
        return np.zeros((0, 192)), []

    embs = np.stack(embs, axis=0)
    return embs, segments


# In[ ]:


def ahc_init(
    embs: np.ndarray,
    min_spk: int,
    max_spk: int,
) -> Tuple[np.ndarray, int]:

    from sklearn.metrics import silhouette_score

    best_score = -1.0
    best_labels = None
    best_k = min_spk

    for k in range(min_spk, max_spk + 1):
        clust = AgglomerativeClustering(
            n_clusters=k,
            metric="cosine",
            linkage="average",
        )
        labels = clust.fit_predict(embs)
        try:
            score = silhouette_score(embs, labels, metric="cosine")
        except ValueError:
            score = -1.0

        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

    return best_labels, best_k


# In[ ]:


def vbx_hmm_refine(
    embs: np.ndarray,
    init_labels: np.ndarray,
    n_spk: int,
    loop_prob: float = 0.99,
) -> np.ndarray:

    N, D = embs.shape
    K = n_spk

    means = np.zeros((K, D), dtype=np.float64)
    covars = np.zeros((K, D), dtype=np.float64)

    for k in range(K):
        xk = embs[init_labels == k]
        if xk.size == 0:
            xk = embs  # fallback
        means[k]  = xk.mean(axis=0)
        covars[k] = xk.var(axis=0) + 1e-6

    transmat = np.full((K, K), (1.0 - loop_prob) / (K - 1), dtype=np.float64)
    np.fill_diagonal(transmat, loop_prob)

    startprob = np.full(K, 1.0 / K, dtype=np.float64)

    model = hmm.GaussianHMM(
        n_components=K,
        covariance_type="diag",
        init_params="",
        params="",
    )
    model.startprob_ = startprob
    model.transmat_  = transmat
    model.means_     = means
    model.covars_    = covars

    refined = model.predict(embs.astype(np.float64))
    return refined


# In[ ]:


def write_rttm(
    out_path: str,
    segments: List[Tuple[float, float]],
    labels: np.ndarray,
    uri: str,
):

    assert len(segments) == len(labels)
    with open(out_path, "w") as f:
        if len(segments) == 0:
            return

        cur_spk = labels[0]
        cur_start, cur_end = segments[0]

        for (t0, t1), spk in zip(segments[1:], labels[1:]):
            if spk == cur_spk and abs(t0 - cur_end) < 1e-3:
                cur_end = t1
            else:
                dur = cur_end - cur_start
                f.write(
                    f"SPEAKER {uri} 1 {cur_start:.3f} {dur:.3f} "
                    f"<NA> <NA> spk{cur_spk} <NA>\n"
                )
                cur_spk = spk
                cur_start, cur_end = t0, t1

        dur = cur_end - cur_start
        f.write(
            f"SPEAKER {uri} 1 {cur_start:.3f} {dur:.3f} "
            f"<NA> <NA> spk{cur_spk} <NA>\n"
        )


# In[ ]:


def run_ecapa_vbx_on_split(split, names, out_dir):
    ensure_dir(out_dir)
    model = build_ecapa_model()

    for name in names:
        print(f"[ECAPA+VBx] {split}: {name}")
        wav_path = f"{cfg.dataset_root}/{split}/wav/{name}.wav"
        embs, seg_times = extract_ecapa_embeddings_for_file(
            model,
            wav_path,
            sr=cfg.sample_rate,
            win_sec=cfg.win_sec,
            hop_sec=cfg.hop_sec,
        )

        labels = run_vbx(embs)
        segments = []
        for (start, end), lab in zip(seg_times, labels):
            spk = f"spk{lab}"
            segments.append((start, end, spk))

        out_rttm = os.path.join(out_dir, f"{name}.rttm")
        segments_to_rttm(segments, uri=name, out_path=out_rttm)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script ecapa_vbx_run.ipynb')

