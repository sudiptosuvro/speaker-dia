#!/usr/bin/env python
# coding: utf-8

# In[1]:


# AudioDecoder stub
def _install_audio_stub():
    import torch, soundfile as sf, numpy as np, types, sys
    from types import SimpleNamespace

    class AudioSamples:

        def __init__(self, data: torch.Tensor, sample_rate: int):
            self.data = data
            self.sample_rate = sample_rate

    class AudioDecoder:

        def __init__(self, path_or_file, **_):

            if isinstance(path_or_file, str):
                self.path = path_or_file
            else:
                self.path = path_or_file.get("audio", path_or_file.get("uri", None))

            info = sf.info(self.path)
            dur = info.frames / info.samplerate if info.samplerate > 0 else 0.0

            self.metadata = SimpleNamespace(
                sample_rate=info.samplerate,
                num_channels=info.channels,
                num_frames=info.frames,
                dtype="float32",
                duration_seconds_from_header=dur,
            )
            self.sample_rate = info.samplerate

        def _load_full(self):
            
            data, sr = sf.read(self.path, dtype="float32")
            if data.ndim == 1:
                data = np.expand_dims(data, 0)
            else:
                data = data.T
            tensor = torch.from_numpy(data)
            return tensor, sr

        def get_samples_played_in_range(self, start: float, end: float):

            if start is None:
                start = 0.0
            if end is None or end <= 0:
                data, sr = self._load_full()
                return AudioSamples(data, sr)


            data, sr = self._load_full()
            n_start = int(max(0, round(start * sr)))
            n_end = int(min(data.shape[-1], round(end * sr)))
            if n_end <= n_start:
                silence = torch.zeros((data.shape[0], 1), dtype=data.dtype)
                return AudioSamples(silence, sr)
            sliced = data[..., n_start:n_end]
            return AudioSamples(sliced, sr)

        def get_all_samples(self):
            data, sr = self._load_full()
            return AudioSamples(data, sr)

        def crop(self, file, segment=None, mode="pad"):

            if segment is None:
                samples = self.get_all_samples()
            else:
                samples = self.get_samples_played_in_range(segment.start, segment.end)
            return {"waveform": samples.data, "sample_rate": samples.sample_rate}

    mod = types.ModuleType("pyannote.audio.pipelines.utils.audio")
    mod.AudioDecoder = AudioDecoder
    sys.modules["pyannote.audio.pipelines.utils.audio"] = mod

    import builtins
    builtins.AudioDecoder = AudioDecoder

    print("AudioDecoder stub installed (torchcodec-free).")

_install_audio_stub()


# In[2]:


import warnings
warnings.filterwarnings('ignore')
import os

import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    def list_audio_backends():
        return []
    torchaudio.list_audio_backends = list_audio_backends

from typing import List
from pyannote.audio import Pipeline
from config import cfg
from utils import ensure_dir, save_annotation_to_rttm


# In[3]:


from speechbrain.pretrained import EncoderClassifier
print("SpeechBrain imported OK!")


# In[4]:


def build_pyannote_pipeline() -> Pipeline:

    os.environ["HUGGINGFACE_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN", "")
    cfg.hf_token = os.environ["HUGGINGFACE_TOKEN"]

    if cfg.hf_token is None or not cfg.hf_token.startswith("hf_"):
        raise RuntimeError(
            "\n[ERROR] HuggingFace token missing or invalid.\n"
            "Create one here: https://huggingface.co/settings/tokens\n"
        )

    print(f"[HF] Using HuggingFace token (first 12 chars): {cfg.hf_token[:12]}********")

    try:
        pipeline = Pipeline.from_pretrained(
            cfg.pyannote_pipeline,
            token=cfg.hf_token
        )
        print("[pyannote] Pipeline loaded successfully!")

    except Exception as e:
        raise RuntimeError(
            "\n[ERROR] Failed to load pyannote pipeline.\n"
            "Your token may not have access or may be expired.\n"
            f"Original error:\n{e}\n"
        )

    return pipeline


# In[5]:


pipeline = build_pyannote_pipeline()


# In[6]:


def run_pyannote_on_split(split: str, names: list, out_dir: str):
    ensure_dir(out_dir)

    print(f"\n=== Running pyannote on {split.upper()} split ===")

    pipeline = build_pyannote_pipeline()

    for name in names:
        wav_path = f"{cfg.dataset_root}/{split}/wav/{name}.wav"
        out_path = f"{out_dir}/{name}.rttm"

        print(f"[pyannote] Processing {split}/{name}")

        # 1) Run diarization -> DiarizeOutput
        diarization_out = pipeline(wav_path)

        # 2) Underlying Annotation
        ann = diarization_out.speaker_diarization

        # 3) Save RTTM
        save_annotation_to_rttm(ann, uri=name, out_path=out_path)
        print(f"[pyannote] Saved RTTM → {out_path}")


# In[7]:


get_ipython().system('jupyter nbconvert --to script pyannote_run.ipynb')


# In[ ]:




