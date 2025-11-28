#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dataclasses import dataclass
import os


# In[2]:


@dataclass
class Config:
    # data
    dataset_root: str = "dataset"
    dev_list: str = os.path.join(dataset_root, "dev_list.txt")
    test_list: str = os.path.join(dataset_root, "test_list.txt")
    
    # pyannote
    pyannote_rttm_dev: str  = "outputs/pyannote_dev"
    pyannote_rttm_test: str = "outputs/pyannote_test"
    
    # ECAPA + VBx RTTM directories
    ecapa_vbx_rttm_dev: str  = "outputs/ecapa_vbx_dev"
    ecapa_vbx_rttm_test: str = "outputs/ecapa_vbx_test"

    # WavLM directories
    wavlm_vbx_rttm_dev  = "outputs/wavlm_vbx_dev"
    wavlm_vbx_rttm_test = "outputs/wavlm_vbx_test"

    # audio
    sample_rate: int = 16000

    # ECAPA windowing
    win_sec: float = 1.5
    hop_sec: float = 0.75

    # HMM / VBx-style parameters
    vbx_loop_prob: float = 0.99
    vbx_min_spk: int     = 1
    vbx_max_spk: int     = 8

    # spectral clustering
    max_speakers: int = 8
    sim_threshold: float = 0.3

    # pyannote pipeline
    pyannote_pipeline: str = "pyannote/speaker-diarization-3.1"
    hf_token: str = os.environ.get("HUGGINGFACE_TOKEN", None)
    
    wavlm_model_name: str = "microsoft/wavlm-base-plus"
    # outputs
    out_root: str = "outputs"
    pyannote_rttm_dev: str = "outputs/pyannote_dev"
    pyannote_rttm_test: str = "outputs/pyannote_test"
    ecapa_rttm_dev: str = "outputs/ecapa_dev"
    ecapa_rttm_test: str = "outputs/ecapa_test"


cfg = Config()


# In[3]:


get_ipython().system('jupyter nbconvert --to script config.ipynb')

