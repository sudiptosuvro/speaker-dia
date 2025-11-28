# 🎙️ Speaker Diarization Pipeline  
### ECAPA-TDNN + VBx │ PyAnnote (HF)

This repository contains a complete and extensible **speaker diarization system** that integrates two SOTA pipelines:

- **PyAnnote Speaker Diarization 3.1 (HF)**
- **ECAPA-TDNN + VBx** (Agglomerative clustering with HMM resegmentation)

The project includes:
✔ Embedding extraction  
✔ VAD  
✔ Clustering  
✔ RTTM generation  
✔ DER/JER evaluation  
✔ Result visualizations  

# 📁 Folder Structure

speaker-dia/
│
├── config.py                     # Global experiment config
├── utils.py                      # RTTM writing, helpers
├── metrics.py                    # DER/JER computation
│
├── ecapa_vbx_run.py              # ECAPA embeddings + VBx clustering
├── pyannote_run.py               # PyAnnote diarization pipeline
│
├── main_ecapa_vbx.ipynb          # Run ECAPA+VBx
├── main_pyannote.ipynb           # Run PyAnnote
├── eval.ipynb                    # Evaluation + plotting
│
├── outputs/                      # Generated RTTM files
├── figs/                         # Plots (DER/JER)
├── results_summary.csv           # Final metrics summary
└── dataset/                      # WAV + RTTM (not included)
