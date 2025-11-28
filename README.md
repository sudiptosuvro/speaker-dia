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

## 📁 Folder Structure

- config.py                   # Global experiment config 
- utils.py                    # RTTM writing, helpers  
- metrics.py                  # DER/JER computation  
- ecapa_vbx_run.py            # ECAPA embeddings 
- pyannote_run.py             # PyAnnote diarization pipeline
- main_ecapa_vbx.ipynb        # Run ECAPA+VBx diarization  
- main_pyannote.ipynb         # Run Pyannote diarization  
- eval.ipynb                  # Evaluation + plotting (DER/JER)

### Requirements

torch==2.1.2
torchaudio==2.1.2
transformers==4.37.0
huggingface_hub
pyannote.audio==3.1.1
pyannote.core
pyannote.metrics
