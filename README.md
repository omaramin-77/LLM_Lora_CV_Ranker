# PDF CV Ranker (Llama 3.1 + LoRA)

Analyze and rank PDF CVs against a job description using a local Llama 3.1 8B model with a CV-matching LoRA. Includes a CLI runner and a Flask API.

## Features
- Rank multiple PDF CVs against a job description
- Local-first with optional online model download via Hugging Face
- Robust JSON parsing for imperfect model outputs
- Web API endpoint for uploading a single PDF and getting a result

## Requirements
- Windows 10/11 or Linux/Mac (Windows GPU defaults provided)
- Python 3.10–3.11 (recommended)
- For GPU acceleration: NVIDIA GPU with CUDA 12.1 runtime

## Minimum System Requirements
- OS: Windows 10/11 (64-bit) or Ubuntu 20.04+/Debian 12+; macOS (CPU-only)
- Python: 3.10 or 3.11
- CPU RAM: 12 GB minimum (16 GB recommended)
- GPU (recommended): NVIDIA RTX-class with ≥ 8 GB VRAM (12 GB recommended)
  - Driver: NVIDIA driver supporting CUDA 12.1
  - Runtime: CUDA 12.1 (PyTorch wheels in requirements use cu121)
- Disk Space: 15–25 GB free for model weights/cache and results
- Network: Required on first run to download models if `models/` is empty
  - Offline use: Place base model in `models/llama-3.1-8b-instruct/` and LoRA in `models/lora-cv-match/`

Notes:
- CPU mode is supported but slow; set `force_gpu=False` in `test.py` if no GPU.
- Verify GPU is visible to PyTorch:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```

## Quickstart

### 1) Clone
```bash
git clone <your-repo-url>.git
cd <your-repo-folder>
```

### 2) Create venv and install deps
```bash
python -m venv env
# Windows PowerShell
env\\Scripts\\Activate.ps1
# or cmd
# env\\Scripts\\activate.bat

pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- The requirements include CUDA 12.1 wheels for PyTorch on Windows and bitsandbytes support. On Windows, a community `bitsandbytes-windows` wheel is used.
- If you are on Linux/Mac, the regular `bitsandbytes` wheel will be installed.

### 3) Prepare inputs
- Put your job description text in `job_description.txt`.
- Place PDFs to evaluate in `pdf_cvs/`.

### 4) Run the CLI
```bash
python test.py
```
This will:
- Load local models from `models/` if present, else download from Hugging Face
- Process all PDFs in `pdf_cvs/`
- Save results JSON to `results/` and print a summary

## GPU vs CPU
- By default the CLI requires a GPU. To allow CPU fallback, change `force_gpu` to `False` in `test.py` config or in `cv_ranker.PDFCVRanker` initialization.
- For the API, set env var `FORCE_GPU=1` to require GPU, otherwise it will allow CPU.

## Running the API
```bash
python app.py
```
- Opens on `http://localhost:5000`
- Health check: `GET /health`
- Upload and rank: `POST /api/rank_cv` (multipart/form-data)
  - fields: `email` (str), `job_description` (str), `cv_file` (PDF)

Example curl:
```bash
curl -X POST http://localhost:5000/api/rank_cv \
  -F "email=user@example.com" \
  -F "job_description=$(cat job_description.txt)" \
  -F "cv_file=@pdf_cvs/example.pdf"
```

## Models
- Base: `meta-llama/Llama-3.1-8B-Instruct`
- LoRA: `LlamaFactoryAI/Llama-3.1-8B-Instruct-cv-job-description-matching`

Local paths (optional):
- `models/llama-3.1-8b-instruct/`
- `models/lora-cv-match/`

If not present, they will be downloaded automatically to `models/` cache.

## Project Layout
- `test.py` — CLI entry point for ranking batches of PDFs
- `app.py` — Flask API server
- `cv_ranker.py` — Core ranking logic and model loading
- `robust_json_parser.py` — Fault-tolerant JSON parsing for model output
- `pdf_cvs/` — Input PDFs
- `results/` — Generated results
- `models/` — Optional local model files

## Troubleshooting
- GPU not detected: ensure NVIDIA drivers and CUDA 12.1 runtime are installed. Verify with `python -c "import torch; print(torch.cuda.is_available())"`.
- bitsandbytes on Windows: the `bitsandbytes-windows` package is included. If you encounter issues, consider CPU fallback or Linux.
- Out of memory: reduce `max_new_tokens` in `cv_ranker.generate_ranking` or close other GPU apps.

## License
See the model licenses and project license files where applicable.


