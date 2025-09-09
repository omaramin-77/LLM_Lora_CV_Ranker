# PDF CV Ranker (Llama 3.1 + LoRA)

Analyze and rank PDF CVs against a job description using either a local Llama 3.1 8B model with LoRA or ChatPDF API. Includes automatic language detection and translation for non-English CVs, CLI runner, and Flask API.

## Features
- Rank multiple PDF CVs against a job description
- **Dual ranking modes**: Local Llama model or ChatPDF API
- **Automatic language detection and translation** - supports non-English CVs via ChatPDF API
- Local-first with optional online model download via Hugging Face
- Cloud-ready with ChatPDF-only mode (no local model required)
- Robust JSON parsing for imperfect model outputs
- Web API endpoint for uploading a single PDF and getting a result

## Requirements
- Windows 10/11 or Linux/Mac (Windows GPU defaults provided)
- Python 3.10–3.11 (recommended)
- For GPU acceleration: NVIDIA GPU with CUDA 12.1 runtime

## Minimum System Requirements

### Local Model Mode (USE_LOCAL=true)
- OS: Windows 10/11 (64-bit) or Ubuntu 20.04+/Debian 12+; macOS (CPU-only)
- Python: 3.10 or 3.11
- CPU RAM: 12 GB minimum (16 GB recommended)
- GPU (recommended): NVIDIA RTX-class with ≥ 8 GB VRAM (12 GB recommended)
  - Driver: NVIDIA driver supporting CUDA 12.1
  - Runtime: CUDA 12.1 (PyTorch wheels in requirements use cu121)
- Disk Space: 15–25 GB free for model weights/cache and results
- Network: Required on first run to download models if `models/` is empty
  - Offline use: Place base model in `models/llama-3.1-8b-instruct/` and LoRA in `models/lora-cv-match/`

### ChatPDF API Mode (USE_LOCAL=false)
- OS: Any OS supporting Python 3.10+
- Python: 3.10 or 3.11
- CPU RAM: 2 GB minimum
- GPU: Not required
- Disk Space: 1 GB for application and results
- Network: Required for ChatPDF API calls
- ChatPDF API Key: Required (see configuration section)

Notes:
- CPU mode is supported but slow for local models; set `force_gpu=False` in `test.py` if no GPU.
- ChatPDF mode bypasses all local model requirements and works on any system with internet access.
- Verify GPU is visible to PyTorch (local mode only):
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

### 4) Configure ranking mode (optional)
Create a `.env` file based on `.env.example`:
```bash
# Use local Llama model (true) or ChatPDF API (false) for ranking
USE_LOCAL=true

# Required for ChatPDF API mode and non-English CV translation
CHATPDF_API_KEY=your_chatpdf_api_key_here
```

### 5) Run the CLI
```bash
python test.py
```
This will:
- **Local mode (USE_LOCAL=true)**: Load local models from `models/` if present, else download from Hugging Face
- **ChatPDF mode (USE_LOCAL=false)**: Skip model loading and use ChatPDF API for ranking
- Process all PDFs in `pdf_cvs/`
- Save results JSON to `results/` and print a summary

## Ranking Modes

### Local Model Mode (USE_LOCAL=true)
- Uses local Llama 3.1 8B model with LoRA for CV ranking
- Requires GPU/CPU resources and model downloads
- Fully offline after initial setup
- Higher accuracy for specialized CV ranking tasks

### ChatPDF API Mode (USE_LOCAL=false)
- Uses ChatPDF API for CV ranking
- No local model requirements
- Requires internet connection and API key
- Faster startup and lower resource usage
- Good for cloud deployments and systems without GPU

## GPU vs CPU (Local Mode Only)
- By default the CLI requires a GPU when using local models. To allow CPU fallback, change `force_gpu` to `False` in `test.py` config or in `cv_ranker.PDFCVRanker` initialization.
- For the API, set env var `FORCE_GPU=1` to require GPU, otherwise it will allow CPU.
- ChatPDF mode bypasses GPU/CPU requirements entirely.

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

## API Configuration

### Environment Variables
Create a `.env` file based on `.env.example`:

```bash
# Use local Llama model (true) or ChatPDF API (false) for ranking
USE_LOCAL=true

# Required for ChatPDF API mode and non-English CV translation
CHATPDF_API_KEY=your_chatpdf_api_key_here
```

### Getting ChatPDF API Key
1. Visit [ChatPDF](https://www.chatpdf.com/)
2. Sign up for an account
3. Navigate to API settings to get your API key
4. Add the key to your `.env` file

**Note**: 
- If `USE_LOCAL=false` and no API key is provided, the system will fail to rank CVs
- If `USE_LOCAL=true` and no API key is provided, the system will still work for English CVs but skip non-English CVs with a warning

## Language Support

The system automatically detects the language of uploaded CVs:
- **English CVs**: Processed directly by the ranking model
- **Non-English CVs**: Automatically translated to English using ChatPDF API before ranking
- **Supported languages**: All languages supported by ChatPDF (Arabic, Spanish, French, German, Chinese, etc.)

Translation process:
1. Language detection using `langdetect`
2. If non-English detected, CV is uploaded to ChatPDF
3. Translation request sent to ChatPDF API
4. Translated English text is used for ranking
5. Original filename and language info preserved in results

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
