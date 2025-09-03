#!/usr/bin/env python3
"""
Flask API for PDF CV Ranking

Endpoints:
- GET /           -> Serve simple website to submit and display results
- GET /health     -> Health check
- POST /api/rank_cv (multipart/form-data)
   fields: email, job_description, cv_file (pdf)
   returns: { email: str, result: {...} }

Usage:
  python app.py

Environment variables:
- MODELS_FOLDER: path to models directory (default: models)
- FORCE_GPU: "1" to require GPU, anything else or unset allows CPU fallback
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from cv_ranker import PDFCVRanker


# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------

app = Flask(__name__, static_folder="static", static_url_path="")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_ranker: Optional[PDFCVRanker] = None


def get_ranker() -> PDFCVRanker:
    """Lazily initialize and cache the PDFCVRanker instance."""
    global _ranker
    if _ranker is None:
        models_folder = os.environ.get("MODELS_FOLDER", "models")
        force_gpu = os.environ.get("FORCE_GPU", "0") == "1"
        logger.info(
            f"Initializing PDFCVRanker (models_folder='{models_folder}', force_gpu={force_gpu})"
        )
        _ranker = PDFCVRanker(models_folder=models_folder, force_gpu=force_gpu)
    return _ranker


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> tuple:
    return jsonify({"status": "ok"}), 200


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/rank_cv")
def api_rank_cv():
    try:
        # Validate fields
        email = request.form.get("email", "").strip()
        job_description = request.form.get("job_description", "").strip()
        file = request.files.get("cv_file")

        if not email:
            return jsonify({"error": "Missing 'email'"}), 400
        if not job_description:
            return jsonify({"error": "Missing 'job_description'"}), 400
        if file is None or file.filename == "":
            return jsonify({"error": "Missing 'cv_file'"}), 400

        filename = secure_filename(file.filename)

        # Save to a temp file for processing
        tmp_dir = Path(tempfile.gettempdir()) / "cv_ranker_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".pdf", delete=False) as tmp:
            temp_pdf_path = Path(tmp.name)
            file.save(tmp)

        try:
            # Real model inference
            ranker = get_ranker()
            result = ranker.rank_single_cv(job_description=job_description, cv_path=temp_pdf_path, debug=False)
            payload = {"email": email, "result": result}
            return jsonify(payload), 200

        finally:
            # Cleanup the temp file
            try:
                if temp_pdf_path.exists():
                    temp_pdf_path.unlink()
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove temp file {temp_pdf_path}: {cleanup_err}")

    except Exception as e:
        logger.exception("Error in /api/rank_cv")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    # Allow external access by binding to all interfaces
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=True)