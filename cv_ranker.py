#!/usr/bin/env python3
"""
PDF CV Ranker - Main module for ranking PDF CVs against job descriptions
Uses LlamaFactoryAI/Llama-3.1-8B-Instruct-cv-job-description-matching LoRA model
Updated with robust JSON parsing to handle malformed responses
Enhanced with multi-language support - automatically detects and translates non-English CVs
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import pdfplumber
import PyPDF2
from pathlib import Path
from typing import List, Dict
import logging
import os
import requests
from langdetect import detect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the robust JSON parser (save the previous artifact as robust_json_parser.py)
from robust_json_parser import RobustJSONParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFCVRanker:
    def __init__(self, models_folder: str = "models", force_gpu: bool = True):
        """
        Initialize the PDF CV Ranker with local Llama model and LoRA extension, or download from Hugging Face if not found locally.
        Args:
            models_folder: Path to folder containing llama and lora models
            force_gpu: If True, will raise error if GPU not available
        """
        self.models_folder = Path(models_folder)
        self.json_parser = RobustJSONParser()
        self._setup_device(force_gpu)

        # Model names for Hugging Face
        self.hf_base_model = "meta-llama/Llama-3.1-8B-Instruct"
        self.hf_lora_model = "LlamaFactoryAI/Llama-3.1-8B-Instruct-cv-job-description-matching"

        # Local model paths
        self.base_model_path = self.models_folder / "llama-3.1-8b-instruct"
        self.lora_model_path = self.models_folder / "lora-cv-match"

        # Check if local models exist, else use Hugging Face
        self.use_local_base = self.base_model_path.exists()
        self.use_local_lora = self.lora_model_path.exists()

        if not self.use_local_base:
            logger.warning(f"Base model not found locally at {self.base_model_path}, will use Hugging Face: {self.hf_base_model}")
        if not self.use_local_lora:
            logger.warning(f"LoRA model not found locally at {self.lora_model_path}, will use Hugging Face: {self.hf_lora_model}")

        self._load_models()
    
    def _setup_device(self, force_gpu: bool = True):
        """Setup device configuration with detailed GPU information"""
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1024**3
            
            logger.info(f"🚀 GPU DETECTED!")
            logger.info(f"   Device: {gpu_name}")
            logger.info(f"   Memory: {gpu_memory:.1f} GB")
            logger.info(f"   Available GPUs: {gpu_count}")
            logger.info(f"   Using GPU {current_gpu}")
            
            # Clear GPU cache for maximum memory availability
            torch.cuda.empty_cache()
            
        else:
            if force_gpu:
                raise RuntimeError(
                    "❌ GPU not available but force_gpu=True. "
                    "Install CUDA-compatible PyTorch or set force_gpu=False"
                )
            self.device = "cpu"
            logger.warning("⚠️ GPU not available, falling back to CPU (will be slower)")
        
        logger.info(f"Using device: {self.device}")
    
    def _load_models(self):
        """Load tokenizer and model, using local files if available, else from Hugging Face online."""
        # --- Tokenizer ---
        if self.use_local_base:
            logger.info(f"📥 Loading tokenizer from local: {self.base_model_path}")
            tokenizer_src = str(self.base_model_path)
            tokenizer_cache = None
        else:
            logger.info(f"📥 Loading tokenizer from Hugging Face: {self.hf_base_model}")
            tokenizer_src = self.hf_base_model
            tokenizer_cache = str(self.models_folder)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, cache_dir=tokenizer_cache)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- Model ---
        logger.info("🔄 Loading base model...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.uint8
        )
        model_kwargs = {
            "quantization_config": quantization_config,
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = None

        if self.use_local_base:
            base_model_src = str(self.base_model_path)
            base_model_cache = None
        else:
            base_model_src = self.hf_base_model
            base_model_cache = str(self.models_folder)

        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_src, cache_dir=base_model_cache, **model_kwargs)

        # --- LoRA ---
        # Try to load LoRA if available locally or online
        if self.use_local_lora:
            logger.info(f"🔧 Loading local LoRA adapter: {self.lora_model_path}")
            try:
                self.model = PeftModel.from_pretrained(self.base_model, str(self.lora_model_path))
                logger.info("✅ Local LoRA adapter loaded successfully!")
            except Exception as e:
                logger.error(f"Failed to load local LoRA adapter: {e}")
                raise
        else:
            logger.info(f"🔧 Loading LoRA adapter from Hugging Face: {self.hf_lora_model}")
            try:
                self.model = PeftModel.from_pretrained(self.base_model, self.hf_lora_model, cache_dir=str(self.models_folder))
                logger.info("✅ Hugging Face LoRA adapter loaded successfully!")
            except Exception as e:
                logger.warning(f"Could not load LoRA from Hugging Face: {e}. Using base model only.")
                self.model = self.base_model

        # GPU memory optimization
        if self.device == "cuda":
            self.model = self.model.half()  # Use FP16 for memory efficiency
            torch.cuda.empty_cache()  # Clear cache after loading
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"📊 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")

        logger.info("✅ Model loaded successfully!")

    def detect_language(self, text: str) -> str:
        """
        Detect language of text using langdetect
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (e.g., 'en', 'ar', 'fr') or 'en' if detection fails
        """
        try:
            if not text or len(text.strip()) < 10:
                logger.warning("Text too short for language detection, assuming English")
                return 'en'
            
            # Clean text for better detection
            cleaned_text = ' '.join(text.split())
            detected_lang = detect(cleaned_text)
            logger.info(f"Detected language: {detected_lang}")
            return detected_lang
        except Exception as e:
            logger.warning(f"Language detection failed: {e}, assuming English")
            return 'en'
    
    def upload_pdf_to_chatpdf(self, pdf_path: Path, api_key: str) -> str:
        """
        Upload PDF to ChatPDF and return source ID
        
        Args:
            pdf_path: Path to PDF file
            api_key: ChatPDF API key
            
        Returns:
            Source ID from ChatPDF or None if failed
        """
        try:
            headers = {'x-api-key': api_key}
            with open(pdf_path, 'rb') as pdf_file:
                files = {'file': (pdf_path.name, pdf_file, 'application/pdf')}
                response = requests.post(
                    'https://api.chatpdf.com/v1/sources/add-file',
                    headers=headers,
                    files=files,
                    timeout=120
                )
                if response.status_code == 200:
                    source_id = response.json().get('sourceId')
                    logger.info(f"PDF uploaded to ChatPDF with ID: {source_id}")
                    return source_id
                else:
                    logger.error(f"ChatPDF upload failed: {response.status_code} - {response.text}")
                    return None
        except Exception as e:
            logger.error(f"Error uploading PDF to ChatPDF: {e}")
            return None
    
    def translate_with_chatpdf(self, source_id: str, api_key: str) -> str:
        """
        Request translation from ChatPDF
        
        Args:
            source_id: ChatPDF source ID
            api_key: ChatPDF API key
            
        Returns:
            Translated text or None if failed
        """
        try:
            headers = {'x-api-key': api_key, 'Content-Type': 'application/json'}

            # Now translate
            translate_payload = {
                'sourceId': source_id,
                'messages': [
                    {
                        'role': 'user',
                        'content': 'Convert the above text to English. Translate every single word and section. Format as a proper CV with:\n\nName: [translated name]\nEmail: [keep exact email]\nPhone: [keep exact phone]\nLocation: [translate location]\nObjective: [translate career objective]\nEducation: [translate education with degrees, universities, dates, GPA]\nExperience: [translate work experience with job titles, companies, dates]\nSkills: [translate all skills]\nCertifications: [translate certifications]\nLanguages: [translate languages]\n\nTranslate everything to professional English while keeping numbers, dates, and contact info unchanged.'
                    }
                ]
            }
            
            response = requests.post(
                'https://api.chatpdf.com/v1/chats/message',
                headers=headers,
                json=translate_payload,
                timeout=120
            )
            
            if response.status_code == 200:
                translated_text = response.json().get('content', '').strip()
                logger.info("CV translation completed successfully")
                return translated_text
            else:
                logger.error(f"ChatPDF translation failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error during ChatPDF translation: {e}")
            return None
    
    def cleanup_chatpdf(self, source_id: str, api_key: str):
        """
        Delete uploaded PDF from ChatPDF
        
        Args:
            source_id: ChatPDF source ID
            api_key: ChatPDF API key
        """
        try:
            headers = {'x-api-key': api_key}
            requests.post(
                'https://api.chatpdf.com/v1/sources/delete',
                headers=headers,
                json={'sources': [source_id]},
                timeout=30
            )
            logger.debug(f"Cleaned up ChatPDF source: {source_id}")
        except Exception as e:
            logger.debug(f"ChatPDF cleanup failed (non-critical): {e}")

    def translate_job_description(self, job_description: str, api_key: str) -> str:
        """
        Translate job description to English using ChatPDF API
        
        Args:
            job_description: Job description text to translate
            api_key: ChatPDF API key
            
        Returns:
            Translated job description or original if translation fails
        """
        try:
            headers = {'x-api-key': api_key, 'Content-Type': 'application/json'}
            
            # Create a temporary text-based conversation for translation
            translate_payload = {
                'messages': [
                    {
                        'role': 'user',
                        'content': f'Translate the following job description to professional English. Keep all technical terms, requirements, and qualifications accurate. Maintain the structure and formatting:\n\n{job_description}'
                    }
                ]
            }
            
            response = requests.post(
                'https://api.chatpdf.com/v1/chats/message',
                headers=headers,
                json=translate_payload,
                timeout=120
            )
            
            if response.status_code == 200:
                translated_text = response.json().get('content', '').strip()
                logger.info("Job description translation completed successfully")
                return translated_text
            else:
                logger.warning(f"Job description translation failed: {response.status_code}, using original")
                return job_description
                
        except Exception as e:
            logger.warning(f"Error translating job description: {e}, using original")
            return job_description
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF using multiple methods for best results
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        text = ""
        
        # Method 1: pdfplumber (best for complex layouts)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if text.strip():
                logger.info(f"✅ Text extracted using pdfplumber: {pdf_path.name}")
                return text.strip()
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path.name}: {e}")
        
        # Method 2: PyPDF2 fallback
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                if text.strip():
                    logger.info(f"✅ Text extracted using PyPDF2: {pdf_path.name}")
                    return text.strip()
        except Exception as e:
            logger.error(f"PyPDF2 also failed for {pdf_path.name}: {e}")
        
        return ""
    
    def find_pdf_files(self, folder_path: str) -> List[Path]:
        """
        Find all PDF files in the specified folder
        
        Args:
            folder_path: Path to folder containing PDF CVs
            
        Returns:
            List of PDF file paths
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"CV folder not found: {folder_path}")
        
        pdf_files = list(folder.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in: {folder_path}")
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        return pdf_files
    
    def load_job_description(self, job_desc_path: str) -> str:
        """
        Load job description from text file
        
        Args:
            job_desc_path: Path to job description text file
            
        Returns:
            Job description content
        """
        job_path = Path(job_desc_path)
        if not job_path.exists():
            raise FileNotFoundError(f"Job description file not found: {job_desc_path}")
        
        try:
            with open(job_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                raise ValueError("Job description file is empty")
            
            logger.info(f"✅ Job description loaded: {len(content)} characters")
            return content
            
        except Exception as e:
            raise Exception(f"Error reading job description: {e}")
    
    def create_ranking_prompt(self, job_description: str, cv_text: str) -> str:
        """Create optimized prompt for CV ranking with local model"""
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert HR recruiter specializing in CV analysis. Your task is to analyze the provided CV against the job description and provide a detailed assessment in JSON format.

CRITICAL: You must respond with ONLY valid JSON. No additional text before or after. The JSON must be properly formatted with correct syntax.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Job Description:
{job_description[:3000]}

CV Content:
{cv_text[:5000]}

Analyze this CV against the job requirements. Respond with ONLY this exact JSON format (ensure proper comma placement and quote escaping):

{{
    "matching_analysis": "Your detailed analysis here (keep under 300 characters)",
    "description": "Brief summary here (keep under 200 characters)",
    "score": 100,
    "recommendation": "Your recommendation here (keep under 200 characters)"
}}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def generate_ranking(self, prompt: str, max_retries: int = 2) -> str:
        """Generate CV ranking using the local model with retry logic"""
        
        for attempt in range(max_retries + 1):
            try:
                # Tokenize with GPU-optimized settings
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=3000,
                    padding=True
                )
                
                # Move inputs to GPU
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # GPU memory check before generation
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    memory_before = torch.cuda.memory_allocated() / 1024**3
                    logger.debug(f"Memory before generation: {memory_before:.2f}GB")
                
                with torch.no_grad():
                    # Adjust generation parameters for better JSON output
                    generation_params = {
                        "max_new_tokens": 400,  
                        "temperature": 0.1,
                        "do_sample": True,
                        "top_p": 0.85,
                        "top_k": 40,
                        "repetition_penalty": 1.05,
                        "pad_token_id": self.tokenizer.eos_token_id,
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "use_cache": True,
                        "num_beams": 1,
                    }
                    
                    # Generate with optimized parameters
                    outputs = self.model.generate(
                        **inputs,
                        **generation_params
                    )
                
                # Clear GPU cache after generation
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the assistant response (after the prompt)
                if "<|start_header_id|>assistant<|end_header_id|>" in response:
                    assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
                elif "assistant" in response.lower():
                    parts = response.split("assistant")
                    if len(parts) > 1:
                        assistant_response = parts[-1].strip()
                    else:
                        assistant_response = response.strip()
                else:
                    # Fallback: try to find the JSON part
                    start_idx = response.find('{')
                    if start_idx != -1:
                        assistant_response = response[start_idx:].strip()
                    else:
                        assistant_response = response.strip()
                
                # Quick validation - if response looks like it might be valid JSON, return it
                if '{' in assistant_response and '}' in assistant_response:
                    return assistant_response
                
                if attempt < max_retries:
                    logger.warning(f"Generation attempt {attempt + 1} produced poor output, retrying...")
                    continue
                else:
                    return assistant_response
                    
            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    return '{"score": 0, "matching_analysis": "Generation failed", "description": "Error occurred", "recommendation": "Manual review required"}'
                continue
        
        return assistant_response
    
    def parse_json_response(self, response: str, cv_filename: str) -> Dict:
        """
        Parse and validate JSON response from local model using robust parser
        """
        return self.json_parser.parse_json_response(response, cv_filename)
    
    def rank_single_cv(self, job_description: str, cv_path: Path, debug: bool = False) -> Dict:
        """Rank a single PDF CV against job description"""
        logger.info(f"Processing: {cv_path.name}")
        
        # Extract text from PDF
        cv_text = self.extract_text_from_pdf(cv_path)
        if not cv_text:
            return {
                "cv_filename": cv_path.name,
                "error": f"Could not extract text from PDF: {cv_path.name}",
                "overall_score": 0
            }
        
        # Detect language
        detected_lang = self.detect_language(cv_text)
        if detected_lang != 'en':
            # Translate non-English CVs
            api_key = os.getenv('CHATPDF_API_KEY')
            if not api_key:
                logger.error("CHATPDF_API_KEY not set, skipping translation")
                return {
                    "cv_filename": cv_path.name,
                    "error": "Translation failed due to missing API key",
                    "overall_score": 0
                }
            
            source_id = self.upload_pdf_to_chatpdf(cv_path, api_key)
            if not source_id:
                logger.error("Failed to upload PDF to ChatPDF, skipping translation")
                return {
                    "cv_filename": cv_path.name,
                    "error": "Translation failed due to upload error",
                    "overall_score": 0
                }
            
            translated_text = self.translate_with_chatpdf(source_id, api_key)
            if not translated_text:
                logger.error("Failed to translate CV, skipping ranking")
                self.cleanup_chatpdf(source_id, api_key)
                return {
                    "cv_filename": cv_path.name,
                    "error": "Translation failed",
                    "overall_score": 0
                }
            
            cv_text = translated_text
            self.cleanup_chatpdf(source_id, api_key)
        
        # Generate ranking using chat template with retry logic
        prompt = self.create_ranking_prompt(job_description, cv_text)
        if debug:
            logger.debug(f"Prompt created for {cv_path.name}")
        
        response = self.generate_ranking(prompt)
        
        if debug:
            logger.debug(f"Response length: {len(response)} characters")
            logger.debug(f"Response preview: {response[:300]}...")
        
        # Parse response using robust parser
        result = self.parse_json_response(response, cv_path.name)
        result["cv_filename"] = cv_path.name
        result["raw_response"] = response[:500] + "..." if len(response) > 500 else response
        
        return result
    
    def rank_all_cvs(self, job_desc_path: str, cv_folder_path: str, debug: bool = False) -> List[Dict]:
        """
        Rank all PDF CVs in folder against job description
        
        Args:
            job_desc_path: Path to job description text file
            cv_folder_path: Path to folder containing PDF CVs
            debug: Enable debug logging for troubleshooting
            
        Returns:
            List of ranking results sorted by score
        """
        # Load job description
        job_description = self.load_job_description(job_desc_path)
        
        # Detect language of job description
        detected_lang = self.detect_language(job_description)
        if detected_lang != 'en':
            # Translate non-English job description
            api_key = os.getenv('CHATPDF_API_KEY')
            if not api_key:
                logger.error("CHATPDF_API_KEY not set, skipping translation")
                return []
            
            translated_job_description = self.translate_job_description(job_description, api_key)
            if not translated_job_description:
                logger.error("Failed to translate job description, skipping ranking")
                return []
            
            job_description = translated_job_description
        
        # Find PDF files
        pdf_files = self.find_pdf_files(cv_folder_path)
        
        logger.info(f"Starting analysis of {len(pdf_files)} CVs...")
        logger.info("-" * 50)
        
        results = []
        for i, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"Processing {i}/{len(pdf_files)}: {pdf_file.name}")
            
            try:
                result = self.rank_single_cv(job_description, pdf_file, debug=debug)
                results.append(result)
                
                if 'error' not in result:
                    score = result.get('overall_score', 0)
                    recommendation = result.get('recommendation', 'N/A')
                    logger.info(f"✅ Completed - Score: {score}/100 - {recommendation}")
                    
                    if debug and 'raw_response' in result:
                        logger.debug(f"Raw response for {pdf_file.name}: {result['raw_response']}")
                else:
                    logger.error(f"❌ Error: {result['error']}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_file.name}: {e}")
                results.append({
                    "cv_filename": pdf_file.name,
                    "error": str(e),
                    "overall_score": 0
                })
        
        # Sort by overall score (highest first)
        results.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
        return results