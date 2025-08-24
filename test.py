#!/usr/bin/env python3
"""
Main runner script for PDF CV Ranking
Usage: python main.py
"""

import json
from pathlib import Path
import logging
from datetime import datetime
from cv_ranker import PDFCVRanker

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('cv_ranking.log')
        ]
    )

def save_results(results, output_path="results"):
    """Save results to JSON file with timestamp"""
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cv_rankings_{timestamp}.json"
    filepath = output_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return filepath

def display_results(results):
    """Display formatted results"""
    print("\n" + "="*70)
    print("🏆 PDF CV RANKING RESULTS")
    print("="*70)
    
    if not results:
        print("❌ No results to display")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. 📄 {result.get('cv_filename', 'Unknown')}")
        
        if 'error' in result:
            print(f"   ❌ ERROR: {result['error']}")
            continue
        
        # Main scores
        overall = result.get('overall_score', 0)
        match_pct = result.get('match_percentage', overall)
        
        print(f"   📊 Overall Score: {overall}/100 ({match_pct}% match)")
        print(f"   ✅ Recommendation: {result.get('recommendation', 'N/A')}")
        
        # Handle strengths and gaps (could be strings or lists)
        strengths = result.get('key_strengths', [])
        gaps = result.get('key_gaps', [])
        
        # Convert to list if it's a string
        if isinstance(strengths, str):
            strengths = [strengths]
        elif not isinstance(strengths, list):
            strengths = []
            
        if isinstance(gaps, str):
            gaps = [gaps]
        elif not isinstance(gaps, list):
            gaps = []
        
        # Limit to 3 each and join
        strengths = strengths[:3]
        gaps = gaps[:3]
        
        if strengths:
            # Convert all items to strings before joining
            strengths_str = ' • '.join([str(s) for s in strengths])
            print(f"   💪 Key Strengths: {strengths_str}")
        if gaps:
            # Convert all items to strings before joining
            gaps_str = ' • '.join([str(g) for g in gaps])
            print(f"   ⚠️  Areas for Improvement: {gaps_str}")
        
        # Summary
        summary = result.get('summary', '')
        if summary:
            print(f"   📝 Summary: {summary}")
        
        # Show LlamaFactoryAI specific analysis if available
        llama_analysis = result.get('llama_analysis', '')
        if llama_analysis:
            if isinstance(llama_analysis, str):
                # Show more of the analysis and try to end at a complete sentence
                analysis_preview = llama_analysis[:400]
                if len(llama_analysis) > 400:
                    # Try to find the last complete sentence
                    last_period = analysis_preview.rfind('.')
                    if last_period > 300:  # Only truncate if we have a reasonable sentence
                        analysis_preview = analysis_preview[:last_period + 1]
                    else:
                        analysis_preview += "..."
                print(f"   🔍 Detailed Analysis: {analysis_preview}")
            else:
                print(f"   🔍 Detailed Analysis: {str(llama_analysis)[:400]}...")
        
        llama_recommendation = result.get('llama_recommendation', '')
        if llama_recommendation:
            if isinstance(llama_recommendation, str):
                # Show more of the recommendation and try to end at a complete sentence
                rec_preview = llama_recommendation[:300]
                if len(llama_recommendation) > 300:
                    # Try to find the last complete sentence
                    last_period = rec_preview.rfind('.')
                    if last_period > 200:  # Only truncate if we have a reasonable sentence
                        rec_preview = rec_preview[:last_period + 1]
                    else:
                        rec_preview += "..."
                print(f"   💡 AI Recommendations: {rec_preview}")
            else:
                print(f"   💡 AI Recommendations: {str(llama_recommendation)[:300]}...")

def main():
    """Main function to run CV ranking"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 PDF CV Ranker System")
    print("=" * 50)
    
    # Configuration - modify these paths as needed
    config = {
        'models_folder': 'models',
        'job_description_file': 'job_description.txt',
        'cv_folder': 'pdf_cvs',
        'output_folder': 'results',
        'debug_mode': False  # Set to True for detailed debugging
    }
    
    try:
        # Validate required files and folders exist (models folder is optional)
        job_desc_path = Path(config['job_description_file'])
        cv_folder_path = Path(config['cv_folder'])

        if not job_desc_path.exists():
            raise FileNotFoundError(f"Job description file not found: {config['job_description_file']}")

        if not cv_folder_path.exists():
            raise FileNotFoundError(f"CV folder not found: {config['cv_folder']}")

        # Warn if models folder is missing, but do not fail
        models_path = Path(config['models_folder'])
        if not models_path.exists():
            print(f"⚠️  Models folder '{config['models_folder']}' not found. Will attempt to download models online if needed.")

        print(f"📁 Models folder: {config['models_folder']}")
        print(f"📝 Job description: {config['job_description_file']}")
        print(f"📄 CV folder: {config['cv_folder']}")
        if config['debug_mode']:
            print(f"🐛 Debug mode: Enabled")
        print()

        logger.info("Initializing PDF CV Ranker with GPU...")
        ranker = PDFCVRanker(
            models_folder=config['models_folder'],
            force_gpu=True  # Set to False if you want CPU fallback
        )

        # Run the ranking
        logger.info("Starting CV ranking process...")
        results = ranker.rank_all_cvs(
            job_desc_path=config['job_description_file'],
            cv_folder_path=config['cv_folder'],
            debug=config['debug_mode']
        )

        if not results:
            print("❌ No results generated")
            return

        # Save results
        output_file = save_results(results, config['output_folder'])
        logger.info(f"Results saved to: {output_file}")

        # Display results
        display_results(results)

        # Summary statistics
        total_cvs = len(results)
        successful_analyses = len([r for r in results if 'error' not in r])

        if successful_analyses > 0:
            avg_score = sum(r.get('overall_score', 0) for r in results if 'error' not in r) / successful_analyses
            highly_recommended = len([r for r in results if r.get('recommendation') == 'Highly Recommended'])
            recommended = len([r for r in results if r.get('recommendation') == 'Recommended'])

            print(f"\n📈 SUMMARY STATISTICS")
            print("-" * 30)
            print(f"Total CVs Processed: {total_cvs}")
            print(f"Successful Analyses: {successful_analyses}")
            print(f"Average Score: {avg_score:.1f}/100")
            print(f"Highly Recommended: {highly_recommended}")
            print(f"Recommended: {recommended}")

        print(f"\n✅ Process completed successfully!")
        print(f"📁 Detailed results saved to: {output_file}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ {e}")
        print("\nPlease ensure the following files/folders exist:")
        print(f"  - {config['job_description_file']} (text file with job description)")
        print(f"  - {config['cv_folder']}/ (folder containing PDF CVs)")

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        print(f"\n❌ Error: {e}")
        print("\nCheck the log file 'cv_ranking.log' for detailed error information")

if __name__ == "__main__":
    main()