import json
import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class RobustJSONParser:
    """Robust JSON parser for handling LLM responses with various formatting issues"""
    
    def __init__(self):
        self.default_response = {
            "overall_score": 0.0,
            "key_strengths": ["Analysis completed - please review raw response"],
            "key_gaps": ["Detailed review needed"],
            "recommendation": "Review Required",
            "summary": "JSON parsing required fallback - check detailed fields",
            "match_percentage": 0.0
        }
    
    def clean_json_string(self, json_str: str) -> str:
        """Clean and fix common JSON formatting issues while preserving content length"""
        # Remove any text before the first {
        start_idx = json_str.find('{')
        if start_idx == -1:
            return ""
        
        # Find the last } that could be the end of our JSON
        end_idx = json_str.rfind('}')
        if end_idx == -1 or end_idx <= start_idx:
            # If no closing brace found, try to add one
            potential_json = json_str[start_idx:]
            if potential_json.strip():
                # Count open braces vs close braces
                open_count = potential_json.count('{')
                close_count = potential_json.count('}')
                missing_braces = open_count - close_count
                if missing_braces > 0:
                    potential_json += '}' * missing_braces
                return potential_json
            return ""
        
        # Extract potential JSON
        potential_json = json_str[start_idx:end_idx + 1]
        
        # Clean up common issues while preserving content
        potential_json = potential_json.strip()
        
        # Fix common escaping issues but preserve content length
        potential_json = potential_json.replace('\\"', '"')
        potential_json = potential_json.replace('\\n', ' ')
        potential_json = potential_json.replace('\\r', ' ')
        potential_json = potential_json.replace('\\t', ' ')
        
        # Normalize whitespace but don't over-compress
        potential_json = re.sub(r'\s+', ' ', potential_json)
        
        # Fix trailing commas before closing braces/brackets
        potential_json = re.sub(r',\s*}', '}', potential_json)
        potential_json = re.sub(r',\s*]', ']', potential_json)
        
        return potential_json
    
    def attempt_json_repair(self, json_str: str) -> Optional[str]:
        """Attempt to repair malformed JSON while preserving content"""
        try:
            # First, try basic cleaning
            cleaned = self.clean_json_string(json_str)
            if not cleaned:
                return None
            
            # Try to parse as-is first
            json.loads(cleaned)
            return cleaned
            
        except json.JSONDecodeError as e:
            logger.debug(f"Initial parse failed at position {e.pos}: {e}")
            
            # Strategy 1: Try to fix incomplete strings
            try:
                fixed = self.fix_incomplete_strings(cleaned, e)
                if fixed:
                    json.loads(fixed)
                    return fixed
            except:
                pass
            
            # Strategy 2: Try to fix at the error position
            try:
                fixed = self.fix_at_error_position(cleaned, e)
                if fixed:
                    json.loads(fixed)
                    return fixed
            except:
                pass
            
            # Strategy 3: Try to truncate at last valid position
            try:
                truncated = self.truncate_at_last_valid_position(cleaned)
                if truncated:
                    json.loads(truncated)
                    return truncated
            except:
                pass
            
            return None
    
    def fix_incomplete_strings(self, json_str: str, error: json.JSONDecodeError) -> Optional[str]:
        """Fix incomplete strings that may be causing JSON errors"""
        try:
            # Find the position where the error occurred
            error_pos = getattr(error, 'pos', len(json_str))
            
            # Look around the error position for context
            context_start = max(0, error_pos - 50)
            context_end = min(len(json_str), error_pos + 50)
            context = json_str[context_start:context_end]
            
            # Check if we're in the middle of a string value
            if '"' in context:
                # Count quotes before the error position
                quotes_before = json_str[:error_pos].count('"')
                
                # If odd number of quotes, we might be in an unclosed string
                if quotes_before % 2 == 1:
                    # Find the last quote before error
                    last_quote = json_str.rfind('"', 0, error_pos)
                    if last_quote != -1:
                        # Check if this looks like an incomplete string value
                        after_quote = json_str[last_quote + 1:error_pos + 20]
                        
                        # If it contains no closing quote but has meaningful content
                        if '"' not in after_quote and len(after_quote.strip()) > 5:
                            # Try to close the string at a reasonable point
                            # Look for word boundaries
                            words = after_quote.split()
                            if words:
                                # Close after the last complete word
                                last_word_end = after_quote.rfind(words[-1]) + len(words[-1])
                                if last_word_end < len(after_quote):
                                    fixed = (json_str[:last_quote + 1 + last_word_end] + 
                                           '"' + json_str[error_pos:])
                                    return fixed
                
            return None
            
        except Exception as e:
            logger.debug(f"Error in fix_incomplete_strings: {e}")
            return None
    
    def fix_at_error_position(self, json_str: str, error: json.JSONDecodeError) -> Optional[str]:
        """Try to fix JSON at the specific error position"""
        try:
            error_pos = getattr(error, 'pos', len(json_str))
            
            # Common fixes based on error message
            if "Expecting ',' delimiter" in str(error):
                # Try inserting a comma
                fixed = json_str[:error_pos] + ',' + json_str[error_pos:]
                return fixed
            
            elif "Expecting ':' delimiter" in str(error):
                # Try inserting a colon
                fixed = json_str[:error_pos] + ':' + json_str[error_pos:]
                return fixed
                
            elif "Expecting property name" in str(error):
                # Try closing the current object
                fixed = json_str[:error_pos] + '}'
                return fixed
            
            return None
            
        except Exception as e:
            logger.debug(f"Error in fix_at_error_position: {e}")
            return None
    
    def truncate_at_last_valid_position(self, json_str: str) -> Optional[str]:
        """Find the last valid JSON structure and truncate there"""
        brace_count = 0
        bracket_count = 0
        in_string = False
        escape_next = False
        last_valid_pos = -1
        
        for i, char in enumerate(json_str):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\' and in_string:
                escape_next = True
                continue
                
            if char == '"':
                in_string = not in_string
            elif not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_valid_pos = i + 1
                elif char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
        
        if last_valid_pos > 0:
            return json_str[:last_valid_pos]
        
        return None
    
    def extract_values_with_regex(self, response: str) -> Dict:
        """Extract values using regex patterns while preserving more content"""
        result = self.default_response.copy()
        
        # Extract score with multiple patterns (including decimals)
        score_patterns = [
            r'"score"\s*:\s*(\d+(?:\.\d+)?)',
            r'score["\']?\s*:\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)/100',
            r'(\d+(?:\.\d+)?)%\s*match',
            r'score.*?(\d+(?:\.\d+)?)',
            r'rating["\']?\s*:\s*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    if 0 <= score <= 100:
                        result['overall_score'] = round(score, 2)
                        result['match_percentage'] = round(score, 2)
                        break
                except ValueError:
                    continue
        
        # Extract analysis text - preserve longer content
        analysis_patterns = [
            r'"matching_analysis"\s*:\s*"([^"]{20,800})"',  # Increased limit
            r'"analysis"\s*:\s*"([^"]{20,800})"',
            r'"description"\s*:\s*"([^"]{20,600})"',
            r'analysis[^"]*"([^"]{20,700})"'
        ]
        
        for pattern in analysis_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                analysis = match.group(1).strip()
                # Clean up the analysis but preserve length
                analysis = re.sub(r'\s+', ' ', analysis)
                # Find the last complete sentence to avoid cutting off mid-sentence
                sentences = analysis.split('. ')
                if len(sentences) > 1 and len(analysis) > 600:
                    # Take all complete sentences that fit within 600 chars
                    complete_analysis = ''
                    for sentence in sentences[:-1]:  # Exclude the last incomplete sentence
                        if len(complete_analysis + sentence + '. ') <= 600:
                            complete_analysis += sentence + '. '
                        else:
                            break
                    result['key_strengths'] = [complete_analysis.strip()]
                else:
                    result['key_strengths'] = [analysis[:600]]  # Increased limit
                break
        
        # Extract recommendation with better context
        rec_patterns = [
            r'"recommendation"\s*:\s*"([^"]{20,300})"',
            r'recommend[^"]*"([^"]{20,300})"',
            r'"suggest[^"]*"([^"]{20,300})"'
        ]
        
        for pattern in rec_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                rec_text = match.group(1).strip()
                # Find the last complete sentence to avoid cutting off mid-sentence
                sentences = rec_text.split('. ')
                if len(sentences) > 1 and len(rec_text) > 400:
                    # Take all complete sentences that fit within 400 chars
                    complete_rec = ''
                    for sentence in sentences[:-1]:  # Exclude the last incomplete sentence
                        if len(complete_rec + sentence + '. ') <= 400:
                            complete_rec += sentence + '. '
                        else:
                            break
                    result['key_gaps'] = [complete_rec.strip()]
                else:
                    result['key_gaps'] = [rec_text[:400]]  # Store as gaps for now
                
                # Determine recommendation level
                rec_lower = rec_text.lower()
                if any(word in rec_lower for word in ['highly recommend', 'excellent', 'outstanding', 'top candidate']):
                    result['recommendation'] = 'Highly Recommended'
                elif any(word in rec_lower for word in ['recommend', 'good', 'suitable', 'qualified']):
                    result['recommendation'] = 'Recommended'
                elif any(word in rec_lower for word in ['consider', 'potential', 'maybe']):
                    result['recommendation'] = 'Consider'
                break
        
        # Create a clean summary without duplicating analysis content
        score = result['overall_score']
        result['summary'] = f"CV analysis completed with {score:.2f}/100 score"
        
        return result
    
    def parse_json_response(self, response: str, cv_filename: str, max_retries: int = 3) -> Dict:
        """
        Robust JSON parsing with multiple fallback strategies optimized for complete responses
        """
        if not response.strip():
            logger.warning(f"Empty response for {cv_filename}")
            return self.default_response.copy()
        
        # Log response length for debugging
        logger.debug(f"Parsing response for {cv_filename}: {len(response)} characters")
        
        # Strategy 1: Try direct JSON parsing
        try:
            # Look for JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx + 1]
                result = json.loads(json_str)
                logger.debug(f"Direct JSON parsing successful for {cv_filename}")
                return self.validate_and_standardize_result(result, cv_filename)
                
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parsing failed for {cv_filename}: {e}")
        
        # Strategy 2: Try JSON repair with multiple attempts
        for attempt in range(max_retries):
            try:
                repaired_json = self.attempt_json_repair(response)
                if repaired_json:
                    result = json.loads(repaired_json)
                    logger.info(f"Successfully repaired JSON for {cv_filename} on attempt {attempt + 1}")
                    return self.validate_and_standardize_result(result, cv_filename)
                    
            except json.JSONDecodeError as e:
                logger.debug(f"JSON repair attempt {attempt + 1} failed for {cv_filename}: {e}")
                continue
        
        # Strategy 3: Enhanced regex extraction
        logger.warning(f"All JSON parsing failed for {cv_filename}, using enhanced regex extraction")
        result = self.extract_values_with_regex(response)
        
        # Store partial raw response for manual review if needed
        result['raw_response_preview'] = response[:500] + "..." if len(response) > 500 else response
        
        return result
    
    def validate_and_standardize_result(self, result: Dict, cv_filename: str) -> Dict:
        """Validate and standardize the parsed result with better content preservation"""
        standardized = self.default_response.copy()
        
        try:
            # Validate and extract score
            score = result.get('score', result.get('overall_score', 50.0))
            if isinstance(score, str):
                score_match = re.search(r'(\d+(?:\.\d+)?)', score)
                score = float(score_match.group(1)) if score_match else 50.0
            else:
                score = float(score)
            
            score = min(max(score, 0.0), 100.0)
            standardized['overall_score'] = round(score, 2)
            standardized['match_percentage'] = round(score, 2)
            
            # Extract and preserve longer content
            matching_analysis = result.get('matching_analysis', '')
            description = result.get('description', '')
            recommendation = result.get('recommendation', '')
            
            # Store longer content in appropriate fields
            if matching_analysis:
                analysis_text = str(matching_analysis)
                # Find the last complete sentence to avoid cutting off mid-sentence
                sentences = analysis_text.split('. ')
                if len(sentences) > 1 and len(analysis_text) > 700:
                    # Take all complete sentences that fit within 700 chars
                    complete_analysis = ''
                    for sentence in sentences[:-1]:  # Exclude the last incomplete sentence
                        if len(complete_analysis + sentence + '. ') <= 700:
                            complete_analysis += sentence + '. '
                        else:
                            break
                    standardized['key_strengths'] = [complete_analysis.strip()]
                else:
                    standardized['key_strengths'] = [analysis_text[:700]]
                
            if recommendation:
                rec_text = str(recommendation)
                # Find the last complete sentence to avoid cutting off mid-sentence
                sentences = rec_text.split('. ')
                if len(sentences) > 1 and len(rec_text) > 500:
                    # Take all complete sentences that fit within 500 chars
                    complete_rec = ''
                    for sentence in sentences[:-1]:  # Exclude the last incomplete sentence
                        if len(complete_rec + sentence + '. ') <= 500:
                            complete_rec += sentence + '. '
                        else:
                            break
                    standardized['key_gaps'] = [complete_rec.strip()]
                else:
                    standardized['key_gaps'] = [rec_text[:500]]
            
            standardized['recommendation'] = self.get_recommendation_from_score(score)
            
            # Create a clean, concise summary without duplicating analysis content
            if description:
                # Use the description if available, but limit it to avoid redundancy
                summary = str(description)[:250]
                # Ensure it ends at a complete sentence
                last_period = summary.rfind('.')
                if last_period > 200:
                    summary = summary[:last_period + 1]
                else:
                    summary = summary[:250]
            else:
                # Create a simple summary without duplicating analysis
                summary = f'CV analysis completed with {score:.2f}/100 score'
            
            standardized['summary'] = summary
            
            # Store original model outputs with full content
            standardized['llama_analysis'] = str(matching_analysis)[:1000] if matching_analysis else ''
            standardized['llama_recommendation'] = str(recommendation)[:600] if recommendation else ''
            
            logger.info(f"Successfully validated result for {cv_filename} - Score: {score:.2f}")
            return standardized
            
        except Exception as e:
            logger.error(f"Error validating result for {cv_filename}: {e}")
            return self.default_response.copy()
    
    def get_recommendation_from_score(self, score: int) -> str:
        """Get recommendation based on score"""
        if score >= 85:
            return "Highly Recommended"
        elif score >= 70:
            return "Recommended"
        elif score >= 50:
            return "Consider"
        else:
            return "Not Recommended"