import argparse
import os
import re
import csv
import pandas as pd
import time
import openai
from config.config import Config
from models.openai_model import OpenAIModel
from models.claude_model import ClaudeModel
from models.gemini_model import GeminiModel
from utils.file_utils import FileUtils
from utils.prompt_strategies import PromptStrategies
from utils.claim_validator import ClaimValidator
from utils.wiki_extractor import (
    get_revision_id, 
    get_wiki_content_by_revision, 
    save_content, 
    is_temporal_title_gpt
)

def process_wikipedia_pages(output_folder, temporal_file):
    """Extract Wikipedia content for temporal topics"""
    print("\n=== Starting Wikipedia Content Extraction ===\n")
    
    try:
        # Read CSV with proper quoting to handle commas in text fields
        df = pd.read_csv(temporal_file, quoting=csv.QUOTE_ALL, escapechar='\\')
        
        # Filter for topics with cutoff year after 2010
        df['Cutoff_Year'] = pd.to_numeric(df['Cutoff_Year'], errors='coerce')
        df = df[df['Cutoff_Year'] > 2010]
        
        # Create dictionary of titles and their cutoff years
        valid_topics = dict(zip(df['Title'], df['Cutoff_Year']))
        print(f"\nFound {len(valid_topics)} valid temporal topics after 2010 to process")
        
    except Exception as e:
        print(f"Error loading temporal classifications: {e}")
        return
    
    if not valid_topics:
        print("No valid topics found!")
        return
    
    current_year = 2024
    
    for title, cutoff_year in valid_topics.items():
        print(f"\nProcessing {title} (cutoff year: {cutoff_year})...")
        
        folder_path = os.path.join(output_folder, title.replace(' ', '_'), 'content')
        os.makedirs(folder_path, exist_ok=True)
        
        # Get historical content with retries
        max_retries = 5
        for attempt in range(max_retries):
            historical_revision_id = get_revision_id(title, cutoff_year)
            if historical_revision_id:
                historical_content = get_wiki_content_by_revision(title, historical_revision_id)
                if historical_content:
                    save_content(historical_content, folder_path, title.replace(' ', '_'), cutoff_year)
                    print(f"✓ Saved historical content ({cutoff_year})")
                    break
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1} for historical content...")
                time.sleep(3)
        
        # Get current content with retries
        for attempt in range(max_retries):
            current_content = get_wiki_content_by_revision(title)
            if current_content:
                save_content(current_content, folder_path, title.replace(' ', '_'), current_year)
                print(f"✓ Saved current content ({current_year})")
                break
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1} for current content...")
                time.sleep(2)
    
    print("\n=== Wikipedia Content Extraction Complete ===\n")

class WikiEvaluator:
    def __init__(self, api_key, base_dir, output_dir, start_idx=0, count=None, target_model=None, target_method=None):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.start_idx = start_idx
        self.count = count
        self.target_model = target_model
        self.target_method = target_method
        self.prompt_strategies = {
            'zero_shot': PromptStrategies.zero_shot,
            'instruction_based': PromptStrategies.instruction_based,
            'chain_of_thought': PromptStrategies.chain_of_thought,
            'few_shot': PromptStrategies.few_shot,
            'generate_and_validate': 'generate_and_validate',
            'self_check': 'self_check'
        }

    def get_model_response(self, prompt, model_name="gpt4", temperature=0.7, max_tokens=500):
        """Helper function to get responses from different models"""
        if model_name == "gpt4":
            return OpenAIModel.get_response(prompt, temperature, max_tokens)
        elif model_name == "claude":
            return ClaudeModel.get_response(prompt, temperature, max_tokens)
        elif model_name == "gemini":
            return GeminiModel.get_response(prompt, temperature, max_tokens)
        return None

    def generate_claims(self, title, strategy, logger, model="gpt4"):
        prompt_func = self.prompt_strategies[strategy]
        try:
            content_dir = os.path.join(self.base_dir, title, 'content')
            cutoff_year = FileUtils.get_cutoff_year(content_dir)
            
            if strategy == 'generate_and_validate':
                # First generate claims
                generate_prompt = f"""Generate 5 atomic facts about '{title}'. 
Each fact should be a short, clear statement representing a distinct piece of information.
Format each fact as a numbered list (1-5)."""

                generated_content = self.get_model_response(
                    generate_prompt, 
                    model_name=model, 
                    temperature=0.7
                )
                
                if not generated_content:
                    return [], f"Failed to generate content with {model}"
                    
                generated_claims = generated_content.split('\n')
                generated_claims = [claim for claim in generated_claims if claim and re.match(r'^\d+\.\s', claim)]
                
                # Then validate each claim
                validated_claims = []
                for claim in generated_claims:
                    validate_prompt = f"""You are a knowledgeable person living just before {cutoff_year}.
Evaluate if this fact was known before {cutoff_year}:

Fact: {claim}

Answer in this format:
[Known Before {cutoff_year}]: yes/no
[Explanation]: Brief explanation of your decision
[Confidence]: high/medium/low"""

                    validation = self.get_model_response(
                        validate_prompt, 
                        model_name=model, 
                        temperature=0
                    )
                    
                    if not validation:
                        continue
                        
                    if 'yes' in validation.lower() and 'high' in validation.lower():
                        validated_claims.append(claim)
                    else:
                        # Generate a replacement fact
                        replace_prompt = f"""The following fact was determined to contain post-{cutoff_year} information:
{claim}

Please generate a replacement fact about '{title}' that was definitely known before {cutoff_year}."""
                        
                        new_claim = self.get_model_response(
                            replace_prompt, 
                            model_name=model, 
                            temperature=0.7
                        )
                        
                        if new_claim:
                            validated_claims.append(new_claim)
                
                return validated_claims, f"Generated and validated claims using {model}"
            
            elif strategy == 'self_check':
                # Basic prompt
                prompt = f"""Generate 5 atomic facts about '{title}'. 
Each fact should be a short, clear statement representing a distinct piece of information.
Only use information from before January 1st, {cutoff_year}.
Format each fact as a numbered list (1-5)."""
                
                # First response
                content = self.get_model_response(prompt, model_name=model)
                if not content:
                    return [], f"Failed to generate content with {model}"
                    
                # First self-check
                check_prompt = f"""Review your previous response about {title}. 
Consider if any of your stated facts rely on information from after {cutoff_year}.
If you find any temporal inconsistencies, please revise those facts. 
Format your response as a numbered list (1-5) of atomic facts.
If no issues found, respond with 'No temporal leakage'."""

                first_check = self.get_model_response(check_prompt, model_name=model, temperature=0)
                
                # Second self-check with specific focus on factual accuracy
                verification_prompt = f"""Carefully verify each fact you provided about {title}:
1. Are you completely certain each fact was known before {cutoff_year}?
2. Can you cite specific pre-{cutoff_year} sources for each fact?
3. Are any facts potentially mixing knowledge from different time periods?

If you find any issues, provide corrected facts as a numbered list (1-5).
Otherwise, respond with 'Facts verified'."""

                second_check = self.get_model_response(verification_prompt, model_name=model, temperature=0)
                
                # Third self-check for final formatting
                final_format_prompt = f"""Review the facts about {title}. Ensure each fact:
1. Is a single, atomic piece of information
2. Is clearly stated
3. Was known before {cutoff_year}

Format your response as exactly 5 numbered facts (1-5).
If the current facts are good, just reformat them. If you find any issues, provide corrections."""

                final_check = self.get_model_response(final_format_prompt, model_name=model, temperature=0)
                
                # Process final claims
                if 'No temporal leakage' in first_check and 'Facts verified' in second_check:
                    claims = content.split('\n')
                else:
                    # Use the last correction if any issues were found
                    corrected_content = final_check
                    claims = corrected_content.split('\n')
                
                # Ensure consistent formatting
                claims = [claim.strip() for claim in claims if claim.strip() and re.match(r'^\d+\.\s', claim)]
                # Ensure exactly 5 claims
                while len(claims) < 5:
                    additional_prompt = f"Generate one more atomic fact about {title} known before {cutoff_year}."
                    additional_fact = self.get_model_response(additional_prompt, model_name=model)
                    if additional_fact:
                        claims.append(f"{len(claims)+1}. {additional_fact}")
                claims = claims[:5]  # Limit to 5 claims if more were generated
                
                return claims, content
            
            else:
                # Handle other strategies
                prompt = prompt_func(title, cutoff_year)
                logger.info(f"Generated prompt for {strategy} using {model}: {prompt}")
                
                # First response
                content = self.get_model_response(prompt, model_name=model)
                if not content:
                    return [], f"Failed to generate content with {model}"
                    
                # First self-check
                check_prompt = f"""Review your previous response about {title}. 
Consider if any of your stated facts rely on information from after {cutoff_year}.
If you find any temporal inconsistencies, please revise those facts. 
Format your response as a numbered list (1-5) of atomic facts.
If no issues found, respond with 'No temporal leakage'."""

                first_check = self.get_model_response(check_prompt, model_name=model, temperature=0)
                
                # Second self-check with specific focus on factual accuracy
                verification_prompt = f"""Carefully verify each fact you provided about {title}:
1. Are you completely certain each fact was known before {cutoff_year}?
2. Can you cite specific pre-{cutoff_year} sources for each fact?
3. Are any facts potentially mixing knowledge from different time periods?

If you find any issues, provide corrected facts as a numbered list (1-5).
Otherwise, respond with 'Facts verified'."""

                second_check = self.get_model_response(verification_prompt, model_name=model, temperature=0)
                
                # Third self-check for final formatting
                final_format_prompt = f"""Review the facts about {title}. Ensure each fact:
1. Is a single, atomic piece of information
2. Is clearly stated
3. Was known before {cutoff_year}

Format your response as exactly 5 numbered facts (1-5).
If the current facts are good, just reformat them. If you find any issues, provide corrections."""

                final_check = self.get_model_response(final_format_prompt, model_name=model, temperature=0)
                
                # Process final claims
                if 'No temporal leakage' in first_check and 'Facts verified' in second_check:
                    claims = content.split('\n')
                else:
                    # Use the last correction if any issues were found
                    corrected_content = final_check
                    claims = corrected_content.split('\n')
                
                # Ensure consistent formatting
                claims = [claim.strip() for claim in claims if claim.strip() and re.match(r'^\d+\.\s', claim)]
                # Ensure exactly 5 claims
                while len(claims) < 5:
                    additional_prompt = f"Generate one more atomic fact about {title} known before {cutoff_year}."
                    additional_fact = self.get_model_response(additional_prompt, model_name=model)
                    if additional_fact:
                        claims.append(f"{len(claims)+1}. {additional_fact}")
                claims = claims[:5]  # Limit to 5 claims if more were generated
                
                return claims, content
            
        except Exception as e:
            logger.error(f"Error generating claims with {model}: {str(e)}")
            return [], str(e)

    def process_title(self, content_dir):
        try:
            title = os.path.basename(os.path.dirname(content_dir))
            logger = FileUtils.setup_logging(self.output_dir, title)
            
            # Use only specified model or all models
            models = [self.target_model] if self.target_model and self.target_model != 'all' else ["gpt4", "claude", "gemini"]
            
            for model in models:
                model_folder = os.path.join(self.output_dir, model, title)
                os.makedirs(model_folder, exist_ok=True)
                
                cutoff_year = FileUtils.get_cutoff_year(content_dir)
                cutoff_content_path = FileUtils.get_content_path(content_dir, title, cutoff_year)
                current_content_path = FileUtils.get_content_path(content_dir, title, '2024')
                
                cutoff_content = FileUtils.read_content(cutoff_content_path, logger)
                current_content = FileUtils.read_content(current_content_path, logger)
                
                if not cutoff_content or not current_content:
                    logger.error(f"Missing content for {title}")
                    continue
                
                # Use only specified method or all methods
                methods = [self.target_method] if self.target_method and self.target_method != 'all' else self.prompt_strategies.keys()
                for strategy in methods:
                    csv_path = FileUtils.create_csv_file(model_folder, title, strategy)
                    claims, _ = self.generate_claims(title, strategy, logger, model=model)
                    
                    for claim in claims:
                        validation_before = ClaimValidator.validate_claim(claim, cutoff_content, logger)
                        validation_after = ClaimValidator.validate_claim(claim, current_content, logger)
                        
                        # Initial leakage check
                        potential_leakage = (validation_before['evaluation'] == 'no' and 
                                           validation_after['evaluation'] == 'yes')
                        
                        # Additional derivation check for potential leakage cases
                        initial_leakage = 'Yes' if potential_leakage else 'No'
                        derivation_response = ''
                        derivable = 'yes'
                        
                        if potential_leakage:
                            derivation_check = ClaimValidator.validate_reference_derivation(
                                claim,
                                cutoff_content,
                                validation_after['reference'],
                                logger
                            )
                            derivable = derivation_check['derivable']
                            derivation_response = derivation_check['full_response']
                        
                        # Final leakage determination
                        final_leakage = 'Yes' if (initial_leakage == 'Yes' and derivable == 'no') else 'No'
                        
                        row_data = [
                            claim,
                            validation_before['evaluation'],
                            validation_after['evaluation'],
                            initial_leakage,
                            validation_before['reference'],
                            validation_after['reference'],
                            validation_before['full_response'],
                            validation_after['full_response'],
                            derivation_response,
                            final_leakage
                        ]
                        
                        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(row_data)
                        
                        logger.info(f"Processed claim with evaluations - Before: {validation_before['evaluation']}, After: {validation_after['evaluation']}")
                    
                    logger.info(f"Completed processing {title} with {strategy} strategy")
            
        except Exception as e:
            print(f"Error processing title directory {content_dir}: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Process Wikipedia content files with different prompting strategies')
    parser.add_argument('--start', type=int, default=0, help='Starting index for processing')
    parser.add_argument('--count', type=int, default=-1, 
                       help='Number of files to process. Use -1 to process all files (default: -1)')
    parser.add_argument('--base_dir', type=str, default=Config.DEFAULT_BASE_DIR,
                      help='Base directory containing content files')
    parser.add_argument('--output_dir', type=str, default=Config.DEFAULT_OUTPUT_DIR,
                      help='Output directory for results')
    parser.add_argument('--model', type=str, choices=['gpt4', 'claude', 'gemini', 'all'],
                      default='all',
                      help='Model to use (default: all)')
    parser.add_argument('--method', type=str, 
                      choices=['zero_shot', 'instruction_based', 'chain_of_thought', 'few_shot', 'generate_and_validate', 'self_check', 'all'],
                      default='all',
                      help='Method to use (default: all)')
    parser.add_argument('--extract', action='store_true',
                      help='Extract Wikipedia content before evaluation')
    parser.add_argument('--topics_file', type=str, default=Config.DEFAULT_TOPICS_FILE,
                      help='CSV file containing valid topics and cutoff years')
    args = parser.parse_args()

    # Set up OpenAI API key
    openai.api_key = Config.OPENAI_API_KEY

    # Extract Wikipedia content if requested
    if args.extract:
        process_wikipedia_pages(args.base_dir, args.topics_file)

    # Create output directory for evaluation results
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which models to run
    models = ["gpt4", "claude", "gemini"] if args.model == 'all' else [args.model]

    # Create summary CSV only for selected models
    for model in models:
        model_summary_file = os.path.join(args.output_dir, f'{model}_summary.csv')
        with open(model_summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Title',
                'Strategy',
                'Total Claims',
                'Leaked Claims',
                'Leakage Rate (%)',
                'Cutoff Year'
            ])

    api_key = Config.OPENAI_API_KEY
    
    evaluator = WikiEvaluator(
        api_key=api_key,
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        start_idx=args.start,
        count=args.count,
        target_model=args.model,
        target_method=args.method
    )
    
    try:
        # List all title directories
        title_dirs = []
        for title in os.listdir(args.base_dir):
            content_dir = os.path.join(args.base_dir, title, 'content')
            if os.path.isdir(content_dir):
                title_dirs.append(content_dir)
        
        # Sort for consistent processing
        title_dirs.sort()
        
        # Apply start and count parameters
        if args.count != -1:
            title_dirs = title_dirs[args.start:args.start + args.count]
        else:
            title_dirs = title_dirs[args.start:]  # Process all remaining files
        
        print(f"Found {len(title_dirs)} titles to process")
        for title_dir in title_dirs:
            print(f"\nProcessing content directory: {title_dir}")
            evaluator.process_title(title_dir)
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise 

if __name__ == "__main__":
    main() 