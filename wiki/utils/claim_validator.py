import re
from models.openai_model import OpenAIModel

class ClaimValidator:
    @staticmethod
    def validate_claim(claim, current_content, logger):
        try:
            prompt = f"""Evaluate if this claim is supported by the Wikipedia document.

Wikipedia Document: {current_content}
Claim: {claim}

Answer in this format:
[Supported]: yes/no (yes = claim is supported by or derived from the wikipedia document, else put no.)
[References]: Quote the relevant text from the document that supports to this claim, if you cannot find any, put None.
[Explanation]: Brief explanation of your decision
"""

            logger.info(f"Validating claim: {claim}")
            
            result = OpenAIModel.get_validation_response(prompt)
            if not result:
                return {
                    'evaluation': 'error',
                    'explanation': 'Failed to get validation response',
                    'reference': 'None',
                    'full_response': 'Error'
                }
            
            ref_match = re.search(r'\[References\]: (.*?)(?=\[|$)', result, re.DOTALL)
            eval_match = re.search(r'\[Supported\]: (yes|no)', result, re.IGNORECASE)
            explanation_match = re.search(r'\[Explanation\]: (.*?)(?=\[|$)', result, re.DOTALL)
            
            return {
                'evaluation': eval_match.group(1).lower() if eval_match else 'no',
                'explanation': explanation_match.group(1).strip() if explanation_match else '',
                'reference': ref_match.group(1).strip() if ref_match else 'None',
                'full_response': result
            }
        except Exception as e:
            logger.error(f"Error validating claim: {str(e)}")
            return {
                'evaluation': 'error',
                'explanation': str(e),
                'reference': 'None',
                'full_response': str(e)
            }
    
    @staticmethod
    def validate_reference_derivation(claim, before_content, after_ref, logger):
        try:
            prompt = f"""Analyze if the following claim's reference from a newer source could be logically derived from the older reference.

Old content: {before_content}
New reference: {after_ref}

Please determine:
1. Is the information in the new reference completely novel, or could it be logically derived/inferred from the old reference?
2. If the new reference contains the claim but the old reference doesn't, is this truly new information or just a different way of stating what was known before?

Answer with:
[Derivable]: yes/no
[Explanation]: Your reasoning
"""
            result = OpenAIModel.get_validation_response(prompt)
            if not result:
                return {'derivable': 'error', 'full_response': 'Failed to get validation response'}
            
            derivable_match = re.search(r'\[Derivable\]: (yes|no)', result, re.IGNORECASE)
            
            return {
                'derivable': derivable_match.group(1).lower() if derivable_match else 'yes',
                'full_response': result
            }
        except Exception as e:
            logger.error(f"Error in reference derivation check: {str(e)}")
            return {'derivable': 'error', 'full_response': str(e)} 