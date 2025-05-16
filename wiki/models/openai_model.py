import openai
from config.config import Config

class OpenAIModel:
    @staticmethod
    def get_response(prompt, temperature=0.7, max_tokens=500):
        """Get response from OpenAI model"""
        try:
            openai.api_key = Config.OPENAI_API_KEY
            response = openai.ChatCompletion.create(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            print(f"Error with OpenAI: {str(e)}")
            return None
    
    @staticmethod
    def get_validation_response(prompt, temperature=0, max_tokens=200):
        """Get validation response from OpenAI model"""
        try:
            openai.api_key = Config.OPENAI_API_KEY
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            print(f"Error with OpenAI validation: {str(e)}")
            return None 