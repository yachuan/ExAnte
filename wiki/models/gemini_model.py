import google.generativeai as genai
from config.config import Config

class GeminiModel:
    @staticmethod
    def get_response(prompt, temperature=0.7, max_tokens=500):
        """Get response from Gemini model"""
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-1.5-pro-002')
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return response.text
        except Exception as e:
            print(f"Error with Gemini: {str(e)}")
            return None 