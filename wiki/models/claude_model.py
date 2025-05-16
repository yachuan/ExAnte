from anthropic import Anthropic
from config.config import Config

class ClaudeModel:
    @staticmethod
    def get_response(prompt, temperature=0.7, max_tokens=500):
        """Get response from Claude model"""
        try:
            claude_client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
            response = claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error with Claude: {str(e)}")
            return None 