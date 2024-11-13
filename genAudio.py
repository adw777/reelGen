import requests
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioGenerator:
    def __init__(self):
        self.VOICE_ID = "IKne3meq5aSn9XLyUdCD" # voice id from elevenlabs
        self.API_KEY = "sk_73c5fc4ce4e8916eb713dc96177a2377b4412f80de951eff"
        self.CHUNK_SIZE = 1024

    def text_to_speech(self, text, output_path):
        """Convert text to speech using ElevenLabs API"""
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.VOICE_ID}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.API_KEY
        }

        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.4,
                "similarity_boost": 0.5
            }
        }

        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            
            if 'audio' not in response.headers.get('Content-Type', ''):
                logger.error(f"Error Response: {response.text}")
                return False
                
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        
            logger.info(f"Audio saved successfully to {output_path}")
            return True
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e}")
            logger.error(f"Response content: {response.text}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error occurred: {e}")
            return False

    def process_script(self, script_id):
        """Process audio script and generate audio file"""
        try:
            # Create audio directory if it doesn't exist
            if not os.path.exists('audio'):
                os.makedirs('audio')

            # Construct file paths
            script_path = f"audio_scripts/audioScript_{script_id}.txt"
            output_path = f"audio/audio_{script_id}.mp3"

            # Check if script exists
            if not os.path.exists(script_path):
                logger.error(f"Script file not found: {script_path}")
                return False

            # Read the script
            with open(script_path, 'r', encoding='utf-8') as f:
                script_text = f.read().strip()

            # Generate audio
            logger.info(f"Generating audio for script ID: {script_id}")
            return self.text_to_speech(script_text, output_path)

        except Exception as e:
            logger.error(f"Error processing script: {e}")
            return False

def main():
    try:
        # Get script ID from user
        script_id = input("Enter the script ID: ")
        
        # Initialize generator and process script
        generator = AudioGenerator()
        success = generator.process_script(script_id)
        
        if success:
            print(f"\nAudio generation complete!")
            print(f"Audio saved to: audio/audio_{script_id}.mp3")
        else:
            print("\nFailed to generate audio")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()