import os
from groq import Groq
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePromptGenerator:
    def __init__(self):
        """Initialize with Groq API key"""
        self.client = Groq(api_key="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")
        
    def read_audio_script(self, script_id: str) -> Optional[str]:
        """Read the audio script file"""
        try:
            script_path = f"audio_scripts/audioScript_{script_id}.txt"
            if not os.path.exists(script_path):
                logger.error(f"Audio script not found: {script_path}")
                return None
                
            with open(script_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading audio script: {e}")
            return None
            
    def clean_prompt(self, content: str) -> List[str]:
        """Clean and extract valid prompts from generated content"""
        try:
            # Split content into lines
            lines = content.split('\n')
            
            # Extract only valid prompts (lines containing '|| Style:')
            cleaned_prompts = []
            for line in lines:
                line = line.strip()
                if '|| Style:' in line:
                    cleaned_prompts.append(line)
            
            return cleaned_prompts
            
        except Exception as e:
            logger.error(f"Error cleaning prompts: {e}")
            return []

    def generate_prompts(self, audio_script: str) -> List[str]:
        """Generate and clean image prompts from audio script"""
        try:
            system_prompt = """
            You are an expert at creating visual storytelling prompts for Instagram Reels. Generate EXACTLY 4 pairs of image prompts (start and end states) that will create smooth transitions in the final video.

            Requirements:
            1. Each prompt MUST use format: [Scene description] || Style: [style specifications]
            2. Scene descriptions should be highly detailed and match audio content perfectly
            3. Each pair should show natural progression/transition
            4. Use 3D animation style for consistent look
            5. Include specific details about:
               - Character expressions and poses
               - Environmental details
               - Lighting and atmosphere
               - Camera angles and composition
            6. Style should specify:
               - Animation style (3D, modern)
               - Color palette
               - Lighting setup
               - Quality specifications (4k, cinematic)
            
            Format each prompt pair as:
            [Scene description] || Style: [style specifications]

            [Scene description] || Style: [style specifications]

            Generate all 8 prompts (4 pairs) in this exact format with each prompt on its own line.
            NO additional text, headers, or formatting - ONLY the prompts.
            """

            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Create 4 pairs of image prompts (8 total) for this audio script:\n\n{audio_script}"
                    }
                ],
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=2000,
                top_p=1,
                stream=False
            )
            
            # Get and clean the generated content
            content = completion.choices[0].message.content.strip()
            cleaned_prompts = self.clean_prompt(content)
            
            # Verify we have exactly 8 prompts
            if len(cleaned_prompts) != 8:
                logger.warning(f"Expected 8 prompts, got {len(cleaned_prompts)}. Regenerating...")
                return self.generate_prompts(audio_script)  # Retry generation
            
            return cleaned_prompts
            
        except Exception as e:
            logger.error(f"Error generating prompts: {e}")
            return []

    def save_prompts(self, prompts: List[str], script_id: str) -> bool:
        """Save generated prompts to file"""
        try:
            # Create img_prompts directory if it doesn't exist
            os.makedirs('img_prompts', exist_ok=True)
            
            output_path = f"img_prompts/imgPrompts_{script_id}.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(prompts))
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving prompts: {e}")
            return False

    def process_script(self, script_id: str) -> bool:
        """Process audio script and generate clean image prompts"""
        try:
            # Read audio script
            audio_script = self.read_audio_script(script_id)
            if not audio_script:
                return False
                
            # Generate and clean prompts
            logger.info("Generating and cleaning image prompts...")
            prompts = self.generate_prompts(audio_script)
            if not prompts:
                return False
                
            # Verify prompt count
            if len(prompts) != 8:
                logger.error(f"Invalid number of prompts generated: {len(prompts)}")
                return False
                
            # Save clean prompts
            logger.info("Saving clean image prompts...")
            if not self.save_prompts(prompts, script_id):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error processing script: {e}")
            return False

def main():
    try:
        # Get script ID from user
        script_id = input("Enter the script ID: ")
        
        # Initialize generator and process script
        generator = ImagePromptGenerator()
        success = generator.process_script(script_id)
        
        if success:
            output_path = f"img_prompts/imgPrompts_{script_id}.txt"
            print("\nImage prompts generated successfully!")
            print(f"Clean prompts saved to: {output_path}")
            
            # # Display generated prompts
            # print("\nGenerated Image Prompts:")
            # print("-" * 50)
            # with open(output_path, 'r', encoding='utf-8') as f:
            #     prompts = f.read().split('\n\n')
            #     for i, prompt in enumerate(prompts, 1):
            #         print(f"\nPrompt {i}:")
            #         print(prompt)
            # print("-" * 50)
        else:
            print("\nFailed to generate image prompts")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()