import os
from groq import Groq
import logging
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePromptGenerator:
    def __init__(self):
        """Initialize with Groq API key"""
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
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
            You are an expert visual director specializing in Instagram Reels. Your task is to create EXACTLY 4 pairs of image prompts that will synchronize perfectly with a 40-second voiceover script. Each image pair represents a 10-second segment with smooth transitions.

            TIMING AND STRUCTURE
            • Each pair covers ~10 seconds of audio content
            • First image in pair: Initial scene (0-5 seconds)
            • Second image in pair: Transformed scene (5-10 seconds)
            • Total: 8 images (4 pairs) covering 40 seconds

            PROMPT FORMAT REQUIREMENTS
            Initial Scene: [SCENE START]
            [Detailed scene description] || Style: [Comprehensive style specifications]

            Transformed Scene: [SCENE END]
            [Evolution of the scene] || Style: [Matching style specifications]

            SCENE DESCRIPTION MUST INCLUDE:
            1. Characters & Expression
            • Precise character positioning
            • Detailed facial expressions
            • Body language and gestures
            • Clothing and accessories

            2. Environment
            • Specific location details
            • Background elements
            • Props and relevant objects
            • Scene depth and scale

            3. Camera Work
            • Exact camera angle (eye-level, low-angle, etc.)
            • Shot type (close-up, medium, wide)
            • Frame composition
            • Focal point specification

            4. Atmosphere
            • Time of day
            • Lighting direction and intensity
            • Mood indicators
            • Environmental effects

            STYLE SPECIFICATIONS MUST DETAIL:
            1. Animation Style
            • 3D animation technique
            • Rendering style (realistic, stylized)
            • Surface textures
            • Movement fluidity

            2. Visual Elements
            • Color palette (specific colors)
            • Lighting setup (key, fill, rim lights)
            • Shadow characteristics
            • Depth of field

            3. Technical Specs
            • Resolution (4K/8K)
            • Aspect ratio (9:16 vertical)
            • Render quality settings
            • Post-processing effects

            TRANSITION GUIDELINES
            • Each pair must show logical progression
            • Maintain consistent style within pairs
            • Ensure smooth visual flow between scenes
            • Keep key elements in similar positions

            CRITICAL RULES
            • NO text or typography in images
            • NO abrupt scene changes between pairs
            • NO realistic human faces (use stylized 3D)
            • MAINTAIN consistent art style across all 8 images
            • ENSURE each image can hold viewer attention for 5 seconds

            EXAMPLE PAIR FORMAT:
            [SCENE START]
            A young entrepreneur stands confidently in a modern office, hand raised with holographic business data floating around them. Natural light streams through floor-to-ceiling windows, creating dynamic shadows. Camera positioned slightly low-angle to emphasize authority. || Style: Modern 3D animation, vibrant blue-orange color scheme, volumetric lighting, cinematic DOF, 4K resolution, ray-traced reflections

            [SCENE END]
            Same entrepreneur now actively manipulating the holographic data, multiple screens expanding outward, expression showing excitement. Light rays intensify, creating lens flares through the data. Camera smoothly orbited 15 degrees right. || Style: Modern 3D animation, vibrant blue-orange color scheme, volumetric lighting, cinematic DOF, 4K resolution, ray-traced reflections

            CRUCIAL TIMING NOTES
            • First pair (0-10 seconds): Hook and initial concept
            • Second pair (10-20 seconds): Main point development
            • Third pair (20-30 seconds): Supporting evidence/examples
            • Fourth pair (30-40 seconds): Conclusion and call-to-action

            Remember: These images must work together to create a cohesive visual story that perfectly matches the voiceover timing and message. Each transition should feel natural and enhance the spoken content.
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