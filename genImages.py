import os
from groq import Groq
from text_to_image import FluxImageGenerator
import textwrap
import logging
import time
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptImageGenerator:
    def __init__(self):
        """Initialize with Groq API and Flux generator"""
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.image_generator = FluxImageGenerator()
        self.words_per_chunk = 25  # Adjust based on your needs

    def _read_audio_script(self, script_path: str) -> str:
        """Read the audio script file"""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading audio script: {e}")
            raise

    def _chunk_script(self, script: str) -> list:
        """Split script into chunks for image generation"""
        words = script.split()
        chunks = []
        
        for i in range(0, len(words), self.words_per_chunk):
            chunk = ' '.join(words[i:i + self.words_per_chunk])
            if chunk:
                chunks.append(chunk)
        
        return chunks

    def generate_prompts(self, audio_script: str) -> list:
        """Generate image prompts for script chunks"""
        try:
            system_prompt = """
            You are an expert at creating image generation prompts that convert text into engaging 2D/3D animated visuals.
            For the given text, create a detailed image prompt that:
            1. Captures the key message in a visual way
            2. Uses 2D/3D cartoon/animated style (NO realistic images)
            3. Includes specific style guidance
            4. Creates visually appealing scenes
            5. Matches the text content perfectly
            6. Uses vibrant colors and engaging compositions
            7. DO NOT include text in the images
            
            Format each prompt as:
            [Scene description with characters, actions, and details] || Style: [2D/3D animation style, colors, lighting, quality specifications]
            
            NO additional text, headers, or formatting - ONLY the prompt text.
            
            example prompts =
            Image 1:
            [Scene description: A person, wearing casual clothes, sits on the floor in frustration, surrounded by unpaid bill receipts and a canceled check. A thought bubble above their head shows the Supreme Court building.] || Style: 2D animated Indian characters in comic style, soft colors, and subtle shading.

            ==================================================
            Image 2:
            [Scene description: A person, dressed professionally, stands in a courtroom, arguing with the judge.] || Style: Bright colors, detailed textures, and dynamic composition, with a sense of urgency and determination.

            ==================================================
            Image 3:
            
            """

            chunks = self._chunk_script(audio_script)
            prompts = []
            
            for chunk in chunks:
                completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": f"Create ONE image prompt for this text chunk (remember: cartoon/animated style only), NO other text than the image prompt: {chunk}"
                        }
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.7,
                    max_tokens=300,
                    top_p=1,
                    stream=False
                )
                
                prompt = completion.choices[0].message.content.strip()
                prompts.append(prompt)
                
            return prompts
            
        except Exception as e:
            logger.error(f"Error generating prompts: {e}")
            raise

    def _read_prompts(self, prompts_file: str) -> list:
        """Read and parse image prompts from file"""
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse scenes from the content
            scenes = []
            current_scene = ""
            for line in content.split('\n'):
                if '[SCENE START]' in line or '[SCENE END]' in line:
                    if current_scene:
                        scenes.append(current_scene.strip())
                    current_scene = ""
                elif '||' in line:  # Only add lines containing actual prompts
                    current_scene = line.strip()
            
            # Add the last scene if exists
            if current_scene:
                scenes.append(current_scene.strip())

            return scenes

        except Exception as e:
            logger.error(f"Error reading prompts file: {e}")
            raise

    def generate_images(self, unique_id: str) -> list:
        """Generate images from prompts"""
        try:
            # Construct file paths
            prompts_file = f"img_prompts/imgPrompts_{unique_id}.txt"
            output_folder = f"images/img_{unique_id}"
            os.makedirs(output_folder, exist_ok=True)

            # Read prompts
            prompts = self._read_prompts(prompts_file)
            if not prompts:
                raise ValueError("No valid prompts found in file")

            generated_images = []
            
            # with open(prompts_file, 'r', encoding='utf-8') as f:
            #     prompts = f.read()
            
            for i, prompt in enumerate(prompts, 1):
                output_path = os.path.join(output_folder, f'image_{i}.png')
                logger.info(f"Generating image {i}/{len(prompts)}")
                
                try:
                    success = self.image_generator.generate_image(
                        prompt=prompt,
                        output_path=output_path
                    )
                    
                    if success:
                        generated_images.append(output_path)
                        logger.info(f"Successfully generated image {i}")
                    else:
                        logger.error(f"Failed to generate image {i}")
                    
                    # Add delay between generations to prevent rate limiting
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error generating image {i}: {e}")
                    continue

            return output_folder if generated_images else None

        except Exception as e:
            logger.error(f"Error in generate_images: {e}")
            raise
        
    # def generate_images(self, prompts: list, output_folder: str) -> list:
    #     """Generate images from prompts"""
    #     generated_images = []
            
    #     for i, prompt in enumerate(prompts, 1):
    #         output_path = os.path.join(output_folder, f"image_{i}.png")
                
    #         try:
    #             logger.info(f"Generating image {i}/{len(prompts)}")
    #             success = self.image_generator.generate_image(prompt, output_path)
                    
    #             if success:
    #                 generated_images.append(output_path)
    #                 logger.info(f"Successfully generated image {i}")
    #             else:
    #                 logger.error(f"Failed to generate image {i}")
                    
    #             # Delay to prevent rate limiting
    #             time.sleep(2)
                    
    #         except Exception as e:
    #             logger.error(f"Error generating image {i}: {e}")
    #             continue
                    
    #     return generated_images

def main():
    try:
        # Get script ID
        script_id = input("Enter the script ID: ")
        
        # Create necessary folders
        os.makedirs('img_prompts', exist_ok=True)
        os.makedirs(os.path.join('images', 'final_images'), exist_ok=True)
        
        # Initialize generator
        generator = PromptImageGenerator()
        
        # Construct paths
        audio_script_path = f"audio_scripts/audioScript_{script_id}.txt"
        prompts_path = f"img_prompts/final_prompts_{script_id}.txt"
        output_folder = f"images/final_images/final_images_{script_id}"
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Read audio script
        logger.info("Reading audio script...")
        audio_script = generator._read_audio_script(audio_script_path)
        
        # Generate prompts
        logger.info("Generating image prompts...")
        prompts = generator.generate_prompts(audio_script)
        
        # Save prompts
        with open(prompts_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(prompts))
        
        logger.info(f"Prompts saved to {prompts_path}")
        
        # Generate images
        logger.info("Generating images...")
        generated_images = generator.generate_images(prompts, output_folder)
        
        # Print results
        print("\nProcess complete!")
        print(f"Generated prompts saved to: {prompts_path}")
        print(f"Number of prompts generated: {len(prompts)}")
        print(f"Images saved in: {output_folder}")
        print(f"Number of images generated: {len(generated_images)}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
