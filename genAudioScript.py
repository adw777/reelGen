import os
from groq import Groq
import logging
import json
from typing import List, Dict
from datetime import datetime
import time
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioScriptGenerator:
    def __init__(self):
        """Initialize with Groq API key"""
        self.client = Groq(api_key="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")
        self.words_per_chunk = 10

    def _read_file(self, file_path: str) -> str:
        """Read content from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
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

    def _generate_script_and_prompts(self, content: str) -> tuple:
        """Generate both audio script and image prompts together"""
        try:
            system_prompt = """
            You are an expert Instagram Reels script writer. Create a short, engaging voiceover script.

            Critical requirements:
            1. MAXIMUM 40 SECONDS when spoken - this is crucial!
            2. Each sentence must be short
            3. Use high-energy, Gen-Z creator voice
            4. Start with an attention-grabbing hook
            5. End with a powerful conclusion or call-to-action
            6. Output ONLY the spoken text - no markers/formatting/headers
            7. DO NOT USE: "Here's a 40 SECOND Instagram Reel script:" or similar just output the  script text
            8. Use natural pauses (line breaks) between thoughts
            9. Total word count: 70-90 words MAXIMUM
            10. Make every word count - no filler content

            example script= 

            "Get ready for a healthcare revolution!
            AI is transforming the medical world like never before!
            From detecting tumors in X-rays to AI-powered chatbots, it's a game-changer!
            Imagine having 24/7 access to medical support, personalized treatment plans, and even robotic surgery assistance!
            But here's the thing: AI isn't replacing doctors and nurses - it's empowering them to deliver better care!
            The future is now: AI is the ultimate medical superpower, amplifying human capabilities!
            Join the healthcare revolution and discover the incredible impact of AI!"

            """
            
            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Create an ultra-concise Instagram Reel script (40 SECONDS MAX) from this content:\n\n{content}"
                    }
                ],
                model="llama-3.1-8b-instant",
                temperature=0.8,
                max_tokens=500,
                top_p=1,
                stream=False
            )
            
            audio_script = completion.choices[0].message.content.strip()
            
            image_system_prompt = """
            You are an expert at creating visual storytelling prompts. Generate EXACTLY 4 pairs of images (start and end states) that will transition smoothly in a video.

            FOLLOW THIS EXACT FORMAT FOR OUTPUT:

            [SCENE START]
            A young Indian professional standing in a modern office with Mumbai skyline, gesturing confidently with one hand raised. Wearing traditional-modern fusion attire. Warm lighting from large windows. || Style: Modern 3D animation, vibrant colors, soft lighting, 4k quality, cinematic composition

            [SCENE END]
            The same professional now leaning forward with both hands gesturing expressively, more engaged expression, same office and lighting setting. || Style: Modern 3D animation, vibrant colors, soft lighting, 4k quality, cinematic composition

            Important:
            - Generate EXACTLY 4 pairs using the above format
            - Keep pairs coherent with matching styles
            - Use [SCENE START] and [SCENE END] markers
            - Include detailed scene descriptions
            - Focus on natural pose transitions
            """
            
            image_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": image_system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Create 4 pairs of image prompts to visualize this script:\n\n{audio_script}"
                    }
                ],
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=2000,
                top_p=1,
                stream=False
            )
            
            image_prompts = image_completion.choices[0].message.content.strip()
            
            return audio_script, image_prompts
        
        except Exception as e:
            logger.error(f"Error generating script and prompts: {e}")
            raise
    
    def generate_audio_script(self, script_path: str, output_path: str, img_prompts_path: str) -> bool:
        """Generate synchronized audio script and image prompts"""
        try:
            content = self._read_file(script_path)
            
            audio_script, image_prompts = self._generate_script_and_prompts(content)
            
            # image_prompts = self.generate_prompts(audio_script)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(audio_script)
            logger.info(f"Audio script saved to {output_path}")
            
            image_prompts = image_prompts.replace('\n\n', '\n').strip()
            
            with open(img_prompts_path, 'w', encoding='utf-8', newline='\n') as f:
                # f.write(f"\n\n{'='*50}\n\n")
                f.write(image_prompts)  # Write the entire string at once
            logger.info(f"Image prompts saved to {img_prompts_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating audio script: {e}")
            return False

def main():
    """Test the AudioScriptGenerator with sample content"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir = f"test_output_{timestamp}"
        
        for folder in ['scripts', 'audio_scripts', 'img_prompts']:
            os.makedirs(os.path.join(test_dir, folder), exist_ok=True)

        # Create a sample input script
        sample_script = """
        The Rise of Artificial Intelligence in Healthcare

        AI is revolutionizing the healthcare industry in unprecedented ways. From diagnostic assistance 
        to drug discovery, machine learning algorithms are enhancing medical capabilities.

        One of the most significant applications is in medical imaging. AI systems can now detect 
        potential tumors and abnormalities in X-rays and MRI scans with remarkable accuracy, often 
        catching details that human radiologists might miss.

        In patient care, AI-powered chatbots provide 24/7 support, answering basic health questions 
        and helping to triage cases. This reduces the burden on healthcare workers and improves 
        patient access to information.

        The future looks even more promising. Researchers are developing AI systems that can predict 
        patient outcomes, recommend personalized treatment plans, and even assist in complex surgeries 
        through robotic systems.

        However, it's important to note that AI is not replacing healthcare workers – it's empowering 
        them to provide better care. The human touch in medicine remains irreplaceable, with AI serving 
        as a powerful tool to enhance human capabilities.
        """

        script_path = os.path.join(test_dir, 'scripts', 'sample_script.txt')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(sample_script)
        
        logger.info("Initializing AudioScriptGenerator...")
        start_time = time.time()
        generator = AudioScriptGenerator()
        init_time = time.time() - start_time
        logger.info(f"Initialization complete in {init_time:.2f} seconds")

        output_path = os.path.join(test_dir, 'audio_scripts', 'audio_script.txt')
        img_prompts_path = os.path.join(test_dir, 'img_prompts', 'image_prompts.txt')

        logger.info("\nGenerating audio script and image prompts...")
        start_time = time.time()
        
        success = generator.generate_audio_script(
            script_path=script_path,
            output_path=output_path,
            img_prompts_path=img_prompts_path
        )
        
        generation_time = time.time() - start_time

        logger.info("\n" + "="*50)
        logger.info("Test Results:")
        logger.info(f"Status: {'Successful' if success else 'Failed'}")
        logger.info(f"Generation Time: {generation_time:.2f} seconds")
        logger.info(f"\nOutput Files:")
        logger.info(f"- Test Directory: {test_dir}")
        logger.info(f"- Input Script: {script_path}")
        logger.info(f"- Audio Script: {output_path}")
        logger.info(f"- Image Prompts: {img_prompts_path}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting AudioScriptGenerator test...")
        main()
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")