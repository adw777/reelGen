import os
import uuid
import logging
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from groq import Groq
import torch
import gc

# Import pipeline components
from genSummary import ContentProcessor
from imagePrompts import ImagePromptGenerator
from genImages import PromptImageGenerator
from genAudio import AudioGenerator
from mergeImages import create_video_from_images
from genFinalVid import AudioOverlay
from genCaptions import add_captions_to_video

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContentPipeline:
    def __init__(self):
        """Initialize pipeline components"""
        self.required_dirs = [
            'content',
            'summaries',
            'audio_scripts',
            'img_prompts',
            'images',
            'audio',
            'videos',
            'final'
        ]
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        for directory in self.required_dirs:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Ensuring directory exists: {directory}")

    def _validate_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception as e:
            logger.error(f"Invalid URL format: {e}")
            return False

    def scrape_content(self, url: str, unique_id: str) -> str:
        """Step 1: Scrape content from URL"""
        logger.info("Step 1: Scraping content...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else "Untitled Content"
            
            # Get main content with improved selection
            content = ""
            main_content = soup.find(['article', 'main', 'div'], class_=['content', 'article', 'post'])
            if main_content:
                paragraphs = main_content.find_all('p')
            else:
                paragraphs = soup.find_all('p')
            
            content = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50])
            
            # Save content to file
            content_file = f'content/content_{unique_id}.txt'
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(f"Title: {title}\n\nURL: {url}\n\n{content}")
            
            return content_file
            
        except Exception as e:
            logger.error(f"Error scraping content: {e}")
            raise

    def generate_summary_and_script(self, unique_id: str) -> tuple:
        """Step 2: Generate summary and audio script"""
        logger.info("Step 2: Generating summary and audio script...")
        
        try:
            processor = ContentProcessor()
            success, results = processor.process_content(unique_id)
            
            if not success:
                raise Exception("Failed to generate summary and audio script")
                
            return results['summary_path'], results['audio_script_path']
            
        except Exception as e:
            logger.error(f"Error generating summary and audio script: {e}")
            raise

    def generate_image_prompts(self, unique_id: str) -> str:
        """Step 3: Generate image prompts"""
        logger.info("Step 3: Generating image prompts...")
        
        try:
            prompt_generator = ImagePromptGenerator()
            success = prompt_generator.process_script(unique_id)
            
            if not success:
                raise Exception("Failed to generate image prompts")
                
            return f"img_prompts/imgPrompts_{unique_id}.txt"
            
        except Exception as e:
            logger.error(f"Error generating image prompts: {e}")
            raise

    def generate_images(self, unique_id: str) -> str:
        """Step 4: Generate images from prompts"""
        logger.info("Step 4: Generating images...")
        
        try:
            image_generator = PromptImageGenerator()
            image_folder = image_generator.generate_images(unique_id)
            
            if not image_folder:
                raise Exception("Failed to generate images")
                
            return image_folder
            
        except Exception as e:
            logger.error(f"Error generating images: {e}")
            raise

    def generate_audio(self, unique_id: str) -> str:
        """Step 5: Generate audio"""
        logger.info("Step 5: Generating audio...")
        
        try:
            audio_generator = AudioGenerator()
            success = audio_generator.process_script(unique_id)
            
            if not success:
                raise Exception("Failed to generate audio")
                
            return f"audio/audio_{unique_id}.mp3"
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            raise

    def merge_images(self, image_folder: str, unique_id: str) -> str:
        """Step 6: Merge images into video"""
        logger.info("Step 6: Merging images into video...")
        
        try:
            video_path = f"videos/vids_{unique_id}/final_video_{unique_id}.mp4"
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            
            create_video_from_images(
                image_folder=image_folder,
                output_name=video_path,
                image_duration=6,
                transition_duration=1.5,
                output_size=(576, 1024)
            )
            
            return video_path
            
        except Exception as e:
            logger.error(f"Error merging images: {e}")
            raise

    def generate_video_with_audio(self, unique_id: str) -> str:
        """Step 7: Generate video with audio"""
        logger.info("Step 7: Generating video with audio...")
        
        try:
            audio_overlay = AudioOverlay()
            success = audio_overlay.add_audio_to_video(unique_id)
            
            if not success:
                raise Exception("Failed to generate video with audio")
                
            return f"final/final_video_without_captions_{unique_id}.mp4"
            
        except Exception as e:
            logger.error(f"Error generating video with audio: {e}")
            raise

    def add_captions(self, unique_id: str) -> str:
        """Step 8: Add captions to final video"""
        logger.info("Step 8: Adding captions to video...")
        
        try:
            input_video = f"final/final_video_without_captions_{unique_id}.mp4"
            output_video = f"final/final_video_with_captions_{unique_id}.mp4"
            
            success = add_captions_to_video(
                input_video_path=input_video,
                output_video_path=output_video,
                model="base"
            )
            
            if not success:
                raise Exception("Failed to add captions")
                
            return output_video
            
        except Exception as e:
            logger.error(f"Error adding captions: {e}")
            raise

    def process_content(self, url: str) -> dict:
        """Main pipeline process"""
        if not self._validate_url(url):
            raise ValueError("Invalid URL provided")
            
        unique_id = str(uuid.uuid4())[:8]
        logger.info(f"Starting pipeline process with ID: {unique_id}")
        
        result = {
            'unique_id': unique_id,
            'status': 'processing',
            'files': {}
        }
        
        try:
            # Step 1: Scrape content
            content_file = self.scrape_content(url, unique_id)
            result['files']['content'] = content_file
            
            # Step 2: Generate summary and audio script
            summary_file, audio_script = self.generate_summary_and_script(unique_id)
            result['files']['summary'] = summary_file
            result['files']['audio_script'] = audio_script
            
            # Step 3: Generate image prompts
            img_prompts = self.generate_image_prompts(unique_id)
            result['files']['img_prompts'] = img_prompts
            
            # Step 4: Generate images
            image_folder = self.generate_images(unique_id)
            result['files']['images'] = image_folder
            
            # Step 5: Generate audio
            audio_file = self.generate_audio(unique_id)
            result['files']['audio'] = audio_file
            
            # Step 6: Merge images into video
            video_file = self.merge_images(image_folder, unique_id)
            result['files']['raw_video'] = video_file
            
            # Step 7: Generate video with audio
            video_with_audio = self.generate_video_with_audio(unique_id)
            result['files']['video_with_audio'] = video_with_audio
            
            # Step 8: Add captions
            final_video = self.add_captions(unique_id)
            result['files']['final_video'] = final_video
            
            result['status'] = 'completed'
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)
            return result

def main():
    try:
        # Initialize pipeline
        pipeline = ContentPipeline()
        
        # Get URL from user
        url = input("Enter the article URL: ").strip()
        
        # Process content
        result = pipeline.process_content(url)
        
        # Print results
        print("\nPipeline Process Complete!")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'completed':
            print("\nGenerated Files:")
            for file_type, file_path in result['files'].items():
                print(f"{file_type}: {file_path}")
            print(f"\nFinal video with captions saved at: {result['files']['final_video']}")
        else:
            print(f"\nError: {result.get('error', 'Unknown error occurred')}")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()