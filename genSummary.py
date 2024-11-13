import os
from groq import Groq
import logging
from typing import Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentProcessor:
    def __init__(self):
        """Initialize with Groq API key"""
        self.client = Groq(api_key="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")

    def read_content(self, content_id: str) -> Optional[str]:
        """Read content from the content file"""
        try:
            file_path = f"content/content_{content_id}.txt"
            if not os.path.exists(file_path):
                logger.error(f"Content file not found: {file_path}")
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading content file: {e}")
            return None

    def generate_summary(self, content: str) -> Optional[str]:
        """Generate a concise summary using Groq's LLama model"""
        try:
            system_prompt = """
            You are an expert content summarizer. Create a concise yet comprehensive summary that:
            1. Captures the main ideas and key points
            2. Preserves crucial information and context
            3. Maintains logical flow and coherence
            4. Includes relevant examples or data points
            5. Is between 150-200 words
            
            The summary should be clear, engaging, and retain the essence of the original content.
            DO NOT include any prefixes like "Here's a summary:" - just output the summary text directly.
            """

            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Create a focused summary of this content while preserving its key insights:\n\n{content}"
                    }
                ],
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=300,
                top_p=1,
                stream=False
            )

            return completion.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return None

    def generate_audio_script(self, summary: str) -> Optional[str]:
        """Generate an engaging audio script for reels from the summary"""
        try:
            system_prompt = """
            You are an expert Instagram Reels script writer. Create a short, engaging script that:
            1. Is EXACTLY 45 SECONDS when spoken
            2. Uses short, punchy sentences
            3. Has a Gen-Z creator voice style
            4. Starts with an attention-grabbing hook
            5. Includes the most important information
            6. Ends with a powerful conclusion
            7. Uses natural pauses between thoughts
            8. Is between 70-90 words TOTAL
            9. Makes every word count - no filler
            
            Output ONLY the spoken script text - no markers or formatting.
            """

            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Create an engaging Instagram Reel script from this summary:\n\n{summary}"
                    }
                ],
                model="llama-3.1-8b-instant",
                temperature=0.8,
                max_tokens=200,
                top_p=1,
                stream=False
            )

            return completion.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating audio script: {e}")
            return None

    def process_content(self, content_id: str) -> Tuple[bool, Dict[str, str]]:
        """Process content and generate summary and audio script"""
        try:
            # Create output directories if they don't exist
            os.makedirs('summaries', exist_ok=True)
            os.makedirs('audio_scripts', exist_ok=True)

            # Read content
            content = self.read_content(content_id)
            if not content:
                return False, {}

            # Generate summary
            logger.info("Generating content summary...")
            summary = self.generate_summary(content)
            if not summary:
                return False, {}

            # Save summary
            summary_path = f'summaries/summary_{content_id}.txt'
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)

            # Generate audio script
            logger.info("Generating audio script...")
            audio_script = self.generate_audio_script(summary)
            if not audio_script:
                return False, {}

            # Save audio script
            audio_script_path = f'audio_scripts/audioScript_{content_id}.txt'
            with open(audio_script_path, 'w', encoding='utf-8') as f:
                f.write(audio_script)

            return True, {
                'summary_path': summary_path,
                'audio_script_path': audio_script_path,
                'summary_word_count': len(summary.split()),
                'script_word_count': len(audio_script.split())
            }

        except Exception as e:
            logger.error(f"Error processing content: {e}")
            return False, {}

def main():
    try:
        # Get content ID from user
        content_id = input("Enter the content ID: ")

        # Initialize processor and process content
        processor = ContentProcessor()
        success, results = processor.process_content(content_id)

        if success:
            print("\nProcessing complete!")
            print(f"Summary saved to: {results['summary_path']}")
            print(f"Summary word count: {results['summary_word_count']}")
            print(f"Audio script saved to: {results['audio_script_path']}")
            print(f"Audio script word count: {results['script_word_count']}")

            # # Display generated content
            # print("\nGenerated Summary:")
            # print("-" * 50)
            # with open(results['summary_path'], 'r', encoding='utf-8') as f:
            #     print(f.read())

            # print("\nGenerated Audio Script:")
            # print("-" * 50)
            # with open(results['audio_script_path'], 'r', encoding='utf-8') as f:
            #     print(f.read())
        else:
            print("\nFailed to process content")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()