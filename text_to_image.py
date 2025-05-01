import torch
from diffusers import FluxPipeline
import logging
from PIL import Image
import gc
import numpy as np

logger = logging.getLogger(__name__)

class FluxImageGenerator:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FluxImageGenerator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                # Clear CUDA cache before initialization
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                # Changed from bfloat16 to float16
                self.pipe = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None
                ).to(self.device)

                # Enable memory optimizations
                if hasattr(self.pipe, 'enable_attention_slicing'):
                    self.pipe.enable_attention_slicing()
                if hasattr(self.pipe, 'enable_vae_slicing'):
                    self.pipe.enable_vae_slicing()

                # Try to enable xformers if available
                try:
                    import xformers
                    if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
                        self.pipe.enable_xformers_memory_efficient_attention()
                        logger.info("Xformers optimization enabled")
                except ImportError:
                    logger.info("Xformers not available, using default attention mechanism")

                logger.info(f"FLUX pipeline initialized on {self.device}")
                FluxImageGenerator._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize FLUX pipeline: {e}")
                raise

    def _truncate_prompt(self, prompt, max_length=77):
        """Truncate prompt to fit CLIP's maximum token length"""
        words = prompt.split()
        truncated = []
        current_length = 0
        
        for word in words:
            word_tokens = len(word.split()) + 1
            if current_length + word_tokens > max_length:
                break
            truncated.append(word)
            current_length += word_tokens
        
        return ' '.join(truncated)

    def _process_generated_image(self, image):
        """Process and validate generated image"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Check if image is blank or nearly blank
            if img_array.mean() < 1 or img_array.mean() > 254:
                logger.warning("Generated image appears to be blank or invalid")
                return None
                
            # Ensure proper value range
            if img_array.max() <= 1:
                img_array = (img_array * 255).astype(np.uint8)
            
            # Convert back to PIL
            return Image.fromarray(img_array)
        except Exception as e:
            logger.error(f"Error processing generated image: {e}")
            return None

    def generate_image(self, prompt, output_path, seed=None, num_attempts=3):
        """
        Generate an image using the FLUX pipeline
        
        Args:
            prompt (str): The input prompt for image generation
            output_path (str): Path where the generated image will be saved
            seed (int, optional): Random seed for reproducibility
            num_attempts (int): Number of attempts to generate a valid image
            
        Returns:
            bool: True if successful, False otherwise
        """
        for attempt in range(num_attempts):
            try:
                # Clear some CUDA memory before generation
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()

                # Truncate prompt
                truncated_prompt = self._truncate_prompt(prompt)
                if truncated_prompt != prompt:
                    logger.info(f"Prompt was truncated to: {truncated_prompt}")

                # Set up generator
                if seed is not None:
                    generator = torch.Generator(self.device).manual_seed(seed + attempt)
                else:
                    generator = torch.Generator(self.device)

                # Generate the image
                with torch.inference_mode():
                    output = self.pipe(
                        truncated_prompt,
                        height=1024,
                        width=576,
                        guidance_scale=7.5,  # Increased for better quality
                        num_inference_steps=50,
                        max_sequence_length=77,
                        generator=generator
                    )

                # Process and validate the generated image
                image = self._process_generated_image(output.images[0])
                
                if image is not None:
                    # Save the image
                    image.save(output_path)
                    logger.info(f"Successfully generated and saved image to {output_path}")
                    return True
                else:
                    logger.warning(f"Attempt {attempt + 1} produced invalid image, retrying...")
                    continue

            except Exception as e:
                logger.error(f"Error in attempt {attempt + 1}: {e}")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                if attempt == num_attempts - 1:
                    return False
                continue

        return False

    def __del__(self):
        """Cleanup when the generator is deleted"""
        if hasattr(self, 'pipe'):
            del self.pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

# # Test the generator if running directly
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     try:
#         generator = FluxImageGenerator()
#         print("Generator initialized successfully!")
        
#         prompts = [
#             {
#                 "prompt": "A young boy with a confused expression standing in a school hallway, looking down at the ground as his classmates sing the National Anthem in the background. || Style: Realistic 3D animation, muted colors, soft lighting, high quality, cinematic composition",
#                 "output": "output1.png"
#             },
#             {
#                 "prompt": "A school building with a sign that reads Knowledge is Power in the background, as a teacher stands in front of the camera with a concerned expression. || Style: Realistic 3D animation, muted colors, soft lighting, high quality, cinematic composition",
#                 "output": "output2.png"
#             }
#         ]
        
#         for i, prompt_data in enumerate(prompts):
#             print(f"\nGenerating image {i+1}...")
#             success = generator.generate_image(
#                 prompt=prompt_data["prompt"],
#                 output_path=prompt_data["output"],
#                 seed=42 + i  # Different seed for each image
#             )
#             if success:
#                 print(f"Successfully generated {prompt_data['output']}")
#             else:
#                 print(f"Failed to generate {prompt_data['output']}")

#     except Exception as e:
#         print(f"Error: {e}")