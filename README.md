# reelGen

An automated content pipeline that converts web articles into engaging Gen-Z style Instagram Reel videos with captions and voiceovers.

## Overview

reelGen is a complete content automation system that:

1. Scrapes content from any web URL
2. Generates a concise summary
3. Creates a Gen-Z style voiceover script
4. Generates image prompts based on the script
5. Creates images using the FLUX image generation model (Flux-3.1 Schnell)
6. Converts text to speech using ElevenLabs API
7. Assembles a vertical video with smooth transitions
8. Adds the voiceover to the video
9. Automatically generates captions using Open AI Whisper

The result is a ready-to-post Instagram Reel video that transforms any article into engaging social media content.

## Project Structure

- `app.py` - FastAPI server that exposes the pipeline as an API
- `main.py` - Main pipeline orchestrator
- `genSummary.py` - Content summarization using Groq API
- `imagePrompts.py` - Generates image prompts for each section
- `genImages.py` - Creates images using the Flux-3.1 Schnell 
- `genAudio.py` - Converts script to audio using ElevenLabs
- `mergeImages.py` - Creates video with transitions from images
- `genFinalVid.py` - Adds audio to the video
- `genCaptions.py` - Adds captions to the final video
- `text_to_image.py` - Provides ImageGenerator

## Requirements

- Python 3.10+
- GPU with 48GB vRAM for image generation (see alternatives below)
- API keys for:
  - ElevenLabs (text-to-speech)
  - Groq (LLM API for content generation)

### Image Generation Alternatives

If you don't have a GPU with 48GB vRAM, modify `text_to_image.py` to use one of these alternative APIs:
- OpenAI gpt-image-1 API
- Fal.ai API (FLUX & more models)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/adw777/reelGen.git
   cd reelGen
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up API keys:
   - Create a `.env` file in the root directory based on `.env.example`
   - Add your API keys:
     ```
     ELEVEN_LABS_API_KEY="your_elevenlabs_api_key"
     GROQ_API_KEY="your_groq_api_key"
     OPENAI_API_KEY="your_openai_api_key"  # if using
     ```

## Usage

### Command Line Interface

Run the main script to process an article URL:

```bash
python main.py
```

You'll be prompted to enter the article URL, and the pipeline will execute all steps automatically.

### API Server

Start the FastAPI server:

```bash
python app.py
```

The API will be available at http://localhost:8000 with the following endpoints:

- `POST /process` - Start processing a URL
  ```json
  {
    "url": "https://example.com/article"
  }
  ```
- `GET /status/{job_id}` - Check the status of a processing job
- `GET /health` - Health check endpoint

### API Documentation

Once the server is running, you can access:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Pipeline Steps

1. **Content Scraping**: Extract article content from the provided URL
2. **Summarization**: Generate a concise summary of the content
3. **Script Generation**: Create a Gen-Z style voiceover script
4. **Image Prompt Generation**: Generate prompts for image creation
5. **Image Generation**: Create stylized images using FLUX
6. **Audio Generation**: Convert script to voiceover using ElevenLabs
7. **Video Creation**: Merge images with smooth transitions
8. **Audio Integration**: Add voiceover to the video
9. **Caption Generation**: Add captions using Whisper

## Directory Structure

The pipeline creates the following directories:
- `content/` - Scraped article content
- `summaries/` - Generated summaries
- `audio_scripts/` - Generated voiceover scripts
- `img_prompts/` - Generated image prompts
- `images/` - Generated images
- `audio/` - Generated audio files
- `videos/` - Intermediate video files
- `final/` - Final video output

## Troubleshooting

- **Memory Issues**: If you encounter CUDA out-of-memory errors, consider switching to one of the API alternatives for image generation
- **API Rate Limits**: ElevenLabs and Groq have rate limits, adjust your request frequency if needed
- **Missing Dependencies**: If you encounter errors about missing modules, ensure all requirements are installed

## Credits

This project uses:
- [FLUX](https://github.com/black-forest-labs/FLUX) for image generation
- [ElevenLabs](https://elevenlabs.io/) for text-to-speech
- [Groq](https://groq.com/) for LLM inference
- [Whisper](https://github.com/openai/whisper) for speech recognition and captioning