from moviepy.editor import VideoFileClip, AudioFileClip
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioOverlay:
    def add_audio_to_video(self, unique_id: str) -> bool:
        """Add audio to video with proper sync"""
        try:
            # Construct file paths
            video_path = f"videos/vids_{unique_id}/final_video_{unique_id}.mp4"
            audio_path = f"audio/audio_{unique_id}.mp3"
            output_path = f"final/final_video_without_captions_{unique_id}.mp4"
            
            # Create output directory if it doesn't exist
            os.makedirs('final', exist_ok=True)
            
            logger.info("Loading video and audio files...")
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)
            
            # Check if audio duration needs adjustment
            if audio.duration > video.duration:
                logger.info("Trimming audio to match video duration...")
                audio = audio.subclip(0, video.duration)
            elif audio.duration < video.duration:
                logger.info("Extending video to match audio duration...")
                video = video.subclip(0, audio.duration)
            
            logger.info("Adding audio to video...")
            final_video = video.set_audio(audio)
            
            logger.info("Writing final video...")
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                fps=video.fps,
                threads=4,
                bitrate='8000k'
            )
            
            # Clean up
            video.close()
            audio.close()
            final_video.close()
            
            logger.info(f"Successfully created video with audio at: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding audio to video: {e}")
            return False

def main():
    try:
        unique_id = input("Enter the unique ID: ")
        
        overlay = AudioOverlay()
        success = overlay.add_audio_to_video(unique_id)
        
        if success:
            print("\nVideo processing complete!")
            print(f"Final video saved to: final/final_video_without_captions_{unique_id}.mp4")
        else:
            print("\nFailed to process video")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()