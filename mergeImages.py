import os
from moviepy.editor import ImageClip, VideoFileClip, concatenate_videoclips, CompositeVideoClip
from moviepy.video.fx.all import fadein, fadeout, resize
import cv2
import numpy as np
from PIL import Image
import glob

def create_smooth_zoom(clip, duration=3):
    """Create an extremely smooth, continuous zoom effect"""
    def smoothstep(x):
        """Smoothstep function for extremely smooth interpolation"""
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3 - 2 * x)
    
    def zoom_factory(t):
        # Linear progress
        progress = t / duration
        # Apply multiple smoothstep for ultra-smooth effect
        smooth_progress = smoothstep(smoothstep(progress))
        # Very minimal zoom (5%) spread over the duration
        return 1.0 + (0.2 * smooth_progress)
    
    w, h = clip.size
    
    def effect(gf, t):
        try:
            zoom = zoom_factory(t)
            frame = gf(t)
            
            # Calculate new dimensions
            new_w = int(w * zoom + 0.5)  # Add 0.5 for proper rounding
            new_h = int(h * zoom + 0.5)
            
            # Ensure dimensions are even
            new_w = new_w if new_w % 2 == 0 else new_w + 1
            new_h = new_h if new_h % 2 == 0 else new_h + 1
            
            # High quality resize
            frame = cv2.resize(frame, (new_w, new_h), 
                             interpolation=cv2.INTER_CUBIC)
            
            # Calculate crop coordinates
            x = (new_w - w) // 2
            y = (new_h - h) // 2
            
            # Return perfectly centered crop
            return frame[y:y+h, x:x+w]
        except Exception as e:
            # Return original frame if any error occurs
            return gf(t)
    
    return clip.fl(effect)

def create_transition(img1, img2, transition_duration=1):
    """Create a smooth crossfade transition between two images"""
    clip1 = ImageClip(img1).set_duration(transition_duration)
    clip2 = ImageClip(img2).set_duration(transition_duration)
    
    # Smooth crossfade
    clip1 = clip1.fx(fadeout, transition_duration)
    clip2 = clip2.fx(fadein, transition_duration)
    
    return CompositeVideoClip([clip1, clip2])

def create_video_from_images(image_folder='images', output_name='bg_vid.mp4', 
                           image_duration=3, transition_duration=1, 
                           output_size=(576, 1024)):
    """
    Create a video from images with smooth transitions and zoom
    """
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.[jp][pn][g]*')))
    if not image_files:
        raise Exception(f"No images found in {image_folder}")
    
    print(f"Found {len(image_files)} images")
    clips = []
    
    # Calculate padding size (add 10% padding)
    pad_w = int(output_size[0] * 0.1)
    pad_h = int(output_size[1] * 0.1)
    padded_size = (output_size[0] + 2*pad_w, output_size[1] + 2*pad_h)
    
    for i, image_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}")
        
        try:
            # Open and resize image with padding
            img = Image.open(image_path)
            img = img.convert('RGB')
            
            # Resize with padding for zoom
            img = img.resize(padded_size, Image.Resampling.LANCZOS)
            
            # Crop to original size from center
            left = pad_w
            top = pad_h
            right = left + output_size[0]
            bottom = top + output_size[1]
            img = img.crop((left, top, right, bottom))
            
            img_array = np.array(img)
            
            # Create clip with smooth zoom
            clip = ImageClip(img_array).set_duration(image_duration)
            clip = create_smooth_zoom(clip, image_duration)
            
            # Add fade effects only for first and last clips
            if i == 0:
                clip = clip.fx(fadein, transition_duration * 0.5)
            if i == len(image_files) - 1:
                clip = clip.fx(fadeout, transition_duration * 0.5)
            
            clips.append(clip)
            
            # Add transition to next image
            if i < len(image_files) - 1:
                next_img = Image.open(image_files[i + 1])
                next_img = next_img.convert('RGB')
                next_img = next_img.resize(padded_size, Image.Resampling.LANCZOS)
                next_img = next_img.crop((left, top, right, bottom))
                next_img_array = np.array(next_img)
                
                transition = create_transition(img_array, 
                                            next_img_array, 
                                            transition_duration)
                clips.append(transition)
                
        except Exception as e:
            print(f"Error processing image {i+1}: {e}")
            continue
    
    print("Concatenating clips...")
    final_video = concatenate_videoclips(clips, method="compose")
    
    print("Rendering video...")
    final_video.write_videofile(
        output_name,
        fps=30,
        codec='libx264',
        audio=False,
        preset='slow',  # Higher quality preset
        bitrate='8000k',
        threads=4,
        ffmpeg_params=["-vf", "format=yuv420p"]  # Ensure compatibility
    )
    print(f"Video saved as {output_name}")

# if __name__ == "__main__":
#     try:
#         create_video_from_images(
#             image_folder='images\img_ff519613',
#             output_name='enhanced_vid.mp4',
#             image_duration=3,  # Longer duration for smoother zoom
#             transition_duration=0.4,
#             output_size=(576, 1024)  # Vertical format
#         )
#     except Exception as e:
#         print(f"Error creating video: {e}")