from gradio_client import Client
import json
import base64
from PIL import Image
import io
import os

# Initialize the client
# Note: Replace the URL below with the actual URL of your deployed Gradio app
client = Client("https://3014dee2e277e8cc4b.gradio.live/")

# Create output directory for keyframes if it doesn't exist
os.makedirs('output_keyframes', exist_ok=True)

# Example using one of the demo cases from the original app

try:
    result = client.predict(
        "videos/demo1.mp4",  # Video path
        "Why did the two ladies put their hands above their eyes while staring out?",  # Question
        "practicing cheer.",  # Option 1
        "play ball.",        # Option 2
        "to see better.",    # Option 3
        "32",                # Number of video frames
        "4",                 # Number of keyframes
        fn_index=0
    )
except Exception as e:
    # Write exception to error log file
    with open('sevila_error.txt', 'w') as f:
        f.write(f"Error occurred during SeViLA API call:\n{str(e)}")

"""

# The result contains:
# 1. Keyframes (list of images)
# 2. Timestamps (string)
# 3. Answer (string)

# Write results to output file
with open('sevila_results.txt', 'w') as f:
    # Save timestamps and answer
    if isinstance(result, tuple) and len(result) >= 3:
        f.write("Timestamps: " + str(result[1]) + "\n")
        f.write("Answer: " + str(result[2]) + "\n")
        
        # Save keyframes as images
        if result[0] and isinstance(result[0], list):
            for i, frame_data in enumerate(result[0]):
                if isinstance(frame_data, dict) and 'data' in frame_data:
                    # Remove header from base64 if present
                    img_data = frame_data['data']
                    if ',' in img_data:
                        img_data = img_data.split(',')[1]
                    
                    # Decode and save image
                    img_bytes = base64.b64decode(img_data)
                    img = Image.open(io.BytesIO(img_bytes))
                    img.save(f'output_keyframes/frame_{i}.png')
                    f.write(f"Saved keyframe {i} to output_keyframes/frame_{i}.png\n")
    else:
        f.write("Unexpected response format\n")
        #f.write(f"Raw result: {str(result)}\n")
"""