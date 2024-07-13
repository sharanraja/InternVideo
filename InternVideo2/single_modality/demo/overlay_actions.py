import cv2
import numpy as np
import json

# Load the JSON file
with open('./video1_predictions.json', 'r') as f:
    text_dict = json.load(f)

# Define input and output video file paths
input_video_path = 'video1.mp4'
output_video_path = 'overlayed_video1.mp4'

# Open the input video file
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID' or other codecs
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Define text properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_color = (255, 255, 255)  # White text color
background_color = (0, 0, 0)  # Black background color
text_size = (frame_width, 60)  # Width of the frame and height for the text background

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video
    
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

    # Get the text for the current frame
    text = text_dict.get(str(frame_number), "")

    if text:
        # Create an overlay for the text background
        overlay = frame.copy()
        text_background = np.full((60, frame_width, 3), background_color, dtype=np.uint8)

        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = int((frame_width - text_width) / 2)
        text_y = int((60 + text_height) / 2)

        # Draw the text background
        text_background = cv2.rectangle(text_background, (0, 0), (frame_width, 60), background_color, -1)

        # Put the text on the text background
        cv2.putText(text_background, text, (text_x, text_y + baseline), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Overlay the text background on top of the frame
        frame[:60, :] = text_background

    # Write the frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete. The new video has been saved as:", output_video_path)

