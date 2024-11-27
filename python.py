import gradio as gr
import cv2
from PIL import Image
import numpy as np
import io
import base64
import requests
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from google.colab import files

# Define the VLM class
class VLM:
    def __init__(self, url, api_key):
        """Provide NIM API URL and an API key"""
        self.api_key = api_key
        self.url = url
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}

    def _encode_image(self, image):
        """Resize image, encode as JPEG to shrink size, then convert to Base64 for upload"""
        if isinstance(image, np.ndarray):  # OpenCV frame
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):  # PIL image
            image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image)}")

        image = image.resize((336, 336))
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()
        return image_b64

    def __call__(self, prompt, image):
        """Call VLM object with the prompt and a single frame"""
        image_b64 = self._encode_image(image)
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f'{prompt} Here is the image: <img src="data:image/jpeg;base64,{image_b64}" />'
                }
            ],
            "max_tokens": 128,
            "temperature": 0.20,
            "top_p": 0.70,
            "stream": False
        }

        response = requests.post(self.url, headers=self.headers, json=payload)
        response_json = response.json()
        reply = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        return reply

# Instantiate the VLM with your API Key
api_key = "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"
vlm = VLM("https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b", api_key)

# Load the diffusion pipeline
pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# Function to process the video and generate captions
def process_video_and_generate_captions(video_path):
    cap = cv2.VideoCapture(video_path)  # Open the uploaded video
    frames_processed = 0
    replies = []
    prompt = "Describe the action or object in this frame."

    while True:
        ret, frame = cap.read()
        if not ret:  # End of video
            break

        # Process only every 30th frame to reduce computation
        if frames_processed % 30 == 0:
            caption = vlm(prompt, frame)
            replies.append(caption)

        frames_processed += 1
        if frames_processed > 300:  # Limit to first 300 frames
            break

    cap.release()
    return replies

# Function to generate video from the captions
def generate_video_from_caption(captions):
    # Combine captions into one string
    combined_caption = " ".join(captions)
    
    # Generate video frames using the diffusion pipeline
    video_frames = pipe(combined_caption, num_inference_steps=30, num_frames=50).frames

    # Ensure frames are in the correct format and handle extra dimension
    formatted_frames = []
    for batch_frames in video_frames:
        for frame in batch_frames:
            if isinstance(frame, Image.Image):
                frame = frame.convert("RGB")  # Convert to RGB if not already
            elif isinstance(frame, np.ndarray):
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]  # Remove alpha channel if present
                frame = (frame * 255).astype(np.uint8)
                frame = Image.fromarray(frame)  # Convert numpy array to PIL Image
            formatted_frames.append(frame)

    # Export the video frames to a video file
    video_path = "generated_video.mp4"
    export_to_video(formatted_frames, video_path)

    return video_path

# Define the Gradio interface function
def gradio_process_input(video):
    captions = process_video_and_generate_captions(video)  # Process video to generate captions
    generated_video = generate_video_from_caption(captions)  # Generate video from captions
    return "\n".join(captions), generated_video

# Define the Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("# Text-to-Video Generator with Captioning")
    gr.Markdown("Upload a video, and it will generate captions and produce an AI-generated video based on the captions.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
        with gr.Column():
            captions_output = gr.Textbox(label="Generated Captions", lines=10)
            generated_video_output = gr.Video(label="Generated AI Video")

    submit_button = gr.Button("Process Video")
    submit_button.click(
        fn=gradio_process_input,
        inputs=[video_input],
        outputs=[captions_output, generated_video_output]
    )

# Launch the interface
if __name__ == "__main__":
    interface.launch(debug=True)
