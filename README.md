Text-to-Video Generator with Captioning
This repository contains a Python application that generates AI-based videos from a given input video. The application uses a combination of diffusion models and NVIDIA's VLM API to process video frames, generate captions, and synthesize an AI-generated video based on the captions.

Project Overview

The Text-to-Video Generator with Captioning performs the following tasks:
1. Accepts a video input from the user.
2. Processes each frame of the video, generating captions using the NVIDIA VLM API.
3. Combines the captions into a prompt and generates a video using a pre-trained diffusion model.
4. Outputs the captions and the generated video.
   
Dependencies
To run the code, you will need to install the following dependencies:
- Gradio
- OpenCV
- Pillow
- NumPy
- Diffusers
- Torch
- Requests
- Google Colab (optional for file uploads)


Usage
Once the app is running, you can use the Gradio interface to upload a video. The app will then process the video, generate captions, and produce an AI-generated video based on those captions. The captions and generated video will be displayed in the interface.

Code Explanation
1. **VLM Class**: This class handles interactions with the NVIDIA VLM API to generate captions based on input images (video frames).
2. **Diffusion Pipeline**: A pre-trained diffusion model is used to generate AI-based video frames from the captions.
3. **Gradio Interface**: The Gradio library is used to create the interactive web interface that allows users to upload videos, view captions, and download the generated video.
   
The below is the first paragraph of the caption the model generated for the same video
”The image shows a man in a maroon T-shirt and a backpack walking down the street. The man is wearing sunglasses and has short hair. The background is blurred, but we can see that the man is walking on a busy street”
