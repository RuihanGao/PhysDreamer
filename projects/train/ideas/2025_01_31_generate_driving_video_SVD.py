# %%
import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import imageio


# %%
# Load the pipeline
pipeline = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16
).to("cuda")

# %%
# # Load an image (Replace with your own image path or URL)
# image_url = "https://YOUR_IMAGE_URL_HERE.jpg"  # Replace with your image URL
# response = requests.get(image_url)
# image = Image.open(BytesIO(response.content)).convert("RGB")

obj_name = "octopus_v2"
image_path = f"test_data/{obj_name}_frame_00001.png"
image = Image.open(image_path).convert("RGB")

# Display the input image
plt.imshow(image)
plt.axis("off")
plt.show()

# default output image size for SVD pipeline: height: int = 576, width: int = 1024
# center crop and resize the image to match the aspect ratio
print(f"Image size befor cropping: {image.size}")
aspect_ratio = 1024 / 576
width, height = image.size
if width / height > aspect_ratio:
    new_width = int(aspect_ratio * height)
    image = image.crop(((width - new_width) // 2, 0, (width + new_width) // 2, height))
else:
    new_height = int(width / aspect_ratio)
    image = image.crop((0, (height - new_height) // 2, width, (height + new_height) // 2))
print(f"Image size after cropping: {image.size}")

resize_size = 224
image = image.resize((resize_size, resize_size), Image.LANCZOS)
# display the resized image
plt.imshow(image)
plt.axis("off")
plt.title("Resized Image")
plt.show()

# %%
# Process image into tensor
device = "cuda"
image = np.array(image) / 255.0  # Normalize
image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(device)

# %%
# Generate a batch with different motion_bucket_id. Default is 127
for motion_bucket_id in range(127, 128): # [5, 10]
    # Generate video frames
    video_frames = pipeline(image, num_inference_steps=50, motion_bucket_id=motion_bucket_id).frames

    # %%
    video_frames_0 = video_frames[0]
    video_frames_0_0 = video_frames_0[0] # PIL.Image.Image, size = (1024, 576)
    numpy_images = np.array([np.array(img) for img in video_frames_0])

    # %%
    # Convert frames to video
    video_path = f"{obj_name}_output_video_mb_{motion_bucket_id}.mp4"
    imageio.mimsave(video_path, (numpy_images).astype(np.uint8), fps=10)
    print(f"Video saved to {video_path}")


