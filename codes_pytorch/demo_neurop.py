import argparse
import os
import cv2
import numpy as np
import torch
from models.networks import NeurOP
from tqdm import tqdm


def pre_process(image, device):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Handle tif images and fit images into [0, 1] range
    max_img_val = np.iinfo(image.dtype).max
    image = image.astype(np.float32) / max_img_val
    image = torch.from_numpy(np.ascontiguousarray(np.transpose(image, (2, 0, 1)))).float().unsqueeze(0)
    image = image.to(device)

    return image


def post_process(output_tensor, img_dtype):
    max_img_val = np.iinfo(img_dtype).max

    image_rgb = output_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    image_rgb = (image_rgb * max_img_val).clip(0, max_img_val).astype(img_dtype)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input folder containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output folder")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use e.g. 'cuda:0', 'cuda:1', 'cpu'")
    parser.add_argument("--weights", type=str, required=True, help="Path to weights")
    args = parser.parse_args()

    # Prepare output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and weights
    model = NeurOP()
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    model.to(args.device)

    # Prepare images
    image_paths = [os.path.join(args.input_dir, img_path) for img_path in os.listdir(args.input_dir) if img_path[0] != "."]

    # Model inference
    with torch.no_grad():
        for img_path in tqdm(image_paths, total=len(image_paths), desc="Running NeurOP..."):
            in_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            image_dtype = in_image.dtype

            model_input = pre_process(in_image, args.device)

            model_output = model(model_input, return_vals=False)

            enhanced_image = post_process(model_output, image_dtype)

            output_path = os.path.join(args.output_dir, os.path.basename(img_path))
            cv2.imwrite(output_path, enhanced_image)
