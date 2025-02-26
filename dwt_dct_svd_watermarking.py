import numpy as np
from PIL import Image
import pywt
import cv2
from scipy.linalg import svd
from scipy.stats import pearsonr
import os
from tkinter import Tk, filedialog

def load_and_resize_image(image_path, size):
    """Load and resize an image (watermark) to match the size of the LL subband."""
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    return img.resize(size, Image.Resampling.LANCZOS)  # Resize the image

def apply_dwt_dct(image):
    """Apply DWT and DCT to an image."""
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    dct_LL = cv2.dct(LL)  # Apply DCT to LL subband
    return dct_LL, (LH, HL, HH)

def apply_idwt_dct(dct_LL, subbands):
    """Apply inverse DCT and DWT to reconstruct an image."""
    LL = cv2.idct(dct_LL)  # Inverse DCT
    return pywt.idwt2((LL, subbands), 'haar')

def embed_watermark(image_path, watermark_image_path, output_path, alpha=0.05):
    """Embed the watermark using DWT + DCT + SVD."""
    img = Image.open(image_path).convert("RGB")  # Load image in color (RGB)
    img_array = np.array(img, dtype=np.float32)

    # Split the image into its color channels
    red_channel = img_array[:, :, 0]
    green_channel = img_array[:, :, 1]
    blue_channel = img_array[:, :, 2]

    # Apply DWT + DCT + SVD to each channel
    def embed_channel(channel):
        dct_LL, subbands = apply_dwt_dct(channel)
        U, S, Vt = svd(dct_LL, full_matrices=False)
        watermark = load_and_resize_image(watermark_image_path, (len(S), 1))
        watermark_array = np.array(watermark, dtype=np.float32)
        watermark_array = (watermark_array - watermark_array.min()) / (watermark_array.max() - watermark_array.min())
        watermark_array = watermark_array.flatten()[:len(S)]
        S_marked = S + alpha * watermark_array
        dct_LL_marked = U @ np.diag(S_marked) @ Vt
        return apply_idwt_dct(dct_LL_marked, subbands)

    # Embed watermark in each channel
    red_marked = embed_channel(red_channel)
    green_marked = embed_channel(green_channel)
    blue_marked = embed_channel(blue_channel)

    # Combine the channels back into a color image
    watermarked_img_array = np.stack([red_marked, green_marked, blue_marked], axis=-1)
    watermarked_img_array = np.clip(watermarked_img_array, 0, 255).astype(np.uint8)

    # Save the watermarked image
    watermarked_img = Image.fromarray(watermarked_img_array)
    watermarked_img.save(output_path, "WEBP", lossless=True)
    print("Watermark embedded successfully in color!")

def extract_watermark(image_path, original_image_path, original_watermark_path, extracted_watermark_path, alpha=0.05):
    """Extract the watermark from a watermarked image."""
    watermarked_img = Image.open(image_path).convert("RGB")
    original_img = Image.open(original_image_path).convert("RGB")

    watermarked_array = np.array(watermarked_img, dtype=np.float32)
    original_array = np.array(original_img, dtype=np.float32)

    # Split the images into their color channels
    red_w = watermarked_array[:, :, 0]
    green_w = watermarked_array[:, :, 1]
    blue_w = watermarked_array[:, :, 2]

    red_o = original_array[:, :, 0]
    green_o = original_array[:, :, 1]
    blue_o = original_array[:, :, 2]

    # Apply DWT + DCT + SVD to each channel
    def extract_channel(channel_w, channel_o):
        dct_LL_w, _ = apply_dwt_dct(channel_w)
        dct_LL_o, _ = apply_dwt_dct(channel_o)
        U_w, S_w, Vt_w = svd(dct_LL_w, full_matrices=False)
        U_o, S_o, Vt_o = svd(dct_LL_o, full_matrices=False)
        return (S_w - S_o) / alpha

    # Extract watermark from each channel
    red_watermark = extract_channel(red_w, red_o)
    green_watermark = extract_channel(green_w, green_o)
    blue_watermark = extract_channel(blue_w, blue_o)

    # Average the watermark from all channels
    extracted_watermark_array = (red_watermark + green_watermark + blue_watermark) / 3

    # Reshape the extracted watermark to a 2D array
    # The size of the watermark is determined by the length of the singular values
    watermark_size = int(np.sqrt(len(extracted_watermark_array)))  # Assume square watermark
    extracted_watermark_array = extracted_watermark_array[:watermark_size * watermark_size]  # Ensure correct size
    extracted_watermark_array = extracted_watermark_array.reshape((watermark_size, watermark_size))

    # Normalize for visibility
    extracted_watermark_array = (extracted_watermark_array - extracted_watermark_array.min()) / (extracted_watermark_array.max() - extracted_watermark_array.min()) * 255
    extracted_watermark_array = np.clip(extracted_watermark_array, 0, 255).astype(np.uint8)

    # Resize to match original watermark
    original_watermark_size = Image.open(original_watermark_path).size
    extracted_watermark = Image.fromarray(extracted_watermark_array)
    extracted_watermark = extracted_watermark.resize(original_watermark_size, Image.Resampling.LANCZOS)
    extracted_watermark.save(extracted_watermark_path, "WEBP", lossless=True)
    print("Watermark extracted successfully!")
    
def detect_watermark(image_path, original_image_path, watermark_image_path, threshold=0.7):
    """Detect if the watermark is present using correlation analysis."""
    extracted_watermark_path = "temp_extracted_watermark.webp"
    extract_watermark(image_path, original_image_path, watermark_image_path, extracted_watermark_path)

    # Load original and extracted watermark
    original_watermark = np.array(Image.open(watermark_image_path).convert("L"), dtype=np.float32)
    extracted_watermark = np.array(Image.open(extracted_watermark_path).convert("L"), dtype=np.float32)

    # Flatten images for correlation calculation
    original_watermark = original_watermark.flatten()
    extracted_watermark = extracted_watermark.flatten()

    # Compute Pearson correlation
    correlation, _ = pearsonr(original_watermark, extracted_watermark)
    
    print(f"Correlation Score: {correlation:.4f}")
    os.remove(extracted_watermark_path)  # Cleanup temporary file

    if correlation >= threshold:
        print("Watermark is present!")
        return True
    else:
        print("No watermark detected.")
        return False

def open_file_dialog(title):
    """Open a file dialog and return the selected file path."""
    root = Tk()
    root.withdraw()  # Hide the root window
    root.attributes("-topmost", True)  # Ensure the file dialog is always on top
    file_path = filedialog.askopenfilename(title=title)
    root.destroy()  # Destroy the root window after selection
    return file_path

if __name__ == "__main__":
    print("Welcome to the DWT + DCT + SVD Watermarking Tool!")
    print("1. Embed a watermark")
    print("2. Extract a watermark")
    print("3. Detect watermark presence")

    choice = input("Enter choice (1, 2, or 3): ").strip()

    if choice == "1":
        img_path = open_file_dialog("Select the ORIGINAL IMAGE (the image to be watermarked)")
        wm_path = open_file_dialog("Select the WATERMARK IMAGE (the image to embed as a watermark)")
        output_path = "watermarked_image.webp"
        embed_watermark(img_path, wm_path, output_path)
    elif choice == "2":
        wm_img_path = open_file_dialog("Select the WATERMARKED IMAGE (the image with the embedded watermark)")
        orig_img_path = open_file_dialog("Select the ORIGINAL IMAGE (the image before watermarking)")
        orig_wm_path = open_file_dialog("Select the ORIGINAL WATERMARK IMAGE (the watermark used during embedding)")
        extract_watermark(wm_img_path, orig_img_path, orig_wm_path, "extracted_watermark.webp")
    elif choice == "3":
        wm_img_path = open_file_dialog("Select the SUSPECTED WATERMARKED IMAGE (the image to check for watermark presence)")
        orig_img_path = open_file_dialog("Select the ORIGINAL IMAGE (the image before watermarking)")
        orig_wm_path = open_file_dialog("Select the ORIGINAL WATERMARK IMAGE (the watermark used during embedding)")
        detect_watermark(wm_img_path, orig_img_path, orig_wm_path)
    else:
        print("Invalid choice. Exiting.")