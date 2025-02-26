import numpy as np
from PIL import Image
import pywt
import cv2
from scipy.linalg import svd
from scipy.stats import pearsonr
import os

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
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img_array = np.array(img, dtype=np.float32)

    # Apply DWT + DCT
    dct_LL, subbands = apply_dwt_dct(img_array)

    # Apply SVD to the DCT coefficients
    U, S, Vt = svd(dct_LL, full_matrices=False)

    # Load and normalize watermark
    watermark = load_and_resize_image(watermark_image_path, dct_LL.shape[::-1])
    watermark_array = np.array(watermark, dtype=np.float32)
    watermark_array = (watermark_array - watermark_array.min()) / (watermark_array.max() - watermark_array.min())

    # Modify singular values
    S_marked = S + alpha * watermark_array.flatten()

    # Reconstruct with modified S
    dct_LL_marked = U @ np.diag(S_marked) @ Vt

    # Apply inverse transforms
    watermarked_img_array = apply_idwt_dct(dct_LL_marked, subbands)
    watermarked_img_array = np.clip(watermarked_img_array, 0, 255).astype(np.uint8)

    # Save the watermarked image
    watermarked_img = Image.fromarray(watermarked_img_array)
    watermarked_img.save(output_path, "WEBP", lossless=True)
    print("Watermark embedded successfully!")

def extract_watermark(image_path, original_image_path, original_watermark_path, extracted_watermark_path, alpha=0.05):
    """Extract the watermark from a watermarked image."""
    watermarked_img = Image.open(image_path).convert("L")
    original_img = Image.open(original_image_path).convert("L")

    watermarked_array = np.array(watermarked_img, dtype=np.float32)
    original_array = np.array(original_img, dtype=np.float32)

    # Apply DWT + DCT
    dct_LL_watermarked, _ = apply_dwt_dct(watermarked_array)
    dct_LL_original, _ = apply_dwt_dct(original_array)

    # Apply SVD to both
    U_w, S_w, Vt_w = svd(dct_LL_watermarked, full_matrices=False)
    U_o, S_o, Vt_o = svd(dct_LL_original, full_matrices=False)

    # Extract watermark from singular values
    extracted_watermark_array = (S_w - S_o) / alpha
    extracted_watermark_array = extracted_watermark_array.reshape(dct_LL_watermarked.shape)

    # Normalize for visibility
    extracted_watermark_array = (extracted_watermark_array - extracted_watermark_array.min()) / (extracted_watermark_array.max() - extracted_watermark_array.min()) * 255
    extracted_watermark_array = np.clip(extracted_watermark_array, 0, 255).astype(np.uint8)

    # Resize to match original watermark
    original_watermark_size = Image.open(original_watermark_path).size
    extracted_watermark = Image.fromarray(extracted_watermark_array).resize(original_watermark_size, Image.Resampling.LANCZOS)
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

if __name__ == "__main__":
    print("Welcome to the DWT + DCT + SVD Watermarking Tool!")
    print("1. Embed a watermark")
    print("2. Extract a watermark")
    print("3. Detect watermark presence")

    choice = input("Enter choice (1, 2, or 3): ").strip()

    if choice == "1":
        img_path = input("Enter the path to the original image: ").strip()
        wm_path = input("Enter the path to the watermark image: ").strip()
        output_path = "watermarked_image.webp"
        embed_watermark(img_path, wm_path, output_path)
    elif choice == "2":
        wm_img_path = input("Enter the path to the watermarked image: ").strip()
        orig_img_path = input("Enter the path to the original image: ").strip()
        orig_wm_path = input("Enter the path to the original watermark: ").strip()
        extract_watermark(wm_img_path, orig_img_path, orig_wm_path, "extracted_watermark.webp")
    elif choice == "3":
        wm_img_path = input("Enter the path to the suspected watermarked image: ").strip()
        orig_img_path = input("Enter the path to the original image: ").strip()
        orig_wm_path = input("Enter the path to the original watermark: ").strip()
        detect_watermark(wm_img_path, orig_img_path, orig_wm_path)
    else:
        print("Invalid choice. Exiting.")
