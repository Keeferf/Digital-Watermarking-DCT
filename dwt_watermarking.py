import cv2
import numpy as np
import pywt
from tkinter import Tk, filedialog

def text_to_binary(text):
    return ''.join(format(ord(c) - ord('a'), '05b') for c in text)

def binary_to_text(binary_string, text_length):
    binary_string = binary_string[:text_length * 5]
    chars = [binary_string[i:i + 5] for i in range(0, len(binary_string), 5)]
    return ''.join(chr(int(char, 2) + ord('a')) for char in chars)

def apply_dwt(image):
    return pywt.dwt2(image.astype(float), 'haar')

def apply_idwt(coeffs2):
    return pywt.idwt2(coeffs2, 'haar')

def embed_watermark(image_path, text_watermark, output_path, alpha=0.5):
    binary_string = text_to_binary(text_watermark)
    print(f"Embedded watermark (binary): {binary_string}")
    print(f"Embedded watermark text: {text_watermark}")

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    blue_channel = img[:, :, 0]

    coeffs2 = apply_dwt(blue_channel)
    LL, (LH, HL, HH) = coeffs2

    wm_size = (LL.shape[0] // 4, LL.shape[1] // 4)
    wm_array = np.zeros(wm_size, dtype=np.float32)
    wm_array.flat[:len(binary_string)] = np.array([int(b) for b in binary_string], dtype=np.float32)

    LL[:wm_size[0], :wm_size[1]] += alpha * wm_array

    watermarked_coeffs2 = (LL, (LH, HL, HH))
    watermarked_blue = np.clip(apply_idwt(watermarked_coeffs2), 0, 255).astype(np.uint8)
    watermarked_blue_resized = cv2.resize(watermarked_blue, (blue_channel.shape[1], blue_channel.shape[0]))

    watermarked_img = img.copy()
    watermarked_img[:, :, 0] = watermarked_blue_resized
    cv2.imwrite(output_path, watermarked_img, [cv2.IMWRITE_WEBP_QUALITY, 100, cv2.IMWRITE_WEBP_LOSSLESS, 1])

    return wm_size, binary_string

def extract_watermark(watermarked_image_path, wm_shape, binary_length, similarity_threshold=0.8):
    watermarked_img = cv2.imread(watermarked_image_path, cv2.IMREAD_COLOR)
    watermarked_blue = watermarked_img[:, :, 0]

    coeffs2 = apply_dwt(watermarked_blue)
    LL, (LH, HL, HH) = coeffs2

    extracted_wm = LL[:wm_shape[0], :wm_shape[1]]
    extracted_wm = (extracted_wm - np.min(extracted_wm)) / (np.max(extracted_wm) - np.min(extracted_wm))
    extracted_wm_binary = (extracted_wm > np.mean(extracted_wm)).astype(np.uint8)

    binary_string = ''.join(extracted_wm_binary.flatten().astype(str))[:binary_length]
    print(f"Extracted watermark (binary): {binary_string}")
    extracted_text = binary_to_text(binary_string, len(binary_string) // 5)
    print(f"Extracted watermark text: {extracted_text}")

    match_percentage = np.mean([1 for i in range(len(binary_string)) if binary_string[i] == extracted_wm_binary.flatten()[i].astype(str)])

    if match_percentage >= similarity_threshold:
        print(f"✅ Watermark extraction successful: {match_percentage * 100:.2f}% of the watermark matches!")
    else:
        print(f"❌ Watermark extraction failed: Only {match_percentage * 100:.2f}% of the watermark matches.")

    return extracted_text

def get_valid_watermark():
    while True:
        text_watermark = input("Enter the text watermark (lowercase a-z only): ").strip().lower()
        if text_watermark.isalpha() and all('a' <= c <= 'z' for c in text_watermark):
            return text_watermark
        else:
            print("Invalid input! Please enter only lowercase letters (a-z).")

def main():
    print("Select mode: (1) Embed watermark, (2) Verify watermark")
    mode = input().strip()

    if mode == "1":
        Tk().withdraw()  # Hide the Tkinter root window
        image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("JPEG Files", "*.jpg;*.jpeg"), ("PNG Files", "*.png")])
        if not image_path:
            print("No image selected. Exiting.")
            return

        text_watermark = get_valid_watermark()
        print(f"Embedding watermark text: {text_watermark}")

        webp_path = "temp_image.webp"
        img = cv2.imread(image_path)
        cv2.imwrite(webp_path, img, [cv2.IMWRITE_WEBP_QUALITY, 100, cv2.IMWRITE_WEBP_LOSSLESS, 1])

        wm_shape, binary_string = embed_watermark(webp_path, text_watermark, "watermarked_image.webp")

    elif mode == "2":
        Tk().withdraw()  # Hide the Tkinter root window
        image_path = filedialog.askopenfilename(title="Select a Watermarked Image", filetypes=[("WebP Files", "*.webp")])
        if not image_path:
            print("No image selected. Exiting.")
            return

        text_watermark = get_valid_watermark()
        print(f"Verifying watermark: {text_watermark}")

        extracted_text = extract_watermark(
            image_path,
            wm_shape=(100, 100),  # Example watermark shape (adjust as needed)
            binary_length=len(text_watermark) * 5,
            similarity_threshold=0.8
        )

        if extracted_text == text_watermark:
            print("✅ Watermark extraction successful: The extracted watermark matches the original!")
        else:
            print("❌ Watermark extraction failed: The extracted watermark does not match the original.")

if __name__ == "__main__":
    main()
