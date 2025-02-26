import numpy as np
import pywt
from PIL import Image
from tkinter import Tk, filedialog

def text_to_binary(text):
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_text(binary_string, text_length):
    binary_string = binary_string[:text_length * 8]
    chars = [binary_string[i:i + 8] for i in range(0, len(binary_string), 8)]
    return ''.join(chr(int(char, 2)) for char in chars)

def apply_dwt(image):
    return pywt.dwt2(image.astype(float), 'haar')

def apply_idwt(coeffs2, original_shape):
    reconstructed = pywt.idwt2(coeffs2, 'haar')
    if reconstructed.shape != original_shape:
        reconstructed = reconstructed[:original_shape[0], :original_shape[1]]
    return reconstructed

def embed_watermark(image_path, text_watermark, output_path, alpha=0.5):
    binary_string = text_to_binary(text_watermark)
    print(f"Original binary string: {binary_string}")
    print(f"Embedded watermark (binary): {binary_string}")
    print(f"Embedded watermark text: {text_watermark}")

    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    
    blue_channel = img_array[:, :, 2].astype(np.float32)
    original_shape = blue_channel.shape

    coeffs2 = apply_dwt(blue_channel)
    LL, (LH, HL, HH) = coeffs2

    wm_size = (LL.shape[0] // 4, LL.shape[1] // 4)
    wm_array = np.zeros(wm_size, dtype=np.float32)
    wm_array.flat[:len(binary_string)] = np.array([int(b) for b in binary_string], dtype=np.float32)

    LL_int = LL[:wm_size[0], :wm_size[1]].astype(np.int32)
    LL_int = (LL_int & ~1) | wm_array.astype(np.int32)
    LL[:wm_size[0], :wm_size[1]] = LL_int.astype(np.float32)

    watermarked_coeffs2 = (LL, (LH, HL, HH))
    watermarked_blue = np.clip(apply_idwt(watermarked_coeffs2, original_shape), 0, 255).astype(np.uint8)

    watermarked_img_array = img_array.copy()
    watermarked_img_array[:, :, 2] = watermarked_blue

    watermarked_img = Image.fromarray(watermarked_img_array)
    watermarked_img.save(output_path, "WEBP", lossless=True)

    return wm_size, binary_string

def extract_watermark(watermarked_image_path, wm_shape, binary_length, similarity_threshold=1):
    img = Image.open(watermarked_image_path).convert("RGB")
    img_array = np.array(img)

    watermarked_blue = img_array[:, :, 2].astype(np.float32)

    coeffs2 = apply_dwt(watermarked_blue)
    LL, (LH, HL, HH) = coeffs2

    extracted_wm = LL[:wm_shape[0], :wm_shape[1]]
    extracted_wm_int = extracted_wm.astype(np.int32)
    extracted_wm_binary = (extracted_wm_int & 1).astype(np.uint8)

    binary_string = ''.join(extracted_wm_binary.flatten().astype(str))[:binary_length]
    print(f"Extracted binary string: {binary_string}")
    print(f"Extracted watermark (binary): {binary_string}")
    extracted_text = binary_to_text(binary_string, len(binary_string) // 8)
    print(f"Extracted watermark text: {extracted_text}")

    match_percentage = np.mean([1 for i in range(len(binary_string)) if binary_string[i] == extracted_wm_binary.flatten()[i].astype(str)])

    if match_percentage >= similarity_threshold:
        print(f"✅ Watermark extraction successful: {match_percentage * 100:.2f}% of the binary watermark matches the original!")
    else:
        print(f"❌ Watermark extraction failed: Only {match_percentage * 100:.2f}% of the binary watermark matches the original.")

    return extracted_text

def get_valid_watermark():
    while True:
        text_watermark = input("Enter the text watermark (any ASCII characters): ").strip()
        if all(ord(c) < 128 for c in text_watermark):
            return text_watermark
        else:
            print("Invalid input! Please enter only ASCII characters.")

def open_file_dialog(file_types):
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(filetypes=file_types)
    root.destroy()
    return file_path

def main():
    print("Welcome to the Watermarking Tool!")
    print("1. Embed a watermark into an image")
    print("2. Verify a watermark in an image")
    
    while True:
        mode = input("Select an option (1 or 2): ").strip()
        if mode in ["1", "2"]:
            break
        else:
            print("Invalid option! Please select 1 or 2.")

    if mode == "1":
        print("Please select an image to embed the watermark.")
        image_path = open_file_dialog([("JPEG Files", "*.jpg;*.jpeg"), ("PNG Files", "*.png")])
        if not image_path:
            print("No image selected. Exiting.")
            return

        text_watermark = get_valid_watermark()
        print(f"Embedding watermark text: {text_watermark}")

        wm_shape, binary_string = embed_watermark(image_path, text_watermark, "watermarked_image.webp")
        print("Watermark embedded successfully!")

    elif mode == "2":
        print("Please select an image to verify the watermark.")
        image_path = open_file_dialog([("WebP Files", "*.webp")])
        if not image_path:
            print("No image selected. Exiting.")
            return

        text_watermark = get_valid_watermark()
        print(f"Verifying watermark: {text_watermark}")

        extracted_text = extract_watermark(
            image_path,
            wm_shape=(100, 100),
            binary_length=len(text_watermark) * 8,
            similarity_threshold=0.95
        )

        if extracted_text == text_watermark:
            print("✅ Watermark verification successful: The extracted watermark text matches the original!")
        else:
            print("❌ Watermark verification failed: The extracted watermark text does not match the original.")

if __name__ == "__main__":
    main()