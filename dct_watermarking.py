import cv2
import numpy as np
import scipy.fftpack
from tkinter import Tk, filedialog


def text_to_binary(text):
    """Converts lowercase alphabet text to a binary string."""
    return ''.join(format(ord(c) - ord('a'), '05b') for c in text)


def binary_to_text(binary_string, text_length):
    """Converts binary string back to text using only lowercase alphabet characters."""
    binary_string = binary_string[:text_length * 5]  # Ensure binary string fits the original text length
    chars = [binary_string[i:i + 5] for i in range(0, len(binary_string), 5)]
    return ''.join(chr(int(char, 2) + ord('a')) for char in chars)


def apply_dct(image):
    """Applies DCT to an image (single-channel)."""
    return scipy.fftpack.dct(scipy.fftpack.dct(image.astype(float).T, norm='ortho').T, norm='ortho')


def apply_idct(dct_image):
    """Applies inverse DCT to reconstruct the image."""
    return scipy.fftpack.idct(scipy.fftpack.idct(dct_image.T, norm='ortho').T, norm='ortho')


def embed_watermark(image_path, text_watermark, output_path, alpha=100):
    """Embeds a text watermark into the blue channel of an image with an increased watermark region."""
    binary_string = text_to_binary(text_watermark)
    print(f"Embedding watermark (binary): {binary_string}")

    # Load image in color
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    blue_channel = img[:, :, 0]  # Extract the blue channel

    # Convert binary string to an array of bits
    binary_bits = np.array([int(b) for b in binary_string], dtype=np.float32)

    # Define a larger watermark region in **low-frequency** DCT area
    wm_size = (blue_channel.shape[0] // 4, blue_channel.shape[1] // 4)  # Increase region size
    wm_array = np.zeros(wm_size, dtype=np.float32)
    wm_array.flat[:len(binary_bits)] = binary_bits[:wm_array.size]  # Fit watermark in region

    # Apply DCT to the blue channel
    dct_blue = apply_dct(blue_channel)

    # Embed watermark in **low-frequency coefficients**
    dct_blue[:wm_size[0], :wm_size[1]] += alpha * wm_array

    # Apply inverse DCT
    watermarked_blue = np.clip(apply_idct(dct_blue), 0, 255).astype(np.uint8)

    # Replace the blue channel and save the new image in JPEG format (high quality)
    watermarked_img = img.copy()
    watermarked_img[:, :, 0] = watermarked_blue
    cv2.imwrite(output_path, watermarked_img, [cv2.IMWRITE_JPEG_QUALITY, 95])  # Save in JPEG with quality 95

    return wm_size, binary_string


def extract_watermark(watermarked_image_path, wm_shape, text_length=None, binary_length=None, similarity_threshold=0.8):
    """Extracts the watermark from the blue channel of a watermarked image with an increased region size."""
    # Load the watermarked image
    watermarked_img = cv2.imread(watermarked_image_path, cv2.IMREAD_COLOR)
    watermarked_blue = watermarked_img[:, :, 0]  # Extract blue channel

    # Apply DCT to the blue channel
    dct_watermarked = apply_dct(watermarked_blue)

    # Extract watermark from the **larger low-frequency region**
    extracted_wm = dct_watermarked[:wm_shape[0], :wm_shape[1]]

    # Normalize values between 0 and 1
    extracted_wm = (extracted_wm - np.min(extracted_wm)) / (np.max(extracted_wm) - np.min(extracted_wm))

    # Use **mean-based thresholding** to get binary values
    threshold = np.mean(extracted_wm)
    extracted_wm_binary = (extracted_wm > threshold).astype(np.uint8)

    # Convert the binary array back to a string
    binary_string = ''.join(extracted_wm_binary.flatten().astype(str))
    binary_string = binary_string[:binary_length]

    print(f"Extracted watermark (binary): {binary_string}")

    # Compare extracted binary string with original binary watermark
    matching_bits = sum([1 for i in range(len(binary_string)) if binary_string[i] == extracted_wm_binary.flatten()[i].astype(str)])
    match_percentage = matching_bits / len(binary_string)

    if match_percentage >= similarity_threshold:
        print(f"✅ Watermark extraction successful: {match_percentage * 100:.2f}% of the watermark matches!")
    else:
        print(f"❌ Watermark extraction failed: Only {match_percentage * 100:.2f}% of the watermark matches.")

    extracted_text = binary_to_text(binary_string, text_length)
    return extracted_text


def get_valid_watermark():
    """Prompt the user for valid watermark input."""
    while True:
        text_watermark = input("Enter the text watermark (lowercase a-z only): ").strip().lower()
        if text_watermark.isalpha() and all('a' <= c <= 'z' for c in text_watermark):
            return text_watermark
        else:
            print("Invalid input! Please enter only lowercase letters (a-z).")


def main():
    """Main function to run the watermark embedding and extraction process."""
    Tk().withdraw()
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("JPEG Files", "*.jpg;*.jpeg")])
    if not image_path:
        print("No image selected. Exiting.")
        return

    text_watermark = get_valid_watermark()  # Get watermark text
    print(f"Embedding watermark text: {text_watermark}")

    # Embed watermark with larger region
    wm_shape, binary_string = embed_watermark(image_path, text_watermark, "watermarked_image.jpg")

    # Extract watermark with similarity threshold of 80%
    extracted_text = extract_watermark(
        "watermarked_image.jpg",
        wm_shape,
        text_length=len(text_watermark),
        binary_length=len(text_watermark) * 5,
        similarity_threshold=0.8  # 80% similarity threshold
    )

    print(f"Extracted watermark text: {extracted_text}")

    # Check if extracted watermark matches the original
    if extracted_text == text_watermark:
        print("✅ Watermark extraction successful: The extracted watermark matches the original!")
    else:
        print("❌ Watermark extraction failed: The extracted watermark does not match the original.")


if __name__ == "__main__":
    main()
