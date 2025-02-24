import cv2
import numpy as np
import scipy.fftpack
from tkinter import Tk, filedialog


def text_to_binary(text):
    """Converts lowercase alphabet text to a binary string."""
    binary_string = ''.join(format(ord(c) - ord('a'), '05b') for c in text)  # 5 bits per character ('a' = 00000, 'z' = 11001)
    return binary_string


def binary_to_text(binary_string, text_length):
    """Converts binary string back to text using only lowercase alphabet characters."""
    binary_string = binary_string[:text_length * 5]  # Ensure binary string fits the original text length
    chars = [binary_string[i:i + 5] for i in range(0, len(binary_string), 5)]
    return ''.join(chr(int(char, 2) + ord('a')) for char in chars)


def apply_dct(image):
    """Applies DCT to an image."""
    return np.array([
        scipy.fftpack.dct(scipy.fftpack.dct(channel.astype(float).T, norm='ortho').T, norm='ortho')
        for channel in cv2.split(image)
    ])


def apply_idct(dct_image):
    """Applies inverse DCT to a DCT image."""
    return np.array([
        scipy.fftpack.idct(scipy.fftpack.idct(channel.T, norm='ortho').T, norm='ortho')
        for channel in dct_image
    ])


def embed_watermark(image_path, text_watermark, output_path, alpha=10):
    """Embeds a text watermark into the image at the center."""
    # Convert the text watermark to a binary string
    binary_string = text_to_binary(text_watermark)
    print(f"Embedding watermark (binary): {binary_string}")

    # Load image in color (BGR format)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert binary string to an array of bits
    binary_bits = np.array([int(b) for b in binary_string], dtype=np.float32)

    # Calculate the size of the watermark region at the center of the image
    wm_size = (img.shape[0] // 4, img.shape[1] // 4)

    # Find the top-left corner of the center region
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2
    top_left_x = center_x - wm_size[1] // 2
    top_left_y = center_y - wm_size[0] // 2

    # Ensure the watermark array fits within the image dimensions
    wm_array = np.zeros(wm_size, dtype=np.float32)
    wm_array.flat[:len(binary_bits)] = binary_bits[:wm_array.size]

    # Apply DCT to image
    dct_channels = apply_dct(img)

    # Embed watermark into the DCT coefficients at the center
    for i in range(3):  # Apply watermark to each color channel
        dct_channels[i][top_left_y:top_left_y + wm_size[0], top_left_x:top_left_x + wm_size[1]] += alpha * wm_array

    # Apply inverse DCT to each channel
    watermarked_channels = apply_idct(dct_channels)

    # Clip values and convert to uint8
    watermarked_channels = [np.clip(channel, 0, 255).astype(np.uint8) for channel in watermarked_channels]

    # Merge the color channels back
    watermarked_img = cv2.merge(watermarked_channels)

    # Save as JPEG to ensure lossy compression
    cv2.imwrite(output_path, watermarked_img, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

    return img.shape  # Return the image dimensions for watermark extraction


def extract_watermark(watermarked_image_path, original_image_path, wm_shape, alpha=10, text_length=None):
    """Extracts the watermark from the watermarked image."""
    # Load images in grayscale
    watermarked_img = cv2.imread(watermarked_image_path, cv2.IMREAD_GRAYSCALE)
    original_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

    # Apply DCT to both images
    dct_watermarked = apply_dct(watermarked_img)
    dct_original = apply_dct(original_img)

    # Extract watermark from the difference of DCT coefficients
    extracted_wm = (dct_watermarked[:wm_shape[0], :wm_shape[1]] - dct_original[:wm_shape[0], :wm_shape[1]]) / alpha

    # Normalize and convert extracted bits back to binary
    extracted_wm = (extracted_wm > np.mean(extracted_wm)).astype(np.uint8)

    # Convert binary array back to text
    binary_string = ''.join(extracted_wm.flatten().astype(str))
    extracted_text = binary_to_text(binary_string, text_length)
    
    print(f"Extracted watermark (binary): {binary_string}")  # Debug print for extracted binary
    
    return extracted_text


def get_valid_watermark():
    """Prompt the user for valid watermark input."""
    while True:
        text_watermark = input("Enter the text watermark (lowercase a-z only): ").strip()
        
        # Convert to lowercase and check if all characters are in a-z
        text_watermark = text_watermark.lower()
        
        if text_watermark.isalpha() and all('a' <= c <= 'z' for c in text_watermark):
            return text_watermark
        else:
            print("Invalid input! Please enter only lowercase letters (a-z).")


def main():
    """Main function to run the watermark embedding and extraction process."""
    Tk().withdraw()
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("JPEG Files", "*.jpg;*.jpeg")])
    if image_path:
        text_watermark = get_valid_watermark()  # Get the valid text watermark from user
        print(f"Embedding watermark text: {text_watermark}")  # Print the text watermark
        
        # Embed the watermark and get the image dimensions for watermark extraction
        img_shape = embed_watermark(image_path, text_watermark, "watermarked_image.jpg")

        # Extract watermark from the watermarked image
        extracted_text = extract_watermark("watermarked_image.jpg", image_path, (img_shape[0] // 4, img_shape[1] // 4), text_length=len(text_watermark))
        print(f"Extracted watermark text: {extracted_text}")  # Print the extracted text

        # Check if extracted watermark matches the original
        if extracted_text == text_watermark:
            print("Watermark extraction successful: The extracted watermark matches the original!")
        else:
            print("Watermark extraction failed: The extracted watermark does not match the original.")


if __name__ == "__main__":
    main()
