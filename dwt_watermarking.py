import numpy as np
from PIL import Image
import pywt

# Function to load and resize an image (watermark) to match the size of the LL subband
def load_and_resize_image(image_path, size):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    return img.resize(size, Image.Resampling.LANCZOS)  # Resize the image

# Perform DWT on an image (decompose into subbands)
def apply_dwt(image):
    return pywt.dwt2(image.astype(float), 'haar')  # Perform Haar wavelet decomposition

# Perform IDWT to reconstruct an image from its subbands
def apply_idwt(coeffs2, original_shape):
    reconstructed = pywt.idwt2(coeffs2, 'haar')  # Inverse DWT
    # Ensure the reconstructed image matches the original shape (crop if necessary)
    if reconstructed.shape != original_shape:
        reconstructed = reconstructed[:original_shape[0], :original_shape[1]]
    return reconstructed

# Function to embed the watermark into the image using DWT
def embed_watermark(image_path, watermark_image_path, output_path, alpha=0.1):
    # Load the main image and convert to RGB
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    
    # Get the dimensions of the image
    img_height, img_width, _ = img_array.shape
    
    # Initialize an array to store the watermarked image
    watermarked_img_array = np.zeros_like(img_array, dtype=np.float32)
    
    # Loop through each color channel (R, G, B)
    for i in range(3):  
        # Perform DWT on the color channel
        coeffs2 = pywt.dwt2(img_array[:, :, i].astype(np.float32), 'haar')
        LL, (LH, HL, HH) = coeffs2
        
        # Resize the watermark to match the size of the LL subband
        watermark_gray = load_and_resize_image(watermark_image_path, LL.shape[::-1])  # Ensure correct orientation
        watermark_array = np.array(watermark_gray).astype(np.float32)

        # Normalize the watermark so it fits within the LL subband's range
        watermark_array = (watermark_array - np.min(watermark_array)) / (np.max(watermark_array) - np.min(watermark_array))

        # Embed the watermark into the LL subband
        LL += alpha * watermark_array  # Add the scaled watermark to the LL subband

        # Reconstruct the channel using inverse DWT
        coeffs2 = (LL, (LH, HL, HH))  # Rebuild the subbands with the modified LL
        channel_watermarked = pywt.idwt2(coeffs2, 'haar')
        
        # Resize the reconstructed channel to match the original image dimensions
        channel_watermarked_resized = np.resize(channel_watermarked, (img_height, img_width))

        # Add the watermarked channel to the final image array
        watermarked_img_array[:, :, i] = channel_watermarked_resized
    
    # Clip the pixel values to the valid range [0, 255] and convert to uint8
    watermarked_img_array = np.clip(watermarked_img_array, 0, 255).astype(np.uint8)

    # Save the watermarked image in WebP format
    watermarked_img = Image.fromarray(watermarked_img_array)
    watermarked_img.save(output_path, "WEBP", lossless=True)

    print("Watermark embedded successfully!")

# Function to extract the watermark from a watermarked image using DWT
def extract_watermark(image_path, original_image_path, watermark_image_path, original_watermark_path, alpha=0.1):
    # Load the watermarked image and the original image
    watermarked_img = Image.open(image_path).convert("RGB")
    original_img = Image.open(original_image_path).convert("RGB")
    
    # Convert the images to NumPy arrays
    watermarked_img_array = np.array(watermarked_img)
    original_img_array = np.array(original_img)
    
    # Initialize an empty array for storing the extracted watermark
    extracted_watermark_array = np.zeros_like(original_img_array, dtype=np.float32)
    
    # Perform DWT on both the watermarked and original images
    for i in range(3):  # Loop through each color channel (R, G, B)
        # DWT on the watermarked image
        coeffs2_watermarked = pywt.dwt2(watermarked_img_array[:, :, i].astype(np.float32), 'haar')
        LL_watermarked, (LH_watermarked, HL_watermarked, HH_watermarked) = coeffs2_watermarked
        
        # DWT on the original image
        coeffs2_original = pywt.dwt2(original_img_array[:, :, i].astype(np.float32), 'haar')
        LL_original, (LH_original, HL_original, HH_original) = coeffs2_original
        
        # Crop both LL subbands to the same size (in case they differ slightly)
        min_height = min(LL_watermarked.shape[0], LL_original.shape[0])
        min_width = min(LL_watermarked.shape[1], LL_original.shape[1])
        
        # Crop both LL subbands to the same size
        LL_watermarked_cropped = LL_watermarked[:min_height, :min_width]
        LL_original_cropped = LL_original[:min_height, :min_width]
        
        # Extract the watermark: Subtract the original LL from the watermarked LL and divide by alpha
        extracted_watermark = (LL_watermarked_cropped - LL_original_cropped) / alpha
        
        # Store the extracted watermark in the corresponding channel
        extracted_watermark_array[:min_height, :min_width, i] = extracted_watermark
    
    # Normalize the extracted watermark to the range [0, 255]
    # Use min-max normalization to ensure the values are properly scaled
    for i in range(3):  # Normalize each channel separately
        channel = extracted_watermark_array[:, :, i]
        channel_min = np.min(channel)
        channel_max = np.max(channel)
        if channel_max != channel_min:  # Avoid division by zero
            extracted_watermark_array[:, :, i] = (channel - channel_min) / (channel_max - channel_min) * 255
        else:
            extracted_watermark_array[:, :, i] = 0  # If all values are the same, set to 0
    
    # Clip the values to ensure they are within the valid range [0, 255]
    extracted_watermark_array = np.clip(extracted_watermark_array, 0, 255).astype(np.uint8)
    
    # Resize the extracted watermark to match the original watermark size
    watermark_size = Image.open(original_watermark_path).size
    extracted_watermark_image = Image.fromarray(extracted_watermark_array).resize(watermark_size, Image.Resampling.LANCZOS)
    
    # Save the extracted watermark as an image
    extracted_watermark_image.save(watermark_image_path, "WEBP", lossless=True)

    print("Watermark extracted successfully!")

# Function to open a file dialog for selecting files
def open_file_dialog(file_types):
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()  # Hide the Tkinter root window
    root.attributes("-topmost", True)  # Keep the file dialog on top
    file_path = filedialog.askopenfilename(filetypes=file_types)  # Open file dialog
    root.destroy()
    return file_path

# Main function to drive the watermarking tool
def main():
    print("Welcome to the DWT Watermarking Tool!")
    print("1. Embed a watermark into an image")
    print("2. Extract a watermark from an image")
    
    while True:
        mode = input("Select an option (1 or 2): ").strip()  # Get user input for embedding or extraction
        if mode in ["1", "2"]:
            break
        else:
            print("Invalid option! Please select 1 or 2.")

    if mode == "1":
        # Embed watermark mode
        print("Please select an image to embed the watermark into.")
        image_path = open_file_dialog([("JPEG Files", "*.jpg;*.jpeg"), ("PNG Files", "*.png")])
        if not image_path:
            print("No image selected. Exiting.")
            return

        print("Please select a watermark image.")
        watermark_image_path = open_file_dialog([("JPEG Files", "*.jpg;*.jpeg"), ("PNG Files", "*.png")])
        if not watermark_image_path:
            print("No watermark image selected. Exiting.")
            return

        embed_watermark(image_path, watermark_image_path, "watermarked_image.webp")
        print("Watermark embedded successfully!")

    elif mode == "2":
        # Extract watermark mode
        print("Please select the watermarked image (the image with the embedded watermark) for extraction.")
        watermarked_image_path = open_file_dialog([("WEBP Files", "*.webp")])
        if not watermarked_image_path:
            print("No watermarked image selected. Exiting.")
            return

        print("Please select the original image (the image that was used to embed the watermark) for extraction.")
        original_image_path = open_file_dialog([("JPEG Files", "*.jpg;*.jpeg"), ("PNG Files", "*.png")])
        if not original_image_path:
            print("No original image selected. Exiting.")
            return

        print("Please select the original watermark image (the image that was embedded) for resizing.")
        original_watermark_path = open_file_dialog([("JPEG Files", "*.jpg;*.jpeg"), ("PNG Files", "*.png")])
        if not original_watermark_path:
            print("No original watermark image selected. Exiting.")
            return

        extract_watermark(watermarked_image_path, original_image_path, "extracted_watermark.webp", original_watermark_path)
        print("Watermark extracted successfully!")

# Run the program
if __name__ == "__main__":
    main()