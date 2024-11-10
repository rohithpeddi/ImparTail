import os.path

from tqdm import tqdm
from PIL.Image import open as Image_open

from dataloader.corrupted.image_based.corruptions import *


def visualize_corruptions():
    # original_image = cv2.imread('/home/rxp190007/CODE/NeSyRobSGG/datasets/action_genome/corruptions/image_based/33.png')
    original_image = Image_open(
        '/home/rxp190007/CODE/stl_stsg/datasets/action_genome/corruptions/image_based/33.png').convert('RGB')
    corrupted_images = []

    severity = 3

    for corruption_method in tqdm(corruption_methods):
        corrupted_img = corruption_name_to_function[corruption_method](original_image, severity)
        corrupted_images.append(corrupted_img)

    num_images = len(corrupted_images)
    width, height = original_image.size
    # height, width, channels = original_image.shape

    # Number of columns and rows
    num_columns = 8
    num_rows = (num_images + num_columns - 1) // num_columns  # Round up to ensure enough rows

    # Create a large combined image to hold all the smaller images in a grid
    combined_image = np.zeros((height * num_rows, width * num_columns, 3), dtype=np.uint8)

    # Font settings for labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    font_thickness = 2

    # Place each image into the grid and label it
    for i, (name, img) in enumerate(zip(corruption_methods, corrupted_images)):
        row = i // num_columns
        col = i % num_columns
        start_row = row * height
        start_col = col * width

        # Paste image
        combined_image[start_row:start_row + height, start_col:start_col + width] = img

        # Calculate text position and add label
        text_size = cv2.getTextSize(name, font, font_scale, font_thickness)[0]
        text_x = start_col + int((width - text_size[0]) / 2)
        text_y = start_row + height - 20  # Position the text 20 pixels from the bottom
        cv2.putText(combined_image, name, (text_x, text_y), font, font_scale, font_color, font_thickness)

    # Convert color (OpenCV uses BGR by default)
    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

    # Save or show the composite image
    save_dir_path = os.path.join(os.path.dirname(__file__), '../../../analysis/assets/')
    cv2.imwrite(os.path.join(save_dir_path, f'paper_combined_image_with_legends_{severity}.png'), combined_image)
    

if __name__ == '__main__':
    corruption_methods = [
        const.GAUSSIAN_NOISE, const.SHOT_NOISE, const.IMPULSE_NOISE, const.SPECKLE_NOISE,
	    const.GAUSSIAN_BLUR, const.DEFOCUS_BLUR, const.FOG, const.FROST, const.SPATTER, const.CONTRAST, const.BRIGHTNESS,
	    const.PIXELATE, const.JPEG_COMPRESSION, const.SUN_GLARE, const.DUST, const.SATURATE
    ]

    visualize_corruptions()