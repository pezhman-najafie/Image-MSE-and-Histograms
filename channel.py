import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from colorama import Fore

def calculate_mse_and_display(image1_path, image2_path, color_space):
    # Reading images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Resizing image1 to match the dimensions of image2
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

    # Convert images to the specified color space
    if color_space == 'RGB':
        image1_cs = image1
        image2_cs = image2
    elif color_space == 'HSV':
        image1_cs = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
        image2_cs = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    elif color_space == 'YCbCr':
        image1_cs = cv2.cvtColor(image1, cv2.COLOR_BGR2YCrCb)
        image2_cs = cv2.cvtColor(image2, cv2.COLOR_BGR2YCrCb)
    elif color_space == 'LAB':
        image1_cs = cv2.cvtColor(image1, cv2.COLOR_BGR2Lab)
        image2_cs = cv2.cvtColor(image2, cv2.COLOR_BGR2Lab)
    else:
        raise ValueError("Invalid color space. Supported values: 'RGB', 'HSV', 'YCbCr', 'LAB'")

    # Splitting images into channels
    channels_image1 = cv2.split(image1_cs)
    channels_image2 = cv2.split(image2_cs)

    # Calculating MSE for each channel
    mse_values = []
    avg_mse = 0
    for channel1, channel2 in zip(channels_image1, channels_image2):
        mse = mean_squared_error(channel1.flatten(), channel2.flatten())
        mse_values.append(mse)
        avg_mse += mse

    avg_mse /= len(channels_image1)

    # Printing MSE values for channels
    print(Fore.MAGENTA + '-' * 50 + color_space + '-' * 50)
    for i, channel in enumerate(['Channel 1', 'Channel 2', 'Channel 3']):
        print(Fore.LIGHTWHITE_EX + f'MSE for {channel}: {mse_values[i]:.2f}')

    print(Fore.LIGHTWHITE_EX + f'Average MSE for {color_space} Channels: {avg_mse:.2f}')



# Example usage:
calculate_mse_and_display('E:\\pro\\najafi\\pattern\\3.jpg', 'E:\\pro\\najafi\\pattern\\4.jpg', 'RGB')
calculate_mse_and_display('E:\\pro\\najafi\\pattern\\3.jpg', 'E:\\pro\\najafi\\pattern\\4.jpg', 'HSV')
calculate_mse_and_display('E:\\pro\\najafi\\pattern\\3.jpg', 'E:\\pro\\najafi\\pattern\\4.jpg', 'YCbCr')
calculate_mse_and_display('E:\\pro\\najafi\\pattern\\3.jpg', 'E:\\pro\\najafi\\pattern\\4.jpg', 'LAB')


def plot_histograms(image_path, color_space):
    # Reading the image
    image = cv2.imread(image_path)

    # Convert the image to the specified color space
    if color_space == 'RGB':
        image_cs = image
    elif color_space == 'HSV':
        image_cs = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == 'YCbCr':
        image_cs = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif color_space == 'LAB':
        image_cs = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    else:
        raise ValueError("Invalid color space. Supported values: 'RGB', 'HSV', 'YCbCr', 'LAB'")

    # Splitting the image into channels
    channels = cv2.split(image_cs)

    # Plotting histograms for each channel
    plt.figure(figsize=(15, 5 * len(channels)))

    for i, channel in enumerate(channels):
        plt.subplot(len(channels), 3, i * 3 + 1)
        plt.hist(channel.flatten(), bins=256, color='red', alpha=0.5)
        plt.title(f'{color_space} - {chr(ord("A") + i)} Channel')

        plt.subplot(len(channels), 3, i * 3 + 2)
        plt.imshow(channel, cmap='gray')
        plt.title(f'{color_space} - {chr(ord("A") + i)} Channel Image')

        plt.subplot(len(channels), 3, i * 3 + 3)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Original Image in {color_space}')

    plt.show()

# Example usage:
plot_histograms('E:\\pro\\najafi\\pattern\\3.jpg', 'RGB')
plot_histograms('E:\\pro\\najafi\\pattern\\3.jpg', 'HSV')
plot_histograms('E:\\pro\\najafi\\pattern\\3.jpg', 'YCbCr')
plot_histograms('E:\\pro\\najafi\\pattern\\3.jpg', 'LAB')

plot_histograms('E:\\pro\\najafi\\pattern\\4.jpg', 'RGB')
plot_histograms('E:\\pro\\najafi\\pattern\\4.jpg', 'HSV')
plot_histograms('E:\\pro\\najafi\\pattern\\4.jpg', 'YCbCr')
plot_histograms('E:\\pro\\najafi\\pattern\\4.jpg', 'LAB')
