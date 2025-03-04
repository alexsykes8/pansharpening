import shutil
import numpy as np
from osgeo_utils.gdal_pansharpen import gdal_pansharpen
from osgeo import gdal
from PIL import Image
import cv2
import os


def startup():
    '''
    Create a folder to store intermediate images.
    '''
    folder_path = 'intermediate_images'
    os.makedirs(folder_path, exist_ok=True)

def collect_geo_data(input_path):
    """
    Collect geo-referencing data from the input image.

    :params input_path: Path to the input image.

    :return geodata.
    """
    dataset = gdal.Open(input_path, gdal.GA_ReadOnly)

    geo_transform = dataset.GetGeoTransform()  # (top left x, pixel width, rotation, top left y, rotation, pixel height)
    projection = dataset.GetProjection()

    return geo_transform, projection

def generate_hillshade(input_path, output_path, azimuth=315, altitude=45):
    '''
    Generate a hillshade from a DEM.

    :param input_path: Path to the input DEM.
    :param output_path: Path to save the hillshade.

    :return: Path to the output hillshade.
    '''
    print("Generating hillshade...")
    dem = gdal.Open(input_path)

    # Generate the hillshade
    gdal.DEMProcessing(output_path, dem, "hillshade", zFactor=2, azimuth=azimuth, altitude=altitude)
    print("Hillshade processing complete.")
    return output_path

def convert_8bit_to_16bit(input_path, output_path):
    '''
    Convert an 8-bit image to 16-bit.

    :param input_path: Path to the input 8-bit image.
    :param output_path: Path to save the 16-bit image.

    :return: Path to the output image.
    '''
    print("Converting hillshade from 8-bit to 16-bit...")
    # Open the 8-bit hillshade
    hillshade_ds = gdal.Open(input_path)
    hillshade = hillshade_ds.ReadAsArray().astype(np.uint16)  # Convert to 16-bit

    # Scale 8-bit range (0-255) to 16-bit range (0-65535)
    hillshade = (hillshade / 255.0 * 65535).astype(np.uint16)

    # Save as new 16-bit TIFF
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_path, hillshade_ds.RasterXSize, hillshade_ds.RasterYSize, 1,
                           gdal.GDT_UInt16)
    out_ds.GetRasterBand(1).WriteArray(hillshade)
    out_ds.FlushCache()

    print("Converted from 8-bit to 16-bit.")

    return output_path

def get_image_dimensions(input_path):
    '''
    Get the width and height of an image.

    :param input_path: Path to the input image.

    :return: Width and height of the image.

    '''
    # Open the image using GDAL
    dataset = gdal.Open(input_path, gdal.GA_ReadOnly)

    if dataset is None:
        raise ValueError(f"Unable to open the image at {input_path}")

    # Get the image width and height
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # Close the dataset
    dataset = None

    return width, height

def resize_multispectral(input_path, output_path, width=5000, height=5000):
    """
    Resize an image.

    Parameters:
    :param input_path: Path to the input TIFF file.
    :param output_path: Path to save the image.
    :param width: New width of the image.
    :param height: New height of the image.

    :return: Path to the output image.

    """
    print("Resampling RGB satellite image...")

    # Resample the image using bilinear interpolation
    gdal.Warp(
        output_path,  # Output file path
        input_path,  # Input dataset
        width=height,  # New width
        height=height,  # New height
        resampleAlg=gdal.GRA_Bilinear
    )
    print("Resampling complete.")
    return output_path

def pansharpen_image(lidar_path, multispectral_path, output_path):
    """
    Pansharpen a multispectral image using a LiDAR image.

    Parameters:
    :param lidar_path: Path to the input lidar TIFF file.
    :param multispectral_path: Path to the input multispectral TIFF file.
    :param output_path: Path to save the image.

    :return: Path to the output image.

        """
    print("Pansharpening image...")

    # Pansharpen the multispectral image using the LiDAR image
    gdal_pansharpen([
        '',
        '-b', '3',  # Use the first band of the resized satellite image (RGB)
        '-b', '2',  # Use the second band of the resized satellite image (RGB)
        '-b', '1',  # Use the third band of the resized satellite image (RGB)
        lidar_path,  # LiDAR image (file path)
        multispectral_path,  # Resized satellite image (file path)
        output_path  # Output pansharpened image
    ])
    print("Pansharpening complete.")
    return output_path

def generate_water_mask(input_path, output_path, channel_index, threshold_fraction=0.1):
    """
    Create a binary mask of the water bodies in an image based on the NIR channel.

    Parameters:
    :param input_path: Path to the input TIFF file.
    :param output_path: Path to save the image.
    :param channel_index (int): Index of the channel to focus on (0: Red, 1: Green, 2: Blue, etc.) Allows a mask of the near infared channel to be made.
    :param threshold_fraction (float): Fraction of the minimum value to create the threshold for darkness.

    :return: Path to the output image.

    """
    print("Generating water body mask...")

    # Open the input image
    image = gdal.Open(input_path)

    # Get the number of bands in the image
    num_bands = image.RasterCount

    # Check if the image has enough bands
    if channel_index >= num_bands:
        raise ValueError(f"The image only has {num_bands} bands. Please choose a valid channel index.")

    # Read the specific channel (band) into an array
    band = image.GetRasterBand(channel_index + 1)  # GDAL band index is 1-based, so add 1 to channel_index
    band_data = band.ReadAsArray()

    # Find the minimum and maximum values in the channel
    min_value = band_data.min()
    max_value = band_data.max()

    # Define a threshold to capture the darkest areas
    threshold = min_value + (threshold_fraction * (max_value - min_value))

    # Create a mask for the darkest areas
    dark_mask = band_data <= threshold  # Mask areas darker than the threshold

    # Convert the mask to a format suitable for saving (0 for non-dark, 255 for dark)
    dark_mask_image = (dark_mask * 255).astype(np.uint8)

    # Save the dark areas mask as a PNG
    dark_mask_pil = Image.fromarray(dark_mask_image)
    dark_mask_pil.save(output_path)

    print(f"Water body mask generated.")

    return output_path

def apply_mask(image_path, mask_path, output_path, factor):
    """
    Applies a brightness reduction to a 16-bit image based on a black-and-white mask.

    :param input_path: Path to the input TIFF file.
    :param mask_path: Path to the 8-bit mask image (white areas will have the brightness reduction applied).
    :param output_path: Path to save the image.
    :param factor: Brightness factor. Values below 1 reduce brightness, values above 1 increase brightness.

    :return: Path to the output image.
    """

    print("Applying mask...")

    # Open the 16-bit image (BGRNir or RGB) and the 8-bit mask
    image_ds = gdal.Open(image_path)
    mask_ds = gdal.Open(mask_path)

    # Read the image and mask as numpy arrays
    image = image_ds.ReadAsArray().astype(np.float32)  # Shape: (bands, height, width)
    mask = mask_ds.ReadAsArray().astype(np.float32)  # Shape: (height, width) for single channel mask

    # Normalize the mask to 0 (black) and 1 (white)
    mask = np.where(mask == 255, 1, 0)  # Convert white to 1, black to 0

    # Apply the brightness reduction (multiplying by the factor) in the white areas of the mask
    for band in range(image.shape[0]):
        image[band] *= (1 - mask * (1 - factor))

    # Clip the values to the valid range for a 16-bit image (0-65535)
    image = np.clip(image, 0, 65535)

    # Get the number of channels in the image (3 or 4)
    num_bands = image.shape[0]

    # Save the modified image to the output path
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, image_ds.RasterXSize, image_ds.RasterYSize, num_bands, gdal.GDT_UInt16)

    for band in range(num_bands):  # Write each channel (BGRNir or RGB)
        out_ds.GetRasterBand(band + 1).WriteArray(image[band].astype(np.uint16))

    # Close the datasets
    out_ds.FlushCache()
    image_ds = None
    mask_ds = None
    out_ds = None

    print(f"Finished applying mask.")

    return output_path

def increase_gamma(input_path, output_path, factor=1.5):
    """
    Increases the gamma of a multi-band 16-bit RGB image.

    :param input_path: Path to the input TIFF file.
    :param output_path: Path to save the gamma-corrected image.
    :param factor: Gamma factor (default: 1.5). Values greater than 1 increase brightness in darker areas,
                   values less than 1 decrease brightness in darker areas.

    :return: Path to the output image.
    """
    print("Adjusting gamma...")
    # Open the image using GDAL
    dataset = gdal.Open(input_path, gdal.GA_ReadOnly)
    band_count = dataset.RasterCount

    if band_count < 3:
        raise ValueError("Image must have at least 3 bands (RGB) for gamma correction.")

    # Read all bands into a NumPy array
    img_array = np.stack([dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(3)], axis=2).astype(np.float32)

    # Normalize the 16-bit values to 0-1 range for processing
    img_array = np.clip(img_array, 0, 65535)  # Ensure values are within the 16-bit range
    img_array = img_array / 65535.0  # Scale to 0-1 range

    # Apply gamma correction
    img_array = np.clip(np.power(img_array, factor), 0, 1)  # Apply gamma correction

    # Scale back to the 16-bit range (0-65535)
    img_array = (img_array * 65535).astype(np.uint16)

    # Save the gamma-corrected image as 16-bit
    driver = gdal.GetDriverByName("GTiff")
    out_dataset = driver.Create(output_path, dataset.RasterXSize, dataset.RasterYSize, 3, gdal.GDT_UInt16)

    # Write each band back to the output dataset
    for i in range(3):
        out_band = out_dataset.GetRasterBand(i + 1)
        out_band.WriteArray(img_array[:, :, i])

    # Close datasets
    dataset = None
    out_dataset = None
    print(f"Adjusting gamma complete.")

    return output_path

def finish_image(input_path, output_path):
    """
    Copies a 16-bit RGB image to a new file.

    :param input_path: Path to the input 16-bit RGB image.
    :param output_path: Path to save the copied image.

    :return: Path to the output image.
    """
    print("Saving image...")
    # Open the source image using GDAL
    dataset = gdal.Open(input_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError(f"Unable to open the image at {input_path}")

    # Get the number of bands (should be 3 for RGB)
    band_count = dataset.RasterCount
    if band_count != 3:
        raise ValueError("Image must have 3 bands (RGB)")

    # Get the image dimensions
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # Create a new output image with the same dimensions and 16-bit unsigned integer type
    driver = gdal.GetDriverByName("GTiff")
    out_dataset = driver.Create(output_path, width, height, 3, gdal.GDT_UInt16)

    # Copy each band (R, G, B) from the input to the output
    for i in range(3):
        in_band = dataset.GetRasterBand(i + 1)
        out_band = out_dataset.GetRasterBand(i + 1)

        # Read data from the input band and write it to the output band
        data = in_band.ReadAsArray()
        out_band.WriteArray(data)

    # Close the datasets
    dataset = None
    out_dataset = None

    print(f"Image saved to {output_path}.")

    return output_path

def apply_geo_data(output_path, geo_transform, projection):
    """
    Apply geo-referencing data to the new image.

    :param output_path: Path to the output image.
    :param geo_transform: GeoTransform data.
    :param projection: Projection data.

    :return output_path: Path to the output image.

    """
    print("Re-attaching geo-data...")
    driver = gdal.GetDriverByName("GTiff")
    dataset = gdal.Open(output_path, gdal.GA_Update)

    # Set the geo-referencing and projection to the new image
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)

    dataset = None
    print(f"Geo-data attached.")

    return output_path

def convert_tif(input_path):
    '''
    Convert a GeoTIFF to a JPEG image.
    :param input_path:
    :return:
    '''
    print("Converting image to JPEG...")
    dataset = gdal.Open(input_path)

    # Check if the dataset has 3 bands (RGB)
    if dataset.RasterCount != 3:
        raise ValueError('The GeoTIFF must have 3 bands (RGB)')

    # Read the bands as numpy arrays
    band_r = dataset.GetRasterBand(1).ReadAsArray().astype(np.uint16)  # Red band
    band_g = dataset.GetRasterBand(2).ReadAsArray().astype(np.uint16)  # Green band
    band_b = dataset.GetRasterBand(3).ReadAsArray().astype(np.uint16)  # Blue band

    # Normalize the data to the 8-bit range (0-255) for JPEG
    band_r = np.clip(band_r / 256, 0, 255).astype(np.uint8)
    band_g = np.clip(band_g / 256, 0, 255).astype(np.uint8)
    band_b = np.clip(band_b / 256, 0, 255).astype(np.uint8)

    # Stack the bands to create an RGB image
    rgb_image = np.stack((band_r, band_g, band_b), axis=-1)

    # Convert to a PIL image
    pil_image = Image.fromarray(rgb_image)

    # Save as JPG
    output_jpg_path = 'output_file.jpg'
    pil_image.save(output_jpg_path, 'JPEG')

    print(f"Converted image to JPEG.")

def cleanup():
    '''
    Delete images generated during runtime.
    '''
    print("Cleaning up intermediate images...")
    folder_path = 'intermediate_images'

    # Delete the folder and its contents
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    print("Finished.")