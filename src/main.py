from tools import *

startup()

lidar_path = "lidar.tif"
satellite_path = "satellite.tif"

geo_transform, projection = collect_geo_data(lidar_path)

lidar_path = generate_hillshade(lidar_path, "intermediate_images/hillshade.tif")

lidar_path = convert_8bit_to_16bit(lidar_path, "intermediate_images/hillshade_16bit.tif")

width, height = get_image_dimensions(lidar_path)

multi_spectral_path = resize_multispectral(satellite_path, "intermediate_images/warped.tif", width, height)

output_path = pansharpen_image(lidar_path, multi_spectral_path, "intermediate_images/output.tif")

water_mask_path = generate_water_mask(multi_spectral_path,"intermediate_images/water_mask.tif", 3,0.035)

output_path = apply_mask(output_path, water_mask_path, "intermediate_images/output_water_masked.tif", 0.1)

output_path = increase_gamma(output_path, "intermediate_images/output_water_masked_saturation_gamma.tif", 2)

output_path = finish_image(output_path, "output.tif")

convert_tif(output_path)

apply_geo_data(output_path, geo_transform, projection)

cleanup()