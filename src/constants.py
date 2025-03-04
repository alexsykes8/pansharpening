'''
This file allows you to adjust parameters for image processing.
'''

# The path to the lidar image. This image should be a 1 band, 8 bit tif file.
lidar_path = "lidar.tif"

# The path to the satellite image. This image should be a 3 band, 16 bit tif file.
satellite_path = "satellite.tif"

# The azimuth is the compass direction of the sun in degrees.
azimuth = 180

# The altitude is the angle of the sun above the horizon in degrees. The closer to 0, the longer the shadows.
altitude = 45

# The threshold for marking pixels as water from the NIR band. Increasing it will result in more areas being marked as water.
water_threshold_factor = 0.035

# The gamma factor for increasing the gamma of the image. Increasing it will result in a more intense image.
gamma_factor = 1.8