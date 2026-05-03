import numpy as np
import random
import math


def shear(data_numpy, r=0.5):
    s1_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]
    s2_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]

    R = np.array([[1,          s1_list[0], s2_list[0]],
                  [s1_list[1], 1,          s2_list[1]],
                  [s1_list[2], s2_list[2], 1        ]])

    R = R.transpose()
    data_numpy = np.dot(data_numpy.transpose([1, 2, 3, 0]), R)
    data_numpy = data_numpy.transpose(3, 0, 1, 2)
    return data_numpy


def temperal_crop(data_numpy, temperal_padding_ratio=6):
    C, T, V, M = data_numpy.shape
    padding_len = T // temperal_padding_ratio
    frame_start = np.random.randint(0, padding_len * 2 + 1)
    data_numpy = np.concatenate((data_numpy[:, :padding_len][:, ::-1],
                                 data_numpy,
                                 data_numpy[:, -padding_len:][:, ::-1]),
                                axis=1)
    data_numpy = data_numpy[:, frame_start:frame_start + T]
    return data_numpy


# The above augmentations are default to CrosSCLR
# Below are added augmentations specific to goalkeeper case application

PAIRS_NTU = [
    (4, 8),    # Shoulder: Left(5) <-> Right(9)
    (5, 9),    # Elbow: Left(6) <-> Right(10)
    (6, 10),   # Wrist: Left(7) <-> Right(11)
    (7, 11),   # Hand: Left(8) <-> Right(12)
    (12, 16),  # Hip: Left(13) <-> Right(17)
    (13, 17),  # Knee: Left(14) <-> Right(18)
    (14, 18),  # Ankle: Left(15) <-> Right(19)
    (15, 19),  # Foot: Left(16) <-> Right(20)
    (21, 23),  # HandTip: Left(22) <-> Right(24)
    (22, 24)   # Thumb: Left(23) <-> Right(25)
]

# Native Pairs (29 Joints)
# "midHip", "neck", "nose", "lEye", "rEye", "lEar", "rEar" [0-6]
# l/r Shoulder [7,8], Elbow [9,10], Wrist [11,12]
# lThumb [13], lPinky [14], rThumb [15], rPinky [16] -> Pairs (13,15), (14,16)
# l/r Hip [17,18], Knee [19,20], Ankle [21,22]
# lBigToe[23], lSmallToe[24], lHeel[25]
# rBigToe[26], rSmallToe[27], rHeel[28]
PAIRS_NATIVE = [
    (3, 4),   # Eyes
    (5, 6),   # Ears
    (7, 8),   # Shoulders
    (9, 10),  # Elbows
    (11, 12), # Wrists
    (13, 15), # Thumbs
    (14, 16), # Pinkies
    (17, 18), # Hips
    (19, 20), # Knees
    (21, 22), # Ankles
    (23, 26), # BigToes
    (24, 27), # SmallToes
    (25, 28)  # Heels
]

def mirror(data_numpy):
    """
    Mirroring augmentation.
    Flips the X-axis (C=0) and swaps left/right body parts.
    
    Args:
        data_numpy: numpy array of shape (C, T, V, M)
    """
    # Flip X-axis
    data_numpy[0, :, :, :] = -data_numpy[0, :, :, :]
    
    C, T, V, M = data_numpy.shape
    
    # Select Correct Pairs based on V
    if V == 29:
        pairs = PAIRS_NATIVE
    else:
        pairs = PAIRS_NTU

    # Swap Left/Right Joints
    for left, right in pairs:
        # Copy strictly needed to avoid reference issues during swap
        temp = data_numpy[:, :, left, :].copy()
        data_numpy[:, :, left, :] = data_numpy[:, :, right, :]
        data_numpy[:, :, right, :] = temp
        
    return data_numpy


def random_rotation(data_numpy, max_angle=30):
    """
    Randomly rotate the skeleton around the Y-axis.
    Makes the model action orientation invariable.

    Args:
        data_numpy: numpy array of shape (C, T, V, M)
        max_angle: maximum rotation angle in degrees (default 30)
    """
    # Convert degrees to radians
    theta = random.uniform(-max_angle, max_angle) * (math.pi / 180)
    
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    # Rotation Matrix for Y-axis:
    # x' = x*cos(t) - z*sin(t)
    # y' = y
    # z' = x*sin(t) + z*cos(t)
    
    x = data_numpy[0, :, :, :].copy()
    z = data_numpy[2, :, :, :].copy()
    
    data_numpy[0, :, :, :] = x * cos_t - z * sin_t
    data_numpy[2, :, :, :] = x * sin_t + z * cos_t
    
    return data_numpy


def gaussian_noise(data_numpy, mean=0, std=0.01):
    """
    Add random Gaussian noise to the skeleton coordinates.
    Simulates sensor/tracker noise.
    
    Args:
        data_numpy: numpy array of shape (C, T, V, M)
        std: standard deviation of the noise (0.01 meters = 1cm jitter)
    """
    noise = np.random.normal(mean, std, data_numpy.shape)
    data_numpy += noise
    
    return data_numpy