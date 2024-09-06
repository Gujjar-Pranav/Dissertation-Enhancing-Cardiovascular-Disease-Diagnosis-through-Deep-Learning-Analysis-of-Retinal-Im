import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import local_binary_pattern
from skimage.exposure import rescale_intensity
import imageio
import random
from skimage.util import random_noise
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support, jaccard_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from scipy.ndimage import binary_opening, binary_closing

# Define paths for further process
base_path = '/Users/pranavgujjar/Documents/Dissertation/DRIVE_DATASET/'
training_image_path = os.path.join(base_path, 'training', 'images')
training_manual1_path = os.path.join(base_path, 'training', '1st_manual')
training_mask_path = os.path.join(base_path, 'training', 'mask')
test_image_path = os.path.join(base_path, 'test', 'images')
test_manual1_path = os.path.join(base_path, 'test', '1st_manual')
test_mask_path = os.path.join(base_path, 'test', 'mask')


def read_images(folder_path, file_extension, target_size=None, binary=False):
    files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
    files.sort()
    images = []
    for file in files:
        if file_extension == '.gif':
            image = imageio.mimread(os.path.join(folder_path, file))
            if len(image) > 0:
                img = np.array(image[0], dtype=np.float32) / 255.0
                if binary:
                    img = (img > 0.5).astype(np.uint8)
                if target_size:
                    img = cv2.resize(img, target_size)
                images.append((img, file))
            else:
                images.append((None, file))
        else:
            image = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            if target_size:
                image = cv2.resize(image, target_size)
            images.append((image, file))
        if images[-1][0] is None:
            print(f"Failed to load image: {file}")
    return images


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated


def flip_image(image, horizontal):
    if horizontal:
        return cv2.flip(image, 1)
    else:
        return cv2.flip(image, 0)


def flip_and_rotate(image, angle, flip_horizontal=False, flip_vertical=False):
    if flip_horizontal:
        image = cv2.flip(image, 1)
    if flip_vertical:
        image = cv2.flip(image, 0)
    image = rotate_image(image, angle)
    return image


def complex_flip_and_rotate(image, angle, flip_hv=False, flip_vh=False):
    if flip_hv:
        image = cv2.flip(image, 1)
        image = cv2.flip(image, 0)
    if flip_vh:
        image = cv2.flip(image, 0)
        image = cv2.flip(image, 1)
    image = rotate_image(image, angle)
    return image


def apply_augmentation(image, method, params):
    if method == 'rotate':
        return rotate_image(image, params['angle'])
    elif method == 'flip':
        return flip_image(image, params['horizontal'])
    elif method == 'h_flip_rotate':
        return flip_and_rotate(image, params['angle'], flip_horizontal=params['flip'])
    elif method == 'v_flip_rotate':
        return flip_and_rotate(image, params['angle'], flip_vertical=params['flip'])
    elif method == 'complex_flip_rotate':
        return complex_flip_and_rotate(image, params['angle'], flip_hv=params['flip_hv'], flip_vh=params['flip_vh'])
    else:
        raise ValueError(f"Unsupported augmentation method: {method}")


def extract_multiple_patches_with_locations(image, patch_size, num_patches, seed=None):
    patches = []
    locations = []
    if seed is not None:
        np.random.seed(seed)
    h, w = image.shape[:2]

    for _ in range(num_patches):
        x = np.random.randint(0, w - patch_size)
        y = np.random.randint(0, h - patch_size)
        patch = image[y:y + patch_size, x:x + patch_size]
        patches.append(patch)
        locations.append((x, y))

    return patches, locations


def extract_patches_from_same_location(image, locations, patch_size):
    patches = []
    for (x, y) in locations:
        patch = image[y:y + patch_size, x:x + patch_size]
        patches.append(patch)
    return patches


def extract_feature(image, feature_type):
    if feature_type == 'green':
        green_channel = image[:, :, 1]
        return green_channel
    elif feature_type == 'enhanced_vessels':
        green_channel = image[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
        enhanced = clahe.apply((green_channel * 255).astype(np.uint8)) / 255.0
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        enhanced_vessels = cv2.addWeighted(enhanced, 1.0, blurred, -0.3, 0)
        return enhanced_vessels
    elif feature_type == 'rgb':
        return image
    elif feature_type == 'hsv':
        return rgb2hsv(image)
    elif feature_type == 'lbp':
        gray_image = rgb2gray(image)
        return local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    elif feature_type == 'contrast':
        gray_image = rgb2gray(image)
        p2, p98 = np.percentile(gray_image, (2, 98))
        return rescale_intensity(gray_image, in_range=(p2, p98))
    elif feature_type == 'grayscale':
        return rgb2gray(image)
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")


def normalize_feature(feature):
    feature_min = feature.min()
    feature_max = feature.max()
    normalized_feature = (feature - feature_min) / (feature_max - feature_min)
    return normalized_feature


def apply_mask(patch, mask):
    return patch * mask[..., np.newaxis]


def binary_to_float(mask):
    return mask.astype(np.float32)


def plot_histogram(features, feature_name):
    num_images = len(features)
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols

    plt.figure(figsize=(20, num_rows * 4))
    for i, feature in enumerate(features):
        plt.subplot(num_rows, num_cols, i + 1)
        if feature_name == 'green' or feature_name == 'enhanced_vessels':
            hist, bins = np.histogram(feature, bins=256, range=(0, 1))
            plt.plot(bins[:-1], hist)
        elif feature_name == 'rgb':
            colors = ('r', 'g', 'b')
            for j, color in enumerate(colors):
                hist, bins = np.histogram(feature[:, :, j], bins=256, range=(0, 1))
                plt.plot(bins[:-1], hist, color=color)
        elif feature_name == 'hsv':
            colors = ('h', 's', 'v')
            for j, color in enumerate(colors):
                hist, bins = np.histogram(feature[:, :, j], bins=256, range=(0, 1))
                plt.plot(bins[:-1], hist, label=colors[j])
            plt.legend()
        elif feature_name == 'lbp':
            hist, bins = np.histogram(feature, bins=256, range=(0, np.max(feature)))
            plt.plot(bins[:-1], hist)
        elif feature_name == 'contrast':
            hist, bins = np.histogram(feature, bins=256, range=(0, 1))
            plt.plot(bins[:-1], hist)
        elif feature_name == 'grayscale':
            hist, bins = np.histogram(feature, bins=256, range=(0, 1))
            plt.plot(bins[:-1], hist)
        plt.xlabel('Pixel Intensity', fontsize=10)
        plt.ylabel('Frequency', rotation=90, fontsize=10)
        plt.title(f'{feature_name} {i + 1}', fontsize=10)
        # plt.axis('off')
    plt.tight_layout(pad=3.0, w_pad=2, h_pad=5)
    plt.show()


def process_all_images(images, feature_type, normalize=False):
    features = []
    for image, filename in images:
        feature = extract_feature(image, feature_type)
        if normalize:
            feature = normalize_feature(feature)
        features.append((feature, filename))
    return features


def save_features_to_csv(features, feature_type, output_path):
    data = []
    filenames = []
    for feature, filename in features:
        if feature_type in ['green', 'enhanced_vessels']:
            data.append(feature.flatten())
        elif feature_type in ['rgb', 'hsv']:
            data.append(feature.reshape(-1, feature.shape[2]))
        elif feature_type in ['lbp', 'contrast', 'grayscale']:
            data.append(feature.flatten())
        filenames.append(filename)

    df = pd.DataFrame(data)
    df['filename'] = filenames
    df = df.transpose()
    df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")


def display_images_with_features(images, feature_type, normalize=False):
    num_images = len(images)
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols

    plt.figure(figsize=(20, num_rows * 10))
    for i, (image, filename) in enumerate(images):
        feature = extract_feature(image, feature_type)
        if normalize:
            feature = normalize_feature(feature)

        plt.subplot(num_rows, num_cols * 2, i * 2 + 1)
        plt.imshow(image)
        plt.title(f'Original:\n {filename}', fontsize=9)
        plt.axis('off')

        plt.subplot(num_rows, num_cols * 2, i * 2 + 2)
        if feature_type == 'green' or feature_type == 'enhanced_vessels':
            plt.imshow(feature, cmap='gray')
            plt.title(f'{feature_type.replace("_", " ").title()}\n Channel', fontsize=9)
        elif feature_type == 'rgb':
            plt.imshow(feature)
            plt.title('RGB Image', fontsize=9)
        elif feature_type == 'hsv':
            plt.imshow(feature)
            plt.title('HSV Image', fontsize=9)
        elif feature_type == 'lbp':
            plt.imshow(feature, cmap='gray')
            plt.title('LBP Image', fontsize=9)
        elif feature_type == 'contrast':
            plt.imshow(feature, cmap='gray')
            plt.title('Contrast Image', fontsize=9)
        elif feature_type == 'grayscale':
            plt.imshow(feature, cmap='gray')
            plt.title('Grayscale Image', fontsize=9)
        plt.axis('off')

    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
    plt.show()


def display_augmented_patches_with_manual(images, manual_images, masks, feature_type, patch_size, num_patches,
                                          normalize=False, seed=None, augment_method=None, **kwargs):
    random.seed(seed)

    for idx, ((image, filename), (manual_image, manual_filename), (mask, mask_filename)) in enumerate(
            zip(images, manual_images, masks)):
        if image is None:
            print(f"Skipping image: {filename} because it is None")
            continue
        if manual_image is None:
            print(f"Skipping manual image: {manual_filename} because it is None")
            continue
        if mask is None:
            print(f"Skipping mask image: {mask_filename} because it is None")
            continue

        original_patches, locations = extract_multiple_patches_with_locations(image, patch_size, num_patches, seed=seed)
        manual_patches = extract_patches_from_same_location(manual_image, locations, patch_size)
        mask_patches = extract_patches_from_same_location(mask, locations, patch_size)
        mask_patches = [binary_to_float(patch) for patch in mask_patches]

        params = {}
        if augment_method == 'rotate':
            params['angle'] = random.uniform(kwargs['angle_range'][0], kwargs['angle_range'][1])
        elif augment_method == 'flip':
            params['horizontal'] = kwargs['horizontal']
        elif augment_method == 'h_flip_rotate':
            params['angle'] = random.uniform(kwargs['angle_range'][0], kwargs['angle_range'][1])
            params['flip'] = random.random() > 0.5
        elif augment_method == 'v_flip_rotate':
            params['angle'] = random.uniform(kwargs['angle_range'][0], kwargs['angle_range'][1])
            params['flip'] = random.random() > 0.5
        elif augment_method == 'complex_flip_rotate':
            params['angle'] = random.uniform(kwargs['angle_range'][0], kwargs['angle_range'][1])
            params['flip_hv'] = random.random() > 0.5
            params['flip_vh'] = random.random() > 0.5

        augmented_patches = [apply_augmentation(patch.copy(), augment_method, params) for patch in original_patches]
        augmented_manual_patches = [apply_augmentation(patch.copy(), augment_method, params) for patch in
                                    manual_patches]

        selected_indices = np.random.choice(len(original_patches), 10, replace=False)
        selected_patches = [original_patches[i] for i in selected_indices]
        selected_augmented_patches = [augmented_patches[i] for i in selected_indices]
        selected_manual_patches = [manual_patches[i] for i in selected_indices]
        selected_augmented_manual_patches = [augmented_manual_patches[i] for i in selected_indices]

        num_cols = 4
        num_rows = 10

        plt.figure(figsize=(20, num_rows * 10))
        for i, (patch, manual_patch, augmented_patch, augmented_manual_patch) in enumerate(
                zip(selected_patches, selected_manual_patches, selected_augmented_patches,
                    selected_augmented_manual_patches)):
            feature = extract_feature(patch, feature_type)
            manual_feature = manual_patch
            augmented_feature = extract_feature(augmented_patch, feature_type)
            augmented_manual_feature = augmented_manual_patch

            if normalize:
                feature = normalize_feature(feature)
                manual_feature = normalize_feature(manual_feature)
                augmented_feature = normalize_feature(augmented_feature)
                augmented_manual_feature = normalize_feature(augmented_manual_feature)

            plt.subplot(num_rows, num_cols * 3, i * 4 + 1)
            if feature_type == 'green' or feature_type == 'enhanced_vessels':
                plt.imshow(feature, cmap='gray')
                plt.title(f'Orig_{feature_type}\nPatch {i + 1}_{filename}', fontsize=7)
            elif feature_type == 'rgb':
                plt.imshow(feature)
                plt.title(f'Orig_rgb_Patch {i + 1}\n{filename}', fontsize=7)
            elif feature_type == 'hsv':
                plt.imshow(feature)
                plt.title(f'Orig_hsv_Patch {i + 1}\n{filename}', fontsize=7)
            elif feature_type == 'lbp':
                plt.imshow(feature, cmap='gray')
                plt.title(f'Orig_lbp_Patch {i + 1}\n{filename}', fontsize=7)
            elif feature_type == 'contrast':
                plt.imshow(feature, cmap='gray')
                plt.title(f'Orig_contrast_Patch {i + 1}\n{filename}', fontsize=7)
            elif feature_type == 'grayscale':
                plt.imshow(feature, cmap='gray')
                plt.title(f'Orig_grayscale_Patch {i + 1}\n{filename}', fontsize=7)
            plt.axis('off')

            plt.subplot(num_rows, num_cols * 3, i * 4 + 2)
            plt.imshow(manual_feature, cmap='gray')
            plt.title(f'Manual_Patch {i + 1}\n{manual_filename}', fontsize=7)
            plt.axis('off')

            plt.subplot(num_rows, num_cols * 3, i * 4 + 3)
            if feature_type == 'green' or feature_type == 'enhanced_vessels':
                plt.imshow(augmented_feature, cmap='gray')
                plt.title(f'Aug_{feature_type}\n Patch {i + 1}_{filename}', fontsize=7)
            elif feature_type == 'rgb':
                plt.imshow(augmented_feature)
                plt.title(f'Aug_rgb_Patch {i + 1}\n{filename}', fontsize=7)
            elif feature_type == 'hsv':
                plt.imshow(augmented_feature)
                plt.title(f'Aug_hsv_Patch {i + 1}\n{filename}', fontsize=7)
            elif feature_type == 'lbp':
                plt.imshow(augmented_feature, cmap='gray')
                plt.title(f'Aug_lbp_Patch {i + 1}\n{filename}', fontsize=7)
            elif feature_type == 'contrast':
                plt.imshow(augmented_feature, cmap='gray')
                plt.title(f'Aug_contrast_Patch {i + 1}\n{filename}', fontsize=7)
            elif feature_type == 'grayscale':
                plt.imshow(augmented_feature, cmap='gray')
                plt.title(f'Aug_grayscale_Patch {i + 1}\n{filename}', fontsize=7)
            plt.axis('off')

            plt.subplot(num_rows, num_cols * 3, i * 4 + 4)
            plt.imshow(augmented_manual_feature, cmap='gray')
            plt.title(f'Aug_Manual_Patch {i + 1}\n{manual_filename}', fontsize=7)
            plt.axis('off')

        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)

        plt.show()


def process_and_save_patch_features(images, manual_images, masks, feature_type, patch_size, num_patches,
                                    normalize=False, seed=None, augment=False, augment_method=None, output_path=None,
                                    **kwargs):
    patch_features = []
    filenames = []

    for (image, filename), (manual_image, manual_filename), (mask, mask_filename) in zip(images, manual_images, masks):
        if image is None or manual_image is None or mask is None:
            print(f"Skipping pair: {filename} and corresponding manual or mask image because one of them is None")
            continue
        patches, locations = extract_multiple_patches_with_locations(image, patch_size, num_patches, seed)
        params = {}
        if augment_method == 'rotate':
            params['angle'] = random.uniform(kwargs['angle_range'][0], kwargs['angle_range'][1])
        elif augment_method == 'flip':
            params['horizontal'] = kwargs['horizontal']
        elif augment_method == 'h_flip_rotate':
            params['angle'] = random.uniform(kwargs['angle_range'][0], kwargs['angle_range'][1])
            params['flip'] = random.random() > 0.5
        elif augment_method == 'v_flip_rotate':
            params['angle'] = random.uniform(kwargs['angle_range'][0], kwargs['angle_range'][1])
            params['flip'] = random.random() > 0.5
        elif augment_method == 'complex_flip_rotate':
            params['angle'] = random.uniform(kwargs['angle_range'][0], kwargs['angle_range'][1])
            params['flip_hv'] = random.random() > 0.5
            params['flip_vh'] = random.random() > 0.5

        if augment:
            patches = [apply_augmentation(patch.copy(), augment_method, params) for patch in patches]
            manual_patches = [apply_augmentation(manual_patch.copy(), augment_method, params) for manual_patch in
                              extract_patches_from_same_location(manual_image, locations, patch_size)]
            mask_patches = [apply_augmentation(mask_patch.copy(), augment_method, params) for mask_patch in
                            extract_patches_from_same_location(mask, locations, patch_size)]
        else:
            manual_patches = extract_patches_from_same_location(manual_image, locations, patch_size)
            mask_patches = extract_patches_from_same_location(mask, locations, patch_size)

        mask_patches = [binary_to_float(patch) for patch in mask_patches]

        masked_patches = [apply_mask(patch, mask_patch) for patch, mask_patch in zip(patches, mask_patches)]

        for patch, manual_patch in zip(masked_patches, manual_patches):
            feature = extract_feature(patch, feature_type)
            manual_feature = manual_patch
            if normalize:
                feature = normalize_feature(feature)
                manual_feature = normalize_feature(manual_feature)
            combined_feature = np.hstack((feature.flatten(), manual_feature.flatten()))
            patch_features.append(combined_feature)
            filenames.append(filename)

    df = pd.DataFrame(patch_features)
    df['filename'] = filenames
    df = df.transpose()
    df.to_csv(output_path, index=False)
    print(f"Patch features saved to {output_path}")


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.backend.cast(tf.keras.backend.flatten(y_true), 'float32')
    y_pred_f = tf.keras.backend.cast(tf.keras.backend.flatten(y_pred), 'float32')
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def combined_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    bce = BinaryCrossentropy()(y_true, y_pred)
    return dice + bce


def dice_coefficient(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.backend.cast(tf.keras.backend.flatten(y_true), 'float32')
    y_pred_f = tf.keras.backend.cast(tf.keras.backend.flatten(y_pred), 'float32')
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def unet_model(input_size=(128, 128, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    batch1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(batch1)
    drop1 = Dropout(0.5)(pool1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(drop1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    batch2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(batch2)
    drop2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(drop2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    batch3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(batch3)
    drop3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(drop3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    batch4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(batch4)
    drop4 = Dropout(0.5)(pool4)

    up6 = UpSampling2D(size=(2, 2))(drop4)
    up6 = Conv2D(512, 2, activation='relu', padding='same')(up6)
    merge6 = concatenate([batch4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    batch6 = BatchNormalization()(conv6)

    up7 = UpSampling2D(size=(2, 2))(batch6)
    up7 = Conv2D(256, 2, activation='relu', padding='same')(up7)
    merge7 = concatenate([batch3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    batch7 = BatchNormalization()(conv7)

    up8 = UpSampling2D(size=(2, 2))(batch7)
    up8 = Conv2D(128, 2, activation='relu', padding='same')(up8)
    merge8 = concatenate([batch2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    batch8 = BatchNormalization()(conv8)

    up9 = UpSampling2D(size=(2, 2))(batch8)
    up9 = Conv2D(64, 2, activation='relu', padding='same')(up9)
    merge9 = concatenate([batch1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    batch9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(batch9)

    model = tf.keras.models.Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(learning_rate=1e-4), loss=combined_loss,
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), dice_coefficient])

    return model


def dense_block(x, filters, kernel_size=(3, 3), activation='relu', padding='same', dropout_rate=0.2):
    conv = Conv2D(filters, kernel_size, activation=activation, padding=padding)(x)
    conv = Dropout(dropout_rate)(conv)
    conv = Conv2D(filters, kernel_size, activation=activation, padding=padding)(conv)
    conv = Dropout(dropout_rate)(conv)
    return conv


def dense_unet_model(input_size=(128, 128, 1)):
    inputs = Input(input_size)

    # Encoder
    dense1 = dense_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(dense1)

    dense2 = dense_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(dense2)

    dense3 = dense_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(dense3)

    dense4 = dense_block(pool3, 512)

    # Decoder (adjusted connections)
    up5 = UpSampling2D(size=(2, 2))(dense4)
    merge5 = concatenate([dense3, up5], axis=3)
    dense5 = dense_block(merge5, 256)

    up6 = UpSampling2D(size=(2, 2))(dense5)
    merge6 = concatenate([dense2, up6], axis=3)
    dense6 = dense_block(merge6, 128)

    up7 = UpSampling2D(size=(2, 2))(dense6)
    merge7 = concatenate([dense1, up7], axis=3)
    dense7 = dense_block(merge7, 64)

    conv8 = Conv2D(1, 1, activation='sigmoid')(dense7)

    model = tf.keras.models.Model(inputs=inputs, outputs=conv8)

    model.compile(optimizer=Adam(learning_rate=1e-4), loss=combined_loss,
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), dice_coefficient])

    return model


def prepare_patch_dataset(images, manuals, patch_size, num_patches, feature_type, seed=None):
    X = []
    y = []
    for (image, _), (manual, _) in zip(images, manuals):
        patches, locations = extract_multiple_patches_with_locations(image, patch_size, num_patches, seed)
        manual_patches = extract_patches_from_same_location(manual, locations, patch_size)
        for patch, manual_patch in zip(patches, manual_patches):
            feature = extract_feature(patch, feature_type)
            manual_feature = manual_patch
            X.append(feature)
            y.append(manual_feature)
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)[..., np.newaxis]
    return X, y


def post_process(prediction):
    binary = (prediction > 0.5).astype(np.uint8).squeeze()
    processed = binary_opening(binary, structure=np.ones((1, 1)))
    processed = binary_closing(processed, structure=np.ones((1, 1)))
    return processed[..., np.newaxis]


target_size = (512, 512)
training_images = read_images(training_image_path, '.tif', target_size=target_size)
training_manual1 = read_images(training_manual1_path, '.gif', target_size=target_size, binary=True)
training_masks = read_images(training_mask_path, '.gif', target_size=target_size)
test_images = read_images(test_image_path, '.tif', target_size=target_size)
test_manual1 = read_images(test_manual1_path, '.gif', target_size=target_size, binary=True)
test_masks = read_images(test_mask_path, '.gif', target_size=target_size)

feature_type = 'enhanced_vessels'
patch_size = 128
num_patches = 250
normalize = True
seed = 30
augment_method = 'complex_flip_rotate'
augmentation_params = {
    'rotate': {'angle_range': (-45, 45)},
    'flip': {'horizontal': True},
    'h_flip_rotate': {'angle_range': (-45, 45)},
    'v_flip_rotate': {'angle_range': (-45, 45)},
    'complex_flip_rotate': {'angle_range': (-45, 45)}
}

display_images_with_features(training_images, feature_type, normalize)
training_features = process_all_images(training_images, feature_type, normalize)
plot_histogram([feature for feature, _ in training_features], feature_type)
display_augmented_patches_with_manual(training_images, training_manual1, training_masks, feature_type, patch_size,
                                      num_patches, normalize, seed, augment_method=augment_method,
                                      **augmentation_params[augment_method])
patch_output_csv_path = os.path.join(base_path, 'patch_features.csv')
process_and_save_patch_features(training_images, training_manual1, training_masks, feature_type, patch_size,
                                num_patches, normalize, seed, augment=True, augment_method=augment_method,
                                output_path=patch_output_csv_path, **augmentation_params[augment_method])

X_train, y_train = prepare_patch_dataset(training_images, training_manual1, patch_size, num_patches, feature_type, seed)
X_test, y_test = prepare_patch_dataset(test_images, test_manual1, patch_size, num_patches, feature_type, seed)
y_test_binary = (y_test > 0.5).astype(int)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

batch_size = 8
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=10000).repeat().batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.repeat().batch(batch_size)

# Choose model type
model_type = 'unet'  # 'unet' or 'dense_unet'

if model_type == 'unet':
    model = unet_model(input_size=(128, 128, 1))
elif model_type == 'dense_unet':
    model = dense_unet_model(input_size=(128, 128, 1))
else:
    raise ValueError(f"Unsupported model type: {model_type}")

model.summary()

steps_per_epoch = len(X_train) // batch_size
epochs = 5
validation_steps = len(X_val) // batch_size

history = model.fit(train_dataset, validation_data=val_dataset,
                    epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

# Save the trained model in the new Keras format
model_save_path = os.path.join(base_path, f'{model_type}_model.keras')
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

evaluation_results = model.evaluate(val_dataset, steps=validation_steps)
val_loss = evaluation_results[0]
val_accuracy = evaluation_results[1]
val_mean_iou = evaluation_results[2]
val_dice_coefficient = evaluation_results[3]

print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Validation Mean IoU: {val_mean_iou}")
print(f"Validation Dice Coefficient: {val_dice_coefficient}")

test_results = model.evaluate(tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size))
test_loss = test_results[0]
test_accuracy = test_results[1]
test_mean_iou = test_results[2]
test_dice_coefficient = test_results[3]

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Mean IoU: {test_mean_iou}")
print(f"Test Dice Coefficient: {test_dice_coefficient}")

predictions = model.predict(X_test)
post_processed_predictions = np.array([post_process(pred) for pred in predictions])
y_pred_binary = (post_processed_predictions > 0.5).astype(int)


# Function for accuracy calculation
def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


accuracies = [calculate_accuracy(y_true, y_pred) for y_true, y_pred in zip(y_test_binary, y_pred_binary)]
mean_accuracy = np.mean(accuracies)
precision, recall, f1, _ = precision_recall_fscore_support(y_test_binary.flatten(), y_pred_binary.flatten(),
                                                           average='binary')
jaccard = jaccard_score(y_test_binary.flatten(), y_pred_binary.flatten(), average='binary')
auc = roc_auc_score(y_test_binary.flatten(), predictions.flatten())
cm = confusion_matrix(y_test_binary.flatten(), y_pred_binary.flatten())

# Mean Dice Coefficeint  calculation
dice_coefficients = []
for i in range(len(y_test_binary)):
    dice_coeff = (2. * np.sum(y_test_binary[i] * y_pred_binary[i])) / (
            np.sum(y_test_binary[i]) + np.sum(y_pred_binary[i]) + 1e-6)
    dice_coefficients.append(dice_coeff)
mean_dice_coefficient = np.mean(dice_coefficients)

# Specificity Calculation
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print(f"Precision: {precision}")
print(f"Recall (Sensitivity): {recall}")
print(f"Specificity: {specificity}")
print(f"F1 Score: {f1}")
print(f"Jaccard Index: {jaccard}")
print(f"AUC: {auc}")
print(f"Mean Dice Coefficient: {mean_dice_coefficient}")
print(f"Mean Accuracy: {mean_accuracy}")
print(f"Confusion Matrix:\n{cm}")

# Visualization - Histogram of Predictions
plt.figure(figsize=(10, 5))
plt.hist(predictions.flatten(), bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Predictions')
plt.xlabel('Prediction Values')
plt.ylabel('Frequency')
plt.show()

# Visualization  for Predicted patches  compare with ground truth and  feature type patches
num_images_to_display = min(len(X_test), 20)
num_cols = 4
num_rows = (num_images_to_display + num_cols - 1) // num_cols
plt.figure(figsize=(20, num_rows * 10))
for i in range(num_images_to_display):
    plt.subplot(num_rows, num_cols * 3, i * 3 + 1)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.title('Original Image', fontsize=9)
    plt.axis('off')

    plt.subplot(num_rows, num_cols * 3, i * 3 + 2)
    plt.imshow(y_test[i].squeeze(), cmap='gray')
    plt.title('Ground Truth', fontsize=9)
    plt.axis('off')

    plt.subplot(num_rows, num_cols * 3, i * 3 + 3)
    prediction_image = post_processed_predictions[i].squeeze()
    plt.imshow(prediction_image, cmap='gray')
    plt.title('Prediction', fontsize=9)
    plt.axis('off')

plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
plt.show()

# Visualization for AUC-ROC curve
fpr, tpr, _ = roc_curve(y_test_binary.flatten(), predictions.flatten())
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# visualization for Confusion Matrix
plt.figure(figsize=(8, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Background', 'Vessel'])
plt.yticks(tick_marks, ['Background', 'Vessel'], rotation=90)

thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


def display_full_image_comparisons_with_highlight(test_images, test_manuals, predictions, accuracies):
    num_images = len(test_images)
    num_cols = 4
    num_rows = (num_images + num_cols - 1) // num_cols

    plt.figure(figsize=(20, num_rows * 10))

    for i in range(num_images):
        original_image = test_images[i][0].squeeze()
        ground_truth = test_manuals[i][0].squeeze()
        prediction_image = predictions[i].squeeze()
        accuracy = accuracies[i]

        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

        difference_mask = (ground_truth > 0.5) & (prediction_image <= 0.5)
        color_original = cv2.cvtColor((original_image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        color_original[difference_mask] = [255, 0, 0]

        plt.subplot(num_rows, num_cols * 3, i * 3 + 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Org_Img', fontsize=9)
        plt.axis('off')

        plt.subplot(num_rows, num_cols * 3, i * 3 + 2)
        plt.imshow(ground_truth, cmap='gray')
        plt.title('Grd_Truth', fontsize=9)
        plt.axis('off')

        plt.subplot(num_rows, num_cols * 3, i * 3 + 3)
        plt.imshow(color_original)
        plt.title(f'Prd vs GT\nAccu: {accuracy:.2f}', fontsize=9)
        plt.axis('off')

    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
    plt.show()


def predict_full_images(test_images, patch_size, model, feature_type, target_size):
    predicted_full_images = []
    for img, _ in test_images:
        original_size = img.shape[:2]
        if original_size != target_size:
            img = cv2.resize(img, target_size)

        patches, locations = extract_multiple_patches_with_locations(img, patch_size, num_patches, seed)
        patches = [extract_feature(patch, feature_type) for patch in patches]
        patches = np.array(patches)[..., np.newaxis]
        predictions = model.predict(patches)
        predictions = [post_process(pred) for pred in predictions]

        full_image_prediction = np.zeros(img.shape[:2])
        for (x, y), pred in zip(locations, predictions):
            full_image_prediction[y:y + patch_size, x:x + patch_size] = np.maximum(
                full_image_prediction[y:y + patch_size, x:x + patch_size], pred.squeeze())

        if original_size != target_size:
            full_image_prediction = cv2.resize(full_image_prediction, original_size)

        predicted_full_images.append(full_image_prediction)

    return predicted_full_images


# Load the saved model for testing
model = tf.keras.models.load_model(model_save_path, custom_objects={'combined_loss': combined_loss,
                                                                    'dice_coefficient': dice_coefficient})
model.compile(optimizer=Adam(learning_rate=1e-4), loss=combined_loss,
              metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), dice_coefficient])

predicted_full_images = predict_full_images(test_images, patch_size, model, feature_type, target_size)
display_full_image_comparisons_with_highlight(test_images, test_manual1, predicted_full_images, accuracies)
