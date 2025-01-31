import tensorflow as tf
import numpy as np
import random
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import os

# Image dimensions
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TEST_PATH = 'stage1_test/'

# Load test data
test_ids = next(os.walk(TEST_PATH))[1]
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []

# Resizing test images
print('Resizing test images')
for n, id_ in enumerate(test_ids):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Test images resized')

# Load the trained model
model = tf.keras.models.load_model('/home/biswajit/Documents/term_peoject/text_to_vid/U-net/model_for_nuclei.keras')

# Make predictions
preds_test = model.predict(X_test, verbose=1)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Display the test results
for i in range(len(X_test)):
    plt.subplot(1, 2, 1)
    imshow(X_test[i])
    plt.title('Test Image')

    plt.subplot(1, 2, 2)
    imshow(np.squeeze(preds_test_t[i]))
    plt.title('Predicted Mask')

    plt.show()

