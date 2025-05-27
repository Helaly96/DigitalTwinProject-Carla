import easyocr
import cv2
from matplotlib import pyplot as plt
from google.colab import files

# Upload image
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

img = cv2.imread(image_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

reader = easyocr.Reader(['en'])
results = reader.readtext(img)

# Print detected text
print("Detected Text:")
for (bbox, text, confidence) in results:
    print(f"{text} (Confidence: {confidence:.2f})")

#######################################
# Tesseract OCR

import cv2
import pytesseract
from PIL import Image

# Load image
image = cv2.imread('8_comp_inverted.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding (optional but helps for clarity)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# OCR config to detect digits only
custom_config = r'--oem 3 --psm 6 outputbase digits'
text = pytesseract.image_to_string(thresh, config=custom_config)

print("Detected numbers:", text)
