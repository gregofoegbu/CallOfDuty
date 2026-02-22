import matplotlib.pyplot as plt
import cv2


def getimgregion(img, bbox): 
    h, w, c = img.shape
    return img[
        int(bbox['y1']*h):int(bbox["y2"]*h),
        int(bbox['x1']*w):int(bbox["x2"]*w)
    ]

def show_processing_stages(region, title="Region"):

    plt.figure(num=title, figsize=(15,8))

    # 1 Original
    plt.subplot(2,3,1)
    plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    # 2 Grayscale
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    plt.subplot(2,3,2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    # 3 CLAHE contrast boost
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    plt.subplot(2,3,3)
    plt.imshow(clahe_img, cmap="gray")
    plt.title("CLAHE")
    plt.axis("off")

    # 4 Gaussian blur
    blur = cv2.GaussianBlur(clahe_img, (3,3), 0)
    plt.subplot(2,3,4)
    plt.imshow(blur, cmap="gray")
    plt.title("Blur")
    plt.axis("off")

    # 5 Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    plt.subplot(2,3,5)
    plt.imshow(thresh, cmap="gray")
    plt.title("Adaptive Threshold")
    plt.axis("off")

    # 6 Morph close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    plt.subplot(2,3,6)
    plt.imshow(morph, cmap="gray")
    plt.title("Morph Close")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    return morph


def get_clahe_img(region):
    # 2 Grayscale
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # 3 CLAHE contrast boost
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)

    return clahe_img
