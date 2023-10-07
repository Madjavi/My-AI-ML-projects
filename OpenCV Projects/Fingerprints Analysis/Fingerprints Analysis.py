import cv2 as cv
import matplotlib.pyplot as plot
import numpy as npy


# Contains heather and assignment instructions.
class messages:
    message1 = "Option #1: Morphology Operations for Fingerprint Enhancement\n"
    message2 = "Acquire an image of a latent fingerprint. In OpenCV, write algorithms to process the image using " \
               "morphological operations (dilation, erosion, opening, and closing)."
    print(message1)
    print(message2)


# This class load the image that is going to be process.
class imgProcess:
    pic = "fingerprint1.jpg"
    loading = cv.imread(pic)
    img_ready = cv.cvtColor(loading, cv.COLOR_RGB2BGR)


# This class binarize the image, and sets the kernel parameters.
class imgProcess2:
    img1 = imgProcess.img_ready
    (thresh, bin_img) = cv.threshold(img1, 127, 255, cv.THRESH_BINARY)
    kernel_proc = npy.ones((5, 5), npy.uint8)


# This class have the different morphological operations to be performed on the fingerprints image.
class morphology_Operations:
    img_erosion = cv.erode(imgProcess2.bin_img, imgProcess2.kernel_proc, iterations=1)
    img_dilation = cv.dilate(imgProcess2.bin_img, imgProcess2.kernel_proc, iterations=1)
    img_opening = cv.morphologyEx(imgProcess2.bin_img, cv.MORPH_OPEN, imgProcess2.kernel_proc, iterations=1)
    img_closing = cv.morphologyEx(imgProcess2.bin_img, cv.MORPH_CLOSE, imgProcess2.kernel_proc, iterations=1)


# This function shows all the morphological operations in the fingerprints for a thorough analysis comparative analysis.
def imgPlot():
    fig, axes = plot.subplots(2, 4)
    fig.suptitle("Morphology Operations for Fingerprint Enhancement")

    axes[0, 0].set_title("Dilation")
    axes[0, 1].set_title("Erosion")
    axes[0, 2].set_title("Opening")
    axes[0, 3].set_title("Closing")
    axes[0, 0].set_ylabel("Original Fingerprint")
    axes[0, 0].imshow(imgProcess.img_ready)

    axes[0, 1].imshow(imgProcess.img_ready)
    axes[0, 2].imshow(imgProcess.img_ready)
    axes[0, 3].imshow(imgProcess.img_ready)

    axes[1, 0].set_ylabel("Morphological Enhanced")
    axes[1, 0].imshow(morphology_Operations.img_erosion)

    axes[1, 1].imshow(morphology_Operations.img_dilation)
    axes[1, 2].imshow(morphology_Operations.img_opening)
    axes[1, 3].imshow(morphology_Operations.img_closing)

    plot.show()


# This function is defined as main in order to execute the sequence of the code.
def main():
    imgPlot()


if __name__ == "__main__":
    main()
