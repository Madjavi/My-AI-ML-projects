import cv2 as cv
import matplotlib.pyplot as plot


# Contains header and assignment instructions.
class messages:
    message1 = "Option #1: Adaptive Thresholding Scheme for Simple Objects\n"
    message2 = "Find on the internet (or use a camera to take) three different types of images: an indoor scene, " \
               "outdoor scenery, and a close-up scene of a single object. Implement an adaptive thresholding scheme " \
               "to segment the images as best as you can.\n"
    print(message1)
    print(message2)


# This class load the images with their respective parameters.
class imgs_Process:
    loading = cv.imread("In doors.jpg")
    img_ready = cv.cvtColor(loading, cv.COLOR_RGB2BGR)
    img1_Pr = cv.cvtColor(img_ready, cv.COLOR_BGR2GRAY)

    loading2 = cv.imread("outdoors.jpg")
    img_ready2 = cv.cvtColor(loading2, cv.COLOR_RGB2BGR)
    img2_Pr = cv.cvtColor(img_ready2, cv.COLOR_BGR2GRAY)

    loading3 = cv.imread("clock.jpg")
    img_ready3 = cv.cvtColor(loading3, cv.COLOR_RGB2BGR)
    img3_Pr = cv.cvtColor(img_ready3, cv.COLOR_BGR2GRAY)


# This class applies the adaptive thresholding scheme to the three different images.
class Adaptive_Thresh:
    img1_thresh1 = cv.adaptiveThreshold(imgs_Process.img1_Pr,
                                        255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 199, 5)
    img1_thresh2 = cv.adaptiveThreshold(imgs_Process.img1_Pr,
                                        255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 199, 5)

    img2_thresh1 = cv.adaptiveThreshold(imgs_Process.img2_Pr,
                                        255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 199, 5)
    img2_thresh2 = cv.adaptiveThreshold(imgs_Process.img2_Pr,
                                        255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 199, 5)

    img3_thresh1 = cv.adaptiveThreshold(imgs_Process.img3_Pr,
                                        255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 199, 5)
    img3_thresh2 = cv.adaptiveThreshold(imgs_Process.img3_Pr,
                                        255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 199, 5)


# This group of functions plots each group of images with their adaptive implementation for compare.
def imgPlot():
    fig, axes = plot.subplots(2, 2)
    fig.suptitle("Adaptive Thresholding Scheme")

    axes[0, 0].set_title("Adaptive Threshold Mean")
    axes[0, 1].set_title("Adaptive Threshold Gaussian")

    axes[0, 0].set_ylabel("Indoors scene")
    axes[0, 0].imshow(imgs_Process.img_ready)

    axes[0, 1].imshow(imgs_Process.img_ready)

    axes[1, 0].set_ylabel("Adaptive Thresholding")
    axes[1, 0].imshow(Adaptive_Thresh.img1_thresh1)

    axes[1, 1].imshow(Adaptive_Thresh.img1_thresh2)

    plot.show()


def imgPlot_2():
    fig, axes = plot.subplots(2, 2)
    fig.suptitle("Adaptive Thresholding Scheme")

    axes[0, 0].set_title("Adaptive Threshold Mean")
    axes[0, 1].set_title("Adaptive Threshold Gaussian")

    axes[0, 0].set_ylabel("Outdoors scene")
    axes[0, 0].imshow(imgs_Process.img_ready2)

    axes[0, 1].imshow(imgs_Process.img_ready2)

    axes[1, 0].set_ylabel("Adaptive Thresholding")
    axes[1, 0].imshow(Adaptive_Thresh.img2_thresh1)

    axes[1, 1].imshow(Adaptive_Thresh.img2_thresh2)

    plot.show()


def imgPlot_3():
    fig, axes = plot.subplots(2, 2)
    fig.suptitle("Adaptive Thresholding Scheme")

    axes[0, 0].set_title("Adaptive Threshold Mean")
    axes[0, 1].set_title("Adaptive Threshold Gaussian")

    axes[0, 0].set_ylabel("Single Object")
    axes[0, 0].imshow(imgs_Process.img_ready3)

    axes[0, 1].imshow(imgs_Process.img_ready3)

    axes[1, 0].set_ylabel("Adaptive Thresholding")
    axes[1, 0].imshow(Adaptive_Thresh.img3_thresh1)

    axes[1, 1].imshow(Adaptive_Thresh.img3_thresh2)

    plot.show()


# This function is defined as main in order to execute the sequence of the code.
def main():
    imgPlot()
    imgPlot_2()
    imgPlot_3()


if __name__ == "__main__":
    main()
