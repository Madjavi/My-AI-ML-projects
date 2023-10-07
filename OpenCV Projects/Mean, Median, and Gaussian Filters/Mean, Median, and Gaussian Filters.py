import cv2 as cv
import matplotlib.pyplot as plot


# This class load the image that is going to be process
class imgProcess:
    pic = "Mod4CT1.jpg"
    loading = cv.imread(pic)
    img_ready = cv.cvtColor(loading, cv.COLOR_RGB2BGR)
    cv.imshow("Original_img", loading)
    cv.waitKey(0)
    message1 = "Option #1: Mean, Median and Gaussian Filters\n"
    print(message1)


# The stage 1 filters class contains the Mean, Median and Gaussian Filters set as 3x3 kernel.
class stage_1_Filters:
    mean = cv.medianBlur(imgProcess.img_ready, 3)
    median = cv.medianBlur(imgProcess.img_ready, 3)
    gaussian1 = cv.GaussianBlur(imgProcess.img_ready, (3, 3), 1)
    gaussian2 = cv.GaussianBlur(imgProcess.img_ready, (3, 3), 2)


# The stage 1 filters class contains the Mean, Median and Gaussian Filters set as 5x5 kernel.
class stage_2_Filters:
    mean = cv.medianBlur(imgProcess.img_ready, 5)
    median = cv.medianBlur(imgProcess.img_ready, 5)
    gaussian1 = cv.GaussianBlur(imgProcess.img_ready, (5, 5), 1)
    gaussian2 = cv.GaussianBlur(imgProcess.img_ready, (5, 5), 2)


# The stage 1 filters class contains the Mean, Median and Gaussian Filters set as 7x7 kernel.
class stage_3_Filters:
    mean = cv.medianBlur(imgProcess.img_ready, 7)
    median = cv.medianBlur(imgProcess.img_ready, 7)
    gaussian1 = cv.GaussianBlur(imgProcess.img_ready, (7, 7), 1)
    gaussian2 = cv.GaussianBlur(imgProcess.img_ready, (7, 7), 2)


# This function deploys the plot of the different types of filtering stages for easy analysis.
def imgPlottingProcess():
    fig, axes = plot.subplots(3, 4)
    fig.suptitle("Mean, Median and Gaussian Filters")

    axes[0, 0].set_title("Mean Filter")
    axes[0, 1].set_title("Median Filter")
    axes[0, 2].set_title("Gaussian w/Sigma = 1")
    axes[0, 3].set_title("Gaussian w/Sigma = 2")
    axes[0, 0].set_ylabel("3 X 3 Kernel")
    axes[0, 0].imshow(stage_1_Filters.mean)

    axes[0, 1].imshow(stage_1_Filters.median)
    axes[0, 2].imshow(stage_1_Filters.gaussian1)
    axes[0, 3].imshow(stage_1_Filters.gaussian2)

    axes[1, 0].set_ylabel("5 X 5 Kernel")
    axes[1, 0].imshow(stage_2_Filters.mean)

    axes[1, 1].imshow(stage_2_Filters.median)
    axes[1, 2].imshow(stage_2_Filters.gaussian1)
    axes[1, 3].imshow(stage_2_Filters.gaussian2)

    axes[2, 0].set_ylabel("7 X 7 Kernel")
    axes[2, 0].imshow(stage_3_Filters.mean)

    axes[2, 1].imshow(stage_3_Filters.median)
    axes[2, 2].imshow(stage_3_Filters.gaussian1)
    axes[2, 3].imshow(stage_3_Filters.gaussian2)

    plot.show()


# This function is defined as main in order to execute the sequence of the code,
def main():
    imgPlottingProcess()


if __name__ == "__main__":
    main()
