# The goal of this project is to write algorithms for license plate detection and license plate character
# recognition.  Select three color images from the internet that meet the following requirements:
#
#     1. Two images containing vehicles with Russian license plates and one image of vehicles with a non-Russian plate.
#     2. All images should include the entire vehicle and not just the license plate.
#     3. At least one image with Russian plates should display the license plate far away.
#     4. At least one image should include multiple vehicles.
#     5. All images should vary in light illumination and color intensity.

# First, using the appropriate trained cascade classifier
# Links to an external site., write one algorithm to detect the Russian license plate in the gray scaled versions of
# the original images.  Put a red boundary box around the detected plate in the image in order to see what region the
# classifier deemed as a license plate.  If expected results are not achieved on the unprocessed images,
# apply processing steps before implementing the classifier for optimal results.
#
# After the license plates have been successfully detected, you will want to process only the extracted plate region
# before applying character recognition on it.  Although the license plate number classifier Links to an external
# site. is fairly accurate, it is important that all license plates are rotated and scaled so that they are
# horizontally aligned. If expected results are not achieved, implement more image processing for optimal character
# recognition.

import cv2 as cv
import matplotlib.pyplot as plot


# The class below contains the Russian & non-Russian vehicles with license plates in the traffic for processing.
# All the cv2 parameter have been set for reading the images.
class imgList:
    traffic1 = "Russian_2.jpg"
    source1 = cv.imread(traffic1)
    colorSrc = cv.cvtColor(source1, cv.COLOR_RGB2BGR)
    colorSrc2 = cv.cvtColor(source1, cv.IMREAD_COLOR)

    traffic2 = "other_plates.jpg"
    source2 = cv.imread(traffic2, cv.IMREAD_COLOR)
    colorSrc3 = cv.cvtColor(source2, cv.COLOR_RGB2BGR)

    traffic3 = "RUS.jpg"
    source3 = cv.imread(traffic3, cv.IMREAD_COLOR)
    colorSrc4 = cv.cvtColor(source3, cv.COLOR_RGB2BGR)

    message = "Option #1: License Plate Detection and Faked Content\n"
    message2 = "Converted to gray scale."


# This class contains the cascade classifiers required for plate & character detection.
class cascading:
    license_plate = cv.CascadeClassifier('haarcascade_licence_plate_rus_16stages.xml')
    license_number = cv.CascadeClassifier('haarcascade_russian_plate_number.xml')


# This class loads the images and converts them from RGB to gray for detection.
class imgLoad:
    images1 = [imgList.traffic1, imgList.traffic2, imgList.traffic3]

    for imgs in images1:
        source = cv.imread(imgs)
        img_ready = cv.cvtColor(source, cv.COLOR_RGB2BGR)

        if imgs == imgList.traffic1:
            traffic_Gray1 = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
            cv.imwrite("Gray_Rus_Traffic.jpg", traffic_Gray1)

        if imgs >= imgList.traffic2:
            traffic_Gray2 = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
            cv.imwrite("Gray_US_Traffic.jpg", traffic_Gray2)

        if imgs <= imgList.traffic3:
            traffic_Gray3 = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
            cv.imwrite("Gray_Rus2_Traffic.jpg", traffic_Gray3)

    print(imgList.message, imgList.message2)


# The function plots the images for comparison.
def imgPlot():
    fig, axes = plot.subplots(2, 3)
    fig.suptitle("Color to Gray Conversion For Detection")

    axes[0, 0].set_title("Russian Traffic")
    axes[0, 1].set_title("US Traffic")
    axes[0, 2].set_title("Car w/RU plate")

    axes[0, 0].set_ylabel("Color")
    axes[0, 0].imshow(imgList.colorSrc)

    axes[0, 1].imshow(imgList.colorSrc3)
    axes[0, 2].imshow(imgList.colorSrc4)

    axes[1, 0].set_ylabel("Gray Scale")
    axes[1, 0].imshow(imgLoad.traffic_Gray1, cmap='gray')

    axes[1, 1].imshow(imgLoad.traffic_Gray2, cmap='gray')
    axes[1, 2].imshow(imgLoad.traffic_Gray3, cmap='gray')

    plot.show()


# This class reloads the converted gray images for the detection process.
class gray_imgs:
    g_traffic1 = "Gray_Rus_Traffic.jpg"
    g_traffic2 = "Gray_US_Traffic.jpg"
    g_traffic3 = "Gray_Rus2_Traffic.jpg"


# The function detects and extract the detected Russian license plates for character recognition.
def plate_detector1():
    cam = 1

    gray_1 = gray_imgs.g_traffic1

    source3 = cv.imread(gray_1)
    size = 1900.0 / source3.shape[1]
    dim = (1900, int(source3.shape[0] * size))
    resize = cv.resize(source3, dim, interpolation=cv.INTER_AREA)

    num_Plate = cascading.license_plate
    num_Plate_Det1 = num_Plate.detectMultiScale(resize, scaleFactor=1.05, minNeighbors=5)
    print('Number of detected license plates:', len(num_Plate_Det1))

    for (x, y, w, h) in num_Plate_Det1:
        cv.rectangle(resize, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(resize, "RU_License Plate", (x - 20, y - 10),
                   cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

        plate_num1 = resize[y:y + h, x:x + w]
        cv.imshow("RU_License plate", plate_num1)
        cv.imwrite("RU_License plate_1.jpg", plate_num1)
        cv.imshow("Plate detected", resize)
        cv.waitKey(0)
        cv.destroyAllWindows()

        while cam < 2:
            cv.imwrite("RU_License plate_2.jpg", plate_num1)
            cam += 1


# The function detects and extract the detected non-Russian license plates for character recognition.
def plate_detector2():
    cam_1 = 1
    cam_2 = 2
    cam_3 = 3

    gray_2 = gray_imgs.g_traffic2

    source2 = cv.imread(gray_2)
    size = 1900.0 / source2.shape[1]
    dim = (1900, int(source2.shape[0] * size))
    resize_2 = cv.resize(source2, dim, interpolation=cv.INTER_AREA)

    num_Plate_2 = cascading.license_plate
    num_Plate_Det_2 = num_Plate_2.detectMultiScale(resize_2, scaleFactor=1.05, minNeighbors=2)
    print('Number of detected license plates:', len(num_Plate_Det_2))

    for (x, y, w, h) in num_Plate_Det_2:
        cv.rectangle(resize_2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(resize_2, "US_License Plate", (x - 20, y - 10),
                   cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

        plate_num2 = resize_2[y:y + h, x:x + w]
        cv.imshow("US_License plate", plate_num2)
        while cam_1 < 2:
            cv.imwrite("US_License plate_1.jpg", plate_num2)
            cam_1 += 1
            break

        while cam_2 < 4:
            cv.imwrite("US_License plate_2.jpg", plate_num2)
            cam_2 += 1
            break

        while cam_3 < 8:
            cv.imwrite("US_License plate_3.jpg", plate_num2)
            cam_3 += 1
            break

        cv.imshow("Plate detected", resize_2)
        cv.waitKey(0)
        cv.destroyAllWindows()


def plate_detector3():
    cam_4 = 1

    gray_3 = gray_imgs.g_traffic3

    source3 = cv.imread(gray_3)
    size = 1900.0 / source3.shape[1]
    dim = (1900, int(source3.shape[0] * size))
    resize_3 = cv.resize(source3, dim, interpolation=cv.INTER_AREA)

    Plate = cascading.license_plate
    Plate_Det = Plate.detectMultiScale(resize_3, scaleFactor=1.05, minNeighbors=5)
    print('Number of detected license plates:', len(Plate_Det))

    for (x, y, w, h) in Plate_Det:
        cv.rectangle(resize_3, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(resize_3, "RU_License Plate", (x - 20, y - 10),
                   cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

        plate_num_D = resize_3[y:y + h, x:x + w]
        cv.imshow("RU_plate_SINGLE_V", plate_num_D)
        cv.imwrite("RU_plate_SINGLE_1.jpg", plate_num_D)
        cv.imshow("Plate detected", resize_3)
        cv.waitKey(0)
        cv.destroyAllWindows()

        while cam_4 < 2:
            cv.imwrite("RU_plate_SINGLE_2.jpg", plate_num_D)
            cam_4 += 1


# This class reloads all the detected and extracted license plates for character recognition.
class license_Plates:
    RU_plate1 = "RU_License plate_1.jpg"
    RU_plate2 = "RU_License plate_2.jpg"

    RU_plate3 = "RU_plate_SINGLE_1.jpg"
    RU_plate4 = "RU_plate_SINGLE_2.jpg"

    US_plate1 = "US_License plate_1.jpg"
    US_plate2 = "US_License plate_2.jpg"
    US_plate3 = "US_License plate_3.jpg"


# This function detects the characters on the Russian license plates.
def plate_char_reco1():
    sensor_1 = 1
    sensor_2 = 2
    sensor_3 = 3

    plates_RU = [license_Plates.RU_plate1, license_Plates.RU_plate2, license_Plates.RU_plate3, license_Plates.RU_plate4]

    for sensor in plates_RU:
        source_4 = cv.imread(sensor)
        size = 400.0 / source_4.shape[1]
        dim = (400, int(source_4.shape[0] * size))
        resize_RU = cv.resize(source_4, dim, interpolation=cv.INTER_AREA)

        resize_RU_flip1 = cv.flip(resize_RU, 0)
        resize_RU_flip2 = cv.flip(resize_RU, 1)
        resize_RU_flip3 = cv.flip(resize_RU, -1)

        resize_RU_rotate1 = cv.rotate(resize_RU, cv.ROTATE_90_CLOCKWISE)
        resize_RU_rotate2 = cv.rotate(resize_RU, cv.ROTATE_90_COUNTERCLOCKWISE)
        resize_RU_rotate3 = cv.rotate(resize_RU, cv.ROTATE_180)

        rotations = [resize_RU,
                     resize_RU_flip1,
                     resize_RU_flip2,
                     resize_RU_flip3,
                     resize_RU_rotate1,
                     resize_RU_rotate2,
                     resize_RU_rotate3]

        for aligned in rotations:
            num_Plate_3 = cascading.license_number
            num_Plate_Det_3 = num_Plate_3.detectMultiScale(aligned, scaleFactor=1.02, minNeighbors=2)
            print('Number of detected Characters:', len(num_Plate_Det_3))

            for (x, y, w, h) in num_Plate_Det_3:
                cv.rectangle(aligned, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv.putText(aligned, "RU_Charters", (x - 20, y - 10),
                           cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

                plate_num_3 = aligned[y:y + h, x:x + w]
                cv.imshow("Characters detected", aligned)

                while sensor_1 < 6:
                    cv.imwrite("RU_Char_1.jpg", plate_num_3)
                    sensor_1 += 1
                    break

                while sensor_2 < 6:
                    cv.imwrite("RU_Char_2.jpg", plate_num_3)
                    sensor_2 += 1
                    break

                while sensor_3 < 6:
                    cv.imwrite("RU_Char_3.jpg", plate_num_3)
                    sensor_3 += 1
                    break

                cv.imshow("RU_plate_Char", plate_num_3)
                cv.imwrite("RU_Char_4.jpg", plate_num_3)
                cv.waitKey(0)
                cv.destroyAllWindows()


# This function detects the characters on the non-Russian license plates.
def plate_char_reco2():
    sensor_1 = 1
    sensor_2 = 2
    sensor_3 = 3

    plates_US = [license_Plates.US_plate1,
                 license_Plates.US_plate2,
                 license_Plates.US_plate3]

    for sensors in plates_US:
        source_4 = cv.imread(sensors)
        size = 380.0 / source_4.shape[1]
        dim = (380, int(source_4.shape[0] * size))
        resize_US = cv.resize(source_4, dim, interpolation=cv.INTER_AREA)

        resize_US_flip1 = cv.flip(resize_US, 0)
        resize_US_flip2 = cv.flip(resize_US, 1)
        resize_US_flip3 = cv.flip(resize_US, -1)

        resize_US_rotate1 = cv.rotate(resize_US, cv.ROTATE_90_CLOCKWISE)
        resize_US_rotate2 = cv.rotate(resize_US, cv.ROTATE_90_COUNTERCLOCKWISE)
        resize_US_rotate3 = cv.rotate(resize_US, cv.ROTATE_180)

        rotations = [resize_US_flip1, resize_US_flip2, resize_US_flip3, resize_US_rotate1, resize_US_rotate2,
                     resize_US_rotate3, resize_US]

        for alignment in rotations:
            char_Plate_3 = cascading.license_number
            char_Plate_Det_3 = char_Plate_3.detectMultiScale(alignment, scaleFactor=1.02, minNeighbors=2)
            print('Number of detected Characters:', len(char_Plate_Det_3))

            for (x, y, w, h) in char_Plate_Det_3:
                cv.rectangle(alignment, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv.putText(alignment, "RU_Charters", (x - 20, y - 10),
                           cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

                plate_char = alignment[y:y + h, x:x + w]
                cv.imshow("Characters detected", alignment)

                while sensor_1 < 4:
                    cv.imwrite("US_Char_1.jpg", plate_char)
                    sensor_1 += 1
                    break

                while sensor_2 < 5:
                    cv.imwrite("US_Char_2.jpg", plate_char)
                    sensor_2 += 1
                    break

                while sensor_3 < 6:
                    cv.imwrite("US_Char_3.jpg", plate_char)
                    sensor_3 += 1
                    break

                cv.imshow("US_plate_Char", plate_char)
                cv.imwrite("US_Char_4.jpg", plate_char)
                cv.waitKey(0)
                cv.destroyAllWindows()


# This function runs the entire code.
def main():
    imgPlot()
    plate_detector1()
    plate_detector2()
    plate_detector3()
    plate_char_reco1()
    plate_char_reco2()


if __name__ == "__main__":
    main()
