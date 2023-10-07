# The code generates the rest of the images from the original source images.
import cv2 as cv


# This class contains the original RGB images named "Subject 1", and "Subject".
class imgList:
    image1 = "Subject 1.jpg"
    image2 = "Subject 2.jpg"
    message = "Simple Image Processing\n"
    message2 = "Converted to gray scale."


# This class contains the cascading arguments required for faces and eyes detection
class cascading:
    face_Cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    eyes_Cascade = cv.CascadeClassifier("haarcascade_eye.xml")


# This class loads and iterates through the images to convert them into gray scale.
class imgLoad:
    images1 = [imgList.image1, imgList.image2]

    for imgs in images1:
        source = cv.imread(imgs, cv.IMREAD_COLOR)
        cv.imshow("Original_MugShots", source)
        cv.waitKey(0)
        cv.destroyAllWindows()

        if imgs == imgList.image1:
            mug_Gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
            cv.imshow("Gray_Mugshot1", mug_Gray)
            cv.imwrite("Gray_Mugshot1.jpg", mug_Gray)
            cv.waitKey(0)
            cv.destroyAllWindows()

        if imgs >= imgList.image2:
            mug_Gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
            cv.imshow("Gray_Mugshot2", mug_Gray)
            cv.imwrite("Gray_Mugshot2.jpg", mug_Gray)
            cv.waitKey(0)
            cv.destroyAllWindows()

    print(imgList.message, imgList.message2)


# This function also iterates through the gray scale images to detect the faces and crop them into four separate images
# forming a bounding box around each face creating a new image of each respective individual.
def imgProcessor():
    mugS1 = "Gray_Mugshot1.jpg"
    mugS2 = "Gray_Mugshot2.jpg"

    x_con1 = 1
    x_con2 = 1

    process1 = [mugS1, mugS2]

    for mugs in process1:
        source1 = cv.imread(mugs)
        print(type(source1))
        imgs_Shapes = "Dimensions of the images", source1.shape
        print(imgs_Shapes)

        face_Processor = cascading.face_Cascade
        faces = face_Processor.detectMultiScale(source1, 1.1, 2)

        if mugs == mugS1:
            for (x, y, w, h) in faces:
                cv.rectangle(source1, (x, y), (x + w, y + h), (0, 0, 255), 2)
                faces = source1[y:y + h, x:x + w]
                cv.imshow("face", faces)
                cv.imwrite("face_Detection_1.jpg", faces)
                cv.waitKey(0)
                while x_con1 < 2:
                    cv.imwrite("face_Detection_2.jpg", faces)
                    x_con1 += 1
                    cv.waitKey(0)
                    cv.destroyAllWindows()

        if mugs >= mugS2:
            for (x, y, w, h) in faces:
                cv.rectangle(source1, (x, y), (x + w, y + h), (0, 0, 255), 2)
                faces = source1[y:y + h, x:x + w]
                cv.imshow("face", faces)
                cv.imwrite("face_Detection_3.jpg", faces)
                cv.waitKey(0)
                while x_con2 < 2:
                    cv.imwrite("face_Detection_4.jpg", faces)
                    x_con2 += 1
                    cv.waitKey(0)
                    cv.destroyAllWindows()


# this other function process iterates through each image to resize, rotate, and detects faces and eyes
# producing the final images per the assignment instructions.
def face_and_eyes_Detection():
    subJ1 = "face_Detection_1.jpg"
    subJ2 = "face_Detection_2.jpg"
    subJ3 = "face_Detection_3.jpg"
    subJ4 = "face_Detection_4.jpg"

    mugSHTs = [subJ1, subJ2, subJ3, subJ4]

    for mugList in mugSHTs:
        source2 = cv.imread(mugList)
        size = 180.0 / source2.shape[1]
        dim = (180, int(source2.shape[0] * size))
        resize = cv.resize(source2, dim, interpolation=cv.INTER_AREA)

        cv.imshow("resized", resize)
        cv.waitKey(0)
        cv.destroyAllWindows()

        (rows, cols) = resize.shape[:2]
        mtX = cv.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
        rotated = cv.warpAffine(resize, mtX, (cols, rows))

        face_Det = cascading.face_Cascade
        face_Det2 = face_Det.detectMultiScale(rotated, 1.3, 5)

        for (x, y, w, h) in face_Det2:
            cv.rectangle(rotated, (x, y), (x + w, y + h), (255, 0, 0), 2)

        eyes_Det = cascading.eyes_Cascade
        eyes = eyes_Det.detectMultiScale(rotated)

        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(rotated, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            cv.imshow("rotated", rotated)
            cv.waitKey(0)
            cv.destroyAllWindows()

        if mugList == subJ1:
            cv.imwrite("Final_MugShots1.jpg", rotated)
        if mugList == subJ2:
            cv.imwrite("Final_MugShots2.jpg", rotated)
        if mugList == subJ3:
            cv.imwrite("Final_MugShots3.jpg", rotated)
        if mugList == subJ4:
            cv.imwrite("Final_MugShots4.jpg", rotated)


# this last function defines the main arguments used to run the entire sequence to the code.
def main():
    imgLoad()
    imgProcessor()
    face_and_eyes_Detection()


if __name__ == "__main__":
    main()
