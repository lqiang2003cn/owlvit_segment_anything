import cv2 as cv


# initialize the camera
# If you have multiple camera connected with
# current device, assign a value in cam_port
# variable according to that
cam_port = 0
cam = cv.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open camera")
    exit()
#
# # reading the input using the camera
# result, image = cam.read()
#
# # If image will detected without any error,
# # show result
# if result:
#
#     # showing result, it take frame name and image
#     # output
#     cv.imshow("GeeksForGeeks", image)
#
#     # saving image in local storage
#     cv.imwrite("GeeksForGeeks.png", image)
#
#     # If keyboard interrupt occurs, destroy image
#     # window
#     cv.waitKey(0)
#     cv.destroyWindow("GeeksForGeeks")
#
# # If captured image is corrupted, moving to else part
# else:
#     print("No image detected. Please! try again")