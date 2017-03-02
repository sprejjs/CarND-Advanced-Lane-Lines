import matplotlib
matplotlib.use('TkAgg')

from tkinter import Tk, Label, Scale, HORIZONTAL, Button
from PIL import Image
from PIL.ImageTk import PhotoImage
import glob
import matplotlib.image as mpimg
import cv2

from pipeline import process_image

images = glob.glob('video_frames/frame*.jpg')
index = 0

sobel_thresh_min = 0
sobel_thresh_max = 255
s_thresh_min = 0
s_thresh_max = 255


def change_sobel_thresh_min(val):
    global sobel_thresh_min
    sobel_thresh_min = int(val)
    update_gui()


def change_sobel_thresh_max(val):
    global sobel_thresh_max
    sobel_thresh_max = int(val)
    update_gui()


def change_s_thresh_min(val):
    global s_thresh_min
    s_thresh_min = int(val)
    update_gui()


def change_s_thresh_max(val):
    global s_thresh_max
    s_thresh_max = int(val)
    update_gui()


def force_sliding():
    update_gui(force_slide=True)


def update_gui(force_slide=False):
    global index
    original = mpimg.imread(images[index])
    undisorted, thresholded, transformed, detected_lines, projected, left_curverad, right_curverad, position_of_the_car = process_image(original, sobel_thresh_min=sobel_thresh_min, sobel_thresh_max=sobel_thresh_max, s_thresh_min=s_thresh_min, s_thresh_max=s_thresh_max, force_sliding=force_slide)

    original = Image.fromarray(resize_image(original))
    undisorted = Image.fromarray(resize_image(undisorted))
    thresholded = Image.fromarray(resize_image(thresholded))
    transformed = Image.fromarray(resize_image(transformed))
    detected_lines = Image.fromarray(resize_image(detected_lines))
    projected = Image.fromarray(resize_image(projected))

    originalPhotoImage = PhotoImage(original)
    undistortedPhotoImage = PhotoImage(undisorted)
    binaryPhotoImage = PhotoImage(thresholded)
    birdViewPhotoImage = PhotoImage(transformed)
    detectedLinesPhotoImage = PhotoImage(detected_lines)
    projectedPhotoImage = PhotoImage(projected)

    window.originalPhotoImage.configure(image=originalPhotoImage)
    window.originalPhotoImage.image = originalPhotoImage

    window.undistortedPhotoImage.configure(image=undistortedPhotoImage)
    window.undistortedPhotoImage.image = undistortedPhotoImage

    window.binaryPhotoImage.configure(image=binaryPhotoImage)
    window.binaryPhotoImage.image = binaryPhotoImage

    window.birdViewPhotoImage.configure(image=birdViewPhotoImage)
    window.birdViewPhotoImage.image = birdViewPhotoImage

    window.detectedLinesPhotoImage.configure(image=detectedLinesPhotoImage)
    window.detectedLinesPhotoImage.image = detectedLinesPhotoImage

    window.projectedPhotoImage.configure(image=projectedPhotoImage)
    window.projectedPhotoImage.image = projectedPhotoImage


def resize_image(image):
    return cv2.resize(image, (0, 0), fx=0.25, fy=0.25)


def left_button_pressed(e):
    global index

    if (index > 0):
        index -= 1
        update_gui()


def right_button_pressed(e):
    global index
    global images

    if (index < len(images)):
        index += 1
        update_gui()

window = Tk()

window.title("Pre-processing configuration")
window.configure(background='grey')

window.bind("<Left>", left_button_pressed)
window.bind("<Right>", right_button_pressed)

Label(text="Original Image").grid(row=0, column=0, sticky='we')
Label(text="Undistorted Image").grid(row=0, column=1, sticky='we')
Label(text="Binary Image").grid(row=0, column=2, sticky='we')

Label(text="Bird View").grid(row=2, column=0, sticky='we')
Label(text="Found lanes").grid(row=2, column=1, sticky='we')
Label(text="Output").grid(row=2, column=2, sticky='we')

originalPhotoImage = PhotoImage(Image.fromarray(resize_image(mpimg.imread(images[0]))))
window.originalPhotoImage = Label(window, image=originalPhotoImage)
window.originalPhotoImage.grid(row=1, column=0)

undistortedPhotoImage = PhotoImage(Image.fromarray(resize_image(mpimg.imread(images[0]))))
window.undistortedPhotoImage = Label(window, image=undistortedPhotoImage)
window.undistortedPhotoImage.grid(row=1, column=1)

binaryPhotoImage = PhotoImage(Image.fromarray(resize_image(mpimg.imread(images[0]))))
window.binaryPhotoImage = Label(window, image=binaryPhotoImage)
window.binaryPhotoImage.grid(row=1, column=2)

birdViewPhotoImage = PhotoImage(Image.fromarray(resize_image(mpimg.imread(images[0]))))
window.birdViewPhotoImage = Label(window, image=birdViewPhotoImage)
window.birdViewPhotoImage.grid(row=3, column=0)

detectedLinesPhotoImage = PhotoImage(Image.fromarray(resize_image(mpimg.imread(images[0]))))
window.detectedLinesPhotoImage = Label(window, image=detectedLinesPhotoImage)
window.detectedLinesPhotoImage.grid(row=3, column=1)

projectedPhotoImage = PhotoImage(Image.fromarray(resize_image(mpimg.imread(images[0]))))
window.projectedPhotoImage = Label(window, image= projectedPhotoImage)
window.projectedPhotoImage.grid(row=3, column=2)

window.sobelLowThreshold = Scale(window, from_=0, to=255, orient=HORIZONTAL,
                                 command=change_sobel_thresh_min, label="Low Sobel Threshold")
window.sobelLowThreshold.grid(row=4, column=0, columnspan=3, sticky='we')

window.sobelHighThreshold = Scale(window, from_=0, to=255, orient=HORIZONTAL,
                                 command=change_sobel_thresh_min, label="High Sobel Threshold")
window.sobelHighThreshold.grid(row=5, column=0, columnspan=3, sticky='we')
window.sobelHighThreshold.set(255)

window.sLowThreshold = Scale(window, from_=0, to=255, orient=HORIZONTAL,
                                 command=change_s_thresh_min, label="Low S Threshold")
window.sLowThreshold.grid(row=6, column=0, columnspan=3, sticky='we')

window.sHighThreshold = Scale(window, from_=0, to=255, orient=HORIZONTAL,
                                 command=change_s_thresh_max, label="High S Threshold")
window.sHighThreshold.grid(row=7, column=0, columnspan=3, sticky='we')
window.sHighThreshold.set(255)

Button(window, text="Run Sliding Windows", command=force_sliding).grid(row=8, column=0, columnspan=3, sticky='we')

update_gui()

window.mainloop()