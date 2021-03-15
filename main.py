import cv2 as cv
import imutils

mode_init = False


def destroy_special_windows():
    cv.destroyWindow('Gaussian')
    cv.destroyWindow('Canny')
    cv.destroyWindow('Sobel')
    cv.destroyWindow('Adjusts')
    cv.destroyWindow('Negative')
    cv.destroyWindow('Gray')
    cv.destroyWindow('Resize')
    cv.destroyWindow('Rotate')
    cv.destroyWindow('Flip')


def default_image(image):
    global mode_init

    if not mode_init:
        destroy_special_windows()
        mode_init = True

    return image


def gaussian_image(image):
    global mode_init

    if not mode_init:
        destroy_special_windows()
        cv.namedWindow('Gaussian')
        cv.createTrackbar('Kernel', 'Gaussian', 1, 100, lambda x: x)
        mode_init = True

    kernel = cv.getTrackbarPos('Kernel', 'Gaussian')

    if kernel % 2 == 0:
        kernel += 1

    g_image = cv.GaussianBlur(image, (kernel, kernel), 0)

    cv.putText(g_image, 'Gaussian',
               (10, 450),
               cv.FONT_HERSHEY_SIMPLEX,
               1,
               (255, 255, 255),
               2)

    cv.imshow('Gaussian', g_image)

    return g_image


def canny_image(image):
    global mode_init

    if not mode_init:
        destroy_special_windows()
        cv.namedWindow('Canny')
        mode_init = True

    canny_image = cv.Canny(image, 100, 200)
    rgb_canny_image = cv.cvtColor(canny_image, cv.COLOR_GRAY2RGB)

    cv.putText(rgb_canny_image, 'Canny',
               (10, 450),
               cv.FONT_HERSHEY_SIMPLEX,
               1,
               (255, 255, 255),
               2)

    cv.imshow('Canny', rgb_canny_image)

    return rgb_canny_image


def sobel_gradient(image):
    global mode_init

    if not mode_init:
        destroy_special_windows()
        cv.namedWindow('Sobel')
        cv.createTrackbar('Kernel', 'Sobel', 3, 10, lambda x: x)
        mode_init = True

    kernel = cv.getTrackbarPos('Kernel', 'Sobel')

    if kernel % 2 == 0:
        kernel += 1

    grad_x = cv.Sobel(image, cv.CV_16S, 1, 0, ksize=kernel)
    grad_y = cv.Sobel(image, cv.CV_16S, 0, 1, ksize=kernel)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv.putText(grad, 'Sobel',
               (10, 450),
               cv.FONT_HERSHEY_SIMPLEX,
               1,
               (255, 255, 255),
               2)

    cv.imshow('Sobel', grad)

    return grad


def general_adjusts(image):
    global mode_init

    if not mode_init:
        destroy_special_windows()
        cv.namedWindow('Adjusts')
        cv.createTrackbar('Brightness', 'Adjusts', 0, 255, lambda x: x)
        cv.createTrackbar('Contrast', 'Adjusts', 0, 127, lambda x: x)
        mode_init = True

    brightness = cv.getTrackbarPos('Brightness', 'Adjusts')
    contrast = cv.getTrackbarPos('Contrast', 'Adjusts')

    alpha = (255 - brightness) / 255
    gamma = brightness

    result_image = cv.addWeighted(image, alpha, image, 0, gamma)

    alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
    gamma = 127 * (1 - alpha)

    result_image = cv.addWeighted(result_image, alpha, result_image, 0, gamma)

    cv.putText(result_image, 'Brightness and Contrast',
               (10, 450),
               cv.FONT_HERSHEY_SIMPLEX,
               1,
               (255, 255, 255),
               2)

    cv.imshow('Adjusts', result_image)

    return result_image


def negative_filter(image):
    global mode_init

    if not mode_init:
        destroy_special_windows()
        cv.namedWindow('Negative')
        mode_init = True

    result_image = 255 - image

    cv.putText(result_image, 'Negative',
               (10, 450),
               cv.FONT_HERSHEY_SIMPLEX,
               1,
               (255, 255, 255),
               2)

    cv.imshow('Negative', result_image)

    return result_image


def gray_filter(image):
    global mode_init

    if not mode_init:
        destroy_special_windows()
        cv.namedWindow('Gray')
        mode_init = True

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    rgb_gray_image = cv.cvtColor(gray_image, cv.COLOR_GRAY2RGB)

    cv.putText(rgb_gray_image, 'Gray',
               (10, 450),
               cv.FONT_HERSHEY_SIMPLEX,
               1,
               (255, 255, 255),
               2)

    cv.imshow('Gray', rgb_gray_image)

    return rgb_gray_image


def resize(image):
    global mode_init

    if not mode_init:
        destroy_special_windows()
        cv.namedWindow('Resize')
        cv.createTrackbar('Factor', 'Resize', 1, 4, lambda x: x)
        mode_init = True

    factor = cv.getTrackbarPos('Factor', 'Resize')

    if factor == 0:
        factor = 1
    else:
        factor = 1 / factor

    result = cv.resize(image, None, fx=factor, fy=factor, interpolation=cv.INTER_CUBIC)

    cv.imshow('Resize', result)

    return image


def rotate(image):
    global mode_init

    if not mode_init:
        destroy_special_windows()
        cv.namedWindow('Rotate')
        cv.createTrackbar('Degrees', 'Rotate', 0, 360, lambda x: x)
        mode_init = True

    degrees = cv.getTrackbarPos('Degrees', 'Rotate')

    rotated = imutils.rotate(image, degrees)

    cv.putText(rotated, 'Rotate',
               (10, 450),
               cv.FONT_HERSHEY_SIMPLEX,
               1,
               (255, 255, 255),
               2)

    cv.imshow('Rotate', rotated)

    return rotated


def flip(image):
    global mode_init

    if not mode_init:
        destroy_special_windows()
        cv.namedWindow('Flip')
        cv.createTrackbar('Entry', 'Flip', 0, 3, lambda x: x)
        mode_init = True

    entry = cv.getTrackbarPos('Entry', 'Flip')

    if entry != 0:
        if entry == 2:
            entry = 0
        if entry == 3:
            entry = -1
        flip_image = cv.flip(image, entry)
        cv.putText(flip_image, 'Flip',
                   (10, 450),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1,
                   (255, 255, 255),
                   2)
        cv.imshow('Flip', flip_image)
        return flip_image
    else:
        cv.imshow('Flip', image)

    return image


filter_mode = 0

frame_options = {0: default_image,
                 1: gaussian_image,
                 2: canny_image,
                 3: sobel_gradient,
                 4: general_adjusts,
                 5: negative_filter,
                 6: gray_filter,
                 7: resize,
                 8: rotate,
                 9: flip}


def set_filter_mode(new_filter_mode):
    global filter_mode
    filter_mode = new_filter_mode


def show_filter(cap):
    global filter_mode
    return frame_options[filter_mode](cap)


def capture_video():
    global mode_init
    video_list = []
    is_recording = False
    cap = cv.VideoCapture(1)

    while True:
        ret, image = cap.read()

        image_with_filter = show_filter(image)

        if is_recording:
            video_list.append(image_with_filter)
            cv.putText(image, 'Recording',
                       (10, 450),
                       cv.FONT_HERSHEY_SIMPLEX,
                       1,
                       (255, 255, 255),
                       2)

        cv.imshow('Processador de Video', image)

        pressed_key = cv.waitKey(1) & 0xFF

        if pressed_key == ord('q'):
            break
        elif pressed_key == ord('0'):
            mode_init = False
            set_filter_mode(0)
        elif pressed_key == ord('1'):
            mode_init = False
            set_filter_mode(1)
        elif pressed_key == ord('2'):
            mode_init = False
            set_filter_mode(2)
        elif pressed_key == ord('3'):
            mode_init = False
            set_filter_mode(3)
        elif pressed_key == ord('4'):
            mode_init = False
            set_filter_mode(4)
        elif pressed_key == ord('5'):
            mode_init = False
            set_filter_mode(5)
        elif pressed_key == ord('6'):
            mode_init = False
            set_filter_mode(6)
        elif pressed_key == ord('7'):
            mode_init = False
            set_filter_mode(7)
        elif pressed_key == ord('8'):
            mode_init = False
            set_filter_mode(8)
        elif pressed_key == ord('9'):
            mode_init = False
            set_filter_mode(9)
        elif pressed_key == ord('r'):
            is_recording = True
        elif pressed_key == ord('s'):
            is_recording = False
            height, width, layers = image.shape
            size = (width, height)
            out = cv.VideoWriter('project.avi', cv.VideoWriter_fourcc(*'DIVX'), 30, size)
            for i in range(len(video_list)):
                out.write(video_list[i])
            out.release()
            video_list = []

    cap.release()
    cv.destroyAllWindows()


def main():
    cv.namedWindow('Processador de Video')
    capture_video()


main()
