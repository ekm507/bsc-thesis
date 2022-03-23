


# از کتابخانه  opencv استفاده خواهد‌شد.
import cv2

# for working with matrices
import numpy as np

# for making delay while showing frames
from time import sleep

# for cli
import os

# for cli
import sys
import matplotlib.pyplot as plt

# for storing time signal
import pickle

from time import time

from pyramids import pyramid_make, pyramid_rendr


# do not let pixel values of image get out of their limit
def guard_image(image:np.ndarray):

    # any pixel value bigger than 255 will be cut down to 255
    image[image > 255] = 255

    # any pixel value smaller than 0 will be 0
    image[image < 0] = 0


# magnify video. differential method
def differential(input_file_name, alpha,
                 number_of_frames, spatial_frequency_coefficients, buffer_size,
                 number_of_jump_frames=0, time_signal_x=0, time_signal_y=0, show_video=True, pyramid_type='gaussian'):
    """
        # magnify video with differential algorithm

        - `input_file_name`: name of the file  
        - `alpha`: magnification ratio  
        - `number_of_frames`: number of video frames to read and magnify.
        - `spatial_frequency_coefficients`
        - `spatial_frequency_coefficients`: coefficients of summision for pyramid.
        - `buffer size`: size of buffer for calculating correlation.
        - `number_of_jump_frames`: number of initial frames to skip. this is useful when your time of interest
        is not the start of video. also useful when you want to cut video into pieces and magnify each seperately.
        - `time_signal_x` and `time_signal_y`: the pixel on image to extract time signal from.
        three time signals will be extracted. one from y'th row. one from x's column. other from pixel (x,y)
        - `show_video` : if set to True, a video will be shown while rendering. results will always be
        stored in ./results folder.
    
    """    

    spatial_coeff = spatial_frequency_coefficients
    number_of_pyramid_levels = len(spatial_coeff)




    # print_mode = 'none'
    # print_mode = 'time'
    print_mode = 'number'

    # if you provide "camera" as input filename, camera of system will be used indtead of file.

    # if you want to use camera
    if input_file_name == 'camera':

        # open camera as a video file
        videoFile = cv2.VideoCapture(0)
    
    # if a plain filename is provided
    else:

        # open videofile.
        videoFile = cv2.VideoCapture(input_file_name)

    # تاخیر مناسب بین فریم‌ها تا ویدیو با سرعت عادی پخش شود
    # به خاطر وجود تاخیر در حین پردازش، این تاخیر نصف شده‌است.
    frame_sleep_duration = 1.0/60 / 2

    # get frame rate of the video
    # (frames per second. it is 30 for most standard videos)
    video_frame_rate = videoFile.get(cv2.CAP_PROP_FPS)


    # if we need to skip some frames, we can just read a bunch of frames and simply toss them away
    for i in range(number_of_jump_frames):

        # read a frame and toss it out!
        # these frames are going to be skipped. so let it go!
        _, frame_now = videoFile.read()


    # خواندن فریم اول و تعریف متغیرهای اولیه
    _, frame_0 = videoFile.read()


    # frame is basically a matrix of pixels. each pixel has 3 variables for each corresponding channel.
    # the variables are 8-bit unsigned integer numbers between 0 and 255 (uint8)
    # for further processings we need floating point numbers.
    # so here we convert the variable types from what it is into float32
    frame_0_float = np.float32(frame_0)

    # get frame size.
    frame_height, frame_width, number_of_channels = np.shape(frame_0_float)

    just_magnify_image = np.zeros((frame_height, number_of_frames, 3), np.uint8)

    # get input file name
    just_file_name = os.path.basename(input_file_name)

    time_signal = []


    # derivative output will be stored in this file
    derivative_outfile = cv2.VideoWriter(f'results/out_differential_derivative_{just_file_name}.avi', cv2.VideoWriter_fourcc(*'MJPG'), video_frame_rate, (frame_width, frame_height))

    # just amplify output will be stored in this file
    amplify_outfile = cv2.VideoWriter(f'results/out_differential_amplify_{just_file_name}.avi', cv2.VideoWriter_fourcc(*'MJPG'), video_frame_rate, (frame_width, frame_height))
    original_outfile = cv2.VideoWriter(f'results/orig_differential_{just_file_name}.avi', cv2.VideoWriter_fourcc(*'MJPG'), video_frame_rate, (frame_width, frame_height))



    # extracted temporal signals from video will be stored in these arrays.

    # define a temporal signal of zeros
    image_signal_out_x = np.zeros((frame_height, number_of_frames, 3), np.uint8)

    # define a temporal signal of zeros
    image_signal_out_y = np.zeros((number_of_frames, frame_width, 3), np.uint8)

    # define a temporal signal of zeros
    image_signal_original_x = np.zeros((frame_height, number_of_frames, 3), np.uint8)

    # define a temporal signal of zeros
    image_signal_original_y = np.zeros((number_of_frames, frame_width, 3), np.uint8)

    # define a pyramid of black frames.
    # this is for defining pyramid buffer.
    # default value is zero.
    pyramid_zero = np.array([

            np.zeros(
                    (frame_height//2**(j), frame_width//2**(j), number_of_channels),
                dtype=np.float32)

            for j in range(number_of_pyramid_levels)
        ], dtype=np.ndarray)


    # define a buffer of pyramids for storing correlation pyramids.
    pyramid_buffer = np.array([
        pyramid_zero.copy()
        for i in range(buffer_size)
        ], dtype=object)



    # this counter is for buffer to work
    # it is index of buffer where the latest frame is stored
    buffer_count = 0


    t0 = time()

    frame_previous = frame_0_float
    print()

    for i in range(number_of_frames):


        # next frame on buffer. it is also oldest frame stored on buffer that is not popped out yet
        buffer_count_next = (buffer_count + 1) % buffer_size


        _, frame_now = videoFile.read()


        if frame_now is None:
            break
        
        frame_now_float = np.float32(frame_now)
        pyramid_now = pyramid_make(frame_now_float, number_of_pyramid_levels, pyramid_type)
        diff_pyramid = pyramid_now - pyramid_buffer[buffer_count_next]
        diff_frame = pyramid_rendr(diff_pyramid, spatial_coeff=spatial_coeff, pyramid_type=pyramid_type)
        # pyramid_differentially_amplified = pyramid_now + diff_pyramid * alpha
        # amplified_rendered = pyramid_rendr(pyramid_differentially_amplified, spatial_coeff=spatial_coeff, pyramid_type=pyramid_type)
        amplified_rendered = frame_now + diff_frame * alpha 
        guard_image(amplified_rendered)

        pyramid_buffer[buffer_count] = pyramid_now


        # diff = frame_now_float - frame_previous
        # frame_differentially_amplified = frame_now_float + diff * alpha
        # guard_image(frame_differentially_amplified)

        # pyramid_just_amplified = pyramid_now        
        # frame_just_amplified = frame_now_float * alpha  - frame_0_float * (alpha - 1)
        # guard_image(frame_just_amplified)

        if show_video==True:

            cv2.imshow('original', frame_now)
            cv2.imshow('out', np.uint8(amplified_rendered))
            
            # cv2.imshow('just magnify', np.uint8(frame_differentially_amplified))
            # cv2.imshow('just amplify', np.uint8(frame_just_amplified))



        frame_previous = frame_now_float


        # just_magnify_line = np.uint8(frame_made*100)[:,x]
        # just_magnify_line = np.uint8(frame_just_amplified)[:,time_signal_x]
        # guard_image(just_magnify_line)
        # just_magnify_image[:, i] = just_magnify_line

        # time_signal.append(frame_just_amplified[time_signal_y, time_signal_x])

        # # derivative_outfile.write(np.uint8(frame_differentially_amplified))
        # derivative_outfile.write(np.uint8(amplified_rendered))
        # amplify_outfile.write(np.uint8(frame_just_amplified))
        # original_outfile.write(frame_now)


        # output 
        # output_color_int = np.uint(frame_differentially_amplified)
        output_color_int = np.uint8(amplified_rendered)

        # extract time signal informaion from rendered video

        # get a column of pixels from the image.
        line_signal_out_x = output_color_int[:, time_signal_x]

        # get a raw of pixels from the image.
        line_signal_out_y = output_color_int[time_signal_y, :]

        # line_signal_correlation_all = np.uint8(p*100)[:,x]
        # image_signal_correlation_all[:, i] = line_signal_correlation_all

        # store the column into array of time signals
        image_signal_out_x[:, i] = line_signal_out_x

        # store the raw into array of time signals
        image_signal_out_y[i, :] = line_signal_out_y



        # extract time signal informaion from original video

        # get a column of pixels from the image.
        line_signal_original_x = frame_now[:, time_signal_x]

        # get a raw of pixels from the image.
        line_signal_original_y = frame_now[time_signal_y, :]

        # store the column into array of time signals
        image_signal_original_x[:, i] = line_signal_original_x

        # store the raw into array of time signals
        image_signal_original_y[i, :] = line_signal_original_y




        buffer_count = buffer_count_next

        if print_mode == 'time':
            t1 = time()
            t = t1 - t0
            t0 = t1
            print(t)

        elif print_mode == 'number':
            sys.stdout.write("\033[F")  # back to previous line
            sys.stdout.write("\033[K")  # clear line
            print('frame:', i)


        pressed_key = cv2.waitKey(1)
        if pressed_key == ord('q'):
            break
        elif pressed_key == ord('p'):
            cv2.waitKey(0)
        # sleep(frame_sleep_duration)

    videoFile.release()

    with open('time_signal.pickle', 'wb') as tsfile:
        pickle.dump(time_signal, tsfile)

    cv2.imwrite(f'results/out_just_magnify_{just_file_name}_x_{time_signal_x}.jpg', just_magnify_image[:, :i])

    original_outfile.release()
    derivative_outfile.release()
    amplify_outfile.release()



    # image signal might be bigger than number of rendered frames and some of it might be empty.
    # here we crop that image!
    # cut non-rendered parts of image signal out
    image_signal_out_x = image_signal_out_x[:, :i]

    # cut non-rendered parts of image signal out
    image_signal_out_y = image_signal_out_y[:i, :]


    # cut non-needed parts of image signal
    image_signal_original_x = image_signal_original_x[:, :i]

    # cut non-needed parts of image signal
    image_signal_original_y = image_signal_original_y[:i, :]

    # write column of time signals from rendered video as an image
    cv2.imwrite(f'results/out_differential_derivative_{just_file_name}_x_{time_signal_x}.jpg', image_signal_out_x)

    # write raw of time signals from rendered video as an image
    cv2.imwrite(f'results/out_differential_derivative_{just_file_name}_y_{time_signal_y}.jpg', image_signal_out_y)


    # write column of time signals from original video as an image
    cv2.imwrite(f'results/orig_differential_{just_file_name}_x_{time_signal_x}.jpg', image_signal_original_x)

    # write raw of time signals from original video as an image
    cv2.imwrite(f'results/orig_differential_{just_file_name}_y_{time_signal_y}.jpg', image_signal_original_y)



    cv2.destroyAllWindows()

    return (
        f'results/orig_differential_{just_file_name}.avi',
        f'results/out_differential_amplify_{just_file_name}.avi'
    )