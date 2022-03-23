"""
این‌جا سعی می‌کنیم با کمک تابع هم‌بستگی، حرکت‌های نوسانی را آشکار کنیم..
"""




# to magnify a video, we have operation X which is this:

# block diagram of this algorithm is like this:
# 
# 
# 
#                        this is a FIFO algorithm
#                        both inputs and outputs                                                             sum up original frame
#                        are read/written                                                                    with corresponding
#                        frame-by-frame                                                                      correlation frame
# 
#                       ┌────────────────┐                                                                          ┌─────┐
#                       │  input video   │                                                                          │     │      ┌─────────────────┐
#                       │ frame by frame ├──────────────────────────────────────────────────────────────────────────►  ∑  ├──────►  output video   │
#                       └───────┬────────┘                                                                          │     │      │  frame by frame │
#                               │                                                                                   └──▲──┘      └─────────────────┘
#                               │                                                                                      │
#                               │                                                                                      │
#                               │                                                                                      │
#                               │                                                                                      │
#                               │                        to avoid DC        product of two functions                   │
#                               │                       from amplification  can be a fair approach                     │
#                               │                                                                                      │
#  stores latest read    ┌──────▼─────┐whole frames       ┌──────────┐      ┌──────────────┐                ┌────┐     │
#  frames. buffer size   │   Buffer   │in the buffer      │          │      │   calculate  │                │ ×α │     │
#  affects correlation   │            ├───────────────────►difference├──────►  correlation ├────────────────►    ├─────┘
#  function and the      └──────┬─────┘                   └────▲─────┘      └───────▲──────┘                └────┘
#  output                       │                              │                    │
#                               │                              │                    │                 amplify/attenuate
#                               │                              │                    │
#                               │first frame                   │                    │
#                               │in the buffer                 │                    │
#                               │                              │                    │
#                               │                              │               ┌────┴─────┐
#                               └──────────────────────────────┘               │ signal   │
#                                                                              │ generator│
#                                                                              └──────────┘
# 
#                                                                              sin(ω×t) can be
#                                                                              a fair approach
# 
# 




# but here we do not use operation X directly.
# we make video into a pyramid seperating it into different spatial frequency bands.
# then perform operation X on each level
# finally render the pyramid into output video.


#                            ┌──┐                                  ┌──┐
#                            │  │             ┌─────────┐          │  │
#                            │  │ level 1     │         │          │  │
#                            │  ├────────────►│    X    ├─────────►│  │
#                            │  │             │         │          │  │
#                            │  │             └─────────┘          │  │
#                            │  │                                  │  │
#                            │  │                                  │  │
#                  make a    │  │                                  │  │ collapse
#  ┌────────────┐  laplace   │  │             ┌─────────┐          │  │ laplace     ┌────────────┐
#  │ input      │  pyramid   │  │ level 2     │         │          │  │ pyramid     │ output     │
#  │ video      ├───────────►│  ├────────────►│    X    ├─────────►│  ├─────────────► video      │
#  │            │            │  │             │         │          │  │             │            │
#  └────────────┘            │  │             └─────────┘          │  │             └────────────┘
#                            │  │                                  │  │
#                            │  │  ...                             │  │
#                            │  │                                  │  │
#                            │  │             ┌─────────┐          │  │
#                            │  │  level n    │         │          │  │
#                            │  ├────────────►│    X    ├─────────►│  │
#                            │  │             │         │          │  │
#                            │  │             └─────────┘          │  │
#                            └──┘                                  └──┘








# import libraries
# از کتابخانه  opencv استفاده خواهد‌شد.

# for reading and writing videos and gaussian blur
import cv2

# for working with matrices
import numpy as np

# for cli
import os
# for cli
import sys

# for making delay while showing video
from time import sleep

# from pyramids import laplacian_pyramid_make, laplacian_pyramid_rendr
from pyramids import pyramid_make, pyramid_rendr




# do not let pixel values of image get out of their limit
def guard_image(image:np.ndarray):

    # any pixel value bigger than 255 will be cut down to 255
    image[image > 255] = 255

    # any pixel value smaller than 0 will be 0
    image[image < 0] = 0








def correlation(input_file_name, alpha, frequency, spatial_frequency_coefficients,
                buffer_size, number_of_frames, number_of_jump_frames=0,
                time_signal_x=0, time_signal_y=0, show_video=True, pyramid_type='gaussian'):
    

    """
        # magnify video with correlation algorithm

        - `input_file_name`: name of the file  
        - `alpha`: magnification ratio  
        - 'frequency`: frequency of signal generator
        - `spatial_frequency_coefficients`: coefficients of summision for pyramid.
        - `buffer size`: size of buffer for calculating correlation.
        - `number_of_frames`: number of video frames to read and magnify.
        - `number_of_jump_frames`: number of initial frames to skip. this is useful when your time of interest
        is not the start of video. also useful when you want to cut video into pieces and magnify each seperately.
        - `time_signal_x` and `time_signal_y`: the pixel on image to extract time signal from.
        three time signals will be extracted. one from y'th row. one from x's column. other from pixel (x,y)
        - `show_video` : if set to True, a video will be shown while rendering. results will always be
        stored in ./results folder.

    
    """    



    # if you provide "camera" as input filename, camera of system will be used indtead of file.

    # if you want to use camera
    if input_file_name == 'camera':

        # open camera as a video file
        videoFile = cv2.VideoCapture(0)
    
    # if a plain filename is provided
    else:

        # open videofile.
        videoFile = cv2.VideoCapture(input_file_name)




    # number of levels in pyramid is equal to size of spatial coefficients provided.
    # this is for making it more flexible.
    number_of_pyramid_levels = len(spatial_frequency_coefficients)


    # pyramid levels coefficients
    # spatial frequency bands coefficients for rendering
    # NOTE: if not all coefficients are set here,
    #  the default value will be used for all of them which is 1.00
    spatial_coeff = spatial_frequency_coefficients

    spatial_coeff2 = [0, 0, 0, 1, 1]
    spatial_coeff3 = [0, 0, 0, 1, 0]

    spatial_coeff_correlation = spatial_frequency_coefficients
    spatial_coeff_correlation_orig = [1] * len(spatial_frequency_coefficients)
    # spatial_coeff_correlation_orig = spatial_coeff_correlation

    # تاخیر مناسب بین فریم‌ها تا ویدیو با سرعت عادی پخش شود
    # به خاطر وجود تاخیر در حین پردازش، این تاخیر نصف شده‌است.
    frame_sleep_duration = 1.0/60 / 2

    # get frame rate of the video
    # (frames per second. it is 30 for most standard videos)
    video_frame_rate = videoFile.get(cv2.CAP_PROP_FPS)

    # normalize alpha for size of the buffer
    normalized_alpha = alpha / buffer_size

    alpha2 = alpha * 0.01 * 0.5
    normalized_alpha2 = alpha2 / buffer_size

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

    # define frame buffer. actually multiplication values will be stored here.
    frame_buffer = np.zeros((buffer_size, frame_height, frame_width, number_of_channels), dtype=np.float32)
    # correlation_pyramid_buffer = [frame_buffer.copy() for _ in range(number_of_pyramid_levels)]


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
    correlation_pyramid_buffer = np.array([
        pyramid_zero.copy()
        for i in range(buffer_size)
        ])


    # generate pyramid of frame 0

    pyramid_frame = pyramid_make(frame_0_float, number_of_pyramid_levels, pyramid_type)

    # initial value of buffer is going to be frame 0

    # for each frame in buffer
    for i in range(buffer_size):

        # store frame 0 in buffer
        frame_buffer[i] = frame_0_float

    # 
    correlation_pyramid = pyramid_zero.copy()

    def freq_function(f:float, t:float) -> float:
        return np.sin(t * f * 2 * np.pi / video_frame_rate)

    frame_0_float_yuv = cv2.cvtColor(frame_0_float, cv2.COLOR_BGR2YUV)
    frame_0_float_yuv_chans = cv2.split(frame_0_float_yuv)

    print('buffer size -> ', correlation_pyramid_buffer.shape, f'× {frame_0.shape} <- frame size ', 'fps=', np.round(video_frame_rate))

    just_file_name = os.path.basename(input_file_name)

    outFile = cv2.VideoWriter(f'results/out_correlation_{just_file_name}.avi', cv2.VideoWriter_fourcc(*'MJPG'), video_frame_rate, (frame_width, frame_height))
    OrigoutFile = cv2.VideoWriter(f'results/orig_correlation_{just_file_name}.avi', cv2.VideoWriter_fourcc(*'MJPG'), video_frame_rate, (frame_width, frame_height))

    tank_size = number_of_frames
    image_signal_out = np.zeros((frame_height, tank_size, 3), np.uint8)
    image_signal_original = np.zeros((frame_height, tank_size, 3), np.uint8)


    correlation_all = np.zeros(frame_0_float.shape, np.float32)
    image_signal_correlation_all = np.zeros((frame_height, tank_size, 3), np.uint8)



    # extracted temporal signals from video will be stored in these arrays.

    # define a temporal signal of zeros
    image_signal_out_x = np.zeros((frame_height, number_of_frames, 3), np.uint8)

    # define a temporal signal of zeros
    image_signal_out_y = np.zeros((number_of_frames, frame_width, 3), np.uint8)

    # define a temporal signal of zeros
    image_signal_original_x = np.zeros((frame_height, number_of_frames, 3), np.uint8)

    # define a temporal signal of zeros
    image_signal_original_y = np.zeros((number_of_frames, frame_width, 3), np.uint8)




    # this counter is for buffer to work
    # it is index of buffer where the latest frame is stored
    buffer_count = 0

    # for better ui
    print()


    # start rendering output video
    # frames will be read
    # pyramids will be made from frame
    # multiplication with signal will be calculated and stored in buffer
    # correlation will be calculated
    # magnification will be done
    # pyramid will be rendered into frame
    for i in range(number_of_frames):

        # next frame on buffer. it is also oldest frame stored on buffer that is not popped out yet
        buffer_count_next = (buffer_count + 1) % buffer_size

        # read a frame
        _, frame_now = videoFile.read()

        # if there is no frame in video left
        if frame_now is None:

            # break the loop and stop rendering
            break
        
        # convert frame into float
        frame_now_float = np.float32(frame_now)

        # generate a pyramid of the read frame
        pyramid_frame = np.array(pyramid_make(frame_now_float, number_of_pyramid_levels, pyramid_type), dtype=np.ndarray)

        # store the frame in the buffer
        frame_buffer[buffer_count] = frame_now_float

        pyramid_of_minus_frame = np.array(pyramid_make(frame_buffer[buffer_count_next], number_of_pyramid_levels, pyramid_type), dtype=np.ndarray)
        correlation_pyramid_now = (pyramid_frame - pyramid_of_minus_frame) * freq_function(frequency, i) 
        correlation_pyramid_buffer[buffer_count] = correlation_pyramid_now
        correlation_pyramid += correlation_pyramid_now
        correlation_pyramid -= correlation_pyramid_buffer[buffer_count_next]

        correlation_all += (frame_now_float - frame_0_float) * freq_function(frequency, i)


        # spatial_coeff_correlation_orig = [1] * 5
        pyramid_magnified = [
            correlation_pyramid[level] * spatial_coeff_correlation[level] * freq_function(frequency*0.5, i) * normalized_alpha + pyramid_frame[level] * spatial_coeff_correlation_orig[level]
            for level in range(number_of_pyramid_levels)
        ]

        # for level in pyramid_magnified:
        #     level[level < 0] = 0
        #     level[level > 255] = 255


        # rendered_out = laplacian_pyramid_rendr(pyramid_magnified, spatial_coeff=[0,0,0,1,1])
        # rendered_out = rendered_out * 10 + 100
        
        rendered_out = pyramid_rendr(pyramid_magnified, pyramid_type=pyramid_type)

        guard_image(rendered_out)
        # pyramid_magnified = correlation_pyramid * normalized_alpha + pyramid_frame
        # spatial_coeff2 = spatial_coeff
        # rendered_out2 = pyramid_rendr(pyramid_magnified, spatial_coeff=spatial_coeff2, pyramid_type=pyramid_type)

        # guard_image(rendered_out2)

        # magnified_plus_original = pyramid_rendr(pyramid_magnified, spatial_coeff=spatial_coeff[:-1]+ [0], pyramid_type=pyramid_type) + frame_now_float
        # magnified_plus_original = pyramid_rendr(pyramid_magnified, spatial_coeff=spatial_coeff, pyramid_type=pyramid_type) + frame_now_float

        # guard_image(magnified_plus_original)

        # i1 = pyramid_rendr(correlation_pyramid, spatial_coeff=[0 , 0, 0, 1, 0], pyramid_type=pyramid_type)
        # guard_image(i1)
        # i2 = pyramid_rendr(correlation_pyramid, spatial_coeff=[1 , 1, 1, 0, 0], pyramid_type=pyramid_type)
        # i3 = i1 * i2 + i1
        # i4 = frame_now_float + i3 * 2
        # guard_image(i4)

        p = correlation_all / (i + 1)
        # q = frame_now_float * p * 1 + frame_now_float
        # guard_image(p)
        # guard_image(q)


        if show_video == True:

            # cv2.imshow('original', frame_now)
            cv2.imshow('rendered pyramid', np.uint8(rendered_out))
            # cv2.imshow('rendered pyramid+0', np.uint8(magnified_plus_original))
            # cv2.imshow('rendered pyramid2', np.uint8(rendered_out2))
            # cv2.imshow('i4', np.uint8(i4))
                
            # cv2.imshow('correlation all', np.uint8(p*100))
            # cv2.imshow('result', np.uint8(q))






        output_color_int = np.uint8(rendered_out)
        # output_color_int = np.uint8(rendered_out2)


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








        sys.stdout.write("\033[F")  # back to previous line
        sys.stdout.write("\033[K")  # clear line
        print('frame:', i)


        pressed_key = cv2.waitKey(1)
        if pressed_key == ord('q'):
            break
        elif pressed_key == ord('p'):
            cv2.waitKey(0)
        # sleep(frame_sleep_duration)

        buffer_count = buffer_count_next



        outFile.write(np.uint8(rendered_out))
        OrigoutFile.write(frame_now)



        # output_color_int = np.uint8(rendered_out2)
        # line_signal_out = output_color_int[:, x]
        # image_signal_out[:, i] = line_signal_out

        # line_signal_original = frame_now[:, x]
        # image_signal_original[:, i] = line_signal_original

        # line_signal_correlation_all = np.uint8(p*100)[:,x]
        # image_signal_correlation_all[:, i] = line_signal_correlation_all

    correlation_all /= i
    cv2.imwrite(f'results/correlation_all_{just_file_name}.jpg', np.uint8(correlation_all))



    outFile.release()
    OrigoutFile.release()




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
    cv2.imwrite(f'results/out_correlation_{just_file_name}_x_{time_signal_x}.jpg', image_signal_out_x)

    # write raw of time signals from rendered video as an image
    cv2.imwrite(f'results/out_correlation_{just_file_name}_y_{time_signal_y}.jpg', image_signal_out_y)


    # write column of time signals from original video as an image
    cv2.imwrite(f'results/orig_correlation_{just_file_name}_x_{time_signal_x}.jpg', image_signal_original_x)

    # write raw of time signals from original video as an image
    cv2.imwrite(f'results/orig_correlation_{just_file_name}_y_{time_signal_y}.jpg', image_signal_original_y)


    cv2.destroyAllWindows()


    return (f'results/orig_correlation_{just_file_name}.avi',
    f'results/out_correlation_{just_file_name}.avi')