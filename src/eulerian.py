#!/usr/bin/python3

# python source code for eularian video motion magnification with gaussian pyramid as spatial filter.

# usage:
# $ python3 nameOfThisFile.py inputVideoFileName

# written by Erfan Kheyrollahi in 2021

"""
 "unlike people, comupters understand". Erfan Kheyrollahi - 2018
"""

# in this code we are going to implement the block diagram

# algorithm:
#
# step 1: take a video and convert each frame into a pyramid then store all of them in a list.
# step 2: take the pyramid and do a temporal filtering on each level of the pyramid.
# step 3: read original video frame by frame and convert them to pyramids.
#         then add up corresponding filtered pyramid to it and finally collapse pyramid into a rendered frame.
#         store the resulting frames in a video file.


# Block Diagram


#                     ┌──┐  ┌───────┐ temporal filter  ┌───┐ ×α   ┌───┐      ┌──┐
#                     │  ├──┤Level 1├─────────────────►│L.1├──────► + ├─────►│  │
#                     │  │  └───┬───┘                  └───┘      └─▲─┘      │  │
#                     │  │      │                                   │        │  │
#                     │  │      └───────────────────────────────────┘        │  │
#                     │  │                                                   │  │
#                     │  │  ┌───────┐ temporal filter  ┌───┐ ×α   ┌───┐      │  │
#            gaussian │  ├──┤Level 2├─────────────────►│L.2├──────► + ├─────►│  │
# ┌────────┐ pyramid  │  │  └───┬───┘                  └───┘      └─▲─┘      │  │ collapse
# │ video  ├─────────►│  │      │                                   │        │  │ pyramid     ┌──────┐
# └────────┘          │  │      └───────────────────────────────────┘        │  ├────────────►│output│
#                     │  │                                                   │  │             └──────┘
#                     │  │  ┌───────┐ temporal filter  ┌───┐ ×α   ┌───┐      │  │
#                     │  ├──┤  ...  ├─────────────────►│...├──────► + ├─────►│  │
#                     │  │  └───┬───┘                  └───┘      └─▲─┘      │  │
#                     │  │      │                                   │        │  │
#                     │  │      └───────────────────────────────────┘        │  │
#                     │  │                                                   │  │
#                     │  │  ┌───────┐ temporal filter  ┌───┐ ×α   ┌───┐      │  │
#                     │  ├──┤Level n├─────────────────►│L.n├──────► + ├─────►│  │
#                     └──┘  └───┬───┘                  └───┘      └─▲─┘      └──┘
#                               │                                   │
#                               └───────────────────────────────────┘
# 
# drawn using asciiflow.com

# import libraries

# for reading and writing videos and gaussian blur
import cv2

# for working with matrices
import numpy as np

# for cli
from os import path

# for butterworth and other filters
from scipy import signal

# for cli
import sys

# from pyramids import gaussian_pyramid_make, gaussian_pyramid_rendr
# from pyramids import laplacian_pyramid_make, laplacian_pyramid_rendr
from pyramids import pyramid_make, pyramid_rendr

# do not let pixel values of image get out of their limit
def guard_image(image:np.ndarray):

    # any pixel value bigger than 255 will be cut down to 255
    image[image > 255] = 255

    # any pixel value smaller than 0 will be 0
    image[image < 0] = 0



# magnify video. eulerian method
def eulerian(input_file_name:str, alpha:float, temporal_frequency_bands,
             spatial_frequency_coefficients, number_of_frames:int, number_of_jump_frames:int=0,
             filter_band_type:str='bandpass', time_signal_x:int=0, time_signal_y:int=0, show_video:bool=True, pyramid_type='gaussian'
             ):
    """
        # magnify video with eulerian algorithm

        - `input_file_name`: name of the file  
        - `alpha`: magnification ratio  
        - `temporal_frequency_bands`: number or list or tuple of two numbers.
        high and low cutoff frequencies. for lowpass and highpass filters, only one number is enough.
        - `spatial_frequency_coefficients`: coefficients of summision for pyramid.
        - `number_of_frames`: number of video frames to read and magnify.
        - `number_of_jump_frames`: number of initial frames to skip. this is useful when your time of interest
        is not the start of video. also useful when you want to cut video into pieces and magnify each seperately.
        - `filter_band_type`: can be either `lowpass`, `highpass`, `bandpass`, `bandstop`. band type.
        - `time_signal_x` and `time_signal_y`: the pixel on image to extract time signal from.
        three time signals will be extracted. one from y'th row. one from x's column. other from pixel (x,y)
        - `show_video` : if set to True, a video will be shown while rendering. results will always be
        stored in ./results folder.

    
    """    

    # parameters

    # parameters are these:
    # 1. temporal frequency band
    # 2. filter order
    # 3. spatial bands coefficients
    # 4. alpha (amplitution factor)
    # amount of amplification. same as the α factor in block diagram
    # 5. frame range of the video to process


    # lets start the job by opening the video file!

    # open the video file to start the job
    videoFile = cv2.VideoCapture(input_file_name)

    # get frame rate of the video (frames per second. it is 30 for most standard videos)
    video_frame_rate = videoFile.get(cv2.CAP_PROP_FPS)


    # this program will read frames i to j of the video and magnify them
    # the frame program should start reading from

    # number of consecutive frames to read from video and then process
    # these frames will be stored in a tank.
    tank_size = number_of_frames

    # this program uses pyramid for filtering into different spatial frequencies
    # each level of pyramid is half of the previous one by both length and width
    # this parameter sets depth of the pyramid. same as the n in number of pyramid levels in block diagram

    # spatial filters summition coefficients
    # these numbers will be multiplied into pyramid levels whn rendering before summition
    # diagram below depicts how it's done


    #    pyramid  coefficients     resize levels and      rendered
    #    levels                    add them together      frame
    # 
    #   ┌───────┐      ×c_1  ┌──────────────────────┐
    #   │Level 1├───────────►│                      │
    #   └───────┘            │                      │
    #                        │    resize and sum    │
    #   ┌───────┐      ×c_2  │                      │
    #   │Level 2├───────────►│ ___ _   _ _ __ ___   │    ┌────────────┐
    #   └───────┘            │/ __| | | | '_ ` _ \  ├───►│output frame│
    #                        │\__ \ |_| | | | | | | │    └────────────┘
    #      ...               │|___/\__,_|_| |_| |_| │
    #                        │                      │
    #   ┌───────┐      ×c_n  │                      │
    #   │Level n├───────────►│                      │
    #   └───────┘            └──────────────────────┘

    # pyramid levels coefficients
    # spatial frequency bands coefficients for rendering
    # NOTE: if not all coefficients are set here,
    #  the default value will be used for all of them which is 1.00
    spatial_coeff = spatial_frequency_coefficients

    # number of levels in pyramid is equal to size of spatial coefficients provided.
    # this is for making it more flexible.
    number_of_pyramid_levels = len(spatial_coeff)


    # sampling frequency of time-domain signal.
    # this is the same as frame rate of the video.
    # this parameter is necessary to set since filter cutoff frequencies are dependant on this.
    fs = video_frame_rate


    # filter design

    # get cutoff frequencies from function input
    # filter cutoff frequency (Hz)

    # if a list of tuple is provided
    if isinstance(temporal_frequency_bands, (list, tuple)):

        # if there is two numbers in the list or tuple
        if len(temporal_frequency_bands) > 1:

            # filter cutoff frequency (Hz)
            # lower cutoff frequency of bandpass or bandstop filter. also cutoff frequency of lowpass filter
            cutoff_low = temporal_frequency_bands[0]

            # higher cutoff frequency of bandpass or bandstop filter. also cutoff frequency of highpass filter
            cutoff_high = temporal_frequency_bands[1]

        # if there is only one number in list or tuple
        else:

            # filter cutoff frequency (Hz)
            # set low cutoff frequency to the provided number
            cutoff_low = temporal_frequency_bands[0]

            # set high cutoff frequency to the provided number
            cutoff_high = temporal_frequency_bands[0]

    # if a single number is provided
    else:

        # filter cutoff frequency (Hz)
        # set low cutoff frequency to the provided number
        cutoff_low = temporal_frequency_bands

        # set high cutoff frequency to the provided number
        cutoff_high = temporal_frequency_bands


    # filter type ('lowpass' or 'highpass' or 'bandpass' or 'bandstop')
    band_type = filter_band_type

    # nyquist frequency. for filter design
    nyq = 0.5 * fs

    # filter order. the order parameter for butterworth or other filters when designing them.
    order = 5
    # NOTE : you may need to change this.

    # normalized cutoff frequency

    # if filter should be bandpass or bandstop
    if band_type in ['bandpass', 'bandstop']:

        # get a list of normalized cutoff frequencies by deviding provided values by nyqusit frequency
        normal_cutoff = [cutoff_low / nyq, cutoff_high / nyq]

    # if filter should be highpass
    elif band_type in ['high', 'highpass']:

        # get normalized frequency by deviding provided values by nyqusit frequency
        normal_cutoff = cutoff_high / nyq

    # if filter should be lowpass
    elif band_type in ['low', 'lowpass']:

        # get normalized frequency by deviding provided values by nyqusit frequency
        normal_cutoff = cutoff_low / nyq

    # if filter band type is not specified (correctly) then print error message and exit
    else:

        # there is an error!
        print("filter type is not correctly set", file=sys.stderr)

        # exit the program to prevent it from bugs.
        exit(2)

    # design filter and get filter coefficients.
    # this is a butterworth filter
    filter_coef_b, filter_coef_a = signal.butter(order, normal_cutoff, btype=band_type, analog=False)

    # this is the iir filter of scipy.signal lib
    # filter_coef_b, filter_coef_a = signal.iirfilter(order, normal_cutoff, btype=band_type)


    # read first frame of the input video
    _, frame_0 = videoFile.read()

    # convert colorspace of first frame from BGR into gray
    # we do this so that there is only one channel in the frame which is luminance
    frame0_gray = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)

    # get frame height and width of the video. also number of color channels which is usually 3
    frame_h, frame_w, number_of_channels = np.shape(frame_0)

    # print needed detail about video
    print(f'frame size is ({frame_w}, {frame_h}). frame rate is {np.round(video_frame_rate)}')



    # define a list for storing pyramids

    print('allocating memory')

    # define pyramid for a video.
    # dimentions of a video is number of frames × size of a frame
    # a pyramid is like several videos. each with different frame sizes but same number of frames.
    # so it is defined like this:
    # 
    # [ video of level 1 : number of frames × frame height × frame width ]
    # [ video of level 2 : number of frames × frame height × frame width ]
    # [       ...                                                        ]
    # [ video of level n : number of frames × frame height × frame width ]
    # 
    # note that frame size is different for each level

    # generate video pyramid list
    pyramid = [np.zeros((tank_size, frame_h//2**(i), frame_w//2**(i)), np.float32)
            for i in range(number_of_pyramid_levels)]

    # filtered video pyramid will be stored here. it should be the same size of the original video pyramid
    result = [np.zeros((tank_size, frame_h//2**(i), frame_w//2**(i)), np.float32)
            for i in range(number_of_pyramid_levels)]


    # start reading video frames and storing them into pyramid

    print('storing video in pyramids')

    # if we need to skip some frames, we can just read a bunch of frames and simply toss them away
    for i in range(number_of_jump_frames):

        # read a frame and toss it out!
        # these frames are going to be skipped. so let it go!
        _, frame_now = videoFile.read()


    # start reading frames one by one and storing them into pyramid.
    for i in range(tank_size):

        # read one frame from the video file
        _, frame_now = videoFile.read()

        # if there is no more frame in the video legt
        if frame_now is None:

            # change tank size to number of frames read.
            tank_size = i

            # break the loop. stop magnifying frames.
            break

        # frame is basically a matrix of pixels. each pixel has 3 variables for each corresponding channel.
        # the variables are 8-bit unsigned integer numbers between 0 and 255 (uint8)
        # for further processings we need floating point numbers.
        # so here we convert the variable types from what it is into float32
        frame_float = np.float32(frame_now)

        # convert frame from BGR color space into single-channeled gray colorspace
        frame_float_gray = cv2.cvtColor(frame_float, cv2.COLOR_BGR2GRAY)

        # decrement result from the first frame. we do this to remove DC in fourier spectrum
        # when using highpass filters, this is not really needed but in lowpass filtering it is
        # necessary to remove DC. this method is not the best one since we may use FIR filters with
        # windows smaller than tank_size. that way each window will have to be decremented from a 
        # different frame in order to remove DC
        frame_float_gray_diff = frame_float_gray- np.float32(frame0_gray)

        # generate a gaussian pyramid from the frame
        pyramid_frame = pyramid_make(frame_float_gray_diff, number_of_pyramid_levels, pyramid_type)

        # store resulting pyramid in video pyramid array
        # each channel of the pyramid needs to be stored in its corresponding vector
        # so that we can apply temporal filter to them seperately
        # so here we do this
        for level in range(number_of_pyramid_levels):

            # store frame pyramid into video pyramid
            pyramid[level][i] = pyramid_frame[level]


    # we have read all the frames we wanted.
    # release the video file
    videoFile.release()


    # prepare for temporal filtering

    # definition: video: a bunch of frames
    # definition: time signal matrix: a bunch of time signals
    # a video has the same amount of time signal matrix.
    # to get time signal matrix out of a video, it should be transposed

    # for applying a filter all over a signal, we are going to use filtfilt function
    # from scipy.signal . this function works as it should on a vector and filters it.
    # but given a matrix with more than one dimention, it selects latest raws of the matrix
    # which are vectors and filters them seperately. given a video matrix, it will first select
    # one frame, then filter each raw of the frame seperately. this will do some kind of spatial filtering
    # instead of temporal filtering.
    # so to achieve our goal, we first transpose our matrix so that dimentions 
    # change from (number of frames, height, width) into (width, height, number of frames)
    # and latest dimention of this matrix is a time sequence for each pixel on image.
    # and if we filter it, it will filter temporal signal of each pixel seperately and
    # this is exactly what we want

    print('transposing the tensor')

    # transpose each level of video pyramid
    for level in range(number_of_pyramid_levels):

        # note that transposed_video[x,y] is a time signal.
        # so we can get time signals this way.
        # and then process them by filtering or whatever else that we want
        # transpose video matrix
        pyramid[level] = np.transpose(pyramid[level])


    # start filtering

    print('time-domain filtering')
    print()

    # filter each level of the pyramid
    for level in range(number_of_pyramid_levels):

        # write progress
        sys.stdout.write("\033[F")  # back to previous line
        sys.stdout.write("\033[K")  # clear line
        print(f'pyramid level {level}')

        # filter each level of pyramid (which is a video matrix) temporally
        result[level] = signal.filtfilt(filter_coef_b, filter_coef_a, pyramid[level])


    print('transposing it again')

    # remove original pyramid for memory optimization
    del pyramid

    # transpose the result once again to get the original video matrix
    # here again, we need to transpose each video matrix in the video pyramid
    for level in range(number_of_pyramid_levels):

        # transpose video matrix
        # this way we transpose time signals once again into a video.
        result[level] = np.transpose(result[level])

    # if you want to keep the pyramid, you might also want to transpose it again.
    # pyramid = np.transpose(pyramid)


    # start rendering the video

    print('showing the result: (press q to stop)\n')


    # filename to write the result video in
    just_file_name = path.basename(input_file_name)
    # open an empty video file
    outFile = cv2.VideoWriter(f'results/out_eulerian_{just_file_name}.avi', cv2.VideoWriter_fourcc(*'MJPG'), video_frame_rate, (frame_w, frame_h))

    outFile_orig = cv2.VideoWriter(f'results/orig_eulerian_{just_file_name}.avi', cv2.VideoWriter_fourcc(*'MJPG'), video_frame_rate, (frame_w, frame_h))

    # start reading the video file once again.
    # because we need original frames to render the result and for memory optimization we have deallocated them
    videoFile = cv2.VideoCapture(input_file_name)

    # let the first frame out. because when reading at first we did this
    videoFile.read()



    # extracted temporal signals from video will be stored in these arrays.

    # define a temporal signal of zeros
    image_signal_original_x = np.zeros((frame_h, number_of_frames, 3), np.uint8)

    # define a temporal signal of zeros
    image_signal_out_x = np.zeros((frame_h, number_of_frames, 3), np.uint8)

    # define a temporal signal of zeros
    image_signal_original_y = np.zeros((number_of_frames, frame_w, 3), np.uint8)

    # define a temporal signal of zeros
    image_signal_out_y = np.zeros((number_of_frames, frame_w, 3), np.uint8)


    # start rendering output video
    # first we need to read the frames one by one, add them up to its corresponding filtered and
    # amplified frame and show or store the result.
    for i in range(len(result[0])):

        # read a frame
        _, frame_now = videoFile.read()

        # if there is no frame to read
        if frame_now is None:

            # stop reading frames.
            # and get out of the loop.
            break
        
        # convert color space from BGR into gray
        frame_gray = cv2.cvtColor(frame_now, cv2.COLOR_BGR2GRAY)
        
        # convert data type from uint8 into float32
        frame_float_gray = np.float32(frame_gray)

        # get corresponding part from the video pyramid
        pyramid_frame_result = [result[level][i] for level in range(number_of_pyramid_levels)]

        # collapse the pyramid and render it into one single frame
        frame_result_rendered = pyramid_rendr(pyramid_frame_result, spatial_coeff=spatial_coeff, pyramid_type=pyramid_type)

        # amplify the filtered frame and sum it up with the original one
        frame_mag = frame_float_gray * 1 + frame_result_rendered * alpha

        # check for boundaries. since it is an 8bit video, all the pixels need to be between 0 and 255
        # if there is overamplification and distortion because of clipping, we will see it as artifacts.
        guard_image(frame_mag)



        # render luminance layer into BGR image.
        # this is done by first splitting frame channels
        # then adding luminance to each of them
        # finally merging split channels again into a colorful frame.
        output_color = cv2.merge(cv2.split(frame_now) +
                                frame_result_rendered * alpha)

        # check for boundaries. since it is an 8bit video, all the pixels need to be between 0 and 255
        # if there is overamplification and distortion because of clipping, we will see it as artifacts.
        guard_image(output_color)


        # if we need to show the result while rendering
        if show_video == True:

            # show original frame
            cv2.imshow('original', np.uint8(frame_now))

            # show the magnified frame
            cv2.imshow('result', np.uint8(frame_mag))
            
            # show filtered frame
            # cv2.imshow('filtered', np.uint8(frame_result_rendered))

            # show one level of the pyramid
            # cv2.imshow('pyr', np.uint8(result[1][i]))

        # convert type of rendered image from float into 8 bit integer.
        output_color_int = np.uint8(output_color)

        # write original image into the output video file that is going to store original video.
        outFile_orig.write(frame_now)


        # extract time signal informaion from rendered video

        # get a column of pixels from the image.
        line_signal_out_x = output_color_int[:, time_signal_x]

        # get a raw of pixels from the image.
        line_signal_out_y = output_color_int[time_signal_y, :]

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



        # write rendered frame into output file
        outFile.write(np.uint8(output_color))
        sys.stdout.write("\033[F")  # back to previous line
        sys.stdout.write("\033[K")  # clear line
        print('frame:', i)


        # if there is any keypress, catch it.
        pressed_key = cv2.waitKey(1)

        # if pressed key was P
        if pressed_key == ord('p'):

            # pause the process and wait for next keypress
            cv2.waitKey(0)
        
        # if pressed key was Q
        if pressed_key == ord('q'):

            # stop rendering and get out of the rendering loop
            break

    # we are done reading all the frames we needed from original video file.
    # so we can release it
    videoFile.release()

    # we are done writing original frames into output video file.
    # so we can release it
    outFile.release()

    # we are done writing randered frames into output video file.
    # so we can release it
    outFile_orig.release()


    # write column of time signals from rendered video as an image
    cv2.imwrite(f'results/out_eulerian_{just_file_name}_x_{time_signal_x}.jpg', image_signal_out_x)

    # write raw of time signals from rendered video as an image
    cv2.imwrite(f'results/out_eulerian_{just_file_name}_y_{time_signal_y}.jpg', image_signal_out_y)

    # write column of time signals from original video as an image
    cv2.imwrite(f'results/orig_eulerian_{just_file_name}_x_{time_signal_x}.jpg', image_signal_original_x)

    # write raw of time signals from original video as an image
    cv2.imwrite(f'results/orig_eulerian_{just_file_name}_y_{time_signal_y}.jpg', image_signal_original_y)

    cv2.destroyAllWindows()


    # that's it!

    # return output file name
    return (f'results/orig_eulerian_{just_file_name}.avi',
            f'results/out_eulerian_{just_file_name}.avi')
