
"""
 "unlike people, comupters understand". Erfan Kheyrollahi - 2018
"""
import cv2
import numpy as np
from os import path
from scipy import signal
import sys
from pyramids import pyramid_make, pyramid_rendr
def guard_image(image:np.ndarray):
    
    image[image > 255] = 255
    
    image[image < 0] = 0
def eulerian(input_file_name:str, alpha:float, temporal_frequency_bands,
             spatial_frequency_coefficients, number_of_frames:int, number_of_jump_frames:int=0,
             filter_band_type:str='bandpass', time_signal_x:int=0, time_signal_y:int=0, show_video:bool=True, pyramid_type='gaussian'
             ):
    """
        
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
    
    videoFile = cv2.VideoCapture(input_file_name)
    video_frame_rate = videoFile.get(cv2.CAP_PROP_FPS)

    order = 5
    
    spatial_coeff = spatial_frequency_coefficients
    number_of_pyramid_levels = len(spatial_coeff)
    fs = video_frame_rate
    
    if isinstance(temporal_frequency_bands, (list, tuple)):
        if len(temporal_frequency_bands) > 1:
            cutoff_low = temporal_frequency_bands[0]
            cutoff_high = temporal_frequency_bands[1]
        else:
            cutoff_low = temporal_frequency_bands[0]
            cutoff_high = temporal_frequency_bands[0]
    else:
        cutoff_low = temporal_frequency_bands
        cutoff_high = temporal_frequency_bands
    band_type = filter_band_type
    nyq = 0.5 * fs
    
    
    
    if band_type in ['bandpass', 'bandstop']:
        normal_cutoff = [cutoff_low / nyq, cutoff_high / nyq]
    elif band_type in ['high', 'highpass']:
        normal_cutoff = cutoff_high / nyq
    elif band_type in ['low', 'lowpass']:
        normal_cutoff = cutoff_low / nyq
    else:
        print("filter type is not correctly set", file=sys.stderr)
        exit(2)
    
    filter_coef_b, filter_coef_a = signal.butter(order, normal_cutoff, btype=band_type, analog=False)
    window_size = len(filter_coef_a)
    
    _, frame_0 = videoFile.read()
    frame0_gray = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)
    frame_h, frame_w, number_of_channels = np.shape(frame_0)
    print(f'frame size is ({frame_w}, {frame_h}). frame rate is {np.round(video_frame_rate)}')
    

    # filename to write the result video in
    just_file_name = path.basename(input_file_name)
    # open an empty video file
    outFile = cv2.VideoWriter(f'results/out_eulerian_{just_file_name}.avi', cv2.VideoWriter_fourcc(*'MJPG'), video_frame_rate, (frame_w, frame_h))

    # outFile_orig = cv2.VideoWriter(f'results/orig_eulerian_{just_file_name}.avi', cv2.VideoWriter_fourcc(*'MJPG'), video_frame_rate, (frame_w, frame_h))



    
    pyramid_0 = [np.zeros((window_size, frame_h//2**(i), frame_w//2**(i)), np.float32)
            for i in range(number_of_pyramid_levels)]
    
    filtered_pyramid =  [np.zeros((window_size, frame_h//2**(i), frame_w//2**(i)), np.float32)
            for i in range(number_of_pyramid_levels)]
    pyramid =  [np.zeros((window_size, frame_h//2**(i), frame_w//2**(i)), np.float32)
            for i in range(number_of_pyramid_levels)]
    pyramid2 =  [np.zeros((window_size, frame_h//2**(i), frame_w//2**(i)), np.float32)
            for i in range(number_of_pyramid_levels)]

    frame_pyramid_list =  [np.zeros((window_size, frame_h//2**(i), frame_w//2**(i)), np.float32)
            for i in range(number_of_pyramid_levels)]
    
    
    for i in range(number_of_jump_frames):
        _, frame_now = videoFile.read()
    

    print()

    zi = signal.lfilter_zi(filter_coef_a, filter_coef_a)

    for j in range(number_of_frames//window_size):

        filtered_pyramid = pyramid_0.copy()
        pyramid = pyramid_0.copy()


        for i in range(window_size):


            _, frame_now = videoFile.read()
            if frame_now is None:
                window_size = i
                break
            frame_float = np.float32(frame_now)
            frame_float_gray = cv2.cvtColor(frame_float, cv2.COLOR_BGR2GRAY)
            frame_float_gray_diff = frame_float_gray- np.float32(frame0_gray)
            pyramid_frame = pyramid_make(frame_float_gray_diff, number_of_pyramid_levels, pyramid_type)
            
            for level in range(number_of_pyramid_levels):
                pyramid[level][i] = pyramid_frame[level]

            pyramid_frame_now = pyramid_make(frame_float_gray, number_of_pyramid_levels, pyramid_type)
            for level in range(number_of_pyramid_levels):
                frame_pyramid_list[level][i] = pyramid_frame_now[level]

        for level in range(number_of_pyramid_levels):
            pyramid2[level] = pyramid[level]
            pyramid2[level] = np.transpose(pyramid2[level])

        filtered_pyramid = pyramid.copy()
        for level in range(number_of_pyramid_levels):

            _, X, Y = np.shape(pyramid2[level]) 
            for x in range(Y):
                for y in range(X):
                    filtered_pyramid[level][:,y,x], zi = signal.lfilter(filter_coef_b, filter_coef_a, pyramid2[level][x,y], zi=zi)
            







        for i in range(window_size):
            
            sys.stdout.write("\033[F")  
            sys.stdout.write("\033[K")  
            print(f'frame {j*window_size +i}')
            # print(f'signal part {j}', f'frame {i}')
            


            pyramid_mag = [
                filtered_pyramid[level][i] * alpha * spatial_coeff[level] + frame_pyramid_list[level][i]

                for level in range(number_of_pyramid_levels)
            ]


            frame_result_rendered = pyramid_rendr(pyramid_mag, spatial_coeff=[1], pyramid_type=pyramid_type)
            
            # frame_mag = frame_float_gray * 1 + frame_result_rendered * alpha
            
            
            guard_image(frame_result_rendered)
            
            if show_video == True:
                cv2.imshow('result', np.uint8(frame_result_rendered))
            
            outFile.write(np.uint8(cv2.cvtColor(frame_result_rendered, cv2.COLOR_GRAY2BGR)))
      
            pressed_key = cv2.waitKey(1)
            
            if pressed_key == ord('p'):
                
                cv2.waitKey(0)
            
            
            if pressed_key == ord('q'):
                
                break
        
        

    videoFile.release()

    outFile.release()
    
    # cv2.destroyAllWindows()
    
