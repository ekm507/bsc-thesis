# this code is for reproducing results

import videomag
from videomag import eulerian, differential, correlation
import os
import sys
import time

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = '/home/erfan/Downloads/baby.mp4'

if not os.path.exists('results'):
    os.system('mkdir results')


def get_filter_properties(cutoff_low, cutoff_high, order=5, bandtype='lowpass'):
    return (cutoff_low, cutoff_high, order, bandtype)


def reproduce_results(filename, alpha, number_of_frames, filter_properties,
                      frequency, spatial_coefficients, number_of_jump_frames=0,
                      time_signal_x=0, time_signal_y=0, show_video=True, pyramid_type='gaussian', buffer_size=5, methods:list=[
                          'eulerian',
                          'correlation',
                          'differential'
                      ]):

    if 'eulerian' in methods:

        print('------------------------eulerian------------------------')

        time_0 = time.time()

        eulerian(filename, alpha, (filter_properties[0], filter_properties[1]),
                        spatial_coefficients, number_of_frames,
                        number_of_jump_frames=number_of_jump_frames,
                        filter_band_type=filter_properties[3],
                        time_signal_x=time_signal_x, time_signal_y=time_signal_y, show_video=show_video, pyramid_type=pyramid_type)

        time_1 = time.time()

        print('execution time:', time_1 - time_0)
                      

    if 'correlation' in methods:

        print('------------------------correlation---------------------')

        time_0 = time.time()

        correlation(filename, alpha, frequency, spatial_coefficients,
                            buffer_size=buffer_size, number_of_frames=number_of_frames, number_of_jump_frames=number_of_jump_frames,
                            time_signal_x=time_signal_x, time_signal_y=time_signal_y, show_video=show_video, pyramid_type=pyramid_type)

        time_1 = time.time()

        print('execution time:', time_1 - time_0)
        

    if 'differential' in methods:

        print('------------------------differential--------------------')

        time_0 = time.time()

        differential(filename, alpha, number_of_frames, spatial_frequency_coefficients=spatial_coefficients, buffer_size=buffer_size,
                    number_of_jump_frames=number_of_jump_frames,
                    time_signal_x=time_signal_x, time_signal_y=time_signal_y, show_video=show_video, pyramid_type=pyramid_type)


        time_1 = time.time()

        print('execution time:', time_1 - time_0)


alpha = 20
number_of_frames = 500
number_of_jump_frames = 0
frequency = 1
filter_properties = get_filter_properties(frequency, frequency, order=5, bandtype='low')
filter_properties = get_filter_properties(0.7, 0.8, order=5, bandtype='bandpass')
spatial_coefficients = [0, 0, 1, 1, 1]
spatial_coefficients = [1, 1]
time_signal_x = 1
time_signal_y = 1
buffer_size = 5
time_signal_x = 501
time_signal_y = 236
time_signal_x = 0
time_signal_y = 0
pyramid_type='laplacian'
show_video = True
# show_video = False
methods = [
    # 'eulerian',
    'correlation',
    # 'differential'
    ]


reproduce_results(filename, alpha, number_of_frames, filter_properties, frequency,
                  spatial_coefficients, number_of_jump_frames, time_signal_x, time_signal_y, show_video, pyramid_type, buffer_size, methods)

# reproduce_results(filename, alpha, 20, filter_properties, frequency,
#                   spatial_coefficients, number_of_jump_frames, time_signal_x, time_signal_y, show_video)

# reproduce_results(filename, alpha, 50, filter_properties, frequency,
#                   spatial_coefficients, number_of_jump_frames, time_signal_x, time_signal_y, show_video)

# reproduce_results(filename, alpha, 100, filter_properties, frequency,
#                   spatial_coefficients, number_of_jump_frames, time_signal_x, time_signal_y, show_video)

# reproduce_results(filename, alpha, 150, filter_properties, frequency,
#                   spatial_coefficients, number_of_jump_frames, time_signal_x, time_signal_y, show_video)

# reproduce_results(filename, alpha, 200, filter_properties, frequency,
#                   spatial_coefficients, number_of_jump_frames, time_signal_x, time_signal_y, show_video)


# eulerian('/home/erfan/Downloads/baby.mp4', 20, 5, [1, 1, 1], 50, 0, 'low', time_signal_y=100)
# correlation('/home/erfan/Downloads/baby.mp4', 20, 2, [1, 1, 0], 5, 100, time_signal_x=513, time_signal_y=236)
# differential(filename, 20, 1000, time_signal_x=0, time_signal_y=0, number_of_jump_frames=number_of_jump_frames)


# eulerian(filename, alpha, 0.5, [1, 1, 1], 500, 0, 'high', time_signal_y=100)

# correlation(
#     filename, alpha, 2, [1, 1, 1], 5, number_of_frames, time_signal_x=0, time_signal_y=0)

# differential(filename, 20, 1000, time_signal_x=513,
#                       time_signal_y=236, show_video=True)
