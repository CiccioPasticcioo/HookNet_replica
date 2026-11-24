import numpy as np

CLICK_PROBABILITY = 0.1
SEGMENT_LENGTH = 16384
OUTPUT_LENGTH = 128

np.random.seed(42)

# Mono impulsive click injection
def click_adder(audio):

    click_positions = []

    data = audio.copy()
 
    for start in range(0 , len(data) - SEGMENT_LENGTH + 1, SEGMENT_LENGTH):

        segment = data[start : start + SEGMENT_LENGTH]

        if np.random.random() < CLICK_PROBABILITY:

            position = np.random.randint(0 , SEGMENT_LENGTH)
            amplitude = np.random.uniform(0.3 , 1.0)
            sign = np.random.choice([-1 , 1])
            
            current_sign = sign

            if abs(segment[position] + amplitude * sign) > 1: current_sign = -current_sign
                
            data[position + start] += amplitude * current_sign

            click_positions.append((position + start))

    return data, click_positions

# Multi impulsive click injection
def click_adder(audio):

    click_positions = []

    data = audio.copy()

    for start in range(0 , len(data) - SEGMENT_LENGTH + 1, SEGMENT_LENGTH):


        segment = data[start : start + SEGMENT_LENGTH]

        if np.random.random() < CLICK_PROBABILITY:

            length = np.random.randint(1 , 4)
            position = np.random.randint(0 , (SEGMENT_LENGTH - length +1))
            amplitude = np.random.uniform(0.3 , 1.0)
            
            sign = np.random.choice([-1 , 1])
            
            current_sign = sign

            for index in range(length):
                
                if abs(segment[position + index] + amplitude * current_sign) > 1: current_sign = -current_sign

                data[position + start +index] += amplitude * current_sign

                click_positions.append((position + start + index))

    #print("Numero di click inseriti:", len(click_positions))

    return data, click_positions

# Click Target Vector
def ground_truth(click_positions, n_segments):

    samples_x_segment = SEGMENT_LENGTH // OUTPUT_LENGTH

    click_matrix = np.zeros((n_segments, OUTPUT_LENGTH) , dtype = np.float32)

    for position in click_positions:

        segment_index = position // SEGMENT_LENGTH 

        local_position = (position % SEGMENT_LENGTH) // samples_x_segment

        click_matrix[segment_index , local_position] = 1.0

    return click_matrix


