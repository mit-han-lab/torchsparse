import h5py
import numpy as np
import os.path

def minmax(input_list):
    """
    Compute the minimum and maximum values of a list.

    Args:
        input_list (list): A list of numeric values.

    Returns:
        tuple: A tuple containing two values - the minimum and maximum values in the input list.
    """
    mn = min(input_list)
    mx = max(input_list)
    return mn, mx

def linearfit(x, min_orig, max_orig, min_desired, max_desired):
    """
    Perform linear scaling (fitting) of a value within a specified range to a desired range.

    This function takes a value `x` within the original range specified by `min_orig` and `max_orig`, 
    and scales it to a new valuewithin the desired range specified by `min_desired` and `max_desired`. 
    This is achieved through linear interpolation.

    Args:
        x (numpy.ndarray): The value to be scaled.
        min_orig (float): The minimum value of the original range.
        max_orig (float): The maximum value of the original range.
        min_desired (float): The minimum value of the desired range.
        max_desired (float): The maximum value of the desired range.

    Returns:
        numpy.ndarray: A NumPy array containing scaled values within the desired range.
    """
    return ((x - min_orig) / (max_orig - min_orig)) * (max_desired - min_desired) + min_desired


if __name__ == '__main__':
    # lines 27-54 are from pre-existing code written by Ben Wagner
    file = h5py.File('../mg22simulated/output_digi_HDF_Mg22_Ne20pp_8MeV.h5', 'r')
    
    original_keys = list(file.keys())
    original_length = len(original_keys)
    
    event_lens = np.zeros(original_length, int)
    for i in range(original_length):
        event = original_keys[i]
        event_lens[i] = len(file[event])
        
    ISOTOPE = 'Mg22'
    file_name = ISOTOPE + '_w_key_index'
    # **only doing this if the file doens't exist already, as the conversion takes a while**
    if not os.path.exists('../mg22simulated/' + file_name + '.npy'):
        event_data = np.zeros((original_length, np.max(event_lens), 13), float) 
        for n in range(len(original_keys)):
            name = original_keys[n]
            event = file[name]
            ev_len = len(event)
            #converting event into an array
            for i,e in enumerate(event):
                instant = np.array(list(e))
                event_data[n][i][:12] = np.array(instant)
                event_data[n][i][-1] = float(n) #insert index value to find corresponding event ID
        np.save('../mg22simulated/' + file_name, event_data)
    
    data = np.load( '../mg22simulated/' + ISOTOPE + '_w_key_index' + '.npy')
    
    # code written by Ian Heung    
    EVENTS = len(data) # total number of events
    DETECTIONS = len(data[0]) # total number of detections per event (even the empty detections)
    
    x_array = np.zeros((EVENTS, DETECTIONS))
    y_array = np.zeros((EVENTS, DETECTIONS))
    z_array = np.zeros((EVENTS, DETECTIONS))
    amp_array = np.zeros((EVENTS, DETECTIONS))
    track_id_array = np.zeros((EVENTS, DETECTIONS))
    
    for event in range(EVENTS):
        for detection in range(DETECTIONS):
            x_array[event][detection] = data[event][detection][0]
            y_array[event][detection] = data[event][detection][1]
            z_array[event][detection] = data[event][detection][2]
            amp_array[event][detection] = data[event][detection][4]
            track_id_array[event][detection] = data[event][detection][5]
    
    x = []
    y = []
    z = []
    t = []
    amp = []
    
    for event in range(EVENTS):
        for detection in data[event]:
            if not(detection[0] == 0. and detection[1] == 0. and detection[2] == 0. and detection[3] == 0. and detection[4] == 0.):
                x.append(detection[0])
                y.append(detection[1])
                z.append(detection[2])
                t.append(detection[3])
                amp.append(detection[4])
    
    x_min, x_max = minmax(x)
    y_min, y_max = minmax(y)
    z_min, z_max = minmax(z)
    
    x_fit = linearfit(x_array, x_min, x_max, 0, 499).astype(int)
    y_fit = linearfit(y_array, y_min, y_max, 0, 499).astype(int)
    z_fit = linearfit(z_array, z_min, z_max, 0, 499).astype(int)

    amp = np.array(amp)
    amp_mean = amp.mean()
    amp_stdev = amp.std()
    
    # Perform z-score normalization
    amp_array = (amp_array - amp_mean) / amp_stdev
    
    total_coords = np.stack((x_fit, y_fit, z_fit), axis=-1)
    total_features = amp_array.reshape(10000, 1476, 1)
    total_features = np.concatenate((total_coords, total_features), axis=2)
    total_labels = track_id_array.reshape(10000, 1476, 1)
    
    np.save('../mg22simulated/' + ISOTOPE + "_coords.npy", total_coords)
    np.save('../mg22simulated/' + ISOTOPE + "_feats.npy", total_features)
    np.save('../mg22simulated/' + ISOTOPE + "_labels.npy", total_labels)

    print("Coords Shape: ", end="")
    print(total_coords.shape)
    print("Feats Shape: ", end="")
    print(total_features.shape)
    print("Labels Shape: ", end="")
    print(total_labels.shape)
    
    print("data_processing.py: Successful")
