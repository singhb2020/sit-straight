# Utility Functions


# ------------------ Importing Libraries ------------------ #
import json
import numpy as np
import math
import tensorflow as tf
import os
import random
import playsound


# ------------------ Utility Functions ------------------ #
def open_thresholds():
    try:
        threshold_file = open("thresholds.json", 'r')
        thesholds = json.load(threshold_file)
        threshold_file.close()
        return thesholds
    except Exception as e:
        print(e)
        print("An error has occured. Ensure the theshold file is created by calibrating.")
        return None

def def_points_dictionary():
    return {
    "nose" : 0,
    "left_eye": 1,
    "right_eye" : 2,
    "left_ear" : 3,
    "right_ear" : 4,
    "left_shoulder" : 5,
    "right_shoulder" : 6,
    "left_elbow" : 7,
    "right_elbow" : 8,
    "left_wrist" : 9,
    "right_wrist" : 10,
    "left_hip" : 11,
    "right_hip" : 12,
    "left_knee" : 13,
    "right_knee" : 14,
    "left_ankle" : 15,
    "right_ankle" : 16
}

def get_dist_between(frame, keypoints, p1, p2):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    POINTS = def_points_dictionary()
    
    p1 = keypoints[0][0][POINTS[p1]]
    p2 = keypoints[0][0][POINTS[p2]]

    p1 = np.array(p1[:2]*[y,x]).astype(int)
    p2 = np.array(p2[:2]*[y,x]).astype(int)

    dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    return dist

def get_dist_values(frame, keypoints):
    #Shoulder to Ear
    dists_right_ear_dist = get_dist_between(frame, keypoints, "right_shoulder", 'right_ear')
    dists_left_ear_dist = get_dist_between(frame, keypoints, "left_shoulder", 'left_ear')

    #Shoulder to Nose
    dists_right_nose_dist = get_dist_between(frame, keypoints, "right_shoulder", "nose")
    dists_left_nose_dist = get_dist_between(frame, keypoints, "left_shoulder", "nose")

    #Shoulder to Eyes
    dists_right_eyes_dist = get_dist_between(frame, keypoints, "right_shoulder", "right_eye")
    dists_left_eyes_dist = get_dist_between(frame, keypoints, "left_shoulder", "left_eye")

    return [dists_right_ear_dist, dists_left_ear_dist, dists_right_nose_dist, dists_left_nose_dist, dists_right_eyes_dist, dists_left_eyes_dist]

def reshape_image(frame, model):
    #Reshaping the camera input frame to fit the model
    image = frame.copy()
    image = tf.image.resize_with_pad( np.expand_dims(image, axis=0), model.input_dim[0], model.input_dim[1] )
    input_image = tf.cast(image, dtype=tf.float32)

    return input_image

def input_output_details(interpreter):
    #Getting the details from the interperter
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return input_details, output_details

def make_prediction(interpreter, input_details, output_details, input_image):
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_score = interpreter.get_tensor(output_details[0]['index'])

    return keypoints_with_score

def get_audio_list(filepath):
    #audio_list = []

    #for audio_file in os.scandir(filepath):
    #    audio_list.append(audio_file.path)
    #return audio_list
    return [file.path for file in os.scandir(filepath)]

def play_audio_recording(audio_list):
    audio_to_play = random.choice(audio_list)
    print(audio_to_play)
    playsound.playsound(audio_to_play)

def model_name_input():
    model_name = input("What is the model you want to use? lightning fast but bad, thunder slow but good:\n")
    if model_name == "lightning" or model_name == "thunder":
        return str(model_name)
    else:
        print("Try again, not a valid model\n")
        return model_name_input()

def calibration_input():
    if not os.path.exists('thresholds.json'):
        print("No calibration file exsists, running calibration\n")
        return True
    
    run_calibration = input("Do you want to run the calibration? Type yes otherwise defaults to no:\n")
    if run_calibration == "yes":
        return True
    else:
        return False
