# Calibartor Functions


# ------------------ Importing Libraries ------------------ #
from scipy import stats, mean
import matplotlib.pyplot as plt
import cv2
import time
import json


# ------------------ Importing Functions ------------------ #
from utils import get_dist_between, reshape_image, input_output_details, make_prediction
from debug import draw_connections, draw_keypoints, get_edge_dictionary


# ------------------ Calibrator Functions ------------------ #
def calibrate(model, interpretor):
    """
    calibrate: 
        Runs the calibration for each user. Saves a threshold.json file for output.
    """

    def find_trimmed_mean(input_list, trim_percent):
        """
        find_trimmed_mean: 
            Returns the trimmed mean for a list of input data
        """

        mean_list = []
        for dist in input_list:
            mean_list.append(stats.trim_mean(dist, trim_percent))

        return mean_list
    

    def get_dist_lst_values(input_list, frame, keypoints):
            """
            get_dist_lst_values: 
                Adds distance values for each frame in a list and returns a list of the lists.
            """

            #Unwrapping the list
            dists_right_ear, dists_left_ear, dists_right_nose, dists_left_nose, dists_right_eye, dists_left_eye = (x for x in input_list)

            #Shoulder to Ear
            dists_right_ear_dist = get_dist_between(frame, keypoints, "right_shoulder", 'right_ear')
            dists_right_ear.append(dists_right_ear_dist)
            dists_left_ear_dist = get_dist_between(frame, keypoints, "left_shoulder", 'left_ear')
            dists_left_ear.append(dists_left_ear_dist)

            #Shoulder to Nose
            dists_right_nose_dist = get_dist_between(frame, keypoints, "right_shoulder", "nose")
            dists_right_nose.append(dists_right_nose_dist)
            dists_left_nose_dist = get_dist_between(frame, keypoints, "left_shoulder", "nose")
            dists_left_nose.append(dists_left_nose_dist)

            #Shoulder to Eyes
            dists_right_eyes_dist = get_dist_between(frame, keypoints, "right_shoulder", "right_eye")
            dists_right_eye.append(dists_right_eyes_dist)
            dists_left_eyes_dist = get_dist_between(frame, keypoints, "left_shoulder", "left_eye")
            dists_left_eye.append(dists_left_eyes_dist)

            return [dists_right_ear, dists_left_ear, dists_right_nose, dists_left_nose, dists_right_eye, dists_left_eye]


    def calibrator_video():
        """
        calibrator_video: 
            Runs the opencv video for the calibration to run.
        """

        capture_front = cv2.VideoCapture(0)
        start_time = time.perf_counter()

        calibration_time = 30
        current_calibration_time = 0
        
        good_calibration_list = [[], [], [], [], [], []]
        bad_calibration_list = [[], [], [], [], [], []]

        while capture_front.isOpened():
            #ret_side, frame_side = capture_side.read()
            ret_front, frame_front = capture_front.read()

            confidence_threshold=0.4

            input_image_front = reshape_image(frame=frame_front, model=model)
            input_details, output_details = input_output_details(interpreter=interpretor)
            keypoint_score_front = make_prediction(interpreter=interpretor, input_details=input_details, output_details=output_details, input_image=input_image_front)

            EDGES = get_edge_dictionary()
            draw_keypoints(frame=frame_front, keypoints=keypoint_score_front, confidence_threshold=0.4)
            draw_connections(frame=frame_front, keypoints=keypoint_score_front, edges=EDGES, confidence_threshold=confidence_threshold)

            if calibration_time > current_calibration_time:
                #Calibrate Good Posture
                cv2.putText(frame_front, 'Calibrating Good Posture : ' + str(int(calibration_time - current_calibration_time)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                good_calibration_list = get_dist_lst_values(good_calibration_list, frame_front, keypoint_score_front)
            elif (2*calibration_time) > current_calibration_time:
                #Calibrate Bad Posture
                cv2.putText(frame_front, 'Calibrating Bad Posture : '  + str(int(2*calibration_time - current_calibration_time)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                bad_calibration_list = get_dist_lst_values(bad_calibration_list, frame_front, keypoint_score_front)
            else:
                break

            current_calibration_time = time.perf_counter() - start_time
            print(current_calibration_time)
            cv2.imshow("Front", frame_front)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        capture_front.release()
        cv2.destroyAllWindows()

        return good_calibration_list, bad_calibration_list


    def view_calibartion(thresholds, jump_percent, raw_good, raw_bad, trimmed_good, trimmed_bad):
        """
        view_calibration: 
            Shows the results of the calibration using graphs.
        """

        fig, axs = plt.subplots(2,3)
        fig.set_size_inches(18.5, 10.5)

        for i, threshold in enumerate(thresholds):
            axs[int(i%2),int(i/2)].set_title(threshold)
            axs[int(i%2),int(i/2)].plot(raw_good[i] + raw_bad[i], color='orange')
            axs[int(i%2),int(i/2)].axhline(trimmed_good[i], color='green', xmin=0, xmax=len(raw_good[i])/len(raw_good[i]+raw_bad[i]))
            axs[int(i%2),int(i/2)].axhline(trimmed_bad[i], color='red', xmin=len(raw_good[i])/len(raw_good[i]+raw_bad[i]), xmax=1)
            axs[int(i%2),int(i/2)].axhline(trimmed_bad[i] + jump_percent*(trimmed_good[i]-trimmed_bad[i]), color='blue')

    #Finding Threshold
    thresholds = {
        'dists_right_ear' : 0,
        'dists_left_ear' : 0,
        'dists_right_nose' : 0,
        'dists_left_nose' : 0,
        'dists_right_eye' : 0,
        'dists_left_eye' : 0
    }

    raw_good, raw_bad = [], []
    raw_good, raw_bad = calibrator_video()

    #Using trimmed means for threshold Values
    trimmed_percent = 0.1
    trimmed_good = find_trimmed_mean(raw_good, trimmed_percent)
    trimmed_bad = find_trimmed_mean(raw_bad, trimmed_percent)

    jump_percent = 0.5
    for i, threshold_key in enumerate(thresholds):
        thresholds[threshold_key] = trimmed_bad[i] + jump_percent*(trimmed_good[i]-trimmed_bad[i])

    view_calibartion(thresholds, jump_percent, raw_good, raw_bad, trimmed_good, trimmed_bad)
    
    return thresholds


def save_thresholds(input_dict):
    """
    save_thresholds: 
        Saves the theshold.json to be used after inital calibration
    """

    theshold_file = open("thresholds.json", 'w')
    json.dump(input_dict, theshold_file)
    theshold_file.close