#Detector Functions


# ------------------ Importing Libraries ------------------ #
import cv2
import os
import time


# ------------------ Importing Functions ------------------ #
from utils import open_thresholds, get_audio_list, reshape_image, input_output_details, make_prediction, get_dist_values, play_audio_recording
from debug import draw_keypoints, draw_connections, get_edge_dictionary


# ------------------ Detector Function ------------------ #
def detector(model, interpretor, debug):
    """
    detector: 
        Main function that operates the detection and event trigger for the application.
    """

    capture_front = cv2.VideoCapture(0)

    time_threshold = 15
    dist_thresholds = open_thresholds()

    basepath = os.getcwd()

    audio_filepath = os.path.join(basepath, '..\..\Database\Audio Recordings\Converted')
    audio_list = get_audio_list(audio_filepath)
    playing_audio = False

    time_count = 0
    start_time = time.time() 

    while capture_front.isOpened():
        #Read Camera Input
        ret_front, frame_front = capture_front.read()

        #Image Reshape
        input_image_front = reshape_image(frame=frame_front, model=model)

        #Setup Tensor Input and Output
        input_details, output_details = input_output_details(interpreter=interpretor)

        #Make Prediction
        keypoint_score_front = make_prediction(interpreter=interpretor, input_details=input_details, output_details=output_details, input_image=input_image_front)

        if debug:
            #Rendering Points
            confidence_threshold=0.4
            draw_keypoints(frame=frame_front, keypoints=keypoint_score_front, confidence_threshold=0.4)

            #Rendering Edges
            EDGES = get_edge_dictionary()
            draw_connections(frame=frame_front, keypoints=keypoint_score_front, edges=EDGES, confidence_threshold=confidence_threshold)

        #Determine Distances
        current_distances = get_dist_values(frame=frame_front, keypoints=keypoint_score_front)
                
        
        #If all distances are above threshold, then posture is correct. Else, posture is not correct.
        if all([current_distances[i] > dist_thresholds[threshold] for i, threshold in enumerate(dist_thresholds)]):
            time_count = 0
        else:
            if not playing_audio:
                current_time = time.time()
                time_count += current_time - start_time
                start_time = current_time
            else:
                time_count = 0

            if (time_count > time_threshold):
                playing_audio = True
                time_count = 0

                play_audio_recording(audio_list)
                
                playing_audio = False
                time_count = 0
        
        
        if debug:
            print(int(time_count))
            cv2.imshow("Front", frame_front)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture_front.release()
    cv2.destroyAllWindows()