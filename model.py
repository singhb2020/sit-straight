# Model Defenition


# ------------------ Importing Libraries ------------------ #
import os
import wget


# ------------------ Model Class Defenition ------------------ #
class Model():
    # Contains Model

    def __init__(self, name):
        if name == 'lightning':
            self.url = 'https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3?lite-format=tflite'
            self.input_dim = (192, 192)
        elif name == 'thunder':
            self.url = 'https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/3?lite-format=tflite'
            self.input_dim = (256, 256)
        
        self.name = name
        self.file_path = ''

    def download_model(self):
        file_dir = os.path.join( os.getcwd(), f'{self.name}.tflite' )

        if os.path.exists(file_dir):
            print("File Already Downloaded")
        else:
            print("Downloading File")
            wget.download(self.url, file_dir)
        
        self.file_path = file_dir