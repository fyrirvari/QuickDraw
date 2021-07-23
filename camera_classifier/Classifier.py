import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore


class InferenceEngineClassifier:
    def __init__(self, configPath=None, weightsPath=None,
            device='CPU', extension=None, classesPath=None):
        
        # Add code for Inference Engine initialization
        self.ie = IECore()
        
        # Add code for model loading
        self.net = self.ie.read_network(model=configPath)
        self.exec_net = self.ie.load_network(network=self.net, device_name=device)

        # Add code for classes names loading
        with open(classesPath, 'r') as f:
            self.labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
        
        return

    def get_top(self, prob, topN=1):
        result = []
        
        # Add code for getting top predictions
        result = np.squeeze(prob)
        result = np.argsort(result)[-topN:][::-1]
        
        return result

    def _prepare_image(self, image, h, w):
    
        # Add code for image preprocessing
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        
        return image

    def classify(self, image):
        probabilities = None
        
        # Add code for image classification using Inference Engine
        input_blob = next(iter(self.net.input_info)) 
        out_blob = next(iter(self.net.outputs))
        
        n, c, h, w = self.net.input_info[input_blob].input_data.shape
        
        image = self._prepare_image(image, h, w)
        
        output = self.exec_net.infer(inputs = {input_blob: image})
        
        output = output[out_blob]
        
        return output