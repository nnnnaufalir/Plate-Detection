import cv2 
import numpy as np
from time import sleep
from loadModel import *
from ocr import ocr_it

# cap = cv2.VideoCapture(0)
cap=cv2.VideoCapture('hitam.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

threshold = .5
while cap.isOpened(): 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,
                min_score_thresh=threshold,
                agnostic_mode=False)

  
    
    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    # print('scores', detections['detection_scores'])
    # print('threshold terpenuhi') if max(detections['detection_scores'])>.5 else print('not detect object')
    if max(detections['detection_scores'])>threshold :
        temp, region = ocr_it(image_np_with_detections, detections, threshold)
        print(temp)
    else :
        print('nggak ada')
        
    # sleep(1)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break



    
