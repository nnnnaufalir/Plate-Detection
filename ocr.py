import keras_ocr
import imutils as im

pipe = keras_ocr.pipeline.Pipeline()

def ocr_it(image, detections, detection_threshold):
    
    # Scores, boxes and classes above threhold
    scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]
    
    # Full image dimensions
    width = image.shape[1]
    height = image.shape[0]
    
    # Apply ROI filtering and OCR
    for idx, box in enumerate(boxes):
        roi = box*[height, width, height, width]
        region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]

        #cv2.imwrite('preprocessed_image.jpg', region)

        #gambar=cv2.imread("preprocessed_image.jpg")
        gambar=region
        rotated=gambar
        images =[(rotated)]
        temp=[]
        ocr_result = pipe.recognize(images)
        for rotated, ocr_result in zip(images,ocr_result):
            for images in ocr_result:
                temp.append(images[0].upper())
        # plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        print(temp)
        if temp==[] or len(temp)<3:
            
            pass
        else :
            temp = temp[:3]
            i=0
            while temp[0].isnumeric() or temp[1].isalpha() or temp[2].isnumeric or i==15:
                i += 1
                rotated=im.rotate_bound(gambar,i)

                images =[(rotated)]
                temp=[]
                ocr_result = pipe.recognize(images)
                for rotated, ocr_result in zip(images,ocr_result):
                    for images in ocr_result:
                        temp.append(images[0].upper())
                if len(temp)<3:
                    break
                else:
                    temp=temp[:3]
                    
                    break

        
        # print(temp)
        return temp, region