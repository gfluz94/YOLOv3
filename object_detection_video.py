import os
import time
import numpy as np
import cv2

from object_detection_image import make_blob, Yolo, get_bounding_boxes, non_max_suppresion

VIDEO_PATH = "./videos"
OUTPUT_PATH = "./output"
MODEL_PATH = "./yolo-coco-data"

def detect_objects(frame, nms_results, boxes, scores, classes, labels):
    counter = 1
    np.random.seed(1)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
    for i in nms_results.flatten():
        text_label = labels[classes[i]]
        print(f"Object {i}: {text_label}")
        counter+=1
        top_left = tuple(boxes[i][:2])
        bottom_right = tuple(boxes[i][2:])
        color = colors[classes[i]].tolist()
        cv2.rectangle(frame, top_left, bottom_right, color, 2)
        show_text = f"{text_label}: {scores[i]:.3f}"
        cv2.putText(frame, show_text, (top_left[0], top_left[1]-5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)


if __name__=="__main__":
    # Reading video
    video_name = "original.mp4"
    video = cv2.VideoCapture(os.path.join(VIDEO_PATH, video_name))
    writer = None
    height, width = None, None
    total_time = 0
    frames = 1

    # Iterating over frames
    while True:
        start = time.time()
        ret, frame = video.read()
        if not ret:
            break
        if width is None or height is None:
            height, width = frame.shape[:2]

        # Getting blob out of frame
        blob = make_blob(frame, to_shape=(416, 416), rgb=True)

        # Getting Model and Predicting from blob
        labels_file = os.path.join(MODEL_PATH, "coco.names")
        config_file = os.path.join(MODEL_PATH, "yolov3.cfg")
        weights_file = os.path.join(MODEL_PATH, "yolov3.weights")

        yolo = Yolo(config_file, weights_file, labels_file)
        network_output = yolo.predict(blob)
        
        # Getting boxes and applying Non Maximum Suppression
        probability_threshold = 0.5
        boxes, scores, classes = get_bounding_boxes(network_output, yolo.labels, (height, width), probability_threshold)
        results = non_max_suppresion(boxes, scores, probability_threshold, nms_threshold=0.3)

        # Drawing bouding boxes in frame
        print(f"---------------- Frame {frames} ----------------")
        detect_objects(frame, results, boxes, scores, classes, yolo.labels)
        delta_time = time.time() - start
        total_time+=delta_time
        print(f"------- Frame {frames} concluded in {delta_time:.4f} s. -------\n")
        frames+=1

        # Generating new video with detection by adding frame
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(os.path.join(OUTPUT_PATH, video_name), fourcc, 30,
                                                (frame.shape[1], frame.shape[0]), True)
        writer.write(frame)

    print(f"{frames-1} frames edited in {total_time:.4f}!!!")
    video.release()
    writer.release()

    