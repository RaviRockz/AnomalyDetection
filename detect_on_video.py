import cv2
import numpy as np
import tensorflow as tf
import os
from draw_annot import draw_on_image
import glob

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

object_detection_model = 'model/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
video_path = glob.glob('input/*.mp4')
output_path = 'output'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(object_detection_model, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

for vid_p in video_path:
    video = cv2.VideoCapture(vid_p)
    frame_array = []
    frame_count = 1
    ret = True
    while ret:
        ret, frame = video.read()
        if frame is not None:
            print('[INFO] Frame', frame_count)
            image_expanded = np.expand_dims(frame, axis=0)

            (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                     feed_dict={image_tensor: image_expanded})

            annot_list = []
            anomaly_id = [2, 3, 4, 6, 36, 41, 42]
            for i in range(int(num[0])):
                if scores[0][i] >= 0.8 and int(classes[0][i]) in anomaly_id:
                    annot_list.append(boxes[0][i])
            height, width, channels = frame.shape
            size = (width, height)
            frame = draw_on_image(frame, annot_list)
            frame_array.append(frame)
            frame_count += 1
    print('[INFO] Saving Video')
    out = cv2.VideoWriter(output_path + '/' + vid_p.split('/')[-1], cv2.VideoWriter_fourcc(*'MP4V'), 25.0, size)

    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()

