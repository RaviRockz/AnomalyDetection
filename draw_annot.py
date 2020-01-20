import cv2


def draw_on_image(image, boxes):
    (im_hei, im_wid) = image.shape[:2]
    for box in boxes:
        left = int(box[1] * im_wid)
        top = int(box[0] * im_hei)
        right = int(box[3] * im_wid)
        bottom = int(box[2] * im_hei)
        cv2.rectangle(image, (left, top), (right, bottom), color=(0, 0, 255), thickness=2)
    return image
