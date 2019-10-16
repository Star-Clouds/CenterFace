import cv2
from centerface import CenterFace


def camera():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    centerface = CenterFace(h, w)
    while True:
        ret, frame = cap.read()
        dets = centerface(frame, threshold=0.5)
        for det in dets:
            boxes, score = det[:4], det[4]
            cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
        cv2.imshow('out', frame)
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


def test_image():
    frame = cv2.imread('000388.jpg')
    h, w = frame.shape[:2]
    centerface = CenterFace(h, w)
    dets = centerface(frame, threshold=0.5)
    for det in dets:
        boxes, score = det[:4], det[4]
        cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
    cv2.imshow('out', frame)
    cv2.waitKey(0)


if __name__ == '__main__':
    # camera()
    test_image()
