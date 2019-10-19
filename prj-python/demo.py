import cv2
from centerface import CenterFace


def camera():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    centerface = CenterFace(h, w)
    while True:
        ret, frame = cap.read()
        dets, lms = centerface(frame, threshold=0.5)
        for det in dets:
            boxes, score = det[:4], det[4]
            cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
        for lm in lms:
            cv2.circle(frame, (int(lm[0]), int(lm[1])), 2, (0, 0, 255), -1)
            cv2.circle(frame, (int(lm[2]), int(lm[3])), 2, (0, 0, 255), -1)
            cv2.circle(frame, (int(lm[4]), int(lm[5])), 2, (0, 0, 255), -1)
            cv2.circle(frame, (int(lm[6]), int(lm[7])), 2, (0, 0, 255), -1)
            cv2.circle(frame, (int(lm[8]), int(lm[9])), 2, (0, 0, 255), -1)
        cv2.imshow('out', frame)
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


def test_image():
    frame = cv2.imread('000388.jpg')
    h, w = frame.shape[:2]
    centerface = CenterFace(h, w)
    dets, lms = centerface(frame, threshold=0.35)
    for det in dets:
        boxes, score = det[:4], det[4]
        cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
    for lm in lms:
        cv2.circle(frame, (int(lm[0]), int(lm[1])), 2, (0, 0, 255), -1)
        cv2.circle(frame, (int(lm[2]), int(lm[3])), 2, (0, 0, 255), -1)
        cv2.circle(frame, (int(lm[4]), int(lm[5])), 2, (0, 0, 255), -1)
        cv2.circle(frame, (int(lm[6]), int(lm[7])), 2, (0, 0, 255), -1)
        cv2.circle(frame, (int(lm[8]), int(lm[9])), 2, (0, 0, 255), -1)
    cv2.imshow('out', frame)
    cv2.waitKey(0)


if __name__ == '__main__':
    # camera()
    test_image()
