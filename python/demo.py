import cv2
import numpy as np
import datetime


def nms(boxes, scores, nms_thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]
    num_detections = boxes.shape[0]
    suppressed = np.zeros((num_detections,), dtype=np.bool)
    for _i in range(num_detections):
        i = order[_i]
        if suppressed[i]:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]

        for _j in range(_i + 1, num_detections):
            j = order[_j]
            if suppressed[j]:
                continue

            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)

            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= nms_thresh:
                suppressed[j] = True
    keep = np.nonzero(suppressed == 0)[0]
    return keep


def decode(heatmap, scale, offset, size, threshold=0.1):
    heatmap = np.squeeze(heatmap)
    scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
    offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
    c0, c1 = np.where(heatmap > threshold)
    boxes = []
    if len(c0) > 0:
        for i in range(len(c0)):
            s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
            o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
            s = heatmap[c0[i], c1[i]]
            x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
            x1, y1 = min(x1, size[1]), min(y1, size[0])
            boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
        boxes = np.asarray(boxes, dtype=np.float32)
        keep = nms(boxes[:, :4], boxes[:, 4], 0.3)
        boxes = boxes[keep, :]
    return boxes


def transform(h, w):
    img_h_new, img_w_new = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
    scale_h, scale_w = img_h_new / h, img_w_new / w
    return img_h_new, img_w_new, scale_h, scale_w


def camera():
    cap = cv2.VideoCapture(0)
    # net = cv2.dnn.readNetFromONNX('centerface-small.onnx')
    net = cv2.dnn.readNetFromONNX('centerface.onnx')
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    img_h_new, img_w_new, scale_h, scale_w = transform(h, w)
    while True:
        ret, frame = cap.read()
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(img_w_new, img_h_new), mean=(0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)

        begin = datetime.datetime.now()
        # heatmap, height, offset = net.forward(["410", "411", "412"])
        hm, scale, off  = net.forward(["535", "536", "537"])
        end = datetime.datetime.now()
        print("cpu times = ", end - begin)
        dets = decode(hm, scale, off , (img_h_new, img_w_new), threshold=0.5)

        if len(dets) > 0:
            dets[:, 0:4:2] = dets[:, 0:4:2] / scale_w
            dets[:, 1:4:2] = dets[:, 1:4:2] / scale_h
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
        for det in dets:
            boxes, score = det[:4], det[4]
            cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
            cv2.putText(frame, str(score), (int(boxes[0]), int(boxes[1])), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('out', frame)
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


def test_image():

    net = cv2.dnn.readNetFromONNX('centerface.onnx')
    frame = cv2.imread('t3.jpeg')
    h, w = frame.shape[:2]
    img_h_new, img_w_new, scale_h, scale_w = transform(h, w)
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(img_w_new, img_h_new), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    begin = datetime.datetime.now()
    hm, scale, off = net.forward(["535", "536", "537"])
    end = datetime.datetime.now()
    print("cpu times = ", end - begin)
    dets = decode(hm, scale, off, (img_h_new, img_w_new), threshold=0.3)

    if len(dets) > 0:
        dets[:, 0:4:2] = dets[:, 0:4:2] / scale_w
        dets[:, 1:4:2] = dets[:, 1:4:2] / scale_h
    else:
        dets = np.empty(shape=[0, 5], dtype=np.float32)
    for det in dets:
        boxes, score = det[:4], det[4]
        cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
        cv2.putText(frame, str(score), (int(boxes[0]), int(boxes[1])), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('out', frame)
    # Press Q on keyboard to stop recording
    cv2.waitKey(0)


if __name__ == '__main__':
    # camera()
    test_image()
