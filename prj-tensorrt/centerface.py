import numpy as np
import cv2
import datetime
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


class CenterFace(object):
    def __init__(self, landmarks=True):
        self.landmarks = landmarks
        self.trt_logger = trt.Logger()  # This logger is required to build an engine
        f = open("../models/tensorrt/centerface.trt", "rb")
        runtime = trt.Runtime(self.trt_logger)
        self.net = runtime.deserialize_cuda_engine(f.read())
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = 0, 0, 0, 0

    def __call__(self, img, height, width, threshold=0.5):
        h, w = img.shape[:2]
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = height, width, height / h, width / w
        return self.inference_tensorrt(img, threshold)

    def inference_tensorrt(self, img, threshold):

        class HostDeviceMem(object):
            def __init__(self, host_mem, device_mem):
                self.host = host_mem
                self.device = device_mem

            def __str__(self):
                return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

            def __repr__(self):
                return self.__str__()

        def allocate_buffers(engine):
            inputs = []
            outputs = []
            bindings = []
            stream = cuda.Stream()
            for binding in engine:
                size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                # Append the device buffer to device bindings.
                bindings.append(int(device_mem))
                # Append to the appropriate list.
                if engine.binding_is_input(binding):
                    inputs.append(HostDeviceMem(host_mem, device_mem))
                else:
                    outputs.append(HostDeviceMem(host_mem, device_mem))
            return inputs, outputs, bindings, stream

        def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
            # Transfer data from CPU to the GPU.
            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
            # Run inference.
            context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
            # Synchronize the stream
            stream.synchronize()
            # Return only the host outputs.
            return [out.host for out in outputs]

        image_cv = cv2.resize(img, dsize=(self.img_w_new, self.img_h_new))
        blob = np.expand_dims(image_cv[:, :, (2, 1, 0)].transpose(2, 0, 1), axis=0).astype("float32")
        engine = self.net

        # Create the context for this engine
        context = engine.create_execution_context()
        # Allocate buffers for input and output
        inputs, outputs, bindings, stream = allocate_buffers(engine)  # input, output: host # bindings

        # Do inference
        shape_of_output = [(1, 1, int(self.img_h_new / 4), int(self.img_w_new / 4)),
                           (1, 2, int(self.img_h_new / 4), int(self.img_w_new / 4)),
                           (1, 2, int(self.img_h_new / 4), int(self.img_w_new / 4)),
                           (1, 10, int(self.img_h_new / 4), int(self.img_w_new / 4))]
        # Load data to the buffer
        inputs[0].host = blob.reshape(-1)
        begin = datetime.datetime.now()
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)  # numpy data
        end = datetime.datetime.now()
        print("gpu times = ", end - begin)

        heatmap, scale, offset, lms = [output.reshape(shape) for output, shape in zip(trt_outputs, shape_of_output)]
        print(heatmap.shape, scale.shape, offset.shape, lms.shape)
        return self.postprocess(heatmap, lms, offset, scale, threshold)

    def postprocess(self, heatmap, lms, offset, scale, threshold):
        if self.landmarks:
            dets, lms = self.decode(heatmap, scale, offset, lms, (self.img_h_new, self.img_w_new), threshold=threshold)
        else:
            dets = self.decode(heatmap, scale, offset, None, (self.img_h_new, self.img_w_new), threshold=threshold)
        if len(dets) > 0:
            dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2] / self.scale_w, dets[:, 1:4:2] / self.scale_h
            if self.landmarks:
                lms[:, 0:10:2], lms[:, 1:10:2] = lms[:, 0:10:2] / self.scale_w, lms[:, 1:10:2] / self.scale_h
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            if self.landmarks:
                lms = np.empty(shape=[0, 10], dtype=np.float32)
        if self.landmarks:
            return dets, lms
        else:
            return dets

    def decode(self, heatmap, scale, offset, landmark, size, threshold=0.1):
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        if self.landmarks:
            boxes, lms = [], []
        else:
            boxes = []
        if len(c0) > 0:
            for i in range(len(c0)):
                s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
                x1, y1 = min(x1, size[1]), min(y1, size[0])
                boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
                if self.landmarks:
                    lm = []
                    for j in range(5):
                        lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                        lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
                    lms.append(lm)
            boxes = np.asarray(boxes, dtype=np.float32)
            keep = self.nms(boxes[:, :4], boxes[:, 4], 0.3)
            boxes = boxes[keep, :]
            if self.landmarks:
                lms = np.asarray(lms, dtype=np.float32)
                lms = lms[keep, :]
        if self.landmarks:
            return boxes, lms
        else:
            return boxes

    def nms(self, boxes, scores, nms_thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]
        num_detections = boxes.shape[0]
        suppressed = np.zeros((num_detections,), dtype=np.bool)

        keep = []
        for _i in range(num_detections):
            i = order[_i]
            if suppressed[i]:
                continue
            keep.append(i)

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
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                inter = w * h
                ovr = inter / (iarea + areas[j] - inter)
                if ovr >= nms_thresh:
                    suppressed[j] = True

        return keep
