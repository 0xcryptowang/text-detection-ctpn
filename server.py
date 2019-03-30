from flask import Flask, request, render_template
from nets import model_train as model
from functools import lru_cache
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector
import io
import cv2
import numpy as np
import tensorflow as tf
import os
import base64

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

checkpoint_path = '/Users/saycolor/Desktop/text-detection-ctpn/data/model/checkpoints'

app = Flask(__name__)


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    bio = io.BytesIO()
    request.files['image'].save(bio)
    img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
    load_model_and_predict()(img)
    return load_model_and_predict()(img)


@lru_cache()
def load_model_and_predict():
    input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
    input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    bbox_pred, cls_pred, cls_prob = model.model(input_image)
    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    checkpoint_state = tf.train.get_checkpoint_state(checkpoint_path)
    model_path = os.path.join(checkpoint_path, os.path.basename(checkpoint_state.model_checkpoint_path))
    print('Restore from {}'.format(model_path))
    saver.restore(sess, model_path)

    def predict_image(image):
        img, (rh, rw) = resize_image(image)
        h, w, c = img.shape
        im_info = np.array([h, w, c]).reshape([1, 3])
        bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                               feed_dict={input_image: [img], input_im_info: im_info})

        text_seg, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
        scores = text_seg[:, 0]
        text_seg = text_seg[:, 1:5]

        text_detector = TextDetector(DETECT_MODE='H')
        boxes = text_detector.detect(text_seg, scores[:, np.newaxis], img.shape[:2])
        boxes = np.array(boxes, dtype=np.int)

        for i, box in enumerate(boxes):
            cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                          thickness=2)
        img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
        _, buffer = cv2.imencode('.jpg', img)
        pic_str = base64.b64encode(buffer)
        pic_str = 'data:image/jpg;base64,' + pic_str.decode()
        return render_template('result.html', image_base_data=pic_str)

    return predict_image


if __name__ == '__main__':
    app.run(port=8000, debug=True)
