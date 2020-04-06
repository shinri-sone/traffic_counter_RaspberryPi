# python3
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using TF Lite to detect objects with the Raspberry Pi camera."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import re
import time

import numpy as np
import cv2

from tflite_runtime.interpreter import Interpreter
import tflite_runtime.interpreter as tflite

EDGETPU_SHARED_LIB = 'libedgetpu.so.1'

def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels


def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  parser.add_argument(
      '--threshold',
      help='Score threshold for detected objects.',
      required=False,
      type=float,
      default=0.4)
  args = parser.parse_args()

  labels = load_labels(args.labels)
  model_file, *device = args.model.split('@')
  try:
    interpreter = Interpreter(model_file, experimental_delegates=[
                                        tflite.load_delegate(EDGETPU_SHARED_LIB,
                                        {'device': device[0]} if device else {})
                                        ])
  except ValueError:
    interpreter = Interpreter(model_file)
  interpreter.allocate_tensors()
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']


  cap = cv2.VideoCapture(0)
  while True:
    ret, frame = cap.read()
    (CAMERA_WIDTH, CAMERA_HEIGHT) = (frame.shape[1], frame.shape[0])
    image = cv2.resize(frame, 
                       dsize=(input_width, input_height), 
                       interpolation=cv2.INTER_NEAREST)

    start_time = time.monotonic()
    results = detect_objects(interpreter, image, args.threshold)
    elapsed_ms = (time.monotonic() - start_time) * 1000
    for obj in results:
      ymin, xmin, ymax, xmax = obj['bounding_box']
      xmin = int(xmin * CAMERA_WIDTH)
      xmax = int(xmax * CAMERA_WIDTH)
      ymin = int(ymin * CAMERA_HEIGHT)
      ymax = int(ymax * CAMERA_HEIGHT)

      cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
      text = '{} {:.2f}'.format(labels[obj['class_id']], obj['score'])
      (text_width, text_height), baseline = cv2.getTextSize(text,
                                              cv2.FONT_HERSHEY_SIMPLEX,
                                              0.5, 1)
      cv2.rectangle(frame,
                    (xmin, ymin),
                    (xmin + text_width, ymin - text_height - baseline),
                    (255, 0, 0),
                    thickness=cv2.FILLED)
      cv2.putText(frame, 
                  text,
                  (xmin, ymin - baseline), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, 
                '{:.1f}ms'.format(elapsed_ms),
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
 
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()
  return

if __name__ == '__main__':
  main()
