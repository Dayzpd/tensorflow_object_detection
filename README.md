# Tensorflow Heart Gesture Detection

### Summary
This project modifies the SSD Lite Mobilenet v2 model provided by Tensorflow's
[Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
detect heart gestures in a video feed.

### Dataset Preperation
- The dataset is comprised of images from Google Images and also pictures taken
  by myself.
- Images were sorted into train and test subsets using img_sort.py and then
  labeled using LabelImg.
- From the xml files exported by LabelImg, TF Records were made for each feature
  in the dataset.

### Training
I haven't yet switched to use Docker containers for training and as of writing
this have other ongoing projects that utilize dependencies similar to that of
Tensorflow. It follows that this was an issue when trying to use the more up to
date model_main.py in the object_detection directory for training. In the legacy
directory of object_detection, train.py works just fine for this application
though. The following command was executed from the models/research directory
to train the graph:
```
python object_detection/legacy/train.py --logtostderr \
--train_dir=training/ \
--pipeline_config_path=training/ssdlite_mobilenet_v2.config
```

### Graph Export:
Exporting the graph after training converged to a consistent loss of about 1 was
done with the following command from, again, the models/research directory:
```
python object_detection/export_inference_graph.py --input_type=image_tensor \
--pipeline_config_path=training\ssdlite_mobilenet_v2.config \
--trained_checkpoint_prefix=training\model.ckpt-10000 \
--output_directory=training
```

### Testing
- The model can be tested using heart_gesture_detection.py.
- The program will search for available cameras, and then you must select one of
  cameras that was found.
- From the video feed, if a heart gesture is detected, a heart will be resized
  to the size of the box and overlaid onto the heart gesture.
