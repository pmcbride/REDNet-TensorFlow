# REDNet-TensorFlow
## REDNet model for TensorFlow with Keras
My implementation of the REDNet Skip-connection Encoder-Decoder Network for image de-noising and super resolution

#### Directories:
- [models](./models): Trained models. May be a bit out of date.
- [video_utils](./video_utils): Contains rough scripts and notebook files for applying the de-noising model to video files. Currently it will just deconstruct the video into into it's frame images, run inference on the images, and then reconstruct the frames back into the video file. Need to clean up the code and modify it to act on video buffer instead of images.

### Reference
- [Image Restoration Using Convolutional Auto-encoders
   with Symmetric Skip Connections](https://arxiv.org/abs/1606.08921)


