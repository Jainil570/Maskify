# Maskify - Background Remover with Deep Learning 
**[video_link](https://screenrec.com/share/Du5BtCbwXR)**

**Maskify** is a Streamlit-based web application that allows users to remove backgrounds from images using a U-Net-based deep learning segmentation model.

##  Features

- Upload an image and instantly remove the background (Works on humans for now)
- Model trained on human segmentation dataset [https://github.com/VikramShenoy97/Human-Segmentation-Dataset/tree/master]
- Transparent PNG output download
- Clean, responsive UI with real-time results

##  Model

- Architecture: U-Net
- Input size: 128x128 RGB images
- Loss: Binary Crossentropy
- Trained on: Human Segmentation Dataset (resized masks and images)

##  How It Works

1. User uploads an image
2. Image is resized and passed through the model
3. Model predicts a binary mask
4. Background is made transparent using the mask
5. Resulting image is displayed and can be downloaded
