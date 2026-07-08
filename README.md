# Waste Image Classification

A waste image classification system using CNN and transfer learning to identify waste categories such as plastic, paper, cardboard, metal, glass, organic waste, batteries, clothes, shoes, and trash. The model is trained using the public Garbage Classification dataset and can also be tested using self-captured local waste images.

## Project Overview

Waste management is an important environmental problem. Manual waste sorting can be time-consuming and inaccurate. This project uses computer vision and deep learning to classify waste images into different categories. The final system can help identify the type of waste from an uploaded image and support better recycling awareness.

The main goal of this project is to build a simple and practical waste classification model that can:

* Classify waste images into multiple categories
* Use CNN or transfer learning models for better accuracy
* Evaluate the model using standard classification metrics
* Provide a simple demo interface for image upload and prediction
* Test the model using public dataset images and optional local images

## Dataset

The main dataset used in this project is the **Garbage Classification dataset** from Kaggle.

Dataset link:
https://www.kaggle.com/datasets/mostafaabla/garbage-classification

The dataset contains waste images from different categories such as:

* Cardboard
* Paper
* Plastic
* Metal
* Biological / Organic waste
* Trash
* Brown glass
* Green glass
* White glass
* Battery
* Clothes
* Shoes

## Project Remarks

A waste image classification system using CNN and transfer learning to identify waste categories such as plastic, paper, cardboard, metal, glass, organic waste, batteries, clothes, shoes, and trash, evaluated on the Garbage Classification dataset and optional self-captured local images.

## Technologies Used

* Python
* TensorFlow / Keras or PyTorch
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Streamlit or Flask for demo interface

## Model Approach

The project can be implemented using two main approaches:

### 1. Custom CNN Model

A basic Convolutional Neural Network can be built from scratch using convolution layers, pooling layers, dropout, and dense layers. This approach is useful for understanding the basics of image classification.

### 2. Transfer Learning

Pretrained models can be used to improve performance and reduce training time. Suitable models include:

* MobileNetV2
* ResNet50
* EfficientNetB0
* VGG16

Transfer learning is recommended because it performs better with limited image datasets.

## Project Workflow

1. Download the Garbage Classification dataset
2. Organize images into class folders
3. Preprocess images by resizing and normalizing
4. Apply data augmentation
5. Split the dataset into training, validation, and testing sets
6. Train a CNN or transfer learning model
7. Evaluate the model using performance metrics
8. Save the trained model
9. Build a simple image upload demo
10. Test the model using sample images and optional local images

## Folder Structure

```text
waste-image-classification/
│
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── custom_test_dataset/
│   ├── cardboard/
│   ├── paper/
│   ├── plastic/
│   ├── metal/
│   ├── biological/
│   ├── trash/
│   ├── brown-glass/
│   ├── green-glass/
│   ├── white-glass/
│   ├── battery/
│   ├── clothes/
│   └── shoes/
│
├── notebooks/
│   └── waste_classification.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── waste_classifier_model.h5
│
├── demo/
│   └── app.py
│
├── results/
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   └── confusion_matrix.png
│
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/waste-image-classification.git
cd waste-image-classification
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

For Windows:

```bash
venv\Scripts\activate
```

For macOS / Linux:

```bash
source venv/bin/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

## Example Requirements

```text
tensorflow
opencv-python
numpy
pandas
matplotlib
scikit-learn
streamlit
pillow
```

## Training the Model

After preparing the dataset, run:

```bash
python src/train.py
```

The training script should:

* Load the dataset
* Resize images
* Apply augmentation
* Train the model
* Save the trained model inside the `models/` folder

## Evaluation

The trained model can be evaluated using:

```bash
python src/evaluate.py
```

The evaluation can include:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

These metrics help identify how well the model performs for each waste category.

## Prediction

To predict a single image:

```bash
python src/predict.py --image path/to/image.jpg
```

The output will show the predicted waste category.

Example:

```text
Predicted Class: Plastic
Confidence: 94.6%
```

## Demo Interface

A simple demo interface can be created using Streamlit.

Run the demo:

```bash
streamlit run demo/app.py
```

The demo allows users to:

* Upload a waste image
* View the uploaded image
* Get the predicted waste category
* See the confidence score

## Custom Local Test Images

In addition to the Kaggle dataset, a small custom test set can be collected using a phone camera. Images can be captured from:

* Home
* University
* Shops
* Canteens
* Waste collection areas

These images can be used to test whether the model works well in real-world local conditions.

Recommended number of custom images: **300–500 images**

Example folder:

```text
custom_test_dataset/plastic/plastic_001.jpg
custom_test_dataset/paper/paper_001.jpg
custom_test_dataset/metal/metal_001.jpg
```

## Expected Output

The final project will include:

* Trained waste classification model
* Evaluation results
* Accuracy and loss graphs
* Confusion matrix
* Image prediction script
* Simple demo interface

## Results

The results will be updated after model training.

| Model          |      Accuracy |     Precision |        Recall |      F1-score |
| -------------- | ------------: | ------------: | ------------: | ------------: |
| Custom CNN     | To be updated | To be updated | To be updated | To be updated |
| MobileNetV2    | To be updated | To be updated | To be updated | To be updated |
| EfficientNetB0 | To be updated | To be updated | To be updated | To be updated |

## Future Improvements

* Improve model accuracy using better augmentation
* Add more local waste images
* Try different transfer learning models
* Deploy the model as a web application
* Add real-time camera-based waste detection
* Connect predictions with recycling instructions

## Applications

This project can be useful for:

* Smart waste sorting systems
* Recycling awareness applications
* Educational machine learning projects
* Environmental monitoring systems
* Computer vision learning projects

## License

This project is created for academic and learning purposes. Dataset usage should follow the license and terms provided by the original Kaggle dataset owner.

## Acknowledgement

The project uses the Garbage Classification dataset available on Kaggle. This work is developed as a computer vision and deep learning project to explore how artificial intelligence can support waste classification and recycling awareness.
