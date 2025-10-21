# Handwritten-Recognition

This project use **Cpnvolutional network (CNN)** to recognize handwritten digits from **MNIST** dataset. Provide Gradio interface allow user write or upload handwriten notebook for prediction.

## Usage
1. Install lib
 ```bash
pip install -r requirements.txt
 ```
2. To starting train model, run the code below:
```bash
python prj2_handwrite.py
 ```
After the process complete, the model will save as file :
```bash
mnist_model.h5
 ```

The Gradio Interface will automatically run after training. The user can draw digits or upload an image of handwriting, and the model will display 3 predictions with the highest probability. 
