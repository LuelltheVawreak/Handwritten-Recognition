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

3. Accuracy and lost
   After train 10 epochs with SGD optimizer (lr = 0.01, momentum = 0.9) model got:
   Training: accuracy 99.35% and loss 0.021
   Validation: accuracy 98.92% and loss 0.033
   Test: accuracy 98.78% and loss 0.040
    
