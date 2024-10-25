# Neural Network Iris Classifier

## Project Overview
This project demonstrates a simple feed-forward neural network implementation to classify the famous Iris dataset. The project uses Python, NumPy, and Scikit-learn to preprocess data and train a neural network.

## Files
- **neural_network.py**: Contains the `NeuralNetwork` class implementing a feed-forward neural network.
- **dataset.py**: Contains the `load_and_preprocess_data` function to load and preprocess the Iris dataset.
- **iris_classification.ipynb**: A Jupyter Notebook demonstrating the usage of the neural network on the Iris dataset.

## Usage
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn
3. Run the Jupyter notebook to see the classification results on the Iris dataset.

## Neural Network Structure
- **Input Layer**: 4 nodes (one for each feature)
- **Hidden Layer**: 5 nodes with sigmoid activation
- **Output Layer**: 3 nodes with sigmoid activation (one for each class)

## Dataset
The Iris dataset is a multi-class dataset with 150 samples, divided into three classes. The task is to classify samples into one of the three classes based on four features.

## License
This project is for my machine learning course, you can use it freely :)