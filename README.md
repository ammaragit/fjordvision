Hierarchical CNN for Multi-Level Classification

This project implements a hierarchical convolutional neural network (CNN) designed for multi-level taxonomy classification. The model is trained on a dataset organized into hierarchical levels, such as binary, class, genus, and species.

Table of Contents

Installation

Dataset Structure

Training the Model

Evaluation Metrics

Results

Contributing

License

Dataset link:
https://figshare.com/articles/dataset/The_Fjord_Dataset/24072606/4?file=47516684

Installation

Clone the repository:

git clone https://github.com/yourusername/fjordvision.git


Navigate to the project directory:

cd fjordvision


Create and activate a virtual environment:

python -m venv fjordvision-env
source fjordvision-env/bin/activate  # On Windows, use `fjordvision-env\Scripts\activate`


Install the required dependencies:

pip install -r requirements.txt

Dataset Structure

The dataset should be organized into the following hierarchy:

data/
├── cnn/
│   ├── train/
│   │   ├── binary/
│   │   ├── class/
│   │   ├── genus/
│   │   └── species/
│   │
│   └── val/
│       ├── binary/
│       ├── class/
│       ├── genus/
│       └── species/


Each level (binary, class, genus, species) contains subfolders for each class, and images are shared across levels.

Training the Model

To train the model, run the following command:

python train_hierarchical_cnn.py


This will train the hierarchical CNN and generate evaluation metrics, including confusion matrices, classification reports, and accuracy/loss curves.

Evaluation Metrics

After training, the following metrics will be generated and saved in the metrics/ directory:

Confusion matrices for each hierarchical level (binary, class, genus, species)

Classification reports in CSV format

Loss and accuracy curves over epochs

Results

The trained model will be saved in the cnn_weights/ directory. The best performing model, based on validation loss, will be stored as best_hierarchical_cnn.pt.

Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have suggestions or find any bugs.

License

This project is licensed under the MIT License. See the LICENSE file for more information.
