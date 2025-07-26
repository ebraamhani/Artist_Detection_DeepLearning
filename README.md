# 🎨 Artist Detection Deep Learning Model

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A sophisticated deep learning application that identifies the artist of a given art image using a ResNet50-based neural network. The system provides top 3 artist predictions with confidence scores and AI-generated insights about each artist.

## 🌟 Features

- **Multi-Artist Classification**: Identifies artwork from 18 famous artists
- **AI-Powered Insights**: GPT-generated explanations for predictions
- **Modern Web Interface**: Beautiful, responsive UI with dark theme
- **Real-time Processing**: Instant predictions with image upload
- **Docker Support**: Easy deployment with containerization
- **High Accuracy**: ResNet50 model with 84%+ accuracy

## 🎯 Supported Artists

| Artist | Artist | Artist |
|--------|--------|--------|
| Vincent van Gogh | Edgar Degas | Pablo Picasso |
| Pierre-Auguste Renoir | Albrecht Dürer | Paul Gauguin |
| Francisco Goya | Rembrandt | Alfred Sisley |
| Titian | Marc Chagall | Rene Magritte |
| Amedeo Modigliani | Paul Klee | Henri Matisse |
| Andy Warhol | Mikhail Vrubel | Sandro Botticelli |

## 🏗️ Architecture

### Model Architecture
- **Base Model**: ResNet50 (pretrained on ImageNet)
- **Input Shape**: 224×224×3 RGB images
- **Output**: 18-class classification with softmax probabilities
- **Training Strategy**: Two-phase fine-tuning approach

### Data Pipeline
```
Raw Images → Preprocessing → Augmentation → Training/Validation Split → Model Training
```

### Web Application
```
User Upload → Image Processing → Model Prediction → GPT Analysis → Results Display
```

## 📋 Prerequisites

- Python 3.9 or higher
- 8GB+ RAM (recommended for training)
- GPU support (optional, for faster training)
- Azure OpenAI API access (for GPT insights)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Artist_Detection_DeepLearning
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```env
OPENAI_API_TYPE=azure
OPENAI_API_BASE=your_azure_openai_endpoint
OPENAI_API_VERSION=2024-02-15-preview
OPENAI_API_KEY=your_azure_openai_key
DEPLOYMENT_NAME=your_deployment_name
```

### 5. Prepare Data and Model
```bash
# Create required directories
mkdir -p Artist_data/images/images
mkdir -p model
mkdir -p uploads

# Download dataset from Kaggle
# Place artists.csv in Artist_data/
# Place artist images in Artist_data/images/images/
```

## 🎮 Usage

### Training the Model
```bash
python model.py
```
This will:
- Load and preprocess the dataset
- Train the ResNet50 model
- Save the trained model to `model/Artist_Resnet_model.h5`

### Running the Web Application
```bash
python app.py
```
The application will be available at `http://localhost:5000`

### Using Docker
```bash
# Build the Docker image
docker build -t artist-detection .

# Run the container
docker run -p 5000:5000 artist-detection
```

## 📊 Model Performance

### Training Details
- **Dataset**: 18 artists with 150+ paintings each
- **Data Augmentation**: Rotation, shifts, shears, zooms, flips
- **Training Split**: 80% training, 20% validation
- **Optimizer**: Adam with learning rate 0.0001
- **Loss Function**: Categorical Cross-Entropy

### Performance Metrics
- **Accuracy**: 84%+ on validation set
- **Training Time**: ~2-3 hours on GPU
- **Inference Time**: <1 second per image

## 🔧 Configuration

### Model Parameters
```python
# In model.py
input_shape = (224, 224, 3)
batch_size = 32
epochs_phase1 = 20
epochs_phase2 = 50
learning_rate = 0.0001
```

### Web App Settings
```python
# In app.py
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB file limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
```

## 📁 Project Structure

```
Artist_Detection_DeepLearning/
├── app.py                              # Flask web application
├── model.py                            # ResNet50 model training
├── data_preprocessing.py               # Data preprocessing pipeline
├── requirements.txt                    # Python dependencies
├── dockerfile                         # Docker configuration
├── README.md                          # Project documentation
├── .env                               # Environment variables (create this)
├── .gitignore                         # Git ignore rules
├── Artist_DataPreprocessing_Visualization_Model.ipynb  # Jupyter notebook
├── templates/                         # HTML templates
│   ├── index.html                     # Main prediction interface
│   └── about.html                     # About page
├── Artist_data/                       # Dataset directory (not in repo)
│   ├── artists.csv                    # Artist metadata
│   └── images/images/                 # Artist image folders
├── model/                             # Trained models (not in repo)
│   └── Artist_Resnet_model_0.84.h5    # Pre-trained model
└── uploads/                           # User uploads (create this)
```

## 🧪 Testing

### Test Images
Create a `test_images/` folder with sample artwork to test the model:
```bash
mkdir test_images
# Add some art images for testing
```

### API Testing
```bash
# Test the prediction endpoint
curl -X POST -F "file=@test_images/sample.jpg" http://localhost:5000/predict
```

## 🔍 Troubleshooting

### Common Issues

1. **Model not found error**
   ```bash
   # Ensure model file exists
   ls model/Artist_Resnet_model_0.84.h5
   ```

2. **Uploads folder missing**
   ```bash
   mkdir uploads
   ```

3. **Azure OpenAI connection issues**
   - Verify `.env` file configuration
   - Check API key and endpoint validity

4. **Memory issues during training**
   - Reduce batch size in `model.py`
   - Use GPU if available

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement
- UI/UX enhancements
- Model performance optimization
- Additional artist support
- Mobile app development
- API rate limiting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time) on Kaggle
- **Base Model**: ResNet50 from TensorFlow/Keras
- **AI Integration**: Azure OpenAI GPT models
- **Web Framework**: Flask

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the project maintainer
- Check the troubleshooting section above

---

**Made with ❤️ for Art and AI enthusiasts**
