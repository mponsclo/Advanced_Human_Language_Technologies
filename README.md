# Advanced Human Language Technologies (AHLT)

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains implementations for biomedical Natural Language Processing tasks, developed as part of the Advanced Human Language Technologies course at UPC (Universitat Politècnica de Catalunya).

## 🎯 Overview

The project focuses on three core biomedical NLP tasks:

- **Named Entity Recognition (NER)**: Drug name identification and classification
- **Drug-Drug Interaction (DDI) Detection**: Identifying pharmacological interactions between drug entities
- **Neural Networks**: Deep learning approaches for both NER and DDI classification

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Java 8+ (for CoreNLP server, DDI tasks only)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ahlt-1
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (first time only):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

4. **For DDI tasks only**: Set up CoreNLP server:
```bash
# Download Stanford CoreNLP from https://stanfordnlp.github.io/CoreNLP/
# Extract and run:
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

## 📊 Usage

### Named Entity Recognition (NER)

#### Baseline NER (Rule-based)
```bash
cd "1. NER"

# Basic usage
python3 baseline-NER.py train_data_path output_file_name

# With drug database lookup
python3 baseline-NER.py -l train_data_path output_file_name
```

#### CRF-based NER
```bash
cd "1. NER"

# Extract features
python3 feature_extractor.py train_data_path > features_train.txt
python3 feature_extractor.py test_data_path > features_test.txt

# Train CRF model
python3 crf-learner.py model_name features_train.txt

# Run classification
python3 crf-classifier.py model_name features_test.txt entities_file output_file
```

### Drug-Drug Interaction (DDI) Detection

#### Complete Pipeline (Recommended)
```bash
cd "2. DDI"
bash runner.sh
```

#### Manual Steps
```bash
cd "2. DDI"

# Extract features
python3 feature_extractor.py ../../labAHLT/data/train > feats.dat
python3 feature_extractor.py ../../labAHLT/data/test > feats_test.dat

# Train and classify
python3 learner.py feats.dat feats_test.dat output.dat

# Evaluate
python3 evaluator.py DDI ../../labAHLT/data/test output.dat
```

### Evaluation

```bash
# Evaluate NER results
python3 "1. NER/evaluator.py" NER gold_standard_dir predictions_file

# Evaluate DDI results
python3 evaluator.py DDI gold_standard_dir predictions_file
```

## 📁 Project Structure

```
ahlt-1/
├── 1. NER/                    # Named Entity Recognition
│   ├── baseline-NER.py        # Rule-based NER implementation
│   ├── crf-classifier.py      # CRF-based classifier
│   ├── crf-learner.py         # CRF model training
│   ├── feature_extractor.py   # Feature engineering
│   ├── evaluator.py           # Evaluation metrics
│   ├── utils.py               # Utility functions
│   └── *.ipynb                # Jupyter notebooks for experimentation
├── 2. DDI/                    # Drug-Drug Interaction Detection
│   ├── baseline-DDI.py        # Rule-based DDI detection
│   ├── feature_extractor.py   # DDI feature extraction
│   ├── learner.py             # ML model training
│   ├── runner.sh              # Complete pipeline script
│   ├── utils.py               # DDI utilities
│   └── *.ipynb                # Analysis notebooks
├── 3. NN/                     # Neural Network implementations
│   ├── *.nn                   # Pre-trained models
│   ├── *.txt                  # Data splits
│   └── *.ipynb                # NN experimentation notebooks
├── resources/                 # Drug databases
│   ├── HSDB.txt               # Simple drug database
│   └── DrugBank.txt           # Comprehensive drug database
├── requirements.txt           # Python dependencies
└──README.md                  # This file
```

## 🔧 Technical Details

### Data Format

- **Input**: XML files with sentence-level annotations
- **Entities**: Marked with `charOffset`, `text`, and `type` attributes
- **DDI Pairs**: Entity pairs with interaction classifications
- **Features**: Tab-separated feature vectors for ML training

### Key Dependencies

- **NLTK**: Natural language processing toolkit
- **python-crfsuite**: Conditional Random Fields implementation
- **scikit-learn**: Machine learning library
- **networkx**: Graph algorithms for dependency parsing
- **Stanford CoreNLP**: Dependency parsing (external server)

### Drug Databases

- `HSDB.txt`: Simple drug name database
- `DrugBank.txt`: Comprehensive database with categories (drug|brand|group)

## 📈 Performance

The implementations include several approaches with varying performance characteristics:

- **Rule-based methods**: Fast, interpretable, good baseline performance
- **CRF models**: Better sequence modeling, improved NER accuracy
- **Neural networks**: State-of-the-art performance on both tasks
- **Feature engineering**: Syntactic and semantic features for DDI detection

## 🤝 Contributing

This is an academic project for the AHLT course. For development guidelines and technical details, see `CLAUDE.md`.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏫 Academic Context

**Course**: Advanced Human Language Technologies  
**Institution**: Universitat Politècnica de Catalunya (UPC)  
**Focus**: Biomedical Natural Language Processing

