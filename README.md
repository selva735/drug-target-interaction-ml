# Drug-Target Interaction ML

Machine Learning project for Drug-Target Interaction Prediction and Drug Repurposing

## Project Overview

**Objective:** Develop and evaluate machine learning models to predict drug-target interactions for drug discovery and repurposing.

This project aims to leverage computational approaches to accelerate drug discovery by predicting how drugs interact with biological targets, potentially identifying new therapeutic applications for existing drugs.

## Project Structure

```
drug-target-interaction-ml/
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for data analysis and experiments
├── src/                    # Source code scripts
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Dataset Suggestions

- **BindingDB**: Comprehensive database of measured binding affinities
- **DrugBank**: Detailed drug and drug target information
- **KEGG Drug**: Drug information from KEGG database
- **ChEMBL**: Large-scale bioactivity database

## Features & Representations

### Drugs
- **SMILES strings**: Molecular structure representation
- **Molecular fingerprints**: ECFP (Extended Connectivity Fingerprints) and other chemical descriptors
- **Chemical properties**: Molecular weight, logP, solubility, etc.

### Targets
- **Amino acid sequences**: Protein sequence information
- **Protein embeddings**: Pre-trained protein representations
- **Structural features**: Secondary structure, domains, binding sites

## Methods

### Machine Learning Models
- **Traditional ML**: Random Forest, SVM, Logistic Regression
- **Deep Learning**: Neural Networks, Graph Neural Networks (GNNs)
- **Advanced Architectures**: Siamese networks, Dual encoders
- **Ensemble Methods**: Model combination and stacking

### Evaluation Metrics
- **AUROC**: Area Under Receiver Operating Characteristic curve
- **AUPR**: Area Under Precision-Recall curve
- **Accuracy**: Overall classification accuracy
- **F1-score**: Harmonic mean of precision and recall

### Tasks
- **Binary Classification**: Drug-target interaction (yes/no)
- **Regression**: Binding affinity score prediction
- **Multi-class**: Interaction type classification

## Getting Started

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/selva735/drug-target-interaction-ml.git
   cd drug-target-interaction-ml
   ```

2. **Set up the environment**
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset download**
   - Download datasets from the suggested sources
   - Place raw data in the `data/` directory
   - Run preprocessing scripts to prepare data

4. **Run sample code**
   ```bash
   # Explore the notebooks
   jupyter notebook notebooks/
   
   # Or run the main scripts
   python src/data_preprocessing.py
   python src/train.py
   ```

### Quick Start Example

```python
# Example usage
from src.models import DTIPredictor
from src.data_preprocessing import load_and_preprocess

# Load data
X_train, y_train, X_test, y_test = load_and_preprocess()

# Train model
model = DTIPredictor()
model.fit(X_train, y_train)

# Evaluate
scores = model.evaluate(X_test, y_test)
print(f"AUROC: {scores['auroc']:.3f}")
```

## Resources & References

### Key Resources
- **DeepPurpose**: [https://github.com/kexinhuang12345/DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose) - Comprehensive DTI prediction toolkit
- **Paper**: "Machine learning for drug-target interaction prediction" (Yamanishi Y, et al.)
- **Tutorials**: Towards Data Science DTI prediction articles

### Additional Reading
- [Drug Discovery with Machine Learning](https://www.nature.com/articles/s41573-019-0024-5)
- [Graph Neural Networks for Drug Discovery](https://arxiv.org/abs/2007.02456)
- [Molecular Fingerprints Guide](https://www.rdkit.org/docs/GettingStartedInPython.html)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaboration opportunities, please open an issue or contact the repository owner.
