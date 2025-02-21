# SAFAARI
Single-cell Annotation and Fusion with Adversarial Open-Set Domain Adaptation Reliable for single-cell multi-omics Data Integration
![Figure 1](https://github.com/user-attachments/assets/0e966206-9834-4f88-aa74-ff20b85dff54)
# SAFAARI: Single-Cell Annotation & Fusion using Adversarial Open-Set Adaptation



SAFAARI is a **Single-cell Annotation and Fusion with Adversarial Open-Set Domain Adaptation Reliable for single-cell multi-omics Data Integration**. It effectively removes batch effects, adapts to new cell types, and improves cross-modality single-cell analysis. It supports both **open-set** and **closed-set** annotation.

##  Features
- **Open-Set & Closed-Set Adaptation**: Handles novel cell types in the target dataset.
- **Batch Effect Removal**: Uses adversarial learning to mitigate batch effects.
- **Class Imbalance Handling**: Uses **SMOTE oversampling** to balance training data.
- **Novel Cell Type Detection**: Identifies unknown cell types in new datasets.
- **Cross-Modality/Cross-species Integration**

##  Installation
To install SAFAARI, run the following commands:

```bash
git clone https://github.com/VafaeeLab/SAFAARI.git
cd SAFAARI
pip install -e .
```


##  Running SAFAARI
You can run SAFAARI using the command-line interface (CLI):

### 1 **Supervised Integration**
```bash
safaari-supervised_integration
```

### 2️ **Unsupervised Annotation & Integration in open set OR closed set Setting**
closed set mode:
```bash
safaari-unsupervised --open_set False
```
open set mode:
```bash
safaari-unsupervised --open_set True
```
#### Downloading Datasets for Supervised Integration
If you want to obtain the **Supervised Integration** Results for the following datasets:

- Ovary (RNA-ATAC)
- PBMC (ADT-SCT)
- SEURAT_PBMC (RNA-ATAC)

You can download the required datasets from the following link:
**[Dataset Download](https://doi.org/10.6084/m9.figshare.c.7502103.v1)**  

After downloading, you need to **modify** `main_integration.py` and uncomment the relevant dataset section to run the integration process.
### 3️ **Custom Configuration**
You can specify additional parameters:
```bash
safaari-unsupervised --open_set True --epochs 500 --batch_size 512 --cuda 0
```

##  Data Structure

SAFAARI expects input data in **CSV format**, where:  
- **Each row represents a single cell**  
- **Each column represents a gene**  
- **The first column must be labeled** `'cell types'` **and contain cell-type annotations**  

### **Dataset Organization**  
SAFAARI utilizes a subset of the **Tabula Muris** cell atlas, containing **seven tissues**:  
- **Bladder**, **Kidney**, **Heart**, **Mammary Gland**, **Muscle**, **Bone Marrow**, **Spleen**  

Data from **FACS (Fluorescence-Activated Cell Sorting)** serves as the **source domain**, while **10x Genomics** is the **target domain**. The structured datasets are stored in:
data/FACS10X/{Tissue_name}
Each dataset follows the **naming convention**:  
- `dataset_source_CmnGenes.csv` (e.g., `Bladder_FACS_CmnGenes.csv`)  
- `dataset_target_CmnGenes.csv` (e.g., `Bladder_10X_CmnGenes.csv`)  

These files contain **normalized gene expression values** across **common highlu variable genes** between domains.

##  Output Files
Unsupervised results will be stored in `data/results/{Tissue_name}/`:
- `source_embeddings.csv` (Processed source domain features)
- `target_embeddings.csv` (Processed target domain features)
- `FACS_to_10X_labels.csv` or `FACS_to_10X_labels_op.csv` (Open vs. Closed set labels)


Supervised integration results will be stored in `SAFAARI_supervised_Integration_Resultsdata/{Tissue_name}/`:
- `dataset_FACS_to_10X_embeddings_op_source.csv` (supervised source embeddings)
- `dataset_FACS_to_10X_embeddings_op_target.csv` (supervised target embeddings)


##  Example Run
```bash
safaari-unsupervised --open_set True
```


```bash
safaari-supervised_integration
```



##  Citation

If you use SAFAARI in your research, please cite:

```bibtex
@article{Aminzadeh2024SAFAARI,
  author = {Fatemeh Aminzadeh, Jun Wu, Jingrui He, Morteza Saberi, Fatemeh Vafaee},
  title = {Single-Cell Data Integration and Cell Type Annotation through Contrastive Adversarial Open-set Domain Adaptation},
  journal = {bioRxiv},
  year = {2024},
  doi = {10.1101/2024.10.04.616599},
  publisher = {Cold Spring Harbor Laboratory}
}
```
You can also find the preprint at: https://doi.org/10.1101/2024.10.04.616599.


---

###  [SAFAARI GitHub Repository](https://github.com/VafaeeLab/SAFAARI)


