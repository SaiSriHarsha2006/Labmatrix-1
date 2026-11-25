
# Project Report & README: Zomato Hyderabad Analytics

This project provides an end-to-end analytical data pipeline, exploratory analysis, and deep unsupervised learning modeling (deep features and semantic clustering) of Zomato's restaurant landscape in Hyderabad. 

---
##📋 Project Executive Summary

This platform leverages advanced feature engineering, interactive visualizations, and PyTorch-based Deep Learning Autoencoders to group dining outlets and isolate critical market dynamics:
1. **Data Dimensions Analyzed**: 10,000 restaurant customer reviews combined with distinct metadata catalogs covering price points, cuisine classifications, and outlet operating hours.
2. **Clustering Strategy**: Employs two key unsupervised deep neural approaches:
    - **Multi-Feature Profile Autoencoder**: Jointly maps cost, popularity, and user response metrics to categorize outlets into highly distinct economic tiers (Budget, Mid-Range, Premium).
    - **Semantic Review NLP Autoencoder**: Leverages TF-IDF natural language representations to construct semantic clustering spaces based strictly on subjective reviews and text profiles.
3. **Anomaly & Outlier Isolation**: Identifies anomalous restaurants whose combination of high pricing, extreme review lengths, or follower engagement rates deviate significantly from general market behaviors.

---
 🛠️ Technical Architecture & Pipeline


     ┌────────────────┐      ┌─────────────────┐      ┌─────────────────────────┐
     │ Zomato Reviews │ ───> │ Data Cleansing  │ ───> │   Feature Engineering   │
     └────────────────┘      │  & Integration  │      │ (Log Transforms, Tiers) │
     ┌────────────────┐      └─────────────────┘      └─────────────────────────┘
     │ Zomato Metadata│ ──────────────┘                                    │
     └────────────────┘                                                    ▼
     ┌─────────────────────────────────────────────────────────────────────┐
     │                       PyTorch Autoencoders                          │
     │   ├─ Improved Profile Model (Deeps layers, Cosine Annealing, BN)    │
     │   └─ Semantic NLP Text Autoencoder (TF-IDF + Latent Space Reduction)│
     └─────────────────────────────────────────────────────────────────────┘
                                                                           │
                                                                           ▼
                                                              ┌────────────────────────┐
                                                              │ Interactive Plots &    │
                                                              │ Business Action Plans  │
                                                              └────────────────────────┘
    

### 1. Data Processing & Features
* **Handling Outliers & Skewness**: Variables such as `Reviewer_Followers` and `Cost` displayed significant skewness. Logarithmic transformations (`Log_Reviewer_Followers`, `Log_Cost`) were computed to stabilize gradient descent during training.
* **Integration Strategy**: Custom string cleanup steps aligned spatial variations in restaurant nomenclature between datasets, securing an exact inner-join mapping of multi-dimensional profiles.

### 2. Deep Autoencoder Progression
#### Baseline Model
* **Structure**: A basic 1-hidden layer Autoencoder mapping inputs to a 2D bottleneck utilizing simple Relu activations.
* **Evaluation**: Left a high reconstruction MSE loss (~0.2063), though it managed a strong clustering structure (Silhouette score ~0.3994).

#### Upgraded Profile Autoencoder (Optimized Model)
* **Key Improvements**:
    - **Deeper Bottleneck Mapping**: Stacked encoders and decoders featuring Batch Normalization (`BatchNorm1d`) and Dropout (`0.15`) regularizations.
    - **Activation Upgrade**: Replaced Relu with continuous `SiLU` activations to secure non-linear boundaries without gradient death.
    - **Advanced Optimization**: Utilized the `AdamW` optimizer combined with a cyclic `CosineAnnealingLR` learning rate scheduler.
    - **Result**: Drastically reduced overall Mean Squared Reconstruction Error from **0.2063** down to **0.00918**, indicating highly precise latent reconstruction accuracy.

#### Semantic Review NLP Autoencoder
* **Textual Space Mapping**: High-dimensional TF-IDF matrices (150 components) from customer text reviews are compressed into a compact 16-dimensional latent space to discover semantic themes and group restaurants based on subjective guest experiences rather than pure metadata metrics.

---

## 📊 Analytical Insights & Key Business Findings

1. **High-Value Segments**: Deep Profile Autoencoders effectively split premium outlets from budget ones. High-rated places are often accompanied by larger volumes of images, demonstrating that digital visual media strongly correlates with high overall review sentiment.
2. **Engagement vs Price Tiers**: Reviewer descriptive volume scales proportionally with cost tier. Fine-dining restaurants average **81 words per review**, while budget options hover near **31 words per review**, showing that higher premium pricing drives deeper client scrutiny.
3. **Cuisine Concentration**: North Indian, Chinese, and Biryani variants comprise the vast majority of Hyderabad's commercial culinary landscape. Niche cuisines (Italian, Japanese, Mediterranean) are almost exclusively restricted to the premium, higher-budget pricing clusters.

---

## 💻 How to Run the Environment

1. **Environment Setup**: Ensure you run the initial sections to download files directly from Google Drive without manual authentication.
2. **Module Execution**: Run all cells sequentially. The PyTorch modules automatically compile, train, and output dynamic performance figures inline.
3. **Visual Customization**: Feel free to alter hyperparameters such as the `latent_dim` or clustering dimensions directly within the model structure cells to explore new spatial classifications!
```
