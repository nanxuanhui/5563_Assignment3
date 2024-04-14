# CS5563 - Gold Team - Assignment 3

- Submitted 2024-04-14
- for CS5563 - Natural Language Processing
- as taught by Dr. Ye Wang at UMKC

## Table of Contents

## Group Members

- Hui Jin
- Odai Athamneh
- Karthik Chellamuthu
- Bao Ngo

## Abstract

This report delves into the intricate world of word embeddings, a cornerstone of modern Natural Language Processing (NLP). It focuses on the comparative analysis of three pivotal types of embeddings: custom-trained Word2Vec, pre-trained Word2Vec, and GloVe, alongside an evaluation involving BERT at the sentence level. The primary dataset selected originates from an online news platform, featuring over 1 million words, tailored to encapsulate diverse linguistic structures and vocabularies pertinent to contemporary media. The study encompasses two main evaluation metrics: semantic distance calculations and a text classification task, aimed at discerning the practical effectiveness of each embedding in real-world applications. Additionally, the report introduces a comparative task for GloVe and BERT embeddings to assess their performance at the sentence level. Visualizations are employed extensively to elucidate semantic relationships and model accuracies, providing a clear, comparative insight into the functionality and utility of each embedding type. Through rigorous analysis, this report aims to furnish a deeper understanding of how these embeddings can be optimized for various NLP tasks, setting the stage for future explorations and innovations in the field.

## Introduction

Word embeddings represent one of the most significant advancements in Natural Language Processing, offering a nuanced method to capture the semantic properties of words through dense vector representations. These embeddings have revolutionized the way machines understand and process human language, facilitating improvements across a myriad of applications, from sentiment analysis to machine translation. Among the various types of embeddings, Word2Vec and GloVe have been widely adopted due to their efficiency in capturing context and semantic similarity. More recently, contextual embeddings like BERT have emerged, offering even deeper linguistic insights.

This report aims to provide a comprehensive evaluation of three specific types of embeddings: a custom-trained Word2Vec model, pre-trained Word2Vec, and pre-trained GloVe embeddings. Each type will be assessed through semantic distance measurements and a classification task to determine their efficacy and applicability in different linguistic scenarios. The purpose of this analysis is to highlight the strengths and potential limitations of each embedding type, providing a detailed comparative framework.

Moreover, the report extends its analysis to sentence-level embeddings, comparing the performance of GloVe and BERT through a designed NLP task. This comparison aims to explore the adaptability and accuracy of these models beyond individual words, focusing on their ability to handle complex sentence structures.

Through detailed experimental setups, visualizations, and extensive testing, this report will contribute valuable insights into the field of word embeddings. The findings are expected to not only enhance academic understanding but also offer practical guidance for implementing these technologies in various NLP applications. Furthermore, this investigation will pave the way for the subsequent research proposal, focusing on innovative applications of NLP technologies in emerging research domains.

## Results

Dataset: THUC and Sougou news<br/>
Stopwords: from Sichuan University<br/>


result of similarity:<br/>
   glove_similarity  bert_similarity<br/>
0          0.967461         0.977059<br/>
1          0.956530         0.944582<br/>
2          0.942311         0.938288<br/>
3          0.968154         0.966305<br/>
4          0.902482         0.932253<br/>

Results for Word2Vec model:<br/>
Training Loss: 0.6561, Training Accuracy: 0.7722<br/>
Test Loss: 0.7847, Test Accuracy: 0.7411<br/>


## Section 1: Training Custom Word2Vec Embedding
- **1.1 Dataset Selection**
  - Description of the dataset used (source, domain, size, language).
  - Justification for dataset choice.
- **1.2 Preprocessing**
  - Steps taken for text cleaning and preprocessing (tokenization, removing stopwords, etc.).
- **1.3 Model Training**
  - Configuration of the Word2Vec model (architecture, window size, vector size, etc.).
  - Training process and parameters.
- **1.4 Evaluation of Embedding**
  - Qualitative analysis (nearest neighbors, analogy tasks).
  - Quantitative metrics (if applicable).

## Section 2: Comparison of Embeddings
- **2.1 Semantic Distance Calculation and Visualization**
  - **2.1.1 Methodology**
    - Description of semantic distance metrics used (cosine similarity, etc.).
    - Tools and libraries utilized for visualization (e.g., matplotlib, seaborn).
  - **2.1.2 Results**
    - Visual representations (scatter plots, heatmaps) comparing the three embeddings.
    - Interpretation of results.
- **2.2 Classification Task**
  - **2.2.1 Task Description**
    - Choice of classification task and rationale.
  - **2.2.2 Model Configuration**
    - Description of the deep learning model used.
    - Explanation of how embeddings were incorporated.
  - **2.2.3 Results and Discussion**
    - Performance metrics (accuracy, F1-score, etc.).
    - Comparison between models using different embeddings.

## Section 3: Sentence-level Comparison of GloVe and BERT
- **3.1 Task Design**
  - Description of the NLP task selected for comparison.
  - Justification for task choice in evaluating sentence-level embeddings.
- **3.2 Implementation**
  - Overview of model architectures used.
  - Details of dataset and preprocessing.
- **3.3 Results and Analysis**
  - Comparison of performance metrics.
  - Discussion on the strengths and weaknesses of each embedding type at the sentence level.

## Section 4: Proposal for Future NLP Research Projects
- **4.1 Research Idea #1**
  - Detailed description of the first research idea.
  - Objectives and expected outcomes.
  - Methodology and tools/technologies required.
- **4.2 Research Idea #2**
  - Detailed description of the second research idea.
  - Explanation of its relevance and potential impact on the field.
  - Proposed methods and necessary resources.

## Conclusion
- Summary of key findings from the comparisons of embeddings.
- Reflections on the learning outcomes of the assignment.
- Potential implications for future NLP applications and research.

## References
- Comprehensive list of all sources cited in APA/MLA format.

## Appendices
- Any supplementary material (code snippets, additional data visualizations).
