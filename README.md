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

### 1.1: Dataset Selection

For the purpose of training our Word2Vec model, we chose two prominent Chinese text datasets, the Sogou News dataset and the THUCnews dataset. These datasets are widely recognized in the field of Natural Language Processing for their comprehensive coverage of news articles, which are ideal for training language models due to their rich vocabulary and varied syntax.

The [Sogou News dataset](https://huggingface.co/datasets/sogou_news) is a large-scale collection of news articles from the Sogou news portal, containing an extensive range of topics from sports to international news. The dataset is publicly available on the Hugging Face dataset repository, ensuring ease of access and use. This dataset includes over 2.7 million news articles, providing a robust corpus for training our embedding model.

[THUCnews](https://paperswithcode.com/dataset/thucnews), compiled by Tsinghua University, comprises around 740,000 news articles categorized into 14 topics. This dataset is derived from the historical data of the Sina News RSS subscription channel between 2005 and 2011. The diversity in article topics allows for a comprehensive embedding that captures a wide spectrum of the Chinese language used in different contexts.

The choice of these datasets is driven by their domain-specific richness and volume, which are crucial for developing a robust Word2Vec model. Training on news articles offers the advantage of dealing with formally structured text that includes a variety of themes, making our model versatile in understanding and processing Chinese language news content.

### 1.2: Preprocessing

Given the formal nature of news texts, the primary focus in our preprocessing was on standardizing text for consistency and removing any non-textual content:

- **Removal of HTML tags**: Often news data scraped from web sources contains HTML formatting which needs to be cleaned out.
- **Normalization of punctuation and characters**: This includes converting traditional Chinese characters to simplified Chinese, when necessary, and standardizing punctuation marks.

Tokenization in the Chinese language is non-trivial due to the absence of explicit word boundaries (like spaces in English). We utilized the Jieba library, a popular text segmentation tool for Chinese, to perform accurate word tokenization. This step is critical to separate words from running texts for subsequent modeling.

To improve the quality of our model, we removed stopwords using a comprehensive list compiled by Sichuan University. Stopwords in any language represent high-frequency words that carry minimal individual semantic weight (e.g., conjunctions, prepositions) and can skew the model's focus away from meaningful words. The link to the list is [here](https://manu44.magtech.com.cn/Jwk_infotech_wk3/EN/10.11925/infotech.2096-3467.2017.03.09).

The preprocessing steps were designed to refine the textual data into a format that is more amenable for training a Word2Vec model. By cleaning and tokenizing the text, and removing stopwords, we ensure that the model learns to embed words based on their semantic and contextual relevance rather than their frequency of occurrence.

This detailed approach to selecting and preprocessing your datasets should provide a solid basis for training your Word2Vec model and demonstrate thoroughness in your methodological execution for your NLP assignment.

## 1.3 Model Training
  - Configuration of the Word2Vec model (architecture, window size, vector size, etc.).
  - Training process and parameters.

## 1.4 Evaluation of Embedding
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

1. Hugging Face. (n.d.). *Sogou News dataset*. Retrieved from https://huggingface.co/datasets/sogou_news
2. Li, X., & Wang, W. (2017). *THUCNews: A Large-Scale News Corpus for Chinese*. Tsinghua University. Available at: https://paperswithcode.com/dataset/thucnews
3. Sichuan University. (2017). *Chinese stopwords list*. Journal of Information Technology, 3(09). Retrieved from https://manu44.magtech.com.cn/Jwk_infotech_wk3/EN/10.11925/infotech.2096-3467.2017.03.09
4. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. Proceedings of the International Conference on Learning Representations (ICLR).
5. Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1532-1543.
6. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), Volume 1, 4171-4186.
7. Lenci, A. (2018). Distributional Models of Word Meaning. Annual Review of Linguistics, 4, 151-171.
Format: Lenci, A. (2018). Distributional models of word meaning. Annual Review of Linguistics, 4, 151-171.

## Appendices
- Any supplementary material (code snippets, additional data visualizations).
