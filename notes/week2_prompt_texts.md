# Pipeline Evaluation Report
Generated: 2026-04-22 17:30

## Summary table

| # | Test | Retrieved | Max sim | Refused | Title halluc | Think halluc | Translated | Resp len |
|---|------|-----------|---------|---------|--------------|--------------|------------|----------|
| 1 | Clear match – BERT fine-tuning | 5 | 0.7437 | no | no | no | no | 1839 |
| 2 | Clear match – LLM summarization | 5 | 0.5232 | no | YES ⚠ | no | no | 2175 |
| 3 | Clear match – Spanish query (translation fix) | 5 | 0.4019 | no | no | no | yes | 2402 |
| 4 | Clear match – Spanish NLP (translation fix) | 5 | 0.4673 | no | YES ⚠ | no | yes | 2750 |
| 5 | Clear match – knowledge graphs | 5 | 0.6406 | no | YES ⚠ | no | no | 2270 |
| 6 | Clear match – sentiment analysis | 5 | 0.4474 | no | YES ⚠ | no | no | 1970 |
| 7 | Borderline – broad topic | 5 | 0.3456 | YES | no | no | no | 0 |
| 8 | Weak match – unrelated domain (threshold check) | 5 | 0.4328 | no | no | no | no | 2638 |
| 9 | Weak match – very off-topic | 5 | -0.3139 | YES | no | no | no | 0 |
| 10 | Edge case – single word | 3 | 0.2679 | YES | no | no | no | 0 |

---

## Detailed results

### 1. Clear match – BERT fine-tuning
**Query:** `fine-tuning BERT for text classification tasks`  
**Expected:** should answer  
**Retrieved papers (5):**
- How to Fine-Tune BERT for Text Classification?
- Improving BERT Fine-tuning with Embedding Normalization
- Improving BERT Fine-Tuning via Self-Ensemble and Self-Distillation
- To Tune or Not To Tune? How About the Best of Both Worlds?
- Revisiting Few-sample BERT Fine-tuning

**Scores:** avg=0.5746 | max=0.7437  
**Refused:** False  
**Title hallucination detected:** False  
**Think-block hallucination:** False  

**Model response:**
```text
- Research direction:  
  Fine-tuning BERT for text classification tasks  

- Relevant articles:  
  1. "How to Fine-Tune BERT for Text Classification?" — Focuses on experimental investigation of fine-tuning methods for BERT in text classification.  
  2. "Improving BERT Fine-tuning with Embedding Normalization" — Proposes embedding normalization as a technique to enhance fine-tuning performance.  
  3. "Improving BERT Fine-Tuning via Self-Ensemble and Self-Distillation" — Explores self-ensemble and self-distillation strategies to improve fine-tuning.  
  4. "To Tune or Not To Tune? How About the Best of Both Worlds?" — Compares lightweight task-specific heads versus complex models for BERT adaptation.  
  5. "Revisiting Few-sample BERT Fine-tuning" — Analyzes instabilities in few-sample fine-tuning scenarios and identifies contributing factors.  

- Key differences:  
  - The user’s work may focus on **specific fine-tuning methods** (e.g., hyperparameter settings, data preprocessing) not explicitly covered in the articles.  
  - The user’s research may not address **few-sample scenarios** (Article 5) or **embedding normalization** (Article 2), depending on the scope.  
  - The user’s approach may differ in **model architecture choices** (e.g., lightweight vs. complex heads, Article 4).  

- Possible novelty:  
  - The user’s work **may** explore novel combinations of fine-tuning strategies (e.g., integrating self-ensemble with normalization).  
  - The user’s research **could** address **instabilities** in few-sample settings, which is a gap in existing literature (Article 5).  

- Conclusion:  
  The user’s research aligns closely with existing work on BERT fine-tuning for text classification but may introduce novel methods or focus on underexplored aspects like few-sample instability or hybrid techniques.
```

### 2. Clear match – LLM summarization
**Query:** `large language models for abstractive text summarization`  
**Expected:** should answer  
**Retrieved papers (5):**
- Challenges in Domain-Specific Abstractive Summarization and How to
  Overcome them
- Combination of abstractive and extractive approaches for summarization
  of long scientific texts
- Evaluation of Abstractive Summarisation Models with Machine Translation
  in Deliberative Processes
- Abstractive Text Summarization Using the BRIO Training Paradigm
- HunSum-1: an Abstractive Summarization Dataset for Hungarian

**Scores:** avg=0.4727 | max=0.5232  
**Refused:** False  
**Title hallucination detected:** True  
**Think-block hallucination:** False  

**Model response:**
```text
- Research direction:  
  Exploring the use of large language models (LLMs) for abstractive text summarization.  

- Relevant articles:  
  1. *"Challenges in Domain-Specific Abstractive Summarization and How to Overcome them"* — Focuses on limitations of LLMs in domain-specific summarization tasks.  
  2. *"Combination of abstractive and extractive approaches for summarization of long scientific texts"* — Proposes hybrid methods using both extractive and abstractive techniques for long texts.  
  3. *"Evaluation of Abstractive Summarisation Models with Machine Translation in Deliberative Processes"* — Examines abstractive summarization for non-English, deliberative texts with poor grammatical quality.  
  4. *"Abstractive Text Summarization Using the BRIO Training Paradigm"* — Introduces a training method to reduce reliance on reference summaries for abstractive models.  
  5. *"HunSum-1: an Abstractive Summarization Dataset for Hungarian"* — Presents a language-specific dataset for Hungarian abstractive summarization.  

- Key differences:  
  - The user’s focus on **LLMs alone** contrasts with Article 2’s hybrid approach (extractive + abstractive) and Article 4’s BRIO training paradigm (which modifies model training).  
  - The user’s work may not address **domain-specific limitations** (Article 1) or **non-English, deliberative texts** (Article 3) unless explicitly stated.  
  - The user’s idea does not mention **dataset creation** (Article 5), which is specific to Hungarian news articles.  

- Possible novelty:  
  - The user’s idea **may** address domain-specific challenges (Article 1) or improve reference independence (Article 4), but this is **unclear** without further details.  
  - The focus on **LLMs without hybrid methods** could be **novel** if the user’s work emphasizes scalability or generalization beyond existing approaches.  

- Conclusion:  
  The user’s research aligns with existing work on abstractive summarization but may differ in scope (e.g., domain-specific vs. general, hybrid methods vs. standalone LLMs). Novelty depends on addressing specific limitations or expanding to new contexts (e.g., non-English datasets).
```

### 3. Clear match – Spanish query (translation fix)
**Query:** `aprendizaje automático para clasificación de texto`  
**Translated for search:** `Machine learning for text classification`  
**Expected:** should translate and answer in Spanish  
**Retrieved papers (5):**
- Machine Learning in Automated Text Categorization
- Machine learning approach for text and document mining
- Naive Bayes and Text Classification I - Introduction and Theory
- Text Classification Algorithms: A Survey
- A Sequential Algorithm for Training Text Classifiers

**Scores:** avg=0.3682 | max=0.4019  
**Refused:** False  
**Title hallucination detected:** False  
**Think-block hallucination:** False  

**Model response:**
```text
- **Dirección de investigación:**  
  Aplicación del aprendizaje automático para la clasificación de texto.  

- **Artículos relevantes:**  
  1. *"Machine Learning in Automated Text Categorization"* — Enfoque en la categorización automática de textos mediante técnicas de aprendizaje automático, destacando un proceso inductivo para construir clasificadores.  
  2. *"Machine learning approach for text and document mining"* — Definición de la clasificación de texto como tarea de asignar documentos a categorías predefinidas, con mención a clasificación single-label y multi-label.  
  3. *"Naive Bayes and Text Classification I - Introduction and Theory"* — Enfoque en clasificadores Naive Bayes para categorización de documentos, destacando su simplicidad y rendimiento.  
  4. *"Text Classification Algorithms: A Survey"* — Revisión de algoritmos de clasificación de texto, resaltando el crecimiento de documentos complejos y el rol del aprendizaje automático en procesamiento de lenguaje natural.  
  5. *"A Sequential Algorithm for Training Text Classifiers"* — Desarrollo de un algoritmo secuencial para entrenar clasificadores de texto, basado en muestreo de incertidumbre.  

- **Diferencias clave:**  
  - **Enfoque metodológico:** La idea del usuario no especifica técnicas particulares (como Naive Bayes o muestreo de incertidumbre), mientras que los artículos mencionan métodos concretos.  
  - **Aplicaciones:** El usuario no detalla aplicaciones específicas (por ejemplo, recuperación de información o procesamiento de lenguaje natural), aunque los artículos abordan estas áreas.  
  - **Complejidad de datos:** No se menciona explícitamente el manejo de documentos complejos o no lineales, temas abordados en algunos artículos.  

- **Posible novedad:**  
  - **Aplicaciones no especificadas:** La idea del usuario podría explorar aplicaciones no cubiertas en los artículos, como clasificación en dominios específicos o integración con otras tareas.  
  - **Metodologías generales:** Puede enfocarse en enfoques más generales de aprendizaje automático, en lugar de algoritmos específicos como Naive Bayes o muestreo de incertidumbre.  

- **Conclusión:**  
  La idea del usuario se alinea con la investigación existente en clasificación de texto mediante aprendizaje automático, pero podría presentar diferencias en enfoques metodológicos o aplicaciones no detalladas en los artículos.
```

### 4. Clear match – Spanish NLP (translation fix)
**Query:** `redes neuronales para procesamiento del lenguaje natural`  
**Translated for search:** `Neural Networks for Natural Language Processing`  
**Expected:** should translate and answer in Spanish  
**Retrieved papers (5):**
- A Primer on Neural Network Models for Natural Language Processing
- A Survey on Neural Network Language Models
- Using Single Layer Networks for Discrete, Sequential Data: An Example
  from Natural Language Processing
- A fast partial parse of natural language sentences using a connectionist
  method
- Analysis Methods in Neural Language Processing: A Survey

**Scores:** avg=0.3911 | max=0.4673  
**Refused:** False  
**Title hallucination detected:** True  
**Think-block hallucination:** False  

**Model response:**
```text
- **Dirección de investigación:**  
  Explorar el uso de redes neuronales en el procesamiento del lenguaje natural (PLN), incluyendo sus aplicaciones y modelos.  

- **Artículos relevantes:**  
  1. *"A Primer on Neural Network Models for Natural Language Processing"* — Revisa el uso de redes neuronales en PLN, destacando su impacto en tareas como el reconocimiento de imágenes y el procesamiento de voz, y su aplicación reciente al procesamiento de texto.  
  2. *"A Survey on Neural Network Language Models"* — Enfocado en modelos de lenguaje basados en redes neuronales, su estructura y mejoras que superan limitaciones de modelos tradicionales.  
  3. *"Using Single Layer Networks for Discrete, Sequential Data: An Example from Natural Language Processing"* — Describe un sistema híbrido que combina redes neuronales con reglas para análisis de texto, con énfasis en la identificación de sujetos y objetos en textos técnicos.  
  4. *"A fast partial parse of natural language sentences using a connectionist method"* — Presenta un sistema híbrido que utiliza redes neuronales dentro de un marco gramatical para análisis sintáctico parcial de frases.  
  5. *"Analysis Methods in Neural Language Processing: A Survey"* — Aborda métodos para analizar y evaluar modelos neuronales en PLN, destacando su opacidad y la necesidad de interpretación.  

- **Diferencias clave:**  
  - **Enfoque específico:** Los artículos 3 y 4 se centran en aplicaciones concretas (análisis sintáctico y parsing), mientras que el usuario podría estar interesado en un enfoque más general o en otro tipo de tareas (por ejemplo, traducción o generación de texto).  
  - **Arquitecturas:** El usuario no especifica si se enfocará en redes neuronales de capa única (como en el artículo 3) o en arquitecturas más complejas (como las de los modelos de lenguaje en el artículo 2).  
  - **Interpretabilidad:** El artículo 5 destaca la necesidad de análisis de modelos neuronales, algo que no se menciona explícitamente en los demás artículos.  

- **Posible novedad:**  
  - **Aplicaciones específicas:** Si el usuario aborda tareas no cubiertas en los artículos (por ejemplo, generación de texto o análisis de sentimientos), podría representar una novedad.  
  - **Métodos híbridos:** Si combina redes neuronales con técnicas no mencionadas en los artículos (como enfoques no basados en reglas), podría ser una dirección innovadora.  

- **Conclusión:**  
  El enfoque del usuario se alinea con los artículos 1 y 2, que revisan modelos neuronales en PLN, pero podría diferir en aplicaciones específicas o enfoques metodológicos no explícitamente abordados en los artículos proporcionados. La novedad depende de la elección de tareas o técnicas no cubiertas en el contexto.
```

### 5. Clear match – knowledge graphs
**Query:** `knowledge graph embeddings for link prediction`  
**Expected:** should answer  
**Retrieved papers (5):**
- Transformation of Node to Knowledge Graph Embeddings for Faster Link
  Prediction in Social Networks
- LineaRE: Simple but Powerful Knowledge Graph Embedding for Link
  Prediction
- Global and local evaluation of link prediction tasks with neural
  embeddings
- SimplE Embedding for Link Prediction in Knowledge Graphs
- Domain Representation for Knowledge Graph Embedding

**Scores:** avg=0.5869 | max=0.6406  
**Refused:** False  
**Title hallucination detected:** True  
**Think-block hallucination:** False  

**Model response:**
```text
- Research direction:  
  Using knowledge graph embeddings to improve link prediction in structured data.  

- Relevant articles:  
  1. "Transformation of Node to Knowledge Graph Embeddings for Faster Link Prediction in Social Networks" — Focuses on converting node embeddings into knowledge graph embeddings to accelerate link prediction in social networks.  
  2. "LineaRE: Simple but Powerful Knowledge Graph Embedding for Link Prediction" — Proposes a knowledge graph embedding method that achieves strong predictive performance for link prediction tasks.  
  3. "Global and local evaluation of link prediction tasks with neural embeddings" — Explores evaluation methodologies for link prediction using neural embeddings, emphasizing benchmarking and comparison of approaches.  
  4. "SimplE Embedding for Link Prediction in Knowledge Graphs" — Introduces a tensor factorization-based method for link prediction, highlighting its effectiveness for incomplete knowledge graphs.  
  5. "Domain Representation for Knowledge Graph Embedding" — Proposes domain-specific hierarchical representations to enhance knowledge graph embeddings, addressing limitations in existing models.  

- Key differences:  
  - The user’s idea does not specify whether it focuses on **tensor factorization** (as in Article 4) or **neural network-based embeddings** (Articles 1–3).  
  - The user’s work does not explicitly address **domain-specific hierarchical knowledge** (Article 5) or **evaluation methodologies** (Article 3).  
  - The user’s research may differ in **application domains** (e.g., social networks vs. general knowledge graphs).  

- Possible novelty:  
  - The user’s approach **may** combine elements from multiple methods (e.g., neural embeddings + domain representation) not explicitly integrated in the retrieved works.  
  - It **suggests** a focus on **efficiency** (e.g., faster link prediction) or **domain adaptation**, which are not the primary emphasis of all articles.  

- Conclusion:  
  The user’s research aligns with existing work on knowledge graph embeddings for link prediction but may differ in specific methodologies (e.g., tensor factorization vs. neural networks) or application domains. The novelty remains **unclear** without further details.
```

### 6. Clear match – sentiment analysis
**Query:** `sentiment analysis using deep learning`  
**Expected:** should answer  
**Retrieved papers (5):**
- Deep Learning for Sentiment Analysis : A Survey
- Sentiment Analysis Based on Deep Learning: A Comparative Study
- Sentiment Analysis for Sinhala Language using Deep Learning Techniques
- Exploiting Deep Learning for Persian Sentiment Analysis
- BowTie - A deep learning feedforward neural network for sentiment
  analysis

**Scores:** avg=0.3712 | max=0.4474  
**Refused:** False  
**Title hallucination detected:** True  
**Think-block hallucination:** False  

**Model response:**
```text
- Research direction:  
  Development of sentiment analysis techniques using deep learning methods.  

- Relevant articles:  
  1. "Deep Learning for Sentiment Analysis : A Survey" — A survey paper summarizing deep learning applications in sentiment analysis, emphasizing state-of-the-art methods.  
  2. "Sentiment Analysis Based on Deep Learning: A Comparative Study" — Focuses on challenges in social media sentiment analysis and evaluates deep learning efficiency and accuracy.  
  3. "Sentiment Analysis for Sinhala Language using Deep Learning Techniques" — Addresses sentiment analysis for an under-resourced language (Sinhala) with limited prior work.  
  4. "Exploiting Deep Learning for Persian Sentiment Analysis" — Targets sentiment analysis for Persian, highlighting the need for deep learning in handling social media data.  
  5. "BowTie - A deep learning feedforward neural network for sentiment analysis" — Proposes a specific neural network architecture (BowTie) for sentiment analysis.  

- Key differences:  
  - The user’s idea is general, while Articles 3–5 focus on specific languages (Sinhala, Persian) or specific models (BowTie).  
  - Articles 1 and 2 emphasize broader methodological surveys or comparative studies, whereas the user’s idea may not specify a particular model, language, or study type.  
  - The user’s work may not address language-specific challenges (e.g., morphology in Sinhala) or propose novel architectures (e.g., BowTie).  

- Possible novelty:  
  - The user’s idea may **suggest** a general approach to sentiment analysis without language-specific constraints, which differs from Articles 3–4.  
  - It may **possibly** avoid the focus on comparative studies (Article 2) or model-specific innovations (Article 5).  

- Conclusion:  
  The user’s research aligns with general deep learning applications in sentiment analysis but differs from specific language-focused or model-driven approaches in the retrieved works.
```

### 7. Borderline – broad topic
**Query:** `deep learning`  
**Expected:** may answer with generic papers  
**Retrieved papers (5):**
- Introduction to deep learning
- Deep Learning: Computational Aspects
- Improving Deep Learning through Automatic Programming
- A Selective Overview of Deep Learning
- Deep Learning: A Critical Appraisal

**Scores:** avg=0.3089 | max=0.3456  
**Refused:** True  
**Title hallucination detected:** False  
**Think-block hallucination:** False  

**Model response:**
```text
(no response — refused)
```

### 8. Weak match – unrelated domain (threshold check)
**Query:** `quantum computing error correction`  
**Expected:** should now refuse with threshold 0.35  
**Retrieved papers (5):**
- Uncertainty Quantification for Quantum Computing
- Enhancing Quantum Memory Lifetime with Measurement-Free Local Error
  Correction and Reinforcement Learning
- The END: An Equivariant Neural Decoder for Quantum Error Correction
- Learning Encodings by Maximizing State Distinguishability: Variational Quantum Error Correction
- Near-Term Quantum Computing Techniques: Variational Quantum Algorithms,
  Error Mitigation, Circuit Compilation, Benchmarking and Classical Simulation

**Scores:** avg=0.3531 | max=0.4328  
**Refused:** False  
**Title hallucination detected:** False  
**Think-block hallucination:** False  

**Model response:**
```text
- Research direction:  
  The user’s research idea centers on **quantum computing error correction**, focusing on methods to identify, mitigate, or correct errors in quantum systems.  

- Relevant articles:  
  1. **Uncertainty Quantification for Quantum Computing** — Focuses on modeling noise and randomness in quantum systems using mathematical frameworks, but does not directly address error correction techniques.  
  2. **Enhancing Quantum Memory Lifetime with Measurement-Free Local Error Correction and Reinforcement Learning** — Proposes local error correction without global readouts, combining reinforcement learning for error mitigation.  
  3. **The END: An Equivariant Neural Decoder for Quantum Error Correction** — Introduces neural network-based decoders to approximate optimal error correction mappings, emphasizing efficiency.  
  4. **Learning Encodings by Maximizing State Distinguishability: Variational Quantum Error Correction** — Proposes a variational framework to tailor error correction codes to specific noise structures.  
  5. **Near-Term Quantum Computing Techniques: Variational Quantum Algorithms, Error Mitigation, Circuit Compilation, Benchmarking and Classical Simulation** — Broadly discusses error mitigation as part of near-term quantum computing, but does not focus on specific error correction protocols.  

- Key differences:  
  - The user’s idea is **broader** than Articles 1 and 5, which focus on uncertainty quantification or general near-term techniques rather than explicit error correction protocols.  
  - Articles 2, 3, and 4 emphasize **specific methods** (local error correction, neural decoders, variational codes), while the user’s idea may not specify a particular approach.  
  - Article 1 and 5 do not address **error correction mechanisms** directly, focusing instead on noise analysis or general computational techniques.  

- Possible novelty:  
  - The user’s work **may** align with Articles 2–4 if it explores novel methods like local error correction, neural decoders, or variational codes. However, the context does not specify whether the user’s approach differs from these.  
  - If the user’s focus is on **systematic error diagnosis** (as in Article 2) or **adaptive decoding** (as in Article 3), it could represent a **new angle** compared to traditional codes (Article 4).  

- Conclusion:  
  The user’s research idea overlaps with several articles on quantum error correction but may differ in methodological focus or scope. The novelty depends on whether the user’s work introduces new techniques or applications not explicitly covered in the retrieved studies.
```

### 9. Weak match – very off-topic
**Query:** `history of the Roman Empire`  
**Expected:** should refuse  
**Retrieved papers (5):**
- Automated Attribution and Intertextual Analysis
- Dating medieval English charters
- Historia Magistra Vitae: Dynamic Topic Modeling of Roman Literature
  using Neural Embeddings
- Vulgaris: Analysis of a Corpus for Middle-Age Varieties of Italian
  Language
- Chronological Analysis of Rigvedic Mandalas using Social Networks

**Scores:** avg=-0.3452 | max=-0.3139  
**Refused:** True  
**Title hallucination detected:** False  
**Think-block hallucination:** False  

**Model response:**
```text
(no response — refused)
```

### 10. Edge case – single word
**Query:** `attention`  
**Expected:** should answer — attention is a core NLP concept  
**Retrieved papers (3):**
- An Attentive Survey of Attention Models
- Visual Attention Methods in Deep Learning: An In-Depth Survey
- Copy this Sentence

**Scores:** avg=0.2413 | max=0.2679  
**Refused:** True  
**Title hallucination detected:** False  
**Think-block hallucination:** False  

**Model response:**
```text
(no response — refused)
```
