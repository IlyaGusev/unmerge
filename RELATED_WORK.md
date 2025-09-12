# COMPREHENSIVE RELATED WORK FOR "UNMERGE: CAUSAL UNMERGING VIA SPARSE CODING"

## Abstract
This comprehensive survey covers six key research areas relevant to UNMERGE: neural network decomposition and parameter attribution, sparse coding and dictionary learning for neural networks, LoRA and adapter analysis, model merging techniques, causal intervention and model editing, and mechanistic interpretability. We focus on high-quality papers from 2022-2025 alongside foundational works, organizing them by methodology and contribution to establish the theoretical and empirical foundations for causal unmerging via sparse coding.

---

## 1. NEURAL NETWORK DECOMPOSITION AND PARAMETER ATTRIBUTION

### 1.1 Foundational Network Dissection (2017-2020)

**Network Dissection: Quantifying Interpretability of Deep Visual Representations** (arXiv:1704.05796v1)
- *Authors*: David Bau, Bolei Zhou, Aditya Khosla, Antonio Torralba
- *Venue*: CVPR 2017 (Oral Presentation)
- *Key Contribution*: Establishes general framework for quantifying interpretability by evaluating alignment between individual hidden units and semantic concepts
- *Method*: Scores semantics of hidden units across objects, parts, scenes, textures, materials, and colors using broad visual concept datasets
- *Relevance to UNMERGE*: Provides foundation for understanding how individual network components encode specific capabilities

**Understanding the Role of Individual Units in a Deep Neural Network** (arXiv:2009.05041v2)
- *Authors*: David Bau, Jun-Yan Zhu, Hendrik Strobelt, Bolei Zhou, Joshua B. Tenenbaum, Antonio Torralba
- *Venue*: PNAS 2020
- *Key Contribution*: Network dissection framework for systematically identifying semantics of individual hidden units in both discriminative and generative models
- *Method*: Analyzes CNNs and GANs by activating/deactivating small sets of units to understand causal role in output generation
- *Relevance to UNMERGE*: Demonstrates surgical intervention methods for understanding component-wise functionality

### 1.2 Recent Parameter-Space Decomposition (2024-2025)

**Attribution-based Parameter Decomposition (APD)** (arXiv:2501.14926v4)
- *Authors*: Dan Braun, Lucius Bushnaq, Stefan Heimersheim, et al.
- *Date*: January 2025
- *Key Contribution*: Direct decomposition of neural network parameters into mechanistic components that are faithful, minimal, and simple
- *Method*: Optimizes for minimal description length while maintaining (i) faithfulness to original parameters, (ii) minimal components per input, (iii) maximal simplicity via low-rank constraints
- *Results*: Successfully recovers ground-truth mechanisms in superposition, compressed computations, and cross-layer distributed representations
- *Relevance to UNMERGE*: Most directly related work - provides theoretical foundation for parameter-space decomposition with sparsity constraints

**Neuron-Level Knowledge Attribution in Large Language Models** (EMNLP 2024)
- *Key Contribution*: Static method for neuron-level knowledge attribution achieving best performance under three metrics compared to seven static methods
- *Method*: Identifies neurons that directly contribute to specific knowledge without requiring dynamic interventions
- *Relevance to UNMERGE*: Provides evaluation frameworks for assessing quality of neural component attribution

---

## 2. SPARSE CODING AND DICTIONARY LEARNING FOR NEURAL NETWORKS

### 2.1 Foundational Sparse Coding in Neural Networks (2020-2021)

**The Interpretable Dictionary in Sparse Coding** (arXiv:2011.11805v1)
- *Authors*: Edward Kim, Connor Onweller, Andrew O'Brien, et al.
- *Date*: November 2020
- *Key Contribution*: Shows that ANNs trained using sparse coding under specific sparsity constraints yield more interpretable models than standard deep learning
- *Method*: Compares sparse coding model with equivalent feed-forward convolutional autoencoder on same data, demonstrates qualitative and quantitative interpretability benefits
- *Relevance to UNMERGE*: Establishes connection between sparse coding and neural network interpretability

### 2.2 Sparse Autoencoders for Neural Analysis (2024-2025)

**Disentangling Dense Embeddings with Sparse Autoencoders** (arXiv:2408.00657v2)
- *Authors*: Charles O'Neill, Christine Ye, Kartheik Iyer, David Klindt
- *Date*: August 2024
- *Key Contribution*: First application of SAEs to dense text embeddings from LLMs, demonstrating effectiveness in disentangling semantic concepts
- *Method*: Trained on 420,000+ scientific paper abstracts, introduces "feature families" concept for related concepts at varying abstraction levels
- *Applications*: Precise semantic search steering with fine-grained control over query semantics
- *Relevance to UNMERGE*: Demonstrates sparse autoencoder effectiveness for decomposing dense neural representations

**Gradient Sparse Autoencoder (GradSAE)** (arXiv:2505.08080v1)
- *Authors*: Dong Shu, Xuansheng Wu, Haiyan Zhao, et al.
- *Date*: May 2025
- *Key Contribution*: Identifies influential latents by incorporating output-side gradient information beyond input activations
- *Method*: Validates hypotheses that (1) activated latents contribute unequally to model output, (2) only high causal influence latents effective for steering
- *Relevance to UNMERGE*: Provides framework for identifying causally relevant sparse components

**Ensembling Sparse Autoencoders** (arXiv:2505.16077v1)
- *Authors*: Soham Gadgil, Chris Lin, Su-In Lee
- *Date*: May 2025
- *Key Contribution*: Ensemble approaches through naive bagging and boosting for SAEs to improve reconstruction and feature diversity
- *Method*: Multiple SAEs with different initializations (bagging), sequential training to minimize residual error (boosting)
- *Results*: Improved reconstruction quality, feature diversity, SAE stability, and downstream task performance
- *Relevance to UNMERGE*: Shows how multiple sparse dictionaries can be combined effectively

### 2.3 Advanced SAE Applications (2024-2025)

**Causal Interpretation of Sparse Autoencoder Features in Vision** (arXiv:2509.00749v1)
- *Authors*: Sangyu Han, Yearim Kim, Nojun Kwak
- *Date*: August 2025
- *Key Contribution*: Causal Feature Explanation (CaFE) using Effective Receptive Field to identify image patches that causally drive SAE feature activation
- *Method*: Applies input-attribution methods to each SAE feature activation, reveals hidden context dependencies
- *Results*: CaFE more effectively recovers/suppresses feature activations than activation-ranked patches
- *Relevance to UNMERGE*: Demonstrates causal verification methods for sparse feature decompositions

---

## 3. LORA (LOW-RANK ADAPTATION) AND ADAPTER ANALYSIS

### 3.1 Foundational LoRA (2021)

**LoRA: Low-Rank Adaptation of Large Language Models** (arXiv:2106.09685v2)
- *Authors*: Edward J. Hu, Yelong Shen, Phillip Wallis, et al.
- *Venue*: ICLR 2022
- *Key Contribution*: Freezes pre-trained weights, injects trainable rank decomposition matrices: W₀ + ΔW = W₀ + BA where B ∈ ℝᵈˣʳ, A ∈ ℝʳˣᵏ, rank r ≪ min(d,k)
- *Method*: Forward pass h = W₀x + BAx, A initialized Gaussian, B initialized zero, scaled by α/r
- *Results*: Reduces trainable parameters by 10,000x, GPU memory by 3x, matches/exceeds full fine-tuning performance
- *Relevance to UNMERGE*: Establishes low-rank parameter adaptation as effective method for capturing task-specific capabilities

### 3.2 Recent LoRA Analysis and Theory (2024-2025)

**Randomized Asymmetric Chain of LoRA (RAC-LoRA)** (arXiv:2410.08305v1)
- *Authors*: Grigory Malinovsky, Umberto Michieli, Hasan Abed Al Kader Hammoud, et al.
- *Date*: October 2024
- *Key Contribution*: First meaningful theoretical framework for LoRA with provable convergence guarantees
- *Method*: Addresses convergence issues in LoRA/Asymmetric LoRA/Chain of LoRA, provides convergence analysis for smooth non-convex loss functions
- *Theory*: Covers gradient descent, SGD, federated learning settings with convergence rates
- *Relevance to UNMERGE*: Provides theoretical foundation for low-rank adaptation methods

**LoRA Done RITE: Robust Invariant Transformation Equilibration** (arXiv:2410.20625v2)
- *Authors*: Jui-Nan Yen, Si Si, Zhao Meng, et al.
- *Venue*: ICLR 2025 (Oral)
- *Key Contribution*: Adaptive matrix preconditioning for transformation-invariant LoRA optimization
- *Method*: Addresses lack of transformation invariance in current LoRA optimizers where updates depend on factor scaling/rotation
- *Results*: 4.66% BLEU improvement on Super-Natural Instructions, 3.5% across other benchmarks (Gemma-2B)
- *Relevance to UNMERGE*: Shows importance of proper optimization for low-rank parameter adaptations

### 3.3 Mechanistic Understanding of LoRA (2025)

**Behind the Scenes: Mechanistic Interpretability of LoRA-adapted Whisper** (arXiv:2509.08454v1)
- *Authors*: Yujian Ma, Jinqiu Sang, Ruizhe Li
- *Date*: September 2025
- *Key Contribution*: First systematic mechanistic interpretability study of LoRA adaptation mechanisms
- *Method*: Layer contribution probing, logit-lens inspection, representational similarity via SVD and CKA
- *Findings*: (1) Delayed specialization: preserves general features in early layers before task-specific consolidation, (2) Forward alignment, backward differentiation dynamics between LoRA matrices
- *Relevance to UNMERGE*: Provides mechanistic understanding of how low-rank adaptations reshape model capabilities

---

## 4. MODEL MERGING TECHNIQUES

### 4.1 Task Arithmetic - Foundational (2022-2023)

**Editing Models with Task Arithmetic** (arXiv:2212.04089v3)
- *Authors*: Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, et al.
- *Venue*: ICLR 2023
- *Key Contribution*: Task vectors τₜ = θₜᶠᵗ - θₚᵣₑ as directions in weight space for steering model behavior
- *Method*: Arithmetic operations on task vectors - negation (forgetting), addition (multi-task), analogies (A:B::C:D)
- *Results*: 6x reduction in toxic generation via negation, 98.9% accuracy preservation in multi-task via addition
- *Limitations*: Same architecture required, shared pre-trained initialization, learning rate sensitivity
- *Relevance to UNMERGE*: Establishes parameter-space arithmetic as method for combining/separating capabilities

### 4.2 Model Soups (2022)

**Model soups: averaging weights of multiple fine-tuned models** (arXiv:2203.05482v3)
- *Authors*: Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, et al.
- *Venue*: ICML 2022 (1102 citations)
- *Key Contribution*: Weight averaging of multiple fine-tuned models improves accuracy without inference cost
- *Method*: Hyperparameter sweep + weight averaging when models lie in single low error basin
- *Results*: ViT-G achieving 90.94% ImageNet accuracy (new SOTA), improved OOD performance and zero-shot transfer
- *Relevance to UNMERGE*: Shows multiple fine-tuned capabilities can be linearly combined

### 4.3 Advanced Merging Methods (2024)

**TIES-Merging: Resolving Interference When Merging Models** (arXiv:2306.01708v2)
- *Key Contribution*: Addresses interference in model merging through systematic parameter selection and sign agreement
- *Method*: Three steps - (1) Trim redundant parameters, (2) Elect sign based on majority vote, (3) Merge with Task Arithmetic
- *Results*: Reduces interference between conflicting parameter updates during merging
- *Relevance to UNMERGE*: Provides methodology for handling conflicts when combining sparse components

**Language Models are Super Mario: Absorbing Abilities (DARE)** (arXiv:2311.03099v1)
- *Key Contribution*: DARE (Drop And REscale) reduces delta parameter redundancy before merging
- *Method*: (1) Randomly drops proportion of delta parameters, (2) Rescales remaining parameters by 1/(1-p)
- *Applications*: Can be combined with other merging methods, reduces parameter redundancy without retraining
- *Relevance to UNMERGE*: Demonstrates effectiveness of sparsification in parameter-space operations

### 4.4 Large-Scale Merging Analysis (2024)

**What Matters for Model Merging at Scale?** (arXiv:2410.03617v1)
- *Authors*: Prateek Yadav, Tu Vu, Jonathan Lai, et al.
- *Date*: October 2024
- *Key Contribution*: Systematic evaluation across 1B-64B parameters, up to 8 expert models, 4 merging methods
- *Findings*: (1) Strong base models crucial, (2) Larger models easier to merge, (3) Merging improves generalization, (4) Different methods converge at scale
- *Methods Compared*: Averaging, Task Arithmetic, DARE, TIES across held-in and zero-shot tasks
- *Relevance to UNMERGE*: Establishes scaling behaviors and best practices for model merging

---

## 5. CAUSAL INTERVENTION AND MODEL EDITING

### 5.1 Activation Patching and Causal Tracing (2023-2024)

**Towards Best Practices of Activation Patching in Language Models** (arXiv:2309.16042v2)
- *Authors*: Fred Zhang, Neel Nanda
- *Venue*: ICLR 2024
- *Key Contribution*: Systematic examination of activation patching methodology including evaluation metrics and corruption methods
- *Method*: Standardizes hyperparameters and methodology for localization and circuit discovery in language models
- *Impact*: Provides methodological foundation for causal intervention in neural networks
- *Relevance to UNMERGE*: Establishes best practices for causal verification of neural component decompositions

### 5.2 Surgical Model Editing (2024-2025)

**Surgical, Cheap, and Flexible: Mitigating False Refusal via Single Vector Ablation** (arXiv:2410.03415v2)
- *Authors*: Xinpeng Wang, Chengzhi Hu, Paul Röttger, et al.
- *Venue*: ICLR 2025
- *Key Contribution*: Training-free, model-agnostic method for surgical intervention via single vector ablation
- *Method*: Extract false refusal vector, ablate to reduce false refusal while preserving safety and general capabilities
- *Results*: Effective calibration of model safety without requiring retraining
- *Relevance to UNMERGE*: Demonstrates surgical subtraction approach for capability removal

**PMET: Precise Model Editing in a Transformer** (arXiv:2308.08742v6)
- *Authors*: Xiaopeng Li, Shasha Li, Shezheng Song, et al.
- *Venue*: AAAI 2024
- *Key Contribution*: Precise model editing by optimizing Transformer Component hidden states, updating only FFN weights
- *Method*: Separates MHSA and FFN contributions, finding MHSA encodes general knowledge extraction patterns
- *Results*: State-of-the-art performance on COUNTERFACT and zsRE datasets
- *Relevance to UNMERGE*: Shows how to precisely edit specific model components while preserving others

### 5.3 Advanced Causal Analysis (2025)

**Can LLMs Lie? Investigation beyond Hallucination** (arXiv:2509.03518v1)
- *Authors*: Haoran Huan, Mihir Prabhudesai, Mengning Wu, et al.
- *Date*: September 2025
- *Key Contribution*: Mechanistic interpretability techniques for understanding deception vs. hallucination
- *Method*: Logit lens analysis, causal interventions, contrastive activation steering to identify/control deceptive behavior
- *Applications*: Behavioral steering vectors for fine-grained manipulation, Pareto frontier analysis of dishonesty vs. performance
- *Relevance to UNMERGE*: Advanced causal intervention methods for understanding and controlling model behaviors

---

## 6. MODEL INTERPRETABILITY AND MECHANISTIC UNDERSTANDING

### 6.1 Circuit Discovery and Analysis (2024)

**Efficient Automated Circuit Discovery using Contextual Decomposition** (arXiv:2407.00886v3)
- *Authors*: Aliyah R. Hsu, Georgia Zhou, Yeshwanth Cherapanamjeri, et al.
- *Date*: July 2024
- *Key Contribution*: Contextual Decomposition for Transformers (CD-T) reduces circuit discovery runtime from hours to seconds
- *Method*: Mathematical equations to isolate contribution of model features, recursive computation followed by pruning
- *Results*: 97% ROC AUC on circuit recovery for indirect object identification, greater-than comparisons, docstring completion
- *Relevance to UNMERGE*: Provides efficient methods for identifying minimal computational subgraphs

**Circuit Compositions: Exploring Modular Structures** (arXiv:2410.01434v3)
- *Authors*: Philipp Mondorf, Sondre Wold, Barbara Plank
- *Venue*: ACL 2025
- *Key Contribution*: Studies modularity by analyzing circuits for 10 compositional string-edit operations
- *Method*: Demonstrates functionally similar circuits exhibit node overlap and cross-task faithfulness
- *Results*: Circuits can be reused and combined through set operations for complex functions
- *Relevance to UNMERGE*: Shows how sparse computational components can be modularly combined

### 6.2 Advanced Interpretability Frameworks (2025)

**Interpretability as Alignment: Making Internal Understanding a Design Principle** (arXiv:2509.08592v1)
- *Authors*: Aadit Sengupta, Pratinav Seth, Vinay Kumar Sankarapu
- *Date*: September 2025
- *Key Contribution*: Argues mechanistic interpretability should be design principle for alignment, not auxiliary diagnostic
- *Method*: Emphasizes causal insight from circuit tracing/activation patching over correlational explanations (LIME/SHAP)
- *Applications*: Detecting deceptive/misaligned reasoning that behavioral methods (RLHF, red teaming) may miss
- *Relevance to UNMERGE*: Positions mechanistic decomposition as fundamental to trustworthy AI systems

**Measuring Uncertainty in Transformer Circuits** (arXiv:2509.07149v1)
- *Authors*: Anatoly A. Krasnovsky
- *Date*: September 2025
- *Key Contribution*: Effective-Information Consistency Score (EICS) for quantifying when circuits behave coherently
- *Method*: Combines normalized sheaf inconsistency from local Jacobians with Gaussian EI proxy for causal emergence
- *Applications*: White-box, single-pass assessment of circuit trustworthiness
- *Relevance to UNMERGE*: Provides metrics for validating decomposed component reliability

### 6.3 SAE-based Interpretability (2024-2025)

**Sparse Autoencoder Neural Operators: Model Recovery in Function Spaces** (arXiv:2509.03738v1)
- *Authors*: Bahareh Tolooshams, Ailsa Shen, Anima Anandkumar
- *Date*: September 2025
- *Key Contribution*: Extends SAEs to infinite-dimensional function spaces for mechanistic interpretability of neural operators
- *Method*: Compares SAEs, lifted-SAE, and SAE neural operators across inference and training dynamics
- *Results*: Lifting and operator modules enable faster recovery, improved smooth concept recovery, resolution-robust inference
- *Relevance to UNMERGE*: Shows how sparse decomposition can scale to continuous and high-dimensional spaces

---

## 7. AI FOR SCIENCE AND AGENTS4SCIENCE APPLICATIONS

### 7.1 Scientific Discovery Frameworks (2025)

**HypoChainer: Collaborative System for Hypothesis-Driven Scientific Discovery** (arXiv:2507.17209v1)
- *Authors*: Haoran Jiang, Shaohan Shi, Yunjie Yao, et al.
- *Date*: July 2025
- *Key Contribution*: Integrates human expertise, LLM reasoning, and knowledge graphs for hypothesis generation/validation
- *Method*: Three stages - (1) exploration/contextualization with RAGs, (2) hypothesis chain formation, (3) validation prioritization
- *Applications*: Biomedicine and drug development with interpretable, scalable, knowledge-grounded discovery
- *Relevance to UNMERGE*: Shows importance of interpretable decompositions for scientific applications

**Simulation-Based Inference: A Practical Guide** (arXiv:2508.12939v1)
- *Authors*: Michael Deistler, Jan Boelts, Peter Steinbach, et al.
- *Date*: August 2025
- *Key Contribution*: Comprehensive guide for SBI methods enabling scientific discoveries in particle physics, astrophysics, neuroscience
- *Method*: Train neural networks on simulator data without likelihood evaluations, amortized inference for rapid Bayesian parameter estimation
- *Applications*: Enables parameter inference for scientific discovery where traditional methods fail
- *Relevance to UNMERGE*: Demonstrates value of interpretable neural methods for scientific applications

### 7.2 Large-Scale Scientific Applications (2025)

**CosmoBench: Multiscale Cosmology Benchmark** (arXiv:2507.03707v1)
- *Authors*: Ningyuan Huang, Richard Stiskalek, Jun-Young Lee, et al.
- *Date*: July 2025
- *Key Contribution*: Largest cosmological simulation dataset (2+ petabytes, 41M core-hours) for geometric deep learning
- *Applications*: Cosmological parameter prediction, halo velocity prediction, merger tree reconstruction across multiple scales
- *Results*: Shows least-squares with invariant features sometimes outperforms deep architectures with many more parameters
- *Relevance to UNMERGE*: Demonstrates value of interpretable, sparse representations for large-scale scientific problems

---

## SYNTHESIS AND CONNECTIONS TO UNMERGE

### Theoretical Foundations
1. **Parameter Space Decomposition**: APD (2501.14926v4) provides the most direct theoretical foundation, showing how neural network parameters can be decomposed into faithful, minimal, and simple components that solve open problems in mechanistic interpretability.

2. **Sparse Coding Theory**: The line of work from interpretable sparse coding (2011.11805v1) through modern SAE applications (2408.00657v2, 2505.08080v1) establishes that sparse representations enhance interpretability while maintaining performance.

3. **Low-Rank Adaptation**: LoRA theory and extensions (2106.09685v2, 2410.08305v1, 2410.20625v2) demonstrate that neural network adaptations can be captured in low-rank subspaces, suggesting task-specific capabilities have inherent sparsity.

### Methodological Connections
1. **Task Arithmetic Precedent**: Task vectors (2212.04089v3) show that model capabilities can be manipulated through parameter-space arithmetic, directly motivating the mathematical foundation for UNMERGE's approach to capability decomposition.

2. **Causal Verification Methods**: Activation patching best practices (2309.16042v2) and surgical editing techniques (2410.03415v2, 2308.08742v6) provide methodological frameworks for causally verifying that decomposed components truly represent intended capabilities.

3. **Dictionary Learning Applications**: SAE ensembling (2505.16077v1) and feature family identification (2408.00657v2) demonstrate how multiple sparse dictionaries can be effectively combined and organized, directly relevant to UNMERGE's micro-task vector dictionaries.

### Empirical Validation Approaches
1. **Circuit Discovery**: Efficient circuit discovery methods (2407.00886v3) and modular circuit analysis (2410.01434v3) provide frameworks for validating that decomposed components form coherent computational units.

2. **Model Merging Analysis**: Large-scale merging studies (2410.03617v1) and interference resolution techniques (TIES, DARE) establish best practices for combining model components while maintaining performance.

3. **Scientific Applications**: The AI for science applications (2507.17209v1, 2508.12939v1, 2507.03707v1) demonstrate the value of interpretable neural decompositions for high-stakes scientific discovery tasks.

### Open Challenges and Future Directions
1. **Scaling**: While APD demonstrates decomposition on toy models, scaling to large language models remains an open challenge that UNMERGE addresses.

2. **Causal Verification**: Current causal intervention methods focus on activation space; UNMERGE's parameter-space surgical subtraction represents a novel approach to causal verification.

3. **Scientific Interpretability**: The AI for science applications show growing need for interpretable neural methods that can be trusted in scientific discovery contexts.

This comprehensive survey establishes UNMERGE within a rich landscape of complementary and foundational work across neural network decomposition, sparse coding, adapter analysis, model merging, causal intervention, and mechanistic interpretability - positioning it as a novel synthesis that addresses key limitations in existing approaches while building on solid theoretical and empirical foundations.

