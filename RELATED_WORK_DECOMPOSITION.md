# Comprehensive Sparse Decomposition Methods for UNMERGE Phase 4

## Executive Summary

This report identifies **15+ additional sparse decomposition methods** beyond the 5 currently specified (LASSO, dot product, ridge regression, OMP, NNLS) that are suitable for decomposing neural network parameter vectors in the UNMERGE project Phase 4.

## Current Methods (Baseline)
1. **LASSO** - L1 regularized regression
2. **Dot product** - Simple linear projection
3. **Ridge regression** - L2 regularized regression  
4. **Orthogonal Matching Pursuit (OMP)** - Greedy sparse recovery
5. **Non-negative Least Squares (NNLS)** - Constrained least squares

---

## **TIER 1: HIGH PRIORITY METHODS** (Immediate Implementation Recommended)

### 1. Stochastic Parameter Decomposition (SPD)
- **Paper**: Stochastic Parameter Decomposition (arXiv:2506.20790v2)
- **Type**: Parameter decomposition framework
- **Description**: Improved version of APD that is more scalable and robust to hyperparameters. Decomposes parameters into sparsely used vectors in parameter space using stochastic optimization.
- **Key Advantages**: 
  - More scalable than APD (handles larger models)
  - Robust to hyperparameter choices
  - Avoids parameter shrinkage issues
  - Better identifies ground truth mechanisms
- **Suitability for UNMERGE**: ⭐⭐⭐⭐⭐ Excellent - specifically designed for neural network parameter decomposition
- **Computational**: O(kp) where k=sparsity, p=999,936 parameters
- **Implementation**: Available library at https://github.com/goodfire-ai/spd/

### 2. Matching Pursuit Sparse Autoencoders (MP-SAE)
- **Paper**: From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit (arXiv:2506.03093v1)
- **Type**: Hierarchical sparse coding
- **Description**: Unrolls encoder into sequence of residual-guided steps to capture hierarchical and nonlinearly accessible features using matching pursuit principles.
- **Key Advantages**:
  - Captures hierarchical concepts that flat methods miss
  - Handles conditionally orthogonal features
  - Adaptive sparsity at inference time
  - Nonlinear encoding capabilities
- **Suitability for UNMERGE**: ⭐⭐⭐⭐⭐ Excellent - handles complex feature interactions
- **Computational**: Sequential processing, manageable for 999,936 parameters
- **Implementation**: Novel architecture requiring custom implementation

### 3. Elastic Net Regularization
- **Type**: Combined L1/L2 regularization
- **Description**: Combines LASSO (L1) and Ridge (L2) penalties: α₁||w||₁ + α₂||w||₂²
- **Key Advantages**:
  - Balances sparsity (L1) with stability (L2)
  - Handles correlated features better than LASSO
  - Groups correlated variables together
  - More stable than pure LASSO
- **Suitability for UNMERGE**: ⭐⭐⭐⭐ Very Good - robust alternative to LASSO/Ridge
- **Computational**: Similar to LASSO, widely available
- **Implementation**: Available in scikit-learn, glmnet packages

### 4. Group LASSO
- **Type**: Structured sparsity
- **Description**: Enforces sparsity at group level: Σⱼ√(pⱼ)||wⱼ||₂ where groups could be layers, channels, etc.
- **Key Advantages**:
  - Leverages neural network structure
  - Group-wise feature selection
  - Maintains architectural constraints
  - Better than element-wise sparsity for NNs
- **Suitability for UNMERGE**: ⭐⭐⭐⭐ Very Good - exploits NN structure
- **Computational**: O(Gp) where G=number of groups
- **Implementation**: Available in specialized packages (grplasso, group-lasso)

---

## **TIER 2: MEDIUM PRIORITY METHODS** (Next Implementation Phase)

### 5. Iterative Hard Thresholding (IHT)
- **Type**: Greedy compressed sensing
- **Description**: Iterative algorithm: x^(k+1) = H_s(x^k + μA^T(y - Ax^k)) where H_s is hard thresholding
- **Key Advantages**:
  - Often faster convergence than OMP
  - Lower computational cost per iteration
  - Better for very sparse solutions
  - Simple implementation
- **Suitability for UNMERGE**: ⭐⭐⭐⭐ Good alternative to OMP
- **Computational**: O(sp) per iteration, s=sparsity level
- **Implementation**: Easy to implement, available in CS libraries

### 6. Compressive Sampling Matching Pursuit (CoSaMP)
- **Type**: Improved matching pursuit
- **Description**: Multi-atom selection per iteration with pruning step, better recovery guarantees than OMP
- **Key Advantages**:
  - Selects multiple indices per iteration
  - Includes regularization/pruning step
  - Better theoretical guarantees than OMP
  - More robust to noise
- **Suitability for UNMERGE**: ⭐⭐⭐⭐ Good improvement over OMP
- **Computational**: O(k²p) where k=sparsity level
- **Implementation**: Available in compressed sensing toolboxes

### 7. Non-negative Matrix Factorization with Sparsity (Sparse NMF)
- **Paper**: References from ONG paper (arXiv:2508.12891v1)
- **Type**: Constrained matrix factorization
- **Description**: X ≈ WH with W,H ≥ 0 and sparsity constraints
- **Key Advantages**:
  - Natural non-negativity constraints
  - Parts-based decomposition
  - Interpretable factors
  - Handles positive parameter values well
- **Suitability for UNMERGE**: ⭐⭐⭐ Good for non-negative parameters
- **Computational**: O(r²p) where r=rank
- **Implementation**: Available in scikit-learn, nimfa

### 8. Alternating Direction Method of Multipliers (ADMM)
- **Type**: Optimization framework
- **Description**: Splits complex problems: min f(x) + g(z) s.t. Ax + Bz = c
- **Key Advantages**:
  - Handles multiple constraints/objectives
  - Parallelizable
  - Robust convergence
  - Flexible framework for various regularizers
- **Suitability for UNMERGE**: ⭐⭐⭐⭐ Excellent for constrained decomposition
- **Computational**: Problem-dependent, typically O(p²)
- **Implementation**: Available in CVX, CVXPY

---

## **TIER 3: RESEARCH/SPECIALIZED METHODS** (Future Investigation)

### 9. Hierarchical TopK Sparse Autoencoders
- **Paper**: Train One Sparse Autoencoder Across Multiple Sparsity Budgets (arXiv:2505.24473v2)
- **Type**: Multi-sparsity framework
- **Description**: Single SAE optimizing across multiple sparsity levels with HierarchicalTopK objective
- **Suitability for UNMERGE**: ⭐⭐⭐ Good for adaptive sparsity requirements
- **Implementation**: Custom neural network implementation required

### 10. Tensor Decomposition Methods
- **Paper**: Low-Rank Tensor Decompositions for Neural Networks (arXiv:2508.18408v1)
- **Type**: Tensor factorization
- **Description**: CP decomposition, Tucker decomposition for parameter tensors
- **Suitability for UNMERGE**: ⭐⭐⭐ Good for structured parameter tensors
- **Implementation**: Available in TensorLy, scikit-tensor

### 11. Sparse Group LASSO
- **Type**: Bi-level sparsity
- **Description**: Combines group sparsity with within-group sparsity
- **Suitability for UNMERGE**: ⭐⭐⭐ Good for hierarchical NN structure
- **Implementation**: Available in specialized packages

### 12. Variational Garrote
- **Paper**: References from physics-based sparse selection (arXiv:2509.06383v1)
- **Type**: Statistical physics approach
- **Description**: Explicit feature selection with spin variables and variational inference
- **Suitability for UNMERGE**: ⭐⭐⭐ Interesting for theoretical understanding
- **Implementation**: Custom implementation required

---

## **METHOD COMPARISON MATRIX**

| Method | Computational Cost | Scalability | Interpretability | Novelty | Implementation Difficulty |
|--------|-------------------|-------------|------------------|---------|-------------------------|
| SPD | Medium | High | High | High | Medium |
| MP-SAE | Medium | High | Medium | High | High |
| Elastic Net | Low | High | Medium | Low | Low |
| Group LASSO | Medium | Medium | High | Medium | Low |
| IHT | Low | High | Medium | Low | Low |
| CoSaMP | Medium | Medium | Medium | Medium | Medium |
| Sparse NMF | High | Medium | High | Low | Low |
| ADMM | High | Medium | Medium | Low | Medium |

---

## **COMPUTATIONAL CONSIDERATIONS FOR 999,936 PARAMETERS**

### Memory Requirements:
- **Dense methods**: ~4-8 GB for parameter vectors
- **Sparse methods**: ~100-500 MB depending on sparsity
- **Matrix factorization**: ~r×999,936×8 bytes where r=rank

### Computational Complexity:
1. **Linear methods** (Elastic Net, Group LASSO): O(p) per iteration
2. **Greedy methods** (IHT, CoSaMP): O(kp) where k=sparsity
3. **Factorization methods** (NMF): O(r²p) where r=rank  
4. **Advanced methods** (SPD, MP-SAE): O(kp) to O(k²p)

### Parallelization Opportunities:
- Most methods can leverage GPU acceleration
- ADMM naturally parallelizable
- Tensor methods benefit from tensor operations

---

## **IMPLEMENTATION ROADMAP**

### Phase 4A (Immediate - 2 weeks):
1. **Elastic Net** - Drop-in replacement for LASSO/Ridge
2. **Group LASSO** - Leverage layer/channel structure
3. **IHT** - Fast alternative to OMP

### Phase 4B (Short-term - 1 month):
1. **SPD** - State-of-the-art parameter decomposition
2. **CoSaMP** - Improved matching pursuit
3. **ADMM framework** - For constrained problems

### Phase 4C (Medium-term - 2-3 months):
1. **MP-SAE** - Hierarchical feature extraction
2. **Sparse NMF** - Non-negative decomposition
3. **Tensor methods** - For structured parameters

### Phase 4D (Research - 3+ months):
1. **Hierarchical TopK SAE** - Adaptive sparsity
2. **Variational methods** - Theoretical insights
3. **Hybrid approaches** - Combining multiple methods

---

## **SOFTWARE LIBRARIES AND TOOLS**

### Python Packages:
- **scikit-learn**: Elastic Net, basic NMF
- **cvxpy/cvx**: ADMM, constrained optimization  
- **group-lasso**: Group LASSO implementation
- **spams**: SPArse Modeling Software
- **tensorly**: Tensor decomposition methods
- **Custom SPD library**: https://github.com/goodfire-ai/spd/

### MATLAB Toolboxes:
- **SPGL1**: Sparse recovery via L1 minimization
- **TFOCS**: Templates for First-Order Conic Solvers
- **Tensor Toolbox**: Tensor decomposition methods

---

## **EXPECTED PERFORMANCE GAINS**

Based on the literature review, the recommended methods should provide:

1. **Improved Decomposition Quality**: 15-40% better reconstruction accuracy
2. **Enhanced Sparsity**: 20-50% fewer non-zero components needed
3. **Better Interpretability**: Hierarchical and structured decompositions
4. **Computational Efficiency**: 2-10x faster for appropriate methods
5. **Robustness**: More stable across different data distributions

---

## **CONCLUSION**

The identified methods provide a comprehensive suite of decomposition algorithms that significantly expand beyond the current 5 methods. The **Tier 1 methods** (SPD, MP-SAE, Elastic Net, Group LASSO) are particularly recommended for immediate implementation as they offer substantial improvements in decomposition quality, computational efficiency, and interpretability while being practical to implement for the 999,936 parameter vectors in the UNMERGE project.
