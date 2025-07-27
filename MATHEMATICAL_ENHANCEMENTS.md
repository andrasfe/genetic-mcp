# Mathematical Enhancements to Genetic Algorithm

## Overview

This document describes the mathematical enhancements implemented in the genetic algorithm to improve convergence, diversity preservation, and solution quality.

## 1. Advanced Selection Strategies

### 1.1 Boltzmann Selection
**Mathematical Basis**: 
- Probability distribution: P(i) = exp(f_i/T) / Σ(exp(f_j/T))
- Temperature annealing: T(t) = T_0 * α^t, where α = 0.95
- Prevents numerical overflow by scaling fitnesses

**Benefits**:
- Temperature-controlled selection pressure
- Smooth transition from exploration to exploitation
- Handles negative fitness values correctly

### 1.2 Stochastic Universal Sampling (SUS)
**Mathematical Basis**:
- Places N equally spaced pointers on the fitness wheel
- Pointer spacing: 1/N
- Reduces selection bias compared to roulette wheel

**Benefits**:
- Guarantees selection spread
- Lower variance than roulette wheel selection
- O(N) time complexity

### 1.3 Rank-Based Selection
**Mathematical Basis**:
- Linear ranking: P(i) = (2-SP)/N + 2*rank(i)*(SP-1)/(N*(N-1))
- Selection pressure SP ∈ [1, 2]
- Rank from worst (1) to best (N)

**Benefits**:
- Independent of fitness scale
- Prevents premature convergence
- Controllable selection pressure

## 2. Adaptive Parameter Control

### 2.1 Mutation Rate Adaptation
**Mathematical Basis**:
- If diversity < 0.3: μ = min(0.5, μ * 1.2)
- If diversity > 0.7: μ = max(0.05, μ * 0.9)
- Self-adaptive mutation: μ(t+1) = μ(t) * exp(τ * N(0,1))

### 2.2 Crossover Rate Adaptation
**Mathematical Basis**:
- Based on fitness improvement rate: Δf/f
- If stagnating (Δf < 0.01): increase crossover
- If improving (Δf > 0.1): maintain current rate

### 2.3 Elitism Adaptation
**Mathematical Basis**:
- If improving rapidly: elite_count = min(elite + 1, pop_size/4)
- If stagnating: elite_count = max(1, elite - 1)

## 3. Fitness Sharing

**Mathematical Basis**:
- Shared fitness: f'(i) = f(i) / Σ(sh(d_ij))
- Sharing function: sh(d) = 1 - (d/σ)^α if d < σ, else 0
- Niche radius σ determines sharing threshold

**Benefits**:
- Maintains population diversity
- Prevents crowding in fitness peaks
- Enables multi-modal optimization

## 4. Crowding Distance (NSGA-II)

**Mathematical Basis**:
- Measures perimeter of cuboid formed by nearest neighbors
- For each objective m: d_i^m = (f_{i+1}^m - f_{i-1}^m) / (f_max^m - f_min^m)
- Total distance: d_i = Σ(d_i^m)

**Benefits**:
- Preserves diversity in objective space
- No parameters to tune
- Computationally efficient

## 5. Convergence Detection

**Multi-Criteria Approach**:
1. **Fitness Plateau**: |f_best(t) - f_best(t-w)| < ε * f_best(t)
2. **Low Diversity**: diversity < 0.2
3. **Low Variance**: Var(f) / Mean(f) < 0.01

**Convergence if**: At least 2 criteria satisfied for w generations

## 6. Diversity Metrics

### 6.1 Simpson's Diversity Index
**Formula**: D = 1 - Σ(n_i/N)²
- n_i = count of type i
- N = total population
- D ∈ [0, 1], higher is more diverse

### 6.2 Shannon Diversity
**Formula**: H = -Σ(p_i * log(p_i))
- p_i = proportion of type i
- Measures both richness and evenness

### 6.3 Average Pairwise Distance
**Formula**: APD = (2/N(N-1)) * Σ(d_ij)
- Measures spread in solution space
- Uses cosine distance for semantic similarity

## 7. Mathematical Correctness Improvements

### 7.1 Numerical Stability
- Fitness normalization to prevent overflow in exponential functions
- Proper handling of zero and negative fitness values
- Epsilon values to prevent division by zero

### 7.2 Probability Normalization
- Ensures all selection probabilities sum to 1.0
- Handles edge cases (empty population, all zero fitness)
- Validates probability distributions

### 7.3 Distance Metrics
- Consistent use of cosine similarity for semantic comparison
- Jaccard distance for content-based diversity
- Proper normalization of all distance measures

## Implementation Notes

1. **Selection Method Scheduling**:
   - Early generations (< 30%): SUS or rank-based (exploration)
   - Middle generations (30-70%): Tournament (balanced)
   - Late generations (> 70%): Boltzmann (exploitation)
   - On stagnation: Rotate methods

2. **Parameter Ranges**:
   - Mutation rate: [0.05, 0.5]
   - Crossover rate: [0.5, 0.95]
   - Temperature: [0.1, 1.0]
   - Niche radius: [0.1, 0.5]

3. **Computational Complexity**:
   - Selection methods: O(N) to O(N log N)
   - Fitness sharing: O(N²)
   - Crowding distance: O(MN log N), M = objectives
   - Diversity metrics: O(N²)

## Validation

The mathematical correctness of these implementations ensures:
- Proper probability distributions (sum to 1)
- Numerical stability (no overflow/underflow)
- Convergence guarantees (elitism preserves best)
- Diversity preservation (measurable metrics)
- Parameter bounds (within valid ranges)