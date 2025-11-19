# CRP Implementation for Bayesian Decipherment

This document explains the Chinese Restaurant Process (CRP) implementation for Bayesian decipherment, following the paper "Bayesian Inference for Zodiac and Other Homophonic Ciphers" by Sravana Reddy and Kevin Knight.

## Overview

The CRP-based approach models both the source (plaintext) and channel (substitution) processes using Bayesian nonparametric methods. This allows the model to:

1. **Favor sparse distributions** - The low β prior encourages deterministic (one-to-one) substitutions
2. **Leverage context adaptively** - The cache tracks patterns in the current hypothesis
3. **Model the full generative process** - Scores both P(p) and P(c|p) jointly

## Architecture

### Core Components

```
src/lm_models/
├── crp_cache.py              # Cache tracking for CRP (C^{i-1}_1 counts)
├── crp_source_model.py       # CRP source model P(p) with α prior
├── crp_channel_model.py      # CRP channel model P(c|p) with β prior
├── crp_joint_model.py        # Combined P(p,c) = P(p) * P(c|p)
└── incremental_scorer.py     # Efficient incremental scoring (TODO: full optimization)

src/search/
└── crp_bayesian_sampler.py   # Main CRP-based sampler
```

## Mathematical Formulation

### Source Model: P(p)

The source model uses CRP to score plaintext sequences:

```
P(pi | pi-1) = (α * P0(pi|pi-1) + C^{i-1}_1(pi-1, pi)) / (α + C^{i-1}_1(pi-1))
```

Where:
- **α = 10,000** (high value favors base distribution P0)
- **P0** = Pre-trained n-gram language model
- **C^{i-1}_1(context, char)** = Count of how many times 'char' followed 'context' before position i

**Implementation**: `CRPSourceModel` in `crp_source_model.py`

### Channel Model: P(c|p)

The channel model scores substitutions:

```
P(ci | pi) = (β * P0(ci|pi) + C^{i-1}_1(pi, ci)) / (β + C^{i-1}_1(pi))
```

Where:
- **β = 0.01** (low value favors sparse/deterministic mappings)
- **P0** = Uniform distribution (1 / num_cipher_symbols)
- **C^{i-1}_1(plain, cipher)** = Count of substitutions before position i

**Implementation**: `CRPChannelModel` in `crp_channel_model.py`

### Joint Model: P(p, c)

The complete score combines both models:

```
log P(p, c) = w_source * log P(p) + w_dict * log P_dict(p) + w_channel * log P(c|p)
```

Default weights:
- w_source = 0.5 (CRP n-gram source)
- w_dict = 0.4 (Dictionary model)
- w_channel = 0.1 (CRP channel)

**Implementation**: `CRPJointModel` in `crp_joint_model.py`

## Cache Mechanism

The cache is central to the CRP formulation. It tracks two types of counts:

### 1. N-gram Cache (Source Model)

Tracks character n-grams seen in the current plaintext hypothesis:

```python
class CRPCache:
    # Maps (context_tuple) -> {char: count}
    ngram_counts: Dict[Tuple[str, ...], Dict[str, int]]
```

**Operations**:
- `add_ngram(context, char)` - Increment count when processing text
- `get_count(context, char)` - Get count for scoring
- `get_context_total(context)` - Get total count for normalization

### 2. Channel Cache (Channel Model)

Tracks substitutions in the current cipher-plaintext alignment:

```python
class ChannelCache:
    # Maps plaintext_char -> {cipher_symbol: count}
    channel_counts: Dict[str, Dict[int, int]]
```

**Operations**:
- `add_substitution(plain_char, cipher_symbol)` - Track substitution
- `get_count(plain_char, cipher_symbol)` - Get count for scoring
- `get_plaintext_total(plain_char)` - Get total for normalization

## Sampling Algorithm

### Type Sampling (Key Updates)

For each unique cipher symbol:

1. Propose a new plaintext character mapping
2. Rebuild caches with proposed hypothesis
3. Score using joint model: P(p, c) = P(p) * P(c|p)
4. Accept/reject using Metropolis-Hastings:
   - Accept if score improves
   - Accept probabilistically if score worsens (temperature-dependent)

### Space Sampling (Word Boundaries)

For multiple random positions:

1. Toggle space at position (add if absent, remove if present)
2. Rebuild caches with updated plaintext
3. Score using joint model
4. Accept/reject using Metropolis-Hastings

### Simulated Annealing

Temperature decreases linearly from 10.0 → 1.0 over 5000 iterations:

```python
temperature = 10.0 - (9.0 * iteration / num_iterations)
```

Higher temperature at start allows exploration; lower temperature at end focuses on exploitation.

## Usage

### Basic Usage

```python
from search.crp_bayesian_sampler import CRPBayesianSampler
from lm_models.n_gram_model import load_ngram_model
from lm_models.dictionary_model import DictionaryLanguageModel

# Load models
ngram_model = load_ngram_model("models/ngram_model_n3_chunks101956_20251118_115213.pkl")
dict_model = DictionaryLanguageModel("data/wordlists/en_10k.txt")

# Load cipher
cipher, ground_truth_key = load_cipher(Path("c_400_30.json"))

# Create sampler
sampler = CRPBayesianSampler(
    ciphertext=cipher,
    ngram_model=ngram_model,
    dict_model=dict_model,
    ground_truth_key=ground_truth_key,
    seed=42,
    use_crp=True  # Set False to disable CRP (use standard LM)
)

# Run decipherment
best_key, best_plaintext = sampler.run(num_iterations=5000)
```

### Comparing CRP vs Standard

The `CRPBayesianSampler` has a `use_crp` flag to enable/disable CRP:

```python
# With CRP (paper's approach)
sampler_crp = CRPBayesianSampler(..., use_crp=True)

# Without CRP (standard LM approach)
sampler_standard = CRPBayesianSampler(..., use_crp=False)
```

This allows direct comparison between the two approaches.

## Hyperparameters

From `utils/constants.py`:

```python
ALPHA = 10000.0  # Source model Dirichlet prior
BETA = 0.01      # Channel model Dirichlet prior
INITIAL_TEMPERATURE = 10.0
TOTAL_ITERATIONS = 5000
```

These values are from the paper and should generally not be changed unless experimenting.

## Performance Considerations

### Current Implementation

The current implementation rebuilds caches on every proposal for correctness:

```python
# For each proposal:
self.model.clear_caches()
self.model.initialize_caches(ciphertext, proposed_plaintext)
proposed_score = self.model.log_score(ciphertext, proposed_plaintext, proposed_key)
```

**Time Complexity**: O(n) per proposal, where n = plaintext length

### Future Optimization: Incremental Scoring

The paper describes using "exchangeability property" to only rescore affected regions:

1. **Identify affected positions** - Where the changed symbol appears
2. **Find context window** - Extend by n-1 chars in each direction
3. **Rescore only window** - Use shared cache for unchanged regions

This would reduce complexity to O(k) per proposal, where k = affected window size (typically << n).

**Status**: Skeleton implementation in `incremental_scorer.py` - not yet integrated.

## Differences from Original Implementation

### Original (Standard LM Approach)

- Uses pre-trained interpolated language model
- Only scores P(p), not P(c|p)
- No cache tracking
- No Dirichlet priors
- Rescores entire plaintext every proposal

### CRP Implementation (Paper's Approach)

- Uses CRP with cache tracking
- Scores both P(p) and P(c|p)
- Explicit Dirichlet priors (α, β)
- Cache adapts to current hypothesis
- Still rescores fully (optimization pending)

## Testing

The CRP models can be tested individually:

```python
# Test source model
from lm_models.crp_source_model import CRPSourceModel

crp_source = CRPSourceModel(ngram_model, alpha=10000.0)
score = crp_source.log_score_text("the quick brown fox")

# Test channel model
from lm_models.crp_channel_model import CRPChannelModel

crp_channel = CRPChannelModel(num_cipher_symbols=30, beta=0.01)
score = crp_channel.log_score_key(ciphertext, plaintext, key)
```

## Debugging

The sampler logs detailed information during execution:

```
2025-11-19 10:00:00 - INFO - Starting CRP Bayesian sampling for 5000 iterations...
2025-11-19 10:00:00 - INFO - Using CRP: True
2025-11-19 10:00:00 - INFO - Initial score: -8.45
2025-11-19 10:00:00 - INFO - Initial SER: 0.8667 (86.67% errors)
2025-11-19 10:00:00 - INFO - Initial CRP source score: -4.23
2025-11-19 10:00:00 - INFO - Initial dict score: -3.11
2025-11-19 10:00:00 - INFO - Initial CRP channel score: -1.11
```

Component scores help diagnose issues:
- **High source score** = Plaintext looks like English
- **High dict score** = Words are valid
- **High channel score** = Substitutions are consistent

## Known Limitations

1. **Performance**: Current implementation rebuilds caches fully on each proposal (O(n) per proposal). Incremental scoring would reduce this to O(k) for affected windows.

2. **Space sampling**: When spaces change, entire cache must rebuild even though only word boundaries changed. Could optimize by caching per-word.

3. **Memory**: Caches can grow large for long texts. Consider implementing cache size limits or pruning.

## Future Improvements

1. **Implement full incremental scoring** using exchangeability property
2. **Optimize space sampling** to avoid full cache rebuilds
3. **Add cache statistics** to monitor performance
4. **Implement parallel tempering** for better exploration
5. **Add visualizations** of cache evolution during sampling

## References

Reddy, S., & Knight, K. (2011). "What We Know About The Zodiac Killer." Proceedings of the 5th ACL-HLT Workshop on Language Technology for Cultural Heritage, Social Sciences, and Humanities.

Paper describes CRP formulation in Section 3.1: "Bayesian Decipherment"
