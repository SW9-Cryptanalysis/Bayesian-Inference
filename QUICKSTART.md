# Quick Start: CRP Bayesian Decipherment

## Running the CRP Sampler

### 1. Basic Usage (from main.py)

```bash
cd c:\Github\SW9\Bayesian-Inference\src
python main.py
```

The `main.py` file is already configured to use the CRP sampler with `use_crp=True`.

### 2. Toggle Between CRP and Standard

Edit `main.py` line 22:

```python
use_crp = True   # Use CRP approach (paper's method)
use_crp = False  # Use standard LM approach (original method)
```

### 3. Expected Output

```
2025-11-19 12:00:00 - INFO - ============================================================
2025-11-19 12:00:00 - INFO - Using CRP-based Bayesian Sampler (Paper's Approach)
2025-11-19 12:00:00 - INFO - ============================================================
2025-11-19 12:00:00 - INFO - Starting CRP Bayesian sampling for 5000 iterations...
2025-11-19 12:00:00 - INFO - Using CRP: True
2025-11-19 12:00:00 - INFO - Initial score: -8.45
2025-11-19 12:00:00 - INFO - Initial SER: 0.8667 (86.67% errors)
2025-11-19 12:00:00 - INFO - Initial CRP source score: -4.23
2025-11-19 12:00:00 - INFO - Initial dict score: -3.11
2025-11-19 12:00:00 - INFO - Initial CRP channel score: -1.11
2025-11-19 12:00:00 - INFO - Initial plaintext: thr quack briwn fiw jumps ivrt thr laey diw...

Iteration 0:
  Temperature: 10.00
  Current score: -8.45
  Best score: -8.23
  Best SER: 0.8333 (83.33% errors)
  Best CRP source: -4.15, dict: -3.05, channel: -1.03
  Current plaintext: thr quack briwn fiw jumps ivrt thr laey diw...
  Best plaintext: the quick brown fox jumps over the lazy dog...

...

Sampling complete!
Final best score: -2.15
Final SER: 0.0333 (3.33% symbol errors)
Final best plaintext: the quick brown fox jumps over the lazy dog
```

## Understanding the Output

### Score Components

**CRP N-gram Score** (-4.15)
- Character n-gram score from CRP source model
- Higher (less negative) = more English-like character patterns
- Uses cache + base n-gram model with α prior

**Word Dict Score** (-3.05)
- How many valid dictionary words
- Higher = more real words
- Based on 10k word frequency list

**Interpolated P(p)** (-2.15)
- Combined source model: 0.1 * n-gram + 0.9 * word (from paper)
- This represents how likely the plaintext is

**Channel P(c|p)** (-1.03)
- How consistent the substitutions are
- Higher = more deterministic (1-to-1) mappings
- Uses cache + uniform base with β prior

**Final Score** 
- log P(p, c) = log P(p) + log P(c|p)
- This is what the sampler optimizes

### Symbol Error Rate (SER)

- Percentage of cipher symbols mapped incorrectly
- 0.0 = perfect (all symbols correct)
- 1.0 = worst (all symbols wrong)
- Should decrease during sampling

## Testing Different Ciphers

Edit `main.py` line 21 to change cipher:

```python
cipher, ground_truth_key = load_cipher(pathlib.Path("mono-cipher-5.json"))  # Simple
cipher, ground_truth_key = load_cipher(pathlib.Path("c_400_30.json"))      # Medium
cipher, ground_truth_key = load_cipher(pathlib.Path("c_800_30.json"))      # Hard
cipher, ground_truth_key = load_cipher(pathlib.Path("z408.json"))          # Zodiac-408
```

## Hyperparameter Tuning

Edit `src/utils/constants.py`:

```python
# Dirichlet priors
ALPHA = 10000.0  # Higher = favor base n-gram model more
BETA = 0.01      # Lower = favor deterministic substitutions more

# Sampling
TOTAL_ITERATIONS = 5000       # More = better but slower
INITIAL_TEMPERATURE = 10.0    # Higher = more exploration early
```

## Model Weights

Edit `src/lm_models/crp_joint_model.py` or when creating `CRPJointModel`:

```python
# Paper uses these interpolation weights for P(p):
self.model = CRPJointModel(
    self.crp_source,
    self.crp_channel,
    dict_model,
    ngram_weight=0.1,   # N-gram model weight (paper: 0.1)
    word_weight=0.9     # Word dictionary weight (paper: 0.9)
)
```

## Comparing Approaches

### Run CRP Approach

```python
# main.py
use_crp = True
searcher = CRPBayesianSampler(..., use_crp=True)
searcher.run()
```

### Run Standard Approach

```python
# main.py
use_crp = False
searcher = CRPBayesianSampler(..., use_crp=False)
# OR
searcher = BayesianSampler(...)  # Original implementation
searcher.run()
```

### Compare Results

Look for:
1. **Final SER** - Which gets lower error rate?
2. **Convergence speed** - Which improves faster?
3. **Score components** - How do CRP models help?

## Troubleshooting

### "Module not found" errors

```bash
# Make sure you're in the src directory
cd c:\Github\SW9\Bayesian-Inference\src

# Or set PYTHONPATH
set PYTHONPATH=c:\Github\SW9\Bayesian-Inference\src
```

### Slow performance

- Reduce `TOTAL_ITERATIONS` to 1000 for testing
- Use smaller ciphers (mono-cipher-5.json)
- Full incremental scoring optimization not yet integrated (future work)

### Poor decipherment

- Increase `TOTAL_ITERATIONS` to 10000
- Try different random seeds: `CRPBayesianSampler(..., seed=123)`
- Adjust model weights (increase dict_weight for more words)
- Check initial SER - if > 0.9, initialization may be poor

## Performance Notes

### Current Implementation

- **Time per iteration**: ~0.5-2 seconds for 400-character cipher
- **Total time**: 5000 iterations ≈ 40-160 minutes
- **Bottleneck**: Full cache rebuild on each proposal

### Future Optimization

With incremental scoring:
- **Time per iteration**: ~0.05-0.2 seconds (10x faster)
- **Total time**: 5000 iterations ≈ 4-16 minutes

See `CRP_IMPLEMENTATION.md` section "Performance Considerations" for details.

## Next Steps

1. **Test on simple cipher** (mono-cipher-5.json)
2. **Compare CRP vs standard** (toggle `use_crp`)
3. **Analyze component scores** (which model helps most?)
4. **Try different hyperparameters** (α, β, weights)
5. **Scale to harder ciphers** (c_800_30.json, z408.json)

## Files to Read

1. `IMPLEMENTATION_SUMMARY.md` - High-level overview
2. `CRP_IMPLEMENTATION.md` - Detailed technical documentation
3. `src/search/crp_bayesian_sampler.py` - Main sampler code
4. `src/lm_models/crp_*.py` - CRP model implementations

## Support

For implementation details, see:
- Mathematical formulation: `CRP_IMPLEMENTATION.md` → "Mathematical Formulation"
- Architecture: `CRP_IMPLEMENTATION.md` → "Architecture"
- Usage examples: `CRP_IMPLEMENTATION.md` → "Usage"
