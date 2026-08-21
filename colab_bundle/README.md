# Colab Pro Preliminary Bundle

## Overview

This bundle contains everything needed to run the preliminary three-seed experiment on Google Colab Pro.

## Files

- `random_event_gppo_preliminary.ipynb` - Main training notebook
- `requirements.txt` - Python dependencies

## Usage

1. Open `random_event_gppo_preliminary.ipynb` in Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Run all cells sequentially
4. Results will be saved to Google Drive

## Configuration

The notebook trains the following variants:
- GPPO-NoGate
- GPPO-Adaptive  
- Fair PPO-MLP

With training seeds: 1101, 2202, 3303

## Important Notes

- This is a **preliminary** run with only 3 training seeds
- Results should NOT be used as final conclusions
- A formal run with 5 seeds is required for publication
- The label "preliminary" must appear in all output files

## Output

The notebook generates:
- Training checkpoints
- Validation evaluation results
- Test evaluation results
- Summary statistics

All results are saved to Google Drive at:
`/content/drive/MyDrive/random_event_gppo_preliminary/`

## P0 Gate

The notebook verifies the P0 gate before training. If the gate is not passed, training will not start.

## Checkpoint Selection

Checkpoints are selected based on validation results using the following criteria:
1. Lowest final infeasible rate
2. Lowest cumulative weighted vacancy
3. Lowest recovery delay
4. Lowest fixed J
5. If tied, earliest checkpoint

## Test Evaluation

The final evaluation uses the frozen test bank with the following sets:
- Test-Single (40 tapes)
- Test-Sequential (40 tapes)
- Test-Overlap (40 tapes)
- Test-Burst (40 tapes)
- Test-Unseen (40 tapes)

## Statistics

The notebook reports:
- Raw seed values
- Mean and standard deviation
- 95% confidence intervals
- Paired differences
- Effect sizes

## Contact

For questions or issues, please refer to the project documentation.
