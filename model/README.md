---
license: apache-2.0
tags:
- speech
- audio
- mos
- non-verbal-vocalization
- speech-quality-assessment
---

# NVMOS SPEAR-L9 Scorer

This repository hosts the released NVMOS downstream scorer checkpoint for non-verbal vocalization quality assessment.

Files:

- `nvmos_spear_l9.pt`: PyTorch state dict for the text-query cross-attention scorer.
- `config.json`: inference configuration, including upstream encoder model IDs and scorer dimensions.
- `training_run_config.json`: training-time configuration record.

The full inference code is available at https://github.com/yongaifadian1/NVMOS.
