# Bounding Box Annotation Noise Research

## Overview

An independent computer vision research project investigating
the effect of controlled bounding-box annotation noise on
YOLO11n object detection performance.

## Experimental Setup

- Dataset: COCO2017 subset (1,500 images)
- Training: 1,000 images
- Validation: 200 images
- Independent test: 300 images
- Model: YOLO11n
- Noise levels: 0%, 5%, 10%, 20%
- Training seeds: 42, 123, 456, 789, 2026
- Training experiments: 20
- Epochs: 20
- Image size: 640
- Batch size: 8

## Main Findings

Independent test mean mAP50-95:

| Noise | mAP50-95 |
|---|---|
| 0% | 0.395969 |
| 5% | 0.396229 |
| 10% | 0.395525 |
| 20% | 0.394158 |

No paired comparison reached conventional statistical
significance (p < 0.05).

These exploratory results do not establish universal
robustness to annotation noise.

## Repository Contents

- `scripts/` - Research scripts
- `results/` - Validation and independent test results
- `reports/` - Research draft, figures and statistical tables

## Limitations

- Small COCO2017 subset
- Five training seeds
- One model architecture
- Training-time and standalone validation discrepancies
- MPS nondeterminism may affect reproducibility
- Multiple comparisons: Holm correction applied to the three primary independent-test mAP50-95 comparisons; other reported comparisons remain unadjusted

## Research Status

Independent research manuscript.
Not peer-reviewed or formally published.

## Interactive Research Demo

A public Streamlit application demonstrates predictions from four YOLO11n models trained with 0%, 5%, 10%, and 20% bounding-box annotation noise.

**Live Demo:** https://yolo11n-public-demo-12.streamlit.app/

**Demo Source Code:** https://github.com/tanvirshadgit/yolo11n-public-demo

The demo provides qualitative comparisons on user-provided images. Quantitative research conclusions should be based on the independent test-set evaluations rather than individual demo predictions.
