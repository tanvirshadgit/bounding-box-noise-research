## Abstract

Annotation quality is an important consideration in supervised object detection, particularly when training data contains inaccurate bounding box coordinates. This study investigates the relationship between controlled bounding box annotation noise and object detection performance using a YOLO-based model trained on a COCO128-derived dataset.

Four noise conditions (0%, 5%, 10%, and 20%) were evaluated using three matched random seeds, producing 12 experimental runs. Model performance was assessed using independently revalidated best-checkpoint mAP50 and mAP50-95.

Mean mAP50 was 0.772522 for the clean baseline, 0.773181 at 5% noise, 0.758847 at 10% noise, and 0.768870 at 20% noise. Corresponding mean mAP50-95 values were 0.617714, 0.620972, 0.618203, and 0.622071, respectively.

Paired statistical comparisons with Holm-Bonferroni correction did not establish statistically significant differences at the 0.05 threshold. Because only three seeds were available per condition, statistical power was limited.

These findings suggest that the measured performance response was non-monotonic under the tested experimental conditions. The results should be interpreted as exploratory rather than as evidence that annotation noise improves model performance or that the model is generally robust to annotation errors.

## Introduction

### Background

Object detection is a fundamental computer vision task that involves identifying and localizing objects within images. Modern object detection models rely on annotated training datasets containing object categories and bounding box coordinates.

The quality of these annotations can influence the information available during model optimization. In practical annotation workflows, bounding boxes may contain inaccuracies arising from inconsistent labeling guidelines, human judgment, object boundaries, or annotation tool limitations.

### Research Problem

Although bounding box annotations are essential for supervised object detection, the relationship between annotation inaccuracies and measured detection performance depends on multiple factors, including dataset characteristics, model architecture, training procedure, and the nature of the annotation errors.

Controlled experiments are therefore useful for investigating how specific annotation modifications relate to model performance.

### Research Objective

The objective of this study is to examine how controlled bounding box annotation noise relates to YOLO-based object detection performance on a COCO128-derived dataset.

The study evaluates four annotation noise levels using three matched random seeds and reports performance through mAP50 and mAP50-95.

### Research Questions

RQ1: How does independently revalidated best-checkpoint object detection performance vary across the tested annotation noise levels?

RQ2: Are the observed performance differences statistically distinguishable from the clean baseline under the applied paired testing procedure?

RQ3: What limitations arise from the dataset size, number of random seeds, and observed training dynamics?

### Study Scope

This study focuses on controlled bounding box modifications in training annotations. Validation annotations were maintained unchanged across the tested noise conditions.

The analysis is limited to the evaluated dataset, YOLO11n model configuration, noise procedure, three random seeds, and best-checkpoint validation metrics. The validation split contained 26 images and was also used for checkpoint selection; therefore, the results do not represent independent test-set performance.

The findings are exploratory and are not intended to establish universal robustness to annotation noise.

## Related Work

### 1. YOLO-Based Object Detection

Redmon et al. [1] introduced YOLO as a unified real-time
object detection framework. This work established an
important direction for single-stage detection systems.

The present study uses YOLO11n to investigate how
controlled changes to bounding-box training annotations
relate to detection performance.

### 2. Object Detection Datasets and Annotation Quality

Lin et al. [2] introduced Microsoft COCO, a widely used
dataset containing objects in complex everyday scenes.

Accurate bounding-box annotations are important because
object detectors learn localization from these coordinates.

He et al. [3] investigated uncertainty in bounding-box
regression, highlighting the importance of localization
uncertainty in object detection.

### 3. Detection Under Inaccurate Bounding-Box Supervision

Xu et al. [4] investigated robust object detection under
noisy category labels and imprecise bounding boxes.

Wu et al. [5] proposed spatial self-distillation for
object detection with inaccurate bounding-box annotations.

These studies demonstrate that inaccurate bounding-box
supervision is an established research problem.

### 4. Research Context and Motivation

The present study examines four controlled training
annotation conditions: 0%, 5%, 10%, and 20% noise.
Each condition was evaluated using three random seeds
with the YOLO11n architecture.

Unlike methods designed to correct inaccurate annotations,
this study focuses on measuring the relationship between
controlled annotation modifications and independently
revalidated best-checkpoint metrics.

The results are exploratory and are limited by the small
COCO128-derived dataset, three random seeds, and the
absence of an independent test-set evaluation.

The study provides a controlled preliminary comparison
rather than proposing a new noise-correction architecture.



## Methodology

### 1. Research Design

This study uses a controlled experimental design to investigate the relationship between bounding box annotation noise and YOLO-based object detection performance.

Four annotation noise conditions were evaluated: 0%, 5%, 10%, and 20%.

Three random seeds (42, 123, and 456) were used for each condition, producing 12 experimental runs.

The experiments were designed to compare model performance across annotation noise conditions while maintaining a consistent evaluation dataset.

### 2. Dataset Preparation

The experiments used a COCO128-derived dataset containing 80 object categories.

Dataset audits identified 102 unique training images and 26 unique validation images for the examined noise conditions.

Exact image-hash comparisons did not identify duplicate image content between the training and validation splits.

Image filenames and image content were consistent across the evaluated noise levels.

The original COCO128 dataset contained 128 images. The dataset was split using a seeded random shuffle (Python random seed 42), with an 80% training ratio and integer truncation, producing 102 training images and 26 validation images. Independent reconstruction of this procedure reproduced the exact saved split membership, with no train-validation image overlap. SHA-256 comparisons confirmed that all 128 saved images and all 126 available label files were identical to their original source files. Two training images (000000000250 and 000000000508) had no label files in either the original source or saved split; their absence was not caused by dataset splitting.

### 3. Annotation Noise Generation

Controlled annotation noise was introduced into the training bounding box annotations.

The evaluated noise levels were 0%, 5%, 10%, and 20%.

Annotation comparisons identified observed bounding box modification rates of approximately 5.06%, 10.00%, and 19.87% for the three nonzero noise conditions.

Validation annotations remained unchanged.

The coordinate modification procedure, boundary handling, and randomization implementation were verified against scripts/generate_coco128_noise.py and independently reproduced against the saved experimental annotations. Detailed implementation and observed modification rates are documented in the Verified Bounding-Box Noise Generation section.

### 4. Object Detection Model

The experiments used the YOLO11n object detection architecture initialized from the pretrained yolo11n.pt checkpoint.

The saved experiment configurations were audited across all 12 runs. Model training used 20 epochs, batch size 8, input resolution 640 × 640 pixels, and random seeds 42, 123, and 456.

The independently revalidated checkpoints were evaluated using Ultralytics 8.4.207 on Apple MPS.

### 5. Training Protocol

Each noise condition was evaluated using three random seeds.

The primary experimental comparisons use independently revalidated best.pt checkpoints rather than final-epoch training-log metrics.

For each condition, the selected best checkpoint was evaluated on the same unchanged validation split. Final-epoch training-log metrics were retained as a separate evaluation record and were not mixed with the primary results.

Training and validation loss histories were also examined to identify potential generalization issues.

Several examined runs showed decreasing training loss alongside increasing validation loss, suggesting possible overfitting.

The saved experiment configurations verified 20 training epochs, batch size 8, and image resolution 640 × 640 pixels across all 12 runs. Evaluation used Apple MPS. All 12 saved best-checkpoint metadata records identified AdamW as the resolved optimizer, with an initial learning rate of 0.000119 and a final learning-rate factor of 0.01. The training logs independently confirmed consistent recorded learning-rate schedules across all runs.

### 6. Evaluation Metrics

Object detection performance was evaluated using:

- mAP50: Mean average precision at an intersection-over-union threshold of 0.50.
- mAP50-95: Mean average precision averaged over IoU thresholds from 0.50 to 0.95 in increments of 0.05.

For each noise condition, independently revalidated best-checkpoint results were summarized using the mean and sample standard deviation across three random seeds.

The validation split contained 26 images and 139 ground-truth objects. Checkpoint revalidation used image size 640, batch size 8, and Apple MPS.

The primary metrics were mAP50 and mAP50-95. The validation split was also involved in checkpoint selection, so these measurements must not be described as independent test-set performance.

### 7. Statistical Analysis

Matched-seed comparisons were conducted between the clean baseline and each noisy annotation condition.

Paired t-tests were used for the comparisons.

Holm-Bonferroni correction was applied separately within each evaluation metric to address multiple comparisons.

The significance threshold was set at 0.05.

Because only three paired observations were available per comparison, statistical power was limited and normality assumptions could not be reliably evaluated.

Non-significant results were not interpreted as evidence of equivalence.

### Verified Training Configuration

The experiments used the YOLO11n object detection model with pretrained weights.

A total of 12 experimental runs were conducted across four annotation noise conditions (0%, 5%, 10%, and 20%), using three random seeds (42, 123, and 456).

The following training parameters were verified across all 12 saved experiment configurations:

- Model: YOLO11n (yolo11n.pt)
- Training epochs: 20
- Batch size: 8
- Input image size: 640 × 640 pixels
- Random seeds: 42, 123, and 456

An automated configuration audit confirmed that all 12 experiments matched these settings, with zero configuration mismatches.

### 8. Dataset Integrity Verification

Dataset integrity checks examined:

- Training and validation image overlap.
- Cross-noise image consistency.
- Annotation modification rates.
- Unchanged validation annotations.
- Experimental result availability.

These audits support the internal consistency of the examined experimental dataset.

### 9. Reproducibility and Limitations

The study includes 12 experimental runs across four noise levels and three seeds.

The available results support a preliminary comparison of annotation noise conditions.

However, the small dataset, 26-image validation split, and limited number of random seeds restrict the strength and generalizability of statistical conclusions.

The validation split was used for checkpoint selection and subsequent revalidation. Consequently, the reported best-checkpoint metrics are validation results rather than independent test-set estimates.

The dataset splitting procedure, source-to-split file integrity, noise-generation algorithm, and resolved optimizer and initial learning-rate settings were independently audited using available source code, datasets, training logs, and checkpoint metadata. Complete historical reproducibility still requires documentation of remaining software dependencies and execution-environment details. The study does not establish performance equivalence or robustness to annotation noise.

### Verified Bounding-Box Noise Generation

The experimental datasets were derived from the COCO128
training and validation split. Four annotation-noise
conditions were examined: 0%, 5%, 10%, and 20%.

The noise-generation implementation is documented in
`scripts/generate_coco128_noise.py`. A fixed random seed
of 42 was used to shuffle the training annotation
identifiers. The first rounded percentage of this
shuffled list was selected at each noise level, resulting
in nested selections across noise conditions. The random
number generator was initialized once and reused across
noise conditions, so perturbation directions were not
necessarily identical for overlapping annotations.

For each selected annotation, the normalized bounding-box
center coordinates were shifted horizontally and
vertically by 10% of the original box width and height,
respectively. The direction of each shift was selected
randomly from positive and negative directions.

The shifted center coordinates were clipped to keep the
bounding box within the normalized image boundaries.
Bounding-box width, height, and class identity were
preserved. Image content and validation annotations
remained unchanged.

A comparison against the clean training annotations
verified 790 total training annotations and the following
actual modification counts:

| Target noise | Modified annotations | Actual rate |
|---|---:|---:|
| 0% | 0 / 790 | 0.00% |
| 5% | 40 / 790 | 5.06% |
| 10% | 79 / 790 | 10.00% |
| 20% | 157 / 790 | 19.87% |

At the nominal 20% noise level, 158 annotations were
selected for perturbation, but only 157 stored annotations
differed from the clean baseline. One selected annotation
remained unchanged because positive shifts on both axes
were constrained by boundary clipping.

The observed mean absolute displacement relative to
bounding-box dimensions ranged approximately from
9.22% to 9.56% horizontally and from 9.45% to 9.60%
vertically across noisy conditions.

The boundary precision audit identified maximum
normalized boundary excess of 0.0000005, consistent
with six-decimal coordinate rounding. No annotation
exceeded image boundaries by more than 1e-6.

These findings establish the observed properties of the
saved experimental datasets. They do not independently
prove the complete historical execution provenance of
the noise-generation script.

## Results

### Experimental Setup

This study investigated the effect of controlled bounding box annotation noise on YOLO11n object detection performance using a COCO128-derived dataset.

Four training annotation noise conditions were evaluated: 0%, 5%, 10%, and 20%. Each condition was evaluated using three random seeds (42, 123, and 456), producing 12 experimental runs.

All runs used 20 training epochs, batch size 8, and image resolution 640 × 640.

The primary evaluation used the saved best.pt checkpoints, independently revalidated on the unchanged validation split using Ultralytics 8.4.207, image size 640, batch size 8, and Apple MPS.

The validation split contained 26 images and 139 ground-truth objects. Results are reported as mean ± sample standard deviation across the three seeds.

### Quantitative Results

| Noise Level | mAP50 Mean | mAP50 SD | mAP50-95 Mean | mAP50-95 SD |
|---|---:|---:|---:|---:|
| 0% | 0.772522 | 0.009473 | 0.617714 | 0.005505 |
| 5% | 0.773181 | 0.008217 | 0.620972 | 0.004160 |
| 10% | 0.758847 | 0.017542 | 0.618203 | 0.008532 |
| 20% | 0.768870 | 0.017999 | 0.622071 | 0.012405 |

These results represent best-checkpoint revalidation, not final-epoch training-log metrics.

### Effect of Annotation Noise

Mean mAP50 was 0.772522 for the clean baseline, 0.773181 at 5% noise, 0.758847 at 10% noise, and 0.768870 at 20% noise.

Mean mAP50-95 was 0.617714 for the clean baseline, 0.620972 at 5% noise, 0.618203 at 10% noise, and 0.622071 at 20% noise.

The observed changes were small and non-monotonic. The slightly higher mean mAP50-95 values under noisy conditions do not establish a beneficial effect of annotation noise.

### Statistical Interpretation

Paired t-tests compared each noisy condition with the clean baseline using matched random seeds. Holm-Bonferroni correction was applied separately to the three comparisons within each metric.

### mAP50 Comparisons

| Noise | Mean Difference | Raw p-value | Holm-adjusted p |
|---|---:|---:|---:|
| 5% | +0.000659 | 0.466127 | 0.932253 |
| 10% | -0.013675 | 0.275615 | 0.826844 |
| 20% | -0.003652 | 0.551428 | 0.932253 |

### mAP50-95 Comparisons

| Noise | Mean Difference | Raw p-value | Holm-adjusted p |
|---|---:|---:|---:|
| 5% | +0.003258 | 0.303799 | 0.911397 |
| 10% | +0.000488 | 0.909853 | 0.911397 |
| 20% | +0.004357 | 0.417929 | 0.911397 |

No comparison reached statistical significance at the 0.05 threshold.

All unadjusted paired 95% confidence intervals included zero. With only three paired observations per comparison, statistical power was limited and normality assumptions could not be reliably assessed.

Non-significant findings must not be interpreted as evidence of equivalence or demonstrated robustness.

### Paired 95% Confidence Intervals

The following unadjusted 95% confidence intervals were
calculated from seed-matched differences between each noisy
condition and the clean baseline (n = 3 paired observations
per comparison; t distribution with 2 degrees of freedom).

| Metric | Noise | Mean Difference | 95% CI Lower | 95% CI Upper |
|---|---:|---:|---:|---:|
| mAP50 | 5% | +0.000659 | -0.002516 | 0.003834 |
| mAP50 | 10% | -0.013675 | -0.053271 | 0.025921 |
| mAP50 | 20% | -0.003652 | -0.025791 | 0.018487 |
| mAP50-95 | 5% | +0.003258 | -0.006962 | 0.013477 |
| mAP50-95 | 10% | +0.000488 | -0.015930 | 0.016907 |
| mAP50-95 | 20% | +0.004357 | -0.014161 | 0.022875 |

All six intervals included zero. These intervals are
unadjusted; the separately reported Holm-adjusted p-values
account for multiple comparisons within each metric.
The wide intervals and small number of paired observations
limit precision. Failure to detect a statistically significant
difference does not establish equivalence or robustness.

### Training Dynamics and Overfitting

The examined training logs showed generally decreasing training losses alongside increasing validation losses in several runs, suggesting possible overfitting.

Final-epoch metrics differed from the independently revalidated best-checkpoint results. These evaluation protocols must not be combined into a single performance comparison.

The observed training dynamics do not establish the underlying cause of performance changes.

### Dataset Integrity

Dataset audits found no exact image overlap between the examined training and validation splits.

Validation image content and annotations remained consistent across the noise conditions.

The observed training bounding box modification rates were approximately 5.06%, 10.00%, and 19.87% for the respective noisy conditions.

These checks support controlled comparisons across the evaluated conditions.

### Limitations

The study has several limitations:

1. Only three random seeds were evaluated per condition.
2. The validation split contained only 26 images.
3. The small sample limits statistical power and generalizability.
4. The validation split was also involved in checkpoint selection, so the results do not constitute independent test-set performance.
5. The results may not generalize to larger datasets, different architectures, or other annotation error types.
6. The effects of checkpoint selection and training dynamics require further investigation.

## Discussion

The independently revalidated best-checkpoint results showed relatively small, non-monotonic differences across the four annotation noise levels.

At 20% noise, mean mAP50-95 was slightly above the clean baseline, whereas mean mAP50 was slightly below it. Neither difference was statistically significant.

The findings do not establish that bounding box annotation noise improves performance, nor do they demonstrate that the model is insensitive to annotation errors.

The limited dataset and small number of paired seeds make the results exploratory.

## Conclusion

Across 12 YOLO11n experiments, best-checkpoint revalidation showed no statistically significant differences between the clean baseline and the evaluated annotation noise conditions.

The observed performance differences were small and non-monotonic.

Future work should use larger datasets, additional random seeds, an independent test split, and broader annotation noise procedures.

This study provides a preliminary, independently audited experimental investigation of the relationship between controlled bounding box annotation noise and object detection performance. Complete historical reproducibility remains subject to verification of the full execution environment.

## Figures

**Figure 1.** Effect of bounding box annotation noise
on mAP50 across 0%, 5%, 10%, and 20% noise levels.
Points show mean performance across three seeds.
Error bars represent +/- one sample standard deviation.

![Noise vs mAP50](statistical_graphs_verified/noise_vs_mAP50.png)

**Figure 2.** Effect of bounding box annotation noise
on mAP50-95 across the same noise levels.
Points show mean performance across three seeds.
Error bars represent +/- one sample standard deviation.

![Noise vs mAP50-95](statistical_graphs_verified/noise_vs_mAP50-95.png)

## References

[1] Redmon, J., Divvala, S., Girshick, R., and Farhadi, A. (2016).
You Only Look Once: Unified, Real-Time Object Detection.
Proceedings of CVPR, 779-788.
https://doi.org/10.1109/CVPR.2016.91

[2] Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P.,
Ramanan, D., Dollár, P., and Zitnick, C. L. (2014).
Microsoft COCO: Common Objects in Context.
Proceedings of ECCV, 740-755.
https://doi.org/10.1007/978-3-319-10602-1_48

[3] He, Yihui, Zhu, Chenchen, Wang, Jianren, Savvides, Marios, and Zhang, Xiangyu. (2019).
Bounding Box Regression With Uncertainty for Accurate Object Detection.
Proceedings of CVPR, 2888-2897.
https://doi.org/10.1109/CVPR.2019.00300

[4] Xu, Y., Zhu, L., Yang, Y., and Wu, F. (2021).
Training Robust Object Detectors From Noisy Category Labels
and Imprecise Bounding Boxes.
IEEE Transactions on Image Processing, 30, 5782-5792.
https://doi.org/10.1109/TIP.2021.3085208

[5] Wu, Di, Chen, Pengfei, Yu, Xuehui, Li, Guorong, Han, Zhenjun, and Jiao, Jianbin. (2023).
Spatial Self-Distillation for Object Detection with Inaccurate Bounding Boxes.
Proceedings of ICCV, 6855-6865.
https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Spatial_Self-Distillation_for_Object_Detection_with_Inaccurate_Bounding_Boxes_ICCV_2023_paper.html
