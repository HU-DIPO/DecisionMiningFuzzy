# Implementation Documentation

## 1. Purpose
This documents the repository implementation that accompanies the 2024 publication on decision discovery using clinical decision support system log data. The dissertation revisits the implementation in the current environment and provides a more explicit methodological description, including the prototype.

The purpose of this annex is not to replace the published paper. Instead, it clarifies the implementation. The 2024 paper presents the scientific baseline: the problem framing, the method, the fuzzy decision-mining approach, and the catheterization synthetic case study. The repository linked from that paper is the implementation artifact that accompanies the publication.

For the dissertation, this repository was revisited and revalidated in the current environment. The additional work in the dissertation is focused on:
1. A working Docker and Python environment,
2. A clearer explanation of the implementation workflow,
3. Documentation that make the method easier to follow for academic review.

This should be understood as an extension of the implementation documentation, not as a reinterpretation of the paper’s scientific contribution.

## 3. Model inference and reporting
A key methodological distinction in the implementation is the separation between the raw rules and the reportin layer.

### 3.1 raw rules
The DMFuzzy model is responsible for extracting the raw rules. These raw rules represent the direct output of the learning process and are preserved as the model-level result.

### 3.2 Reporting layer
For readability, a post-processing step is applied after rule extraction. This reporting step was intended for the medical audience of the journal, not for data scientists.

## 4. Dataset used for the demonstration
The dissertation uses a synthetic catheterization dataset to demonstrate the pipeline in a controlled way. The dataset contains numeric risk values so that the fuzzy classifier can derive linguistic risk regions automatically. The dataset is synthetic and should not be interpreted as clinical evidence.

## 5. Parameterization
The linguistic encoder maps fuzzy positions to the labels:
1. `Low`
2. `Medium`
3. `High`

These labels are used for interpretability and should be read as a presentation layer over the fuzzified numeric input.

## 6. Revalidation in the current environment
When the repository was revisited for the dissertation, the code was re-executed in the current Docker and Python environment. This revealed a dependency mismatch in the backend stack that prevented the application from starting as originally configured.

This issue is best understood as an environment-related reproducibility problem rather than a change in the scientific content of the paper. To restore reproducibility, the backend dependencies were aligned to compatible versions and the prototype was tested and debugged again.

## 7. Validity and limitations
1. The synthetic dataset supports controlled demonstration but does not prove clinical generalizability.
2. The paper-style transformation is a reporting convention and should not be interpreted as additional model learning.
3. The results depend on the selected columns, the label distribution, and the DMFuzzy hyperparameters.
4. The dissertation version adds documentation and reproducibility detail; it does not claim a new scientific algorithmic contribution beyond the original paper.