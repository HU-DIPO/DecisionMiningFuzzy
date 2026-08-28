# ANNEX

## 1. Purpose of this note
The published paper on this topic in 2024. This annex documents the repository version that was revisited for the dissertation, where the implementation had to be revalidated in the current environment and explained in more detail. We documented the method more explicitly.

## 2. Research question addressed in this implementation
Given a structured decision log (CSV), can the system Discover decision rules using the DMFuzzy classifier.

## 3. Separation of model inference and reporting
A strict separation is implemented between:
1. Inference layer (raw mined rules), and
2. Reporting layer (paper-style unique rules).

### 3.1 Inference layer
The DMFuzzy model performs training and rule extraction from the parsed data. Raw rules are preserved and serialized into `xml_raw`.

### 3.2 Reporting layer
A deterministic transformation (`paper_unique_rules`) is applied to make the output readable for the intended audience, which is medidal staff rather than data scientists. The 2024 paper already used a readable decision-table presentation; this dissertation-oriented version keeps that idea and documents it more explicitly. The aim was to keep the representation technically correct while remaining accessible enough for a non-technical journal context and public.

## 4. API behavior and backward compatibility
Endpoint: `POST /rules`

The response contains:
1. `xml_raw`: XML generated from raw mined rules.
3. `xml`: backward-compatible field, selected by user preference.
4. `rules_view`: `raw` or `paper`.
5. `accuracy`: model scoring output.

Selection policy:
1. If `normalize_bool = false`, `xml` is `xml_raw`.
2. If `normalize_bool = true`, `xml` is `xml_paper`.

This design preserves prior frontend behavior while making both views explicit and auditable.

## 5. Revalidation in the current environment
When the repository was revisited for the dissertation, the code was re-executed in the current Docker and Python environment. This revealed dependency mismatches that prevented the backend from starting and working correctly. We tried to update the dependencies, but sticked with forcing specific versions of dependencies in the requirements.txt.

## 6. Parameterization choices
In `DMFuzzy`, the default `minimal_gain_ratio` is set to `0.2`.

Rationale:
1. This setting improves stability of multi-region fuzzy splitting for the dataset.

The linguistic encoder maps fuzzy positional terms to interpretable labels:
1. `Low`
2. `Medium`
3. `High`

## 7. External validity and limitations
1. Synthetic data supports controlled demonstration but does not prove clinical generalizability, as also mentioned in the paper.
2. The paper-style transformation is a reporting convention and should not be interpreted as additional model learning.
3. Results depend on selected columns, label distributions, and DMFuzzy hyperparameters.
