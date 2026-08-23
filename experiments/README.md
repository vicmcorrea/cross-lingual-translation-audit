# Experiments

This directory contains the Hydra pipeline, analysis package, frozen configurations, and tests. Each executing invocation creates a unique append-only run with checkpoints, manifests, checksums, and a terminal seal. Study data are not distributed.

The sensitivity stage verifies five named sealed dependencies and emits only text-free aggregate artifacts. Target-group retrieval scores a normalized group by the maximum similarity among its member rows. Filtered bootstraps retain the complete survey-specific respondent-record sampling universe and the fixed retrieval candidate corpus.

The historical internal fields `participant_id` and `n_participants` refer to survey-specific respondent records. They do not identify unique people across survey cohorts.

The immutable configuration retains pre-analysis provenance labels containing `preservation`, `participant`, and `leave_one_cohort_out`. These labels reproduce the sealed runs. In the paper, they correspond to reference-free audit evidence, exploratory affective profiles, survey-specific respondent records, and an exploratory ridge artifact that is omitted from the central manuscript.
