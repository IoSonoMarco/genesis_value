# GENESIS: Semantic-Episodic Interaction for Reward-Driven Behavior

This project explores how episodic memory can support decision-making. It extends
[GENESIS](https://arxiv.org/abs/2510.15828), a generative account of cortical-semantic and
hippocampal-episodic interaction, with mechanisms for retrieving individual experiences, reconstructing their
rewards, and using those sampled memories to choose actions.

The scientific program is motivated by two complementary accounts of memory-guided choice. Bornstein et al. (2017)
showed that reward decisions can be constructed from samples of individual past episodes and biased by incidental
reminders. Nicholas and Mattar (2026) showed how detailed episodic memories support flexible decisions when it is not
known during encoding which event features will later matter. This repository uses Shapes3D stimuli to study both
problems in a common computational framework.

The current code is a **decision-making adaptation of GENESIS**, not a complete reimplementation of every component
in the original model. The implemented bandit agent combines a frozen cortical stimulus CVAE, a learned reward
encoder/decoder, explicit temporal embeddings, and retrieval-augmented episodic sampling. It stores compressed
episodes directly and does not currently instantiate the original Hippocampal-VAE.

---

# Part 1: Technical Guide

Part 1 describes the codebase, the training sequence, and the current GENESIS implementation. Part 2 reorganizes the
same machinery around the scientific questions that it is intended to address.

## Repository Layout

The tree below follows the maintained training and experiment pipeline. Descriptions identify the contents of each
directory or file without expanding beyond one level, except for the experiment-facing configuration and script files.

```text
.
|-- configs/                                      # Training, simulation, and analysis parameters.
|   |-- analysis_simulation_genesis_igt_task_temporal_only.yaml
|   |-- analysis_simulation_genesis_nicholas_mattar_task.yaml
|   |-- analysis_simulation_genesis_reward_bandit_task.yaml
|   |-- analysis_simulation_genesis_reward_bandit_task_temporal_only.yaml
|   |-- analysis_simulation_genesis_reward_reconstruction.yaml
|   |-- cvae_shape3d_model_selection.yaml
|   |-- simulation_genesis_igt_task_models.yaml
|   |-- simulation_pvl_delta_igt_task.yaml
|   |-- simulation_genesis_igt_task_temporal_only.yaml
|   |-- simulation_genesis_nicholas_mattar_task.yaml
|   |-- simulation_genesis_novel_stimuli.yaml
|   |-- simulation_genesis_reward_bandit_task.yaml
|   |-- simulation_genesis_reward_bandit_task_temporal_only.yaml
|   |-- simulation_genesis_reward_reconstruction.yaml
|   `-- train_cvae_shape3d_genesis.yaml
|-- data/                                         # External dataset adapters and task inputs.
|   |-- datasets/                                 # Dataset loading, filtering, factor remapping, and splits.
|   `-- IGT-task-config.csv                       # Fixed Iowa Gambling Task payoff schedules.
|-- experiments/                                  # Raw model-selection and simulation records.
|   |-- cvae_shape3d_model_selection/             # CVAE parameter-sweep records.
|   |-- genesis/                                  # Genesis experiment records.
|   |-- limited_capacity/                          # Limited-capacity experiment records.
|   |-- simulation_genesis_igt_task_models/                 # Six unified IGT model conditions.
|   |   `-- pvl_delta_healthy_reference/                    # Fixed PVL-Delta reference histories.
|   |-- simulation_genesis_igt_task_temporal_only/           # Temporal-only IGT histories.
|   |-- simulation_genesis_nicholas_mattar_task/             # Feature-choice simulation records.
|   |-- simulation_genesis_novel_stimuli/                     # Novel-stimulus reconstruction records.
|   |-- simulation_genesis_reward_bandit_task/                # Reminder-biased bandit records.
|   |-- simulation_genesis_reward_bandit_task_temporal_only/ # Temporal-only bandit records.
|   `-- simulation_genesis_reward_reconstruction/            # Feature-reward reconstruction records.
|-- models/                                       # CVAE, reward, memory, and policy model components.
|   |-- agent/                                    # Decision-policy components.
|   |-- memory/                                   # Episodic storage, retrieval, and temporal-context components.
|   |-- base.py                                   # Shared model base classes.
|   |-- cvae.py                                   # FiLM conditional VAE implementation.
|   |-- decoders.py                               # Stimulus feature decoders.
|   |-- embedding_vae.py                          # Embedding-VAE implementation.
|   |-- encoders.py                               # Stimulus feature encoders.
|   |-- linear_decoder.py                         # Scalar reward encoder and decoder.
|   `-- utils.py                                  # Model utilities.
|-- output/                                       # Checkpoints, diagnostics, analysis tables, and figures.
|   |-- analysis_simulation_genesis_feature_recombination/   # Reconstruction analysis outputs.
|   |-- analysis_simulation_genesis_igt_task_models/        # Unified IGT model analyses and comparisons.
|   |   `-- pvl_delta_healthy_reference/                    # PVL behavior-only analyses.
|   |-- analysis_simulation_genesis_igt_task_temporal_only/     # Temporal-only IGT analyses.
|   |-- analysis_simulation_genesis_nicholas_mattar_task/       # Feature-choice analyses.
|   |-- analysis_simulation_genesis_novel_stimuli/               # Novel-stimulus analyses.
|   |-- analysis_simulation_genesis_reward_bandit_task/          # Reminder-biased bandit analyses.
|   |-- analysis_simulation_genesis_reward_bandit_task_temporal_only/ # Temporal-only bandit analyses.
|   |-- cvae/                                     # Trained stimulus-CVAE checkpoints and summaries.
|   |-- diagnostic_igt_retrieval_replay/          # IGT retrieval-replay diagnostics.
|   `-- reward_linear_decoder/                    # Trained reward-model checkpoints.
|-- scripts/                                      # Training, simulation, model-selection, and analysis entry points.
|   |-- analysis_simulation_genesis_feature_recombination.py
|   |-- analysis_simulation_genesis_igt_task_temporal_only.py
|   |-- analysis_simulation_genesis_igt_task_temporal_valence.py
|   |-- analysis_simulation_genesis_igt_task_models.py
|   |-- analysis_simulation_genesis_nicholas_mattar_task.py
|   |-- analysis_simulation_genesis_novel_stimuli.py
|   |-- analysis_simulation_genesis_reward_bandit_task_replacement.py
|   |-- analysis_simulation_genesis_reward_bandit_task_temporal_only_replacement.py
|   |-- cvae_shape3d_model_selection.py
|   |-- simulation_genesis_feature_recombination.py
|   |-- simulation_genesis_igt_task_temporal_only.py
|   |-- simulation_genesis_igt_task_temporal_valence.py
|   |-- simulation_genesis_igt_task_models.py
|   |-- simulation_pvl_delta_igt_task.py
|   |-- simulation_genesis_nicholas_mattar_task.py
|   |-- simulation_genesis_novel_stimuli.py
|   |-- simulation_genesis_reward_bandit_task.py
|   |-- train_cvae_shape3d_genesis.py
|   `-- train_reward_linear_decoder.py
|-- training/                                     # Shared optimization, evaluation, checkpointing, and summaries.
|   `-- trainer.py                                # CVAE training loop.
`-- utils/                                        # Paths, constants, metrics, and simulation configuration helpers.
    |-- constants.py                              # Shared constants.
    |-- filepaths.py                              # Project, dataset, experiment, output, and visualization paths.
    |-- metrics.py                                # Drifting-reward trajectory metrics.
    `-- simulation_config.py                      # Simulation configuration loading helpers.
```

## Setup and Shapes3D Preparation

Create an environment and install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The repository does not generate the upstream Shapes3D dataset. Obtain the HDF5 file separately, usually as
`3dshapes.h5`, and set `SHAPES3D_FILE_PATH` in `utils/filepaths.py`.

```python
SHAPES3D_FILE_PATH = r'C:\path\to\3dshapes.h5'
```

Project artifacts are rooted through the same module:

- model-selection and simulation data: `experiments/`;
- trained models and analysis tables: `output/`;
- reconstruction and experiment figures: `visuals/`.

### Dataset representation

`Shapes3DDataset` reads the HDF5 images and labels, discretizes the original factors, filters the selected
subpartition, and returns:

- an RGB image tensor;
- a `Shapes3DFactors` object containing `object_shape`, `object_color`, `floor_color`, and `wall_color`;
- a scalar reward.

Scale and orientation are marginalized rather than returned. Global factor identifiers are preserved in the dataset
metadata, while model-facing labels are remapped to local indices. The default split is 80% training and 20% testing,
stratified by the four returned factors with `random_state=44`.

## Training and Experiment Workflow

The scripts were designed to be used in the following order.

| Stage | Script | Purpose |
| ---: | --- | --- |
| 1 | `data/datasets/shapes3d.py` through the training scripts | Load and split the external Shapes3D data |
| 2 | `scripts/cvae_shape3d_model_selection.py` | Sweep the stimulus CVAE architecture and KL weight |
| 3 | `scripts/cvae_shape3d_model_selection.py --summarize-only` | Consolidate per-cell training summaries |
| 4 | `scripts/cvae_shape3d_model_selection.py --analyze-only` | Select and save the best model parameters |
| 5 | `scripts/train_cvae_shape3d_genesis.py` | Train the final cortical stimulus CVAE |
| 6 | `scripts/train_reward_linear_decoder.py` | Train the scalar reward latent encoder/decoder |

The final CVAE config stores its selected architecture explicitly, so retraining the current model does not
technically depend on rerunning model selection. Model selection remains part of the provenance of those choices and
is required when selecting a new architecture.

# Part 2: Experiment Implementation Guide

This section maps the manuscript's experiments to their repository implementations. The manuscript is the source for methods,
results, and interpretation; this guide identifies scripts, configurations, and artifacts. Paths are relative to the
repository root. Simulations write raw records under <code>experiments/</code>; analyses write tables and figures under
<code>output/</code>.

## Shared Prerequisites

| Script location | Purpose | Configuration | Output |
| --- | --- | --- | --- |
| <code>scripts/cvae_shape3d_model_selection.py</code> | Selects stimulus-CVAE parameters. | <code>configs/cvae_shape3d_model_selection.yaml</code> | <code>experiments/cvae_shape3d_model_selection/</code>: selection records. |
| <code>scripts/train_cvae_shape3d_genesis.py</code> | Trains the Shapes3D stimulus CVAE. | <code>configs/train_cvae_shape3d_genesis.yaml</code> | <code>output/cvae/shape3d_genesis_2/</code>: checkpoints and summaries. |
| <code>scripts/train_reward_linear_decoder.py</code> | Trains scalar reward representations; <code>--igt</code> creates the IGT model. | Script defaults and <code>--igt</code>. | <code>output/reward_linear_decoder/</code>: reward checkpoints. |

## Feature-Rich Flexible Decision-Making

This study implements the feature-rich episodic-memory simulations, from feature-reward reconstruction to feature
choice.

### Feature-Reward Reconstruction from Stored Episodes

| Script location | Purpose | Configuration | Output |
| --- | --- | --- | --- |
| <code>scripts/simulation_genesis_feature_recombination.py</code> | Creates sessions and runs the reconstruction grid. | <code>configs/simulation_genesis_reward_reconstruction.yaml</code> | <code>experiments/simulation_genesis_reward_reconstruction/sample_replacement_true/</code>: records. |
| <code>scripts/analysis_simulation_genesis_feature_recombination.py</code> | Summarizes reconstruction records. | <code>configs/analysis_simulation_genesis_reward_reconstruction.yaml</code> | <code>output/analysis_simulation_genesis_feature_recombination/sample_replacement_true/</code>: tables and figures. |

### Nicholas-Mattar Feature-Choice Simulation

This follow-up implements paired feature-reward reconstruction for feature choice.

| Script location | Purpose | Configuration | Output |
| --- | --- | --- | --- |
| <code>scripts/simulation_genesis_nicholas_mattar_task.py</code> | Creates feature-choice sessions. | <code>configs/simulation_genesis_nicholas_mattar_task.yaml</code> | <code>experiments/simulation_genesis_nicholas_mattar_task/sample_replacement_true/</code>: records. |
| <code>scripts/analysis_simulation_genesis_nicholas_mattar_task.py</code> | Summarizes feature-choice records. | <code>configs/analysis_simulation_genesis_nicholas_mattar_task.yaml</code> | <code>output/analysis_simulation_genesis_nicholas_mattar_task/sample_replacement_true/</code>: tables and figures. |

### Novel-Stimulus Value Reconstruction

This experiment constructs the cube-only 5×5×5 semantic stimulus space independently in each session and compares
value reconstruction for the same target under two episodic-memory conditions. In the `unseen` condition, the target
conjunction is excluded and retrieval operates over the remaining 124 episodes. In the `seen` condition, the exact
target image, conjunction, and reward are included once among all 125 episodes. The session seed, selected images,
target conjunction, reward assignment, query, and retrieval procedure are otherwise identical across conditions.

Both conditions use a reward-free joint image-latent and color-feature query. GENESIS samples episodes with
replacement from the query-specific retrieval distribution, averages their reward embeddings, and decodes the result;
the seen condition therefore remains probabilistic rather than performing a direct reward lookup. Use
`--condition {unseen,seen}` with either script. Omitting the flag preserves the original unseen behavior.

- Simulation: `scripts/simulation_genesis_novel_stimuli.py`, configured by
  `configs/simulation_genesis_novel_stimuli.yaml`. Unseen records retain
  `experiments/simulation_genesis_novel_stimuli/sample_replacement_true/`; seen records use the corresponding
  `seen/sample_replacement_true/` subdirectory.
- Analysis: `scripts/analysis_simulation_genesis_novel_stimuli.py`, configured by the simulation YAML's `analysis`
  section. Unseen outputs retain the original analysis path, seen outputs use its `seen/` subdirectory, and paired
  outputs use `comparison/`.

```powershell
python scripts/simulation_genesis_novel_stimuli.py --condition unseen
python scripts/simulation_genesis_novel_stimuli.py --condition seen

python scripts/analysis_simulation_genesis_novel_stimuli.py --condition unseen
python scripts/analysis_simulation_genesis_novel_stimuli.py --condition seen
```

When valid records from both conditions are available, either analysis run also exports the matched comparison under
`output/analysis_simulation_genesis_novel_stimuli/comparison/sample_replacement_true/`. Pairing is validated by session
ID, target conjunction, and target HDF5 image rather than by row order.

## Sampling-Based Decisions in a Two-Armed Bandit

This study implements non-stationary two-armed bandit simulations. The two subsections share a simulation runner and
differ through configuration and analysis.

### Temporal-Only Retrieval

This subsection implements temporal-context retrieval with replacement sampling.

| Script location | Purpose | Configuration | Output |
| --- | --- | --- | --- |
| <code>scripts/simulation_genesis_reward_bandit_task.py</code> | Generates temporal-only environments and histories. | <code>configs/simulation_genesis_reward_bandit_task_temporal_only.yaml</code> | <code>experiments/simulation_genesis_reward_bandit_task_temporal_only/sample_replacement_true/</code>: records. |
| <code>scripts/analysis_simulation_genesis_reward_bandit_task_temporal_only_replacement.py</code> | Analyzes replacement-sampled temporal-only conditions. | <code>configs/analysis_simulation_genesis_reward_bandit_task_temporal_only.yaml</code> | <code>output/analysis_simulation_genesis_reward_bandit_task_temporal_only/sample_replacement_true/</code>: tables and figures. |

### Reminder-Biased Retrieval

This subsection implements reminder-cued retrieval and its decision-effect analyses.

| Script location | Purpose | Configuration | Output |
| --- | --- | --- | --- |
| <code>scripts/simulation_genesis_reward_bandit_task.py</code> | Generates reminder-cued environments and histories. | <code>configs/simulation_genesis_reward_bandit_task.yaml</code> | <code>experiments/simulation_genesis_reward_bandit_task/sample_replacement_true/</code>: records. |
| <code>scripts/analysis_simulation_genesis_reward_bandit_task_replacement.py</code> | Produces replacement-sampled reminder summaries. | <code>configs/analysis_simulation_genesis_reward_bandit_task.yaml</code> | <code>output/analysis_simulation_genesis_reward_bandit_task/sample_replacement_true/</code>: tables and figures. |

## Iowa Gambling Task

This study implements four-deck Iowa Gambling Task simulations using the IGT reward checkpoint.

### Temporal-Only IGT Simulation

This subsection implements the temporal-context IGT baseline and its derivative analyses.

| Script location | Purpose | Configuration | Output |
| --- | --- | --- | --- |
| <code>scripts/simulation_genesis_igt_task_temporal_only.py</code> | Runs the temporal-only IGT simulation. | <code>configs/simulation_genesis_igt_task_temporal_only.yaml</code> | <code>experiments/simulation_genesis_igt_task_temporal_only/</code>: histories. |
| <code>scripts/analysis_simulation_genesis_igt_task_temporal_only.py</code> | Aggregates temporal-only histories. | <code>configs/analysis_simulation_genesis_igt_task_temporal_only.yaml</code> | <code>output/analysis_simulation_genesis_igt_task_temporal_only/</code>: tables and figures. |

### Unified Valence- and Magnitude-Selective IGT Models

The unified experiment compares loss-focused, win-focused, and balanced attention in two families. Valence-selective
models use the existing categorical positive, negative, and zero representations. Magnitude-selective models use the
learned reward embeddings and query with the scaled payoff-schedule extrema.

| Script location | Purpose | Configuration | Output |
| --- | --- | --- | --- |
| <code>scripts/simulation_genesis_igt_task_models.py</code> | Runs all six conditions sequentially, with optional parallel temporal-weight cells inside the active model. | <code>configs/simulation_genesis_igt_task_models.yaml</code> | <code>experiments/simulation_genesis_igt_task_models/&lt;model&gt;/</code>: histories. |
| <code>scripts/analysis_simulation_genesis_igt_task_models.py</code> | Exports aligned per-model analyses, retrieval composition, query diagnostics, and cross-model comparisons. | The same unified YAML. | <code>output/analysis_simulation_genesis_igt_task_models/</code>: six model subfolders and root comparison files. |

```powershell
python scripts/simulation_genesis_igt_task_models.py --parallel --n-jobs 5
python scripts/simulation_pvl_delta_igt_task.py
python scripts/analysis_simulation_genesis_igt_task_models.py
```

### PVL-Delta Reference Model

The seventh condition is a fixed-parameter Prospect Valence Learning model with Delta updating. Its active `A`,
`w_loss`, `a`, and `c` parameter vector is loaded from `configs/simulation_pvl_delta_igt_task.yaml`. The model transforms
each experienced net payoff once, updates only the chosen deck expectancy, and samples choices from
`softmax(theta_t * Q)`, where the response sensitivity is `theta = 3**c - 1`.

To match the six GENESIS conditions' first-10-trial exploration schedule, PVL uses
`theta_t = theta * (0.5 / T_t)`, with the shared temperature schedule `T_t` decreasing from 5.0 to 0.5. The configured
theta is reached on trial 10 and remains fixed thereafter. This warmup is an environment-matching adaptation; the
four parameters remain fixed within each simulated agent population and are never fitted.

PVL is simulated independently and writes one fixed-parameter cell under the common experiment root. Every invocation
overwrites that cell using the parameter vector currently stored in the PVL YAML. Rerun the analysis separately to
refresh its tables and figures. Unified analysis adds PVL behavior to the cross-model comparison while omitting all
retrieval-specific tables and figures.

---

# Appendix: Implementation Details

This appendix describes the implementation mechanics shared by the manuscript's experiments. It focuses on how GENESIS
encodes episodes, how the task-specific RAG module retrieves and recombines them, and how the experiment schedule
constructs each decision query. All documented episodic samples use replacement sampling.

## Feature-Rich Flexible Decision-Making

### GENESIS Representation

The feature-rich simulations use the frozen Shapes3D CVAE and scalar reward encoder/decoder. The CVAE represents each
image with its image-latent embedding and separate embeddings for object shape, object color, floor color, and wall
color. The reward model encodes the scalar session reward before it enters episodic memory and decodes the averaged
retrieved reward embedding into the reconstructed feature value.

### Episodic RAG and Query

<code>CompressedEpisodeFactorized</code> stores the image-latent, the four factor embeddings, the reward embedding,
and a temporal embedding for every session item. <code>EpisodicSamplingModuleFeatureRecombination</code> constructs
one episodic memory from the complete encoded session, computes factor-specific query-key similarities, converts the
configured weighted scores into replacement-sampling probabilities, and averages the sampled reward embeddings before
decoding them.

The retrieval queries activate the semantic factor named by the task cue. Temporal embeddings are stored for each
episode but have zero retrieval weight in both feature-rich configurations, so the query is semantic rather than
temporally driven. The active configuration uses z-score normalization of component similarities and replacement
sampling.

### Experiment Schedule

The feature-reward reconstruction study builds 20 independently generated, 200-item Shapes3D sessions. Session rewards
are assigned from the balanced discrete reward set, then a cube-shape query reconstructs the reward associated with
that feature. Each session and sampling budget is evaluated with 500 stochastic retrieval repetitions.

The Nicholas--Mattar feature-choice study uses the same 200-item session structure. It generates cube-dominant and
sphere-dominant sessions, queries the cube and sphere features independently, and compares the two decoded rewards to
produce a choice. The configured true feature-reward gaps span 0.1 to 1.0, and each session-budget condition uses 500
paired retrieval decisions.

## Sampling-Based Decisions in a Two-Armed Bandit

### GENESIS Representation

The bandit simulations use the same frozen Shapes3D CVAE and reward encoder/decoder. Every experienced trial is
compressed into a stimulus embedding, an encoded scalar outcome, and temporal context. The policy receives only the
two action values reconstructed by episodic memory; it then samples an arm from their difference.

### Episodic RAG and Query

<code>EpisodicSamplingModuleTwoArmedBandit</code> stores compressed episodes and partitions eligible memories by the
arm chosen in the original trial. For each decision, it samples an arm-specific set of remembered reward embeddings,
averages each set, and decodes the two averages into LEFT and RIGHT action values.

Normal trials query the RAG with temporal context only. Reminder trials add the encoded held-out Shapes3D image to the
same temporal query; the temporal-weight condition determines the balance between reminder-stimulus and temporal
similarity. The temporal-only study uses a temporal query exclusively. Both bandit configurations use replacement
sampling with a fixed retrieval budget of 20 and vary retrieval temperature.

### Experiment Schedule

Each environment contains 130 two-armed choice trials. The left-arm win probability follows a reflected random walk
between 0.3 and 0.7, and the right-arm probability is its complement. The first 20 trials are random-choice warm-up
trials; later trials reconstruct action values from the preceding episodic buffer.

The temporal-only simulation uses 100 environments with 50 agents per environment and evaluates the configured
retrieval-temperature ladder. The reminder-biased simulation uses 50 environments with 50 agents per environment.
Its schedule reserves same-class Shapes3D images as reminders, attaches them to later trials, and queries the earlier
episode they identify through the reminder-plus-temporal cue.

## Iowa Gambling Task

### GENESIS Representation

The IGT simulations use the learned IGT reward encoder/decoder rather than the Shapes3D stimulus model. Each observed
deck payoff is scaled by the task configuration, encoded into the learned reward representation, and later decoded only
after episodic reward embeddings have been recombined.

### Episodic RAG and Query

<code>EpisodicSamplingModuleIowaGamblingTask</code> stores each episode under its selected deck and reconstructs one
value per deck from its corresponding episodic candidates. The module samples and averages reward embeddings but does
not select a deck; <code>IowaGamblingTaskPolicy</code> converts the four reconstructed values into a temperature-
controlled softmax choice.

The temporal-only baseline queries every deck with the current temporal embedding. The unified models add one or two
attention components. Focused conditions assign temporal similarity weight <code>w</code> and their single attention
component weight <code>1-w</code>. Balanced conditions assign temporal similarity weight <code>w</code> and assign each
positive and negative component weight <code>(1-w)/2</code>.

Valence-selective attention compares categorical queries with the stored positive, negative, or zero key. Magnitude-
selective attention compares the learned reward embeddings of the schedule minimum and maximum with stored learned
reward keys. With the current schedule and scaling these queries are exactly -11.5 and +1.0. Each active component is
z-scored independently over all candidates from the relevant deck, after which the weighted similarities are summed.
The combined score produces one softmax distribution and one replacement sample of 20 episodes per nonempty deck.
Candidates are never filtered by reward sign.
Temporal-only IGT uses legacy query-key normalization, whereas the temporal-plus-valence studies use component-wise
z-score normalization. All IGT retrievals use replacement samples with a budget of 20.

### Experiment Schedule

The IGT schedule reads the four fixed payoff sequences from <code>data/IGT-task-config.csv</code>. Each simulated
agent makes 100 choices, and each deck maintains an independent payoff pointer that advances only when that deck is
chosen.

The temporal-only baseline simulates 500 agents at each configured retrieval temperature. Every unified model
simulates the same 100 agents, fixes retrieval temperature at 0.25, and evaluates temporal weights of 0.25, 0.75, and
1.0. The six model conditions execute consecutively; only temporal-weight cells belonging to the active model may run
in parallel. Both IGT variants anneal the policy temperature during the first 10 choices before using the condition's
minimum temperature.

---

# References

- D'Alessandro, M., D'Amato, L., Elkano, M., Uriz, M., and Pezzulo, G. (2025).
  [*GENESIS: A Generative Model of Episodic-Semantic Interaction*](https://arxiv.org/abs/2510.15828).
- Bornstein, A. M., Khaw, M. W., Shohamy, D., and Daw, N. D. (2017).
  [*Reminders of past choices bias decisions for reward in humans*](https://www.nature.com/articles/ncomms15958).
- Nicholas, J., and Mattar, M. G. (2026).
  *Episodic memory facilitates flexible decision-making via access to detailed events*.
  [Nature Human Behaviour][nicholas-mattar].

[nicholas-mattar]: https://www.nature.com/articles/s41562-025-02383-3
[choice-accuracy-and-error]:
  output/analysis_simulation_genesis_nicholas_mattar_task/visuals/choice_accuracy_and_error.pdf
[gap-reconstruction]:
  output/analysis_simulation_genesis_nicholas_mattar_task/visuals/gap_reconstruction.pdf
[margin-confidence-variability]:
  output/analysis_simulation_genesis_nicholas_mattar_task/visuals/margin_confidence_variability.pdf
[igt-temporal-choice-probabilities]:
  output/analysis_simulation_genesis_igt_task_temporal_only/visuals/figure_1_igt_deck_choice_probability_trajectories.pdf
[igt-temporal-net-score]:
  output/analysis_simulation_genesis_igt_task_temporal_only/visuals/figure_2_igt_net_score_learning.pdf
[igt-temporal-frequency]:
  output/analysis_simulation_genesis_igt_task_temporal_only/visuals/figure_3_igt_frequency_and_retrieval_mechanism.pdf
[igt-loss-choice-probabilities]:
  output/analysis_simulation_genesis_igt_task_models/valence_selective_attention_to_loss/visuals/figure_1_igt_deck_choice_probability_trajectories.pdf
[igt-loss-net-score]:
  output/analysis_simulation_genesis_igt_task_models/valence_selective_attention_to_loss/visuals/figure_2_igt_net_score_learning.pdf
[igt-loss-frequency]:
  output/analysis_simulation_genesis_igt_task_models/valence_selective_attention_to_loss/visuals/figure_3_igt_frequency_and_retrieval_mechanism.pdf
[igt-win-choice-probabilities]:
  output/analysis_simulation_genesis_igt_task_models/valence_selective_attention_to_win/visuals/figure_1_igt_deck_choice_probability_trajectories.pdf
[igt-win-net-score]:
  output/analysis_simulation_genesis_igt_task_models/valence_selective_attention_to_win/visuals/figure_2_igt_net_score_learning.pdf
[igt-win-frequency]:
  output/analysis_simulation_genesis_igt_task_models/valence_selective_attention_to_win/visuals/figure_3_igt_frequency_and_retrieval_mechanism.pdf
