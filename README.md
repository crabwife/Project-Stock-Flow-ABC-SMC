# Productivity Garden: likelihood-free calibration

This is a clean-slate Principled Data Processing project for the AARC Productivity Garden. The only observed process is annual harvest count, `H_it`. Garden membership, arrivals, abandonment, effort, soil, and capacity remain latent.

The project tests five mechanistic prior families rather than treating `H_it ~ Geometric(p)` as a primitive annual count model. A geometric waiting-time law appears as a special case of a constant project-level harvest hazard.

## Observation boundary

- The included run uses the attached AARC-derived summaries and the empirical age-coverage distribution.
- It does not reconstruct scientist-level harvest histories from spell summaries.
- To enable trajectory-level dependence targets and the ABC particle filter, place `scientist_trajectories.csv` in `a_import/input/`. Expected form: one row per scientist, optional identifier column, and annual harvest columns named `0` through `11`; use `NaN` outside observed follow-up and retain observed zero years.
- The default observable is the unweighted integer harvest count. Fractional credit requires a distinct observation model.

## Hypothesis ladder

1. `competing_risks`: Poisson arrivals, hard capacity, and constant project-level harvest and death hazards.
2. `effort_threshold`: finite annual effort is allocated compositionally; projects below the fixed maintenance unit die; surplus effort raises harvest intensity.
3. `age_hazards`: discrete Weibull harvest and death hazards; shape `1` recovers a constant geometric hazard.
4. `project_frailty`: project-level lognormal harvest/death frailty creates mixtures of geometric waiting times without scholar-specific creativity.
5. `dynamic_context`: persistent annual effort and soil states; effort changes hazards, soil changes capacity only, and their innovations may be correlated.

All models hold the seed-arrival rate population-level. None fits a scholar-specific creativity parameter.

## Pipeline

- `a_import`: validate and copy empirical inputs.
- `b_prepare`: build a shared target table, empirical follow-up distribution, and optional raw trajectory matrix.
- `c_competing_risks` through `g_dynamic_context`: prior-predictive checks for each hypothesis.
- `h_abc_smc`: separate ABC-SMC calibration for each hypothesis using the same target blocks.
- `i_particle_filter`: ABC bootstrap particle filter for raw harvest paths when supplied.
- `j_posterior_predict`: posterior-predictive checks and descriptive model screening.
- `k_sensitivity`: leave-one-block-out rankings, Monte Carlo stability, and prior-width screening.

Each task has `input/`, `output/`, and `src/`. From `b_prepare` onward, inputs are relative symlinks to upstream outputs. Every notebook is self-contained at the task level and writes tables and figures to its own `output/`.

## Run

From the project root:

```bash
make all
```

The checked-in configuration uses the `smoke` profile. Change `run_profile` to `analysis` in `config.json` for a substantive VACC run. The analysis profile is intentionally much larger; submit its tasks through the scheduler rather than a login node.

Dependencies are `numpy`, `pandas`, `scipy`, `seaborn`, `matplotlib`, `jupyter`, and `nbconvert`.

## Interpretation

ABC output is used for within-hypothesis parameter learning and posterior prediction. Cross-hypothesis tables are descriptive screening results, not posterior model probabilities or Bayes factors: the summaries are deliberately non-sufficient and strongly dependent. The project ranks hypotheses by blockwise posterior-predictive discrepancy, coverage, stability, and whether latent mechanisms remain prior-dominated.

The particle-filter score is also likelihood-free. It uses an explicit discrepancy kernel around the observed harvest count; it is not an exact marginal likelihood. It becomes a building block for later particle MCMC only after the observation model and state transition are frozen and validated.

## Literature basis

- Toni et al. (2009), *Journal of the Royal Society Interface*, doi:10.1098/rsif.2008.0172.
- Beaumont et al. (2009), *Biometrika*, doi:10.1093/biomet/asp052.
- Andrieu, Doucet, and Holenstein (2010), *JRSS B*, doi:10.1111/j.1467-9868.2009.00736.x.
- Fearnhead and Prangle (2012), *JRSS B*, doi:10.1111/j.1467-9868.2011.01010.x.
- Robert et al. (2011), *PNAS*, doi:10.1073/pnas.1102900108.
- Vaupel, Manton, and Stallard (1979), *Demography*, doi:10.2307/2061224.
- Sousa et al. (2010), *Philosophical Transactions B*, doi:10.1098/rstb.2010.0167.
