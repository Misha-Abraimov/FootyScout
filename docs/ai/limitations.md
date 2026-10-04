# AI Scout governed limitations

Status: Phase 1 deterministic foundation

These limitations apply to the deterministic tools now and to any future AI
interpretation built on them.

## Dataset and cohort

- FootyScout is based on available StatsBomb Open Data, not an exhaustive global
  scouting database.
- The V1 product cohort contains 369 player profiles from the available
  Bundesliga sample. Coverage, match counts, minutes, and event samples differ
  by player.
- Broader modeling corpora used to train xG and possession-value models do not
  imply that every player/team from those competitions is available in the
  product cohort.
- Missing or ineligible metrics are not zero. They remain unavailable when
  sample or same-position peer requirements are not met.

## Reliability and sample support

- Passing, shooting, attacking-impact, percentile, archetype, similarity, and
  Role Fit outputs have different support requirements.
- Same-position percentiles are calculated per metric. Peer counts and
  eligibility can therefore differ between metrics for the same player.
- Similarity and Role Fit expose match support. “Limited” support must remain
  visible and must not be converted into a confidence probability.
- Goalkeepers and small-sample players have intentionally limited intelligence
  coverage.

## Similarity and archetypes

- Similarity means same-position playing-style similarity in the frozen
  production feature space.
- Similarity is not player quality, potential, future performance, transfer
  success, or proof that players are interchangeable.
- A similarity score is not a probability.
- Archetypes are playing-style groups. They are not rankings or quality ratings.
- Archetype separation measurements describe geometry in the clustering space,
  not predictive confidence.

## Team intelligence and Role Fit

- Team intelligence describes observed event data. It does not reveal coaching
  intent, tactical instructions, causal effects, or future plans.
- Current production team intelligence is limited to the qualified Bayer
  Leverkusen profile and its DEF, MID, and FWD role profiles.
- Role Fit is outfield-only. No goalkeeper Role Fit or goalkeeper recommendation
  is available.
- Role Fit measures resemblance to an observed positional-role playing style.
  Lower raw role distance means closer resemblance.
- Role Fit does not predict transfer success, future performance, adaptation,
  lineup selection, or tactical success.
- Current-team players use a leave-self-out target role where the persisted
  calculation indicates that scope. External recommendation rankings exclude
  target-team players.

## Models and experiments

- XGBoost is the current production xPass model.
- XGBoost is the current production xG model.
- XGBoost is the current production possession-value model.
- The GRU and PyTorch causal Transformer are offline experiments. Neither is the
  production possession-value model.
- Attacking impact is derived from persisted possession-value differences. It is
  not causal attribution and is not recalculated by AI Scout.
- Provider StatsBomb xG may be used as an external diagnostic benchmark but is
  not a FootyScout xG feature or label.

## Data FootyScout does not contain

AI Scout has no authoritative data for:

- transfer fees, market values, wages, budgets, or agent fees;
- contracts, availability, negotiations, or willingness to transfer;
- injuries, medical status, suspensions, or fitness forecasts;
- future performance, transfer success, adaptation, or lineup guarantees;
- private coaching intent or tactical instructions.

The system must state that evidence is unavailable rather than infer or invent
these facts.

## Evidence and future generated language

FootyScout analytics, PostgreSQL records, and packaged runtime metadata remain
the source of truth. Deterministic tools retrieve those values. Future generated
interpretation is secondary to the structured evidence and must keep numeric
claims traceable to tool results and methodology claims traceable to versioned
documentation.

No future LLM may calculate replacement xPass, xG, possession value, attacking
impact, player metrics, percentiles, archetypes, similarity, team intelligence,
Role Fit, or recommendations.
