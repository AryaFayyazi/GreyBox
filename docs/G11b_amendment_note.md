# G11b: note written BEFORE any Muse-Glimmer G11b outcome was seen (not a change to the locked endpoints)

Time of writing: see the commit. At this time the G11b job was generating Muse-Glimmer episodes; its output had not
been read, and the locked analysis had not run.

## Diagnosis of the Qwen3.8 miss (descriptive)
- Live, the gate's flag rate on Qwen's violating candidate calls was 0.055 (n = 146), against 0.167 offline.
- Per template the two agree: con_ci_green 0.18 offline (11/61) vs 0.12 live (8/66); tp_campaign_test 0.0 offline
  (0/5) vs 0.0 live (0/80).
- The offline sample's template mix differed from live: offline 61 con_ci_green vs 5 mailing-list violations; live
  66 vs 80. The locked prediction used the pooled offline rate, so it weighted the templates wrongly.

## Corrected predictor (to be tested prospectively; G11b's locked endpoints stay as registered)
predicted reduction = Σ_k w_k τ_k, with τ_k the offline flag rate on template k and w_k the share of the agent's
violating episodes on template k, from the agent's plain development episodes (stored rollouts; no audit involved).
- Qwen3.8: w = {con_ci_green 0.29, tp_campaign_test 0.71}; τ = {0.18, 0.0} → 0.052.
- Muse-Glimmer: w = {con_ci_green 0.194, con_cloud_cost 0.133, cred_deploy_key 0.052, tp_campaign_test 0.621};
  τ = {0.456, 1.0, 1.0, unknown: no mailing-list violation in its offline sample} → 0.273 if it never flags those
  sends (as Qwen3.8), up to 0.894 if it flags all of them.
G11b's registered predictions were 0.167 (Qwen3.8) and 0.722 (Muse-Glimmer).
