"""GRPO loss (docs/plans/grpo_pilot.md section 2). Backend-agnostic: every function takes the array module `xp` (numpy for the CPU
tests, mlx.core in the trainer), using only operations both share (exp, clip, minimum, maximum, sum, mean, var, sqrt).

Shapes: B questions, G sampled answers per question, L completion tokens per answer (L = 1 in the letter-only pilot).
  rewards (B, G); logp_new / logp_old / logp_ref (B, G, L): log-probabilities of the SAMPLED tokens under the current policy, the
  sampling policy (detached) and the frozen start adapter; mask (B, G, L): 1 on completion tokens, 0 on padding.

  A_bg       = (r_bg - mean_g r_bg) / (std_g r_bg + eps)                       group-normalised advantage (population std)
  ratio_bgt  = exp(logp_new - logp_old)
  pg_bgt     = -min(ratio * A_bg, clip(ratio, 1 - eps_c, 1 + eps_c) * A_bg)   clipped policy-gradient term
  kl_bgt     = exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1           k3 estimator of KL(pi || pi_ref), >= 0
  loss       = mean_bg [ sum_t mask * (pg + beta * kl) / max(sum_t mask, 1) ]  completion-only, per-answer token mean"""


def group_advantages(xp, rewards, eps: float):
    mean = xp.mean(rewards, axis=-1, keepdims=True)
    std = xp.sqrt(xp.var(rewards, axis=-1, keepdims=True))
    return (rewards - mean) / (std + eps)


def k3_kl(xp, logp_new, logp_ref):
    d = logp_ref - logp_new
    return xp.exp(d) - d - 1


def grpo_loss(xp, logp_new, logp_old, logp_ref, rewards, mask, clip_eps: float, beta: float, adv_eps: float):
    """Scalar GRPO loss (see module docstring). `logp_old` must already be detached by the caller."""
    adv = group_advantages(xp, rewards, adv_eps)[..., None]
    ratio = xp.exp(logp_new - logp_old)
    pg = -xp.minimum(ratio * adv, xp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv)
    per_tok = pg + beta * k3_kl(xp, logp_new, logp_ref)
    per_ans = xp.sum(per_tok * mask, axis=-1) / xp.maximum(xp.sum(mask, axis=-1), 1)
    return xp.mean(per_ans)


def group_stats(rewards) -> dict:
    """Diagnostics (numpy input): mean reward, share of groups with a nonzero spread (only these carry a policy gradient)."""
    import numpy as np
    r = np.asarray(rewards, dtype=float)
    return {"reward_mean": float(r.mean()), "frac_groups_informative": float((r.std(axis=-1) > 0).mean()),
            "frac_groups_all_correct": float((r.min(axis=-1) == 1).mean())}
