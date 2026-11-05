# Evaluation protocol

How SentinelMCP is evaluated. Every baseline and every model is reported with exactly these
definitions. Numbers that do not follow this protocol are not comparable.

*Protocol version: 2026-11-05. Update this file whenever a metric rule changes.*

## 1. Definitions

An **event** is one inspectable unit of traffic: a tool-call request, a tool output, or a tool
description. A **session** is one agent run. Every event has a ground-truth label and a gold
action (see [data/LABELING.md](../data/LABELING.md)); labels come from the generating template
and the labelling rules, never from the gateway's own decisions.

"Detected" means the predicted action is anything other than `ALLOW`.

| Metric | Definition |
|---|---|
| **Detection rate (TPR)** | malicious events not `ALLOW`ed ÷ all malicious events |
| **False positive rate (FPR)** | benign events not `ALLOW`ed ÷ all benign events (always split, see 4) |
| **Action accuracy** | predicted action == gold action |
| **Attack success rate (ASR)** | end-to-end: attacker goal achieved in the sandbox ÷ attack sessions |
| **Benign task completion** | benign agent tasks still completed with the gateway on |
| **Latency** | added p50 / p95 milliseconds per event, fast path and guard path reported separately |

ASR is measured only on attacks **verified to succeed with the gateway off**, so it measures
the gateway and not the agent's weakness. Attacks neutralised by the tool itself (for example an
HTML comment stripped by the fetch tool before the agent sees it) are labelled benign and get no
credit.

## 2. Headline: macro average by attack family

Attack-side metrics (TPR, action accuracy, ASR) are reported two ways:

- **Macro by family (the headline):** the unweighted mean of the per-family values. A large
  family such as `multi_step_exfil` cannot dominate the result.
- **Overall (micro):** pooled over all events or sessions, shown next to the headline.

Per-family values are always listed with their confidence intervals. A family with fewer than
**20 sessions** in the evaluated split is flagged **low-n**, and its interval should be read
as wide.

## 3. Confidence intervals

Every reported number carries a **95% bootstrap confidence interval**:

- percentile method, at least 1,000 resamples, fixed seed (results are deterministic);
- the resampling unit is the **session**, never the event, because events within a session are
  not independent;
- for macro averages, sessions are resampled **within each family**, then the family values are
  averaged, so every family keeps its sample size in every resample.

## 4. False positive rate is always split

Benign events are reported in two groups, plus the combined figure, each with its own interval:

- **Ordinary benign:** everyday tasks.
- **Hard negatives:** legitimate actions that look risky, for example reading a config file the
  user asked for, fetching a page that discusses prompt injection, a scoped `DELETE` on a
  temporary table, decoding a harmless base64 string, or a hiding technique used harmlessly.

A detector that only scores well on ordinary benign events is flagged by the hard-negative figure.
Benign-side metrics are additionally broken down by task category.

## 5. Breakdowns that are always reported

- **By event kind:** every metric separately for `request`, `output` and `description` events.
  The three behave very differently and must not be averaged together.
- **By hiding technique (outputs):** for outputs carrying generated content, the technique
  used to hide an instruction (HTML comment, CSS-hidden, zero-width, base64, markdown comment,
  fake system block, addressed note, attribute injection). The technique is ground truth from
  the content generator, not parsed from text.
- **By split:** `test_seen` (unseen sessions of known families) and `test_heldout`
  (entire attack families and one public source never used in training) are always reported
  separately. Generalisation is judged on `test_heldout`.

## 6. Reproducibility requirements

Every result file records:

- the **dataset version and checksum**;
- the **`policy_version`**: a sha256 over the gateway policy file *and* the policy, sanitizer
  and detector code that evaluates it (`gateway.policy.policy_version`). The policy used for
  the rule-based baseline is frozen (revision `2026-11-05.1`); changing it changes every
  baseline number, so a new revision must be stated alongside any new results;
- the **`build_id`** of the data: the git commit plus a composition version (hash of the
  templates, translations, composer, mutators, labelling rules and content generators). The
  dataset builder refuses to mix samples with different `build_id`s, and refuses ids from
  uncommitted code, so splits and metrics always come from one clean capture;
- the random seed, and for LLM baselines temperature 0.

## 7. Splits

Splits are made by **session**, never by event, and mutated versions of a seed attack stay in
the seed's split. Near-duplicates across splits are removed. Train, validation and
`test_seen` share attack families; `test_heldout` contains families and a public benchmark
source absent from training. Each split targets about 60% benign and 40% attack sessions, and
the per-split ratio and the number of distinct benign tasks are reported in the dataset
statistics.
