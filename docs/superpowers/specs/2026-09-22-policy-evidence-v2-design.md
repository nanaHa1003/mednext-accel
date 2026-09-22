# Optimization Policy and Evidence v2 Design

## Goal

Make `mednext-accel` useful without per-batch lookup tables. Runtime dispatch
will use compact, hypothesis-driven policies with explicit confidence and
bounded exceptions. Profiling will produce measurement evidence separately.
Missing measurements will no longer mean automatic reference fallback.

This is a direct schema replacement. There are no external users that require
runtime compatibility with schema-v1 profile documents.

## Why v1 is replaced

The v1 documents combine observations, dispatch policy, reference defaults,
launch-parameter formulas, and provenance:

| Bundled file | Lines | Runtime entries |
|---|---:|---:|
| `generic-nvidia.json` | 1,496 | 50 |
| `sm120.json` | 1,502 | 50 |
| `sm89.json` | 2,537 | 78 |

All generic and SM120 entries choose the same implementation and parameters.
SM89 contains only 19 distinct operator/shape/phase groups. Its 59 dW entries
expand two formulas already present in Python. Reference defaults terminate
resolution before a shared layer can contribute a useful inferred policy.

The new SM86 campaign demonstrates the semantic failure: 46 kernels pass
numerical validation and 18 pass the balanced operator objective, yet one
aggregate model-policy rejection turns every measurement invalid and generates
zero rules.

## Three separate concepts

### Correctness and feasibility

Implementation metadata and runtime guards decide whether an implementation
can legally run for a descriptor and context. Guards include device type,
backend availability, dtype, kernel geometry, export safety, approximation
permission, and parameter validity. They do not depend on benchmark coverage.

### Evidence

Evidence is immutable machine-generated JSON. It describes exactly what ran:
environment, software, campaign, batch-search probes, kernel measurements,
component numerical errors, OOM/error stage, model comparison metrics, and
acceptance reasons. Evidence never participates directly in runtime dispatch.

### Policy

Policy is compact YAML reviewed as source code. It states where an
implementation should be selected, including inferred regions. Rules may cite
evidence and confidence, but confidence does not alter dispatch. The authors
decide which inferred rules are strong enough to ship in `auto`.

## Policy v2 document

Bundled and external runtime policies use YAML:

```yaml
version: 2
kind: mednext-accel-policy
name: sm89
target:
  vendor: nvidia
  sm: [8, 9]
scope:
  dtype: bfloat16
evidence:
  - id: l40s-base-128-b10
    sha256: fffa74b383d74e2bcc01e1edc189c5d01249aea7d10c6dd7cd595b836b5d538e

rules:
  - id: shared-stem-gemm
    when:
      family: pointwise_conv3d
      direction: regular
      spatial_shape: [128, 128, 128]
      channels: [1, 32]
    use:
      training:
        implementation: pointwise_gemm_per_sample
    confidence: validated-cross-sm
```

### Document fields

- `version` is exactly `2`.
- `kind` is exactly `mednext-accel-policy`.
- `name` is a nonempty stable identifier.
- `target.vendor` is required; `target.sm` is optional. Omitted SM means a
  vendor-shared layer.
- `scope` supplies match conditions shared by every rule. Rule-level `when`
  may narrow but never contradict scope.
- `evidence` is descriptive provenance and does not affect matching.
- `rules` are ordered. First matching rule with the requested phase wins.

### Supported match conditions

Current v2 needs only typed equality and numeric ranges:

- `family`, `direction`, `role`;
- `kernel_size`, `stride`, exact `spatial_shape`;
- `channels: [in, out]`;
- `batch`, `spatial_volume`, `work`, and `reduction_work` ranges;
- `total_vram_gib` range;
- `dtype`, `model_family`, `variant`, `checkpointing`.

`work = batch × spatial_volume × in_channels` and
`reduction_work = batch × spatial_volume`. Numeric ranges use `{min, max}`;
an integer means exact equality.

### Selection and tombstones

Each phase selection has an implementation and optional parameters.
`parameters: auto` evaluates the implementation's recipe. An explicit mapping
starts from recipe defaults when a recipe exists, then replaces the named
values; otherwise it is used as-is. Omitted parameters produce `{}`. Policy
loading rejects `auto` for an implementation without a recipe.

`implementation: reference` is an explicit tombstone: it stops lower-priority
inferred rules for a known losing or numerically invalid region.

Omission is different from a tombstone. If no rule in a layer matches, the
resolver continues to the next layer.

### Confidence

Allowed values are:

- `measured-exact-context`
- `interpolated-bounded`
- `inferred-same-sm`
- `validated-cross-sm`
- `extrapolated`
- `guard`

Confidence is reported to users but is not a hidden mode switch. Bundled policy
authors decide which rules are suitable for default `auto` behavior.

## Resolver precedence

The resolver evaluates:

1. hard implementation guards;
2. explicit user policy;
3. exact-SM bundled overlay;
4. shared NVIDIA policy;
5. built-in reference fallback.

An external policy targeting another SM still applies after a once-only warning,
subject to hard guards. Exact-SM and shared bundled layers apply only to their
targets. Policies contain no blanket defaults. All small bundled policies are
loaded once; the resolver chooses the exact-SM layer from each execution
context, so moving a model after construction cannot freeze the wrong policy.

The supported vendor context is CUDA with a known NVIDIA SM. CPU, absent Triton,
unsupported dtype/geometry, inference/export, approximation, and parameter
guards are visible to both resolution reports and execution. Tensor properties
that do not exist until forward remain execution-time guards and are reported as
such rather than as a guaranteed static decision.

The resolver must expose the selected policy, rule, confidence, parameters, and
warning through `explain_optimization()`.

## Launch parameters

Arithmetic is ordinary tested Python, not a data-language expression evaluator.
Profiling and runtime share small recipe functions keyed by implementation and,
where needed, SM/descriptor region. Recipes receive the implementation,
descriptor, and execution context. Spatial volume is the product of all three
dimensions; Python's ties-to-even `round` is used before integer clamping.

Initial recipes:

```python
# regular depthwise dW
splits = clamp(round(batch * spatial_volume / 32768), 1, 512)

# transpose depthwise dW
splits = clamp(round(batch * spatial_volume / 4096), 1, 512)
```

These are the SM86/SM89 recipes and use `dw_block=512`. SM120 retains one compact
shape-aware recipe matching its existing base launches, including its 1024/128
blocks and split scaling. Its two real per-batch exceptions remain explicit.
This is a deliberate preservation requirement, not an assumption that one
formula is optimal across every architecture.

## Independent regular dX and dW execution

Regular depthwise dispatch currently enters the custom autograd path only when
both dX and dW are custom, and that path always computes both custom gradients.
Policy v2 requires phase independence. The backward implementation combines
native `convolution_backward` masks with custom phases for all four cases:

- custom dX + custom dW;
- custom dX + native dW;
- native dX + custom dW;
- native dX + native dW.

Bias-gradient behavior remains native/equivalent. `explain_optimization()` and
actual execution must agree. BF16 autocast, FP32 parameters, compilation, and
uncompiled training receive regression coverage.

## Initial bundled policy hypotheses

### Shared NVIDIA

The shared layer is deliberately small. The initial matrix is:

| Phase | Region | Evidence and inference |
|---|---|---|
| Pointwise training | stem `1→32 @ 128³`, BF16, batch ≥1 | winner on SM86 B12, SM89 B1/B10 and SM120 B2/B4/B6 |
| Downsample dX | supported k3 stride-2, work ≥1,000,000 | about 2.1–2.2× across four scales on SM86/SM89; threshold is the smallest legacy observed work |
| Regular dX | supported k3, work ≥1,500,000 | covers SM89 observed onset at 16³/B2 and 8³/B6 plus all SM86/SM89 selected-batch winners |
| Transpose dW | supported k3, reduction work ≥4,096 | inferred boundary from legacy SM89 onset; new SM86/SM89 points at 5,120/6,144 win |
| Regular dW | exact 64³/C64 and 32³/C128 for batch ≥1; 8³/C512 for reduction work ≥1,024 | stable shared winner regions; 128³ and 16³ are excluded due observed device/batch counterexamples |

Thresholds are policy hypotheses supported by boundary evidence, not claims that
every point was measured. They intentionally cover batch 9 and batches above
10/12 when the work condition remains satisfied.

Unknown NVIDIA GPUs therefore receive a useful, conservative kernel set rather
than the entire SM120 pointwise table or all-reference execution.

### SM86

The SM86 overlay is based on RTX A6000 batch-12 evidence. It keeps shared dX,
transpose dW and stem behavior. It may add the observed 16³ regular-dW winner
only from batch 12 upward; 128³ remains reference because the measured candidate
lost. It does not convert the aggregate policy rejection into numerical kernel
rejection.

### SM89

The SM89 overlay combines legacy dense L40S evidence with the new batch-10 run.
Batch rows using the same parameter recipe collapse into one bounded rule.
Conflicting regular-dW regions use explicit bounded rules or reference
tombstones. New batch-10 pointwise winners are retained as
`measured-exact-context` rules. They are not called exact-device behavior because
the matcher has no GPU product identity. The two validator OOM cases are
evidence gaps, not negative performance rules. The old 432-case raw file was
overwritten locally; the bundled v1 policy and benchmark document are retained
summaries, so a compact launch fixture is captured before those JSON files are
removed.

### SM120

The SM120 overlay preserves current RTX 5090 behavior where supported but
removes confidence-only overrides. Only the two overrides that change launch
parameters remain. Same-SM policies with structural scaling apply to larger
Blackwell devices such as a 96-GB RTX PRO 6000. VRAM establishes feasibility,
not whether GEMM beats native convolution.

Pointwise rules remain bounded to robust shapes and measured or defensibly
inferred batch regions. Non-stem pointwise rules do not expand beyond their
supported batch range merely because a device has more VRAM. Structural SM120
depthwise recipes remain active above the RTX 5090 batch range, so a 96-GB
same-SM device receives useful defaults. The full SM120 pointwise table is not
copied into the shared NVIDIA layer.

## Evidence v1 document

Profiler output uses JSON:

```json
{
  "version": 1,
  "kind": "mednext-accel-evidence",
  "environment": {},
  "campaign": {},
  "batch_searches": [],
  "kernel_measurements": [],
  "model_comparisons": []
}
```

### Batch-search evidence

Every workload records the actual ordered attempts plus a batch-indexed summary,
with batch, status, step time, peak allocated memory, budget outcome, and
selected maximum. Search probes remain
diagnostic and never become kernel measurements automatically.

### Kernel evidence

Kernel records preserve probe status, failure stage/message, timing, peak
memory, component validation metrics, kernel numerical validity and objective
winner status. An aggregate model result never mutates these facts.

Pointwise validation compares forward, dX, dW and dB one component at a time so
native/candidate pairs and FP32 error temporaries are released before the next
component. OOM records identify the stage.

### Model-comparison evidence

Reference and candidate comparisons use the same documented campaign seed and
the same initialization/input protocol. Each side records status, seed, step
time and peak memory. Whole-model numerical equivalence is explicitly
`not-measured`; component kernel validation remains the numerical correctness
evidence. Scalar diagnostics may be recorded, but they are not used as a
correctness gate.

The model comparison records separate performance and memory acceptance plus
final policy acceptance and a deterministic reason. Failed, missing, zero, or
nonfinite timing/memory metrics cannot pass. Model gates are:

- balanced: candidate time `<` reference and peak `≤115%` of reference;
- throughput: candidate time `<` reference and peak `≤125%` of reference;
- memory: candidate peak `<` reference and time `≤110%` of reference.

Kernel gates remain separate:

- balanced: candidate time `≤97%` and peak `≤115%` of reference;
- throughput: candidate time `<` reference;
- memory: candidate peak `<` reference and time `≤110%` of reference.

The name `whole_model_valid` is removed. A policy rejection never rewrites
kernel numerical evidence.

### Campaign phases

The current profiler measures training only. The `phases` workload field is
removed until an inference campaign exists. Generated provenance says
`phase: training` exactly.

## Profiler outputs

`mednext-accel profile campaign.yaml` writes two adjacent files:

- `smXX-local.<timestamp>-<hash>.evidence.json`: complete immutable evidence;
- `smXX-local.policy.yaml`: a compact external overlay.

The provisional overlay contains exact winners plus two kinds of exact reference
tombstone: numerical failures (correctness evidence) and numerically valid
objective losers (performance evidence). Infrastructure errors and validator
OOM remain gaps. The provisional overlay is layered over the same bundled
policies that runtime will use, and the complete effective-policy identity is
stored in the model comparison.

If the provisional effective policy is accepted, its positive rules are
published. If it is rejected, the stable local overlay publishes only negative
tombstones. The evidence records `policy_accepted: false` and explicitly marks
the negative-only overlay's remaining bundled fallthrough as unvalidated by that
comparison. Runtime YAML contains only runtime policy fields. Removing positives
must not be described as a passing model policy. The overlay has no defaults, so
unmatched contexts continue through bundled SM and shared layers.

Evidence is serialized once to deterministic UTF-8 JSON bytes. SHA-256 covers
those exact bytes. It is published first under an immutable unique filename
containing timestamp/hash; the stable policy YAML is then atomically replaced
and references that filename/hash. A crash may leave an orphan evidence file,
but can never publish a policy associated with overwritten evidence. Policy
loading does not require the evidence file to be present.

The CLI reports both paths and the model-comparison acceptance reason. A policy
YAML path remains a direct factory `optimization=` input. Evidence JSON is not
a runtime policy and produces a clear error if supplied there.

The public result becomes:

```python
@dataclass(frozen=True)
class ProfilingArtifacts:
    policy_path: Path
    evidence_path: Path

@dataclass(frozen=True)
class ProfilingResult:
    policy: OptimizationPolicy
    evidence: ProfilingEvidence
    artifacts: ProfilingArtifacts
    environment: Mapping[str, object]

    def save(
        self,
        *,
        policy_path: Path | None = None,
        evidence_directory: Path | None = None,
    ) -> ProfilingArtifacts: ...
```

`save()` follows the immutable-evidence/public-policy order above and returns
both actual paths. Existing `profile`, `measurements`, `output_path`, and
single-`Path` save semantics are removed with schema v1.

## Migration

- Remove bundled schema-v1 JSON profiles.
- Add `shared-nvidia.yaml`, `sm86.yaml`, `sm89.yaml`, and `sm120.yaml`.
- Replace `OptimizationProfile`/`ProfileRegistry` internals with policy-v2
  records and loading. Factory behavior remains `auto`, `reference`, or a path.
- Schema-v1 runtime documents fail with a concise migration error. No permanent
  dual-schema resolver is retained.
- Old raw measurement files remain local evidence. Their source hashes and
  conclusions are recorded in policy provenance and benchmark documentation.
- Obsolete campaign `phases` keys and evidence-as-policy inputs fail before CUDA
  side effects with explicit migration messages.

## Verification

- Policy parser rejects unknown fields, contradictions, invalid ranges,
  unknown implementations and unavailable `auto` recipes.
- Layer tests distinguish omission from a reference tombstone.
- Unknown SMs receive shared rules; SM86/89/120 receive their overlay plus
  shared fallthrough.
- RTX PRO 6000-like SM120 contexts above RTX 5090 VRAM/batch ranges receive
  inferred depthwise decisions rather than blanket reference.
- Current SM120 decisions are preserved where intentionally supported, with
  removed generic overreach tested on SM86 pointwise counterexamples.
- Parameter recipes reproduce every retained dW launch value from old profiles;
  only real exceptions remain explicit.
- Regular dX and dW decisions execute independently in all four combinations,
  and reports match the path actually taken.
- Evidence round-trips and preserves unsuccessful probes without turning them
  into kernel invalidity.
- Streaming pointwise validation passes RTX 5090 batch 1/2/3 and avoids the old
  simultaneous-component retention by construction.
- Profiler output policy falls through to bundled layers outside exact local
  evidence.
- Failure injection between evidence and policy writes cannot expose a policy
  pointing at mismatched evidence bytes.
- Wheel/sdist contain compact YAML policies and no raw local evidence files.
