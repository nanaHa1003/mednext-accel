# Legacy dW launch snapshot

`legacy_dw_launches.csv` is an independent fixture captured from bundled schema-v1
profiles before their removal. It records expected launch values, not runtime
policy rules or measured performance. Tests must not regenerate expected values
with the new recipes.

Source bytes:

| Source | SHA-256 |
|---|---|
| `src/mednext_accel/profiles/sm89.json` | `a84d04d432e396cbc779b0a364387aa7ff964f8b0ab9d3dd61c8fa1f4799d668` |
| `src/mednext_accel/profiles/sm120.json` | `d33cf9e7ea50575de8e48d3ed8a1793054ad2459e0f5d61314660ac258746c1b` |

The snapshot has 209 rows:

- SM89: all 59 dW rules, expanded across every integer in their finite batch
  ranges, producing 72 launches. The source covers batches 1–8 and 10.
- SM120: all six base dW rules evaluated at batches 1–16, 64, 257, 512, 513,
  and 1024, producing 126 launches. Larger probes exercise split clamps and
  preservation of the fixed transpose launch; they are not benchmark evidence.
- SM120: all 11 measured dW overrides at their exact batches, producing 11
  launches. Nine match the base recipe. Only `dw-split-b4-128-32` (four splits)
  and `dw-split-b6-128-32` (eight splits) differ and need explicit overrides.

Values were obtained by evaluating the legacy profile literals and its
`scale_work` evaluator, before implementing the new recipes. The regular SM120
base rates are integer splits per batch, so its old rounding-to-multiple step
is redundant within those exact shape/channel regions. The new recipe preserves
those rates and the 512-split clamp without carrying a formula language into
policy documents. Generic and SM120 legacy profiles had identical launches; the
shared v2 policy is deliberately not a copy of that generic legacy policy.
