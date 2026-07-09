# Repeated-control output agreement

`output_agreement_summary.json` is a compact, machine-readable extraction of
the 1,000-page `full_exact_a` accuracy envelope. The original full envelope is
3.28 MB and approximately 68,000 lines; the four prediction captures total
about 30 MB, so this publication bundle records their SHA-256 identities rather
than duplicating them.

The comparison used two control captures and two optimized full-exact-fusion
captures over the same 1,000 JPEG-byte pages. It evaluates page/region counts,
aligned OCR text, coordinates, and confidence drift. The optimized text
divergence was within or better than ordinary control-to-control divergence,
numeric drift on exact-text pairs stayed within `0.001` coordinate and `0.002`
confidence tolerances, candidate repeatability was within 10% of the control
repeat rate, and the repeat-calibrated verdict was
`no_output_regression_detected`.

The strict pairwise comparator is nevertheless false: control A versus
candidate A contained one segmentation-count difference and 18 sequence edits.
The changed segmentation preserved the page's space-joined text, while the
control A/B repeat had 19 sequence edits. Candidate A/B had 10 edits. Reporting
both facts prevents the repeat-calibrated envelope from being mistaken for
bitwise identity.

This is not a labeled-ground-truth accuracy benchmark and does not establish an
absolute OCR accuracy score. It supports the narrower statement that no output
regression was detected relative to the model's measured run-to-run envelope on
this 1,000-page workload. The gate exercises the optimized model pipeline
directly; it does not separately validate HTTP transport behavior.
