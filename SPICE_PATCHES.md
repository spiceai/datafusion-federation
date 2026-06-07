# Spice patches over upstream `datafusion-federation`

This fork tracks upstream
[`datafusion-contrib/datafusion-federation`](https://github.com/datafusion-contrib/datafusion-federation)
and carries a small number of Spice-specific patches on top of a pinned upstream
release.

Base: **upstream v0.5.3** (DataFusion 53).

The v0.5.x line is an architectural rewrite of the federation pass: the old
analyzer-rule design (`src/analyzer/mod.rs` + a large `src/sql/mod.rs`
table-scan rewriter) was replaced by `FederationOptimizerRule`
(`src/optimizer/mod.rs`) for the federation decision and a dedicated
`RewriteTableScanAnalyzer` (`src/sql/analyzer.rs`) for unparsing. Because of this
rewrite, several patches that were written against the DataFusion-52.5
analyzer-rule architecture are either obsolete (already handled by the redesign)
or had to be re-implemented in the new shape.

## Patch reconciliation (DF52.5 fork → v0.5.3)

| Original patch | Intent | Status on v0.5.3 |
| --- | --- | --- |
| #74 `fix: handle EXISTS/NOT EXISTS subqueries in federation analyzer` | Recognize correlated `EXISTS` / `NOT EXISTS` subqueries and federate them together with the outer query when both sides share a provider. | **Re-implemented.** The federation-decision logic was ported to `FederationOptimizerRule` (`Expr::Exists` arms in `scan_expr_recursively` and `optimize_expr_recursively`, mirroring the existing `ScalarSubquery` handling, including the no-op `Projection` wrapper that lets `decorrelate_predicate_subquery` accept a federated `Extension` node). |
| #73 `Properly handle DML` | Do not federate `LogicalPlan::Dml` as a whole (the unparser's `dml_to_sql` is unimplemented and wrapping DML would hide it from write-permission validators); delegate `delete_from` / `update` through `FederatedTableProviderAdaptor`. | **Re-implemented.** `FederationOptimizerRule` now falls through on `LogicalPlan::Dml(_)` (alongside the existing `Analyze` case), leaving the `Dml` node intact; `FederatedTableProviderAdaptor` gained `delete_from` / `update` delegation matching the existing `insert_into`. |
| #71 `Correctness issues with sorts` / `fix table column naming` / `fix non-sort` | Local sort-preservation + table column naming on the DF52.5 unparser. | **Obsolete (self-cancelling).** #71 was fully reverted by #72 (`git diff` of the pre-#71 tree vs the post-#72 tree on `src/sql/mod.rs` is empty), so the pair contributes nothing. The functions it introduced (`sink_projection_below_sort`, `find_top_sort`, `sink_exprs_below_sort`) do not exist in the v0.5.3 redesign and are not needed. |
| #72 `Revert "Correctness issues with sorts"` | Revert of #71. | **Obsolete.** See above — it is the second half of the self-cancelling pair. |

## Copilot review threads (PR #76)

1. **DF52.5 pin vs DF53 title** — resolved by basing on upstream v0.5.3
   (`datafusion = "53"`, workspace `version = "0.5.3"`); no `[patch]` to a
   v52.5.0-rc1 commit.
2. **`same_provider_join_not_exists` snapshot dropped the inner alias `l2`
   (tautological `lineitem.x = lineitem.x`)** — this was an artifact of the
   DF52.5 analyzer-rule unparser. The v0.5.3 unparser emits the EXISTS subquery
   as a properly-aliased derived table
   (`(SELECT ... FROM lineitem AS l2) AS __correlated_sq_1`), so the bug does not
   occur. Guarded by `same_provider_join_not_exists_keeps_inner_alias`.
3. **`same_provider_aliased_not_exists` snapshot used bare `lineitem.*` where
   `l1` / `l2` were required** — same root cause, also fixed by the v0.5.3
   unparser. Guarded by `same_provider_aliased_not_exists_uses_aliases`.
4. **`plan_federation` no longer enforces top-level ordering** — this is
   upstream-by-design: federation pushes the `ORDER BY` into the remote SQL.
   The Spice patch that added local sort preservation (#71) was reverted (#72)
   for causing correctness issues and is not part of the v0.5.3 redesign.
   Documented/guarded by `top_level_order_by_is_pushed_down`.

## Spice tests added on top of v0.5.3

- `src/sql/mod.rs`: `same_provider_exists_federates_as_single_unit`,
  `same_provider_join_not_exists_keeps_inner_alias`,
  `same_provider_aliased_not_exists_uses_aliases`,
  `cross_provider_not_exists_splits`, `top_level_order_by_is_pushed_down`.
- `src/table_provider.rs`: `delete_from_delegates_to_inner_provider`,
  `update_delegates_to_inner_provider`,
  `delete_from_without_inner_provider_is_not_implemented`.
