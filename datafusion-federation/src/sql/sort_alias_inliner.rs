/// Re-inline projection aliases that appear **inside** `ORDER BY` expressions.
///
/// # The Problem
///
/// PostgreSQL (and the SQL standard) resolve `ORDER BY` sort keys against the
/// SELECT output column list in two different ways depending on context:
///
/// - A **bare** output-column alias *is* allowed as a top-level sort key:
///   `ORDER BY "my_alias"` ✅
/// - An alias used **inside a compound expression** is **not** resolved against
///   the SELECT list — it is resolved against the `FROM` tables only:
///   `ORDER BY CASE WHEN "my_alias" = 0 THEN ... END` ❌ → `42703`
///
/// DataFusion's unparser (`unproject_sort_expr`) only re-inlines `ScalarFunction`
/// projection expressions into sort keys. For any other expression shape (e.g. a
/// `BinaryExpr` like `grouping(a) + grouping(b)`), it falls through and leaves a
/// bare `Column("alias")` inside compound sort expressions. When this column
/// reference is not at the top level of the sort key, PostgreSQL cannot resolve it
/// and returns `ERROR: column "alias" does not exist (SQLSTATE 42703)`.
///
/// See: <https://www.postgresql.org/docs/current/sql-select.html#SQL-ORDERBY>
/// > "Each column can be referenced by name or by number. [...] Note that an
/// > *output column name* must stand alone, that is, it cannot be used in an
/// > expression — for example, `ORDER BY foo + 1` is not valid if the output
/// > column name is `foo`."
///
/// # The Fix
///
/// Before handing the federated sub-plan to the unparser, walk the plan tree.
/// When a `Sort` node sits directly above a `Projection` node, examine every
/// sort expression. Any `Expr::Column` with no relation qualifier that matches a
/// Projection output alias AND maps to a non-trivial projection expression (i.e.
/// not itself a plain `Column`) is replaced inline with that underlying expression
/// (with any alias wrapper stripped). After this substitution the alias name no
/// longer appears inside compound sort keys, so the SQL the unparser generates is
/// valid for any standard-compliant SQL engine.
use datafusion::{
    common::{
        tree_node::{Transformed, TreeNode},
        Column,
    },
    error::Result,
    logical_expr::{Expr, LogicalPlan, Projection, Sort},
};

/// Walk the entire logical plan and, for every `Sort → Projection` pair,
/// replace alias column references **inside compound sort expressions** with the
/// underlying projection expression.  Returns the (possibly rewritten) plan.
pub fn inline_sort_projection_aliases(plan: LogicalPlan) -> Result<LogicalPlan> {
    plan.transform(|node| match node {
        LogicalPlan::Sort(Sort { expr, input, fetch })
            if matches!(input.as_ref(), LogicalPlan::Projection(_)) =>
        {
            let LogicalPlan::Projection(proj) = input.as_ref() else {
                unreachable!("guarded by matches! above")
            };
            let new_exprs = expr
                .iter()
                .map(|sort_expr| inline_aliases_in_sort_expr(sort_expr.clone(), proj))
                .collect::<Result<Vec<_>>>()?;

            if new_exprs == expr {
                Ok(Transformed::no(LogicalPlan::Sort(Sort { expr, input, fetch })))
            } else {
                Ok(Transformed::yes(LogicalPlan::Sort(Sort {
                    expr: new_exprs,
                    input,
                    fetch,
                })))
            }
        }
        other => Ok(Transformed::no(other)),
    })
    .map(|t| t.data)
}

/// Inline aliases for a single `SortExpr` (which wraps an `Expr`).
///
/// If the inner expression is itself a bare `Expr::Column` with no qualifier
/// (i.e. a potential top-level alias reference), we leave it alone — PostgreSQL
/// accepts a bare alias as a top-level sort key and re-inlining it would
/// produce a redundant expression change.
///
/// For any other expression shape we walk its sub-expressions and replace any
/// unqualified column references that map to non-trivial projection aliases.
fn inline_aliases_in_sort_expr(
    sort_expr: datafusion::logical_expr::SortExpr,
    proj: &Projection,
) -> Result<datafusion::logical_expr::SortExpr> {
    // If the sort key is already a bare column reference at the top level,
    // leave it as-is: PostgreSQL handles bare output aliases fine.
    if matches!(&sort_expr.expr, Expr::Column(c) if c.relation.is_none()) {
        return Ok(sort_expr);
    }

    let inlined = sort_expr
        .expr
        .transform(|e| inline_one_column_ref(e, proj))?
        .data;

    Ok(datafusion::logical_expr::SortExpr {
        expr: inlined,
        asc: sort_expr.asc,
        nulls_first: sort_expr.nulls_first,
    })
}

/// If `expr` is an unqualified `Column` that maps to a non-trivial projection
/// expression, replace it with that expression (alias-stripped). Otherwise
/// return `Transformed::no(expr)`.
fn inline_one_column_ref(expr: Expr, proj: &Projection) -> Result<Transformed<Expr>> {
    let Expr::Column(ref col) = expr else {
        return Ok(Transformed::no(expr));
    };

    // Only rewrite unqualified column references (relation is None).
    // A qualified reference like `table.col` cannot be a SELECT-list alias.
    if col.relation.is_some() {
        return Ok(Transformed::no(expr));
    }

    // Look up the column name in the projection schema to find the
    // corresponding projection expression.
    if let Ok(idx) = proj.schema.index_of_column(&Column::new_unqualified(&col.name)) {
        if let Some(proj_expr) = proj.expr.get(idx) {
            let underlying = strip_alias(proj_expr.clone());
            // Only inline if the underlying expression is non-trivial —
            // a plain column-to-column rename does not need inlining.
            if !matches!(underlying, Expr::Column(_)) {
                return Ok(Transformed::yes(underlying));
            }
        }
    }

    Ok(Transformed::no(expr))
}

/// Strip any top-level `Alias` wrapper(s) from an expression, returning the
/// innermost non-alias expression.
fn strip_alias(expr: Expr) -> Expr {
    match expr {
        Expr::Alias(alias) => strip_alias(*alias.expr),
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::{
        arrow::datatypes::{DataType, Field, Schema},
        common::{DFSchema, DFSchemaRef},
        logical_expr::{col, lit, Expr, LogicalPlan, Projection, Sort, SortExpr},
        sql::unparser::{dialect::DefaultDialect, Unparser},
    };

    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn schema_ref(fields: &[(&str, DataType)]) -> DFSchemaRef {
        let arrow_schema = Schema::new(
            fields
                .iter()
                .map(|(name, dt)| Field::new(*name, dt.clone(), true))
                .collect::<Vec<_>>(),
        );
        Arc::new(DFSchema::try_from(arrow_schema).unwrap())
    }

    // ------------------------------------------------------------------
    // Test 1: alias inside a CASE expression in ORDER BY is inlined
    // ------------------------------------------------------------------

    /// Build a plan shaped like:
    ///
    /// ```text
    /// Sort: [CASE WHEN s = 0 THEN a END ASC, s ASC]
    ///   Projection: (a + b) AS s, a
    ///     EmptyRelation (schema: a: Int64, b: Int64)
    /// ```
    ///
    /// The `s` in the CASE expression is an alias for `a + b`.  After
    /// inlining, the sort expression should reference `a + b` directly,
    /// not the alias `s`.
    #[test]
    fn alias_inside_case_is_inlined() {
        let input_schema = schema_ref(&[("a", DataType::Int64), ("b", DataType::Int64)]);

        let empty = LogicalPlan::EmptyRelation(datafusion::logical_expr::EmptyRelation {
            produce_one_row: false,
            schema: Arc::clone(&input_schema),
        });

        // Projection: (a + b) AS s, a
        let proj_exprs = vec![
            (col("a") + col("b")).alias("s"),
            col("a"),
        ];
        let proj_schema = schema_ref(&[("s", DataType::Int64), ("a", DataType::Int64)]);
        let projection = LogicalPlan::Projection(Projection::try_new_with_schema(
            proj_exprs,
            Arc::new(empty),
            proj_schema,
        ).unwrap());

        // Sort: CASE WHEN s = 0 THEN a END ASC, s ASC
        let case_expr = Expr::Case(datafusion::logical_expr::Case {
            expr: None,
            when_then_expr: vec![(
                Box::new(col("s").eq(lit(0i64))),
                Box::new(col("a")),
            )],
            else_expr: None,
        });

        let sort_exprs = vec![
            SortExpr { expr: case_expr, asc: true, nulls_first: false },
            SortExpr { expr: col("s"), asc: true, nulls_first: false },
        ];
        let sort = LogicalPlan::Sort(Sort {
            expr: sort_exprs,
            input: Arc::new(projection),
            fetch: None,
        });

        let rewritten = inline_sort_projection_aliases(sort).unwrap();

        let LogicalPlan::Sort(rewritten_sort) = &rewritten else {
            panic!("expected Sort at root");
        };

        // The CASE expression's `s` should have been replaced with `(a + b)`.
        // The top-level bare `s` should remain unchanged.
        let case_sort = &rewritten_sort.expr[0].expr;
        let bare_sort = &rewritten_sort.expr[1].expr;

        // Bare alias `s` at the top level → NOT rewritten.
        assert!(
            matches!(bare_sort, Expr::Column(c) if c.name == "s"),
            "Top-level bare alias column must NOT be inlined; got: {bare_sort:?}"
        );

        // `s` inside CASE → must be rewritten to `a + b`.
        let case_str = format!("{case_sort:?}");
        assert!(
            !case_str.contains("Column(Column { relation: None, name: \"s\" })"),
            "Alias `s` must not remain inside CASE; got: {case_sort:?}"
        );
        // The `s` inside the CASE when-expr should now be the BinaryExpr (a + b).
        let Expr::Case(case) = case_sort else {
            panic!("expected CASE at sort key 0; got: {case_sort:?}");
        };
        let when_cond = &case.when_then_expr[0].0;
        // when_cond should be `(a + b) = 0`, not `s = 0`
        assert!(
            !matches!(when_cond.as_ref(), Expr::Column(c) if c.name == "s"),
            "The WHEN condition must not be a bare `s` column; got: {when_cond:?}"
        );
    }

    // ------------------------------------------------------------------
    // Test 2: no rewrite when Sort is not directly above a Projection
    // ------------------------------------------------------------------

    #[test]
    fn no_rewrite_without_projection_child() {
        let input_schema = schema_ref(&[("a", DataType::Int64)]);
        let empty = LogicalPlan::EmptyRelation(datafusion::logical_expr::EmptyRelation {
            produce_one_row: false,
            schema: Arc::clone(&input_schema),
        });

        // Sort directly on EmptyRelation (no Projection in between).
        let sort = LogicalPlan::Sort(Sort {
            expr: vec![SortExpr { expr: col("a"), asc: true, nulls_first: false }],
            input: Arc::new(empty),
            fetch: None,
        });

        // Clone for comparison.
        let original_exprs = match &sort {
            LogicalPlan::Sort(s) => s.expr.clone(),
            _ => panic!(),
        };

        let rewritten = inline_sort_projection_aliases(sort).unwrap();
        let rewritten_exprs = match &rewritten {
            LogicalPlan::Sort(s) => s.expr.clone(),
            _ => panic!(),
        };

        assert_eq!(
            original_exprs, rewritten_exprs,
            "Sort with no Projection child must not be modified"
        );
    }

    // ------------------------------------------------------------------
    // Test 3: plain column alias is NOT inlined (it's a trivial rename)
    // ------------------------------------------------------------------

    #[test]
    fn trivial_column_alias_not_inlined() {
        let input_schema = schema_ref(&[("a", DataType::Int64), ("b", DataType::Int64)]);
        let empty = LogicalPlan::EmptyRelation(datafusion::logical_expr::EmptyRelation {
            produce_one_row: false,
            schema: Arc::clone(&input_schema),
        });

        // Projection: a AS x, b
        let proj_exprs = vec![col("a").alias("x"), col("b")];
        let proj_schema = schema_ref(&[("x", DataType::Int64), ("b", DataType::Int64)]);
        let projection = LogicalPlan::Projection(Projection::try_new_with_schema(
            proj_exprs,
            Arc::new(empty),
            proj_schema,
        ).unwrap());

        // Sort: x + 1 — x is a trivial alias of column a, so `x` inside a
        // compound expression is NOT inlined (underlying expr is also a column).
        let sort_expr = Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr {
            left: Box::new(col("x")),
            op: datafusion::logical_expr::Operator::Plus,
            right: Box::new(lit(1i64)),
        });

        let sort = LogicalPlan::Sort(Sort {
            expr: vec![SortExpr { expr: sort_expr, asc: true, nulls_first: false }],
            input: Arc::new(projection),
            fetch: None,
        });

        let rewritten = inline_sort_projection_aliases(sort).unwrap();
        let LogicalPlan::Sort(rs) = &rewritten else { panic!() };
        // `x` is a trivial rename of `a`, so inlining would give `a + 1`
        // which is equivalent — but the underlying expr is a Column so
        // our rule deliberately does NOT inline it.
        let sort_key_str = format!("{:?}", rs.expr[0].expr);
        assert!(
            sort_key_str.contains("\"x\""),
            "Trivial column alias must NOT be inlined; got: {sort_key_str}"
        );
    }

    // ------------------------------------------------------------------
    // Test 4: end-to-end SQL generation does not contain alias in ORDER BY
    // ------------------------------------------------------------------

    /// Verifies that after `inline_sort_projection_aliases` the SQL produced by
    /// the DataFusion unparser does not contain the alias name inside a CASE in
    /// the ORDER BY clause.
    #[test]
    fn unparsed_sql_does_not_use_alias_inside_order_by_case() {
        let input_schema = schema_ref(&[("a", DataType::Int64), ("b", DataType::Int64)]);
        let empty = LogicalPlan::EmptyRelation(datafusion::logical_expr::EmptyRelation {
            produce_one_row: false,
            schema: Arc::clone(&input_schema),
        });

        // Projection: (a + b) AS s, a
        let proj_exprs = vec![(col("a") + col("b")).alias("s"), col("a")];
        let proj_schema = schema_ref(&[("s", DataType::Int64), ("a", DataType::Int64)]);
        let projection = LogicalPlan::Projection(Projection::try_new_with_schema(
            proj_exprs,
            Arc::new(empty),
            proj_schema,
        ).unwrap());

        // Sort: CASE WHEN s = 0 THEN a END ASC, s ASC
        let case_expr = Expr::Case(datafusion::logical_expr::Case {
            expr: None,
            when_then_expr: vec![(
                Box::new(col("s").eq(lit(0i64))),
                Box::new(col("a")),
            )],
            else_expr: None,
        });

        let sort = LogicalPlan::Sort(Sort {
            expr: vec![
                SortExpr { expr: case_expr, asc: true, nulls_first: false },
                SortExpr { expr: col("s"), asc: true, nulls_first: false },
            ],
            input: Arc::new(projection),
            fetch: None,
        });

        let rewritten = inline_sort_projection_aliases(sort).unwrap();

        let dialect = DefaultDialect {};
        let sql = Unparser::new(&dialect)
            .plan_to_sql(&rewritten)
            .unwrap()
            .to_string();

        // The alias `s` must NOT appear inside the CASE expression in ORDER BY.
        // A bare `s` at the top level of ORDER BY is acceptable (Postgres allows it),
        // but the CASE WHEN condition should reference `(a + b)` directly.
        assert!(
            !sql.contains("CASE WHEN s = 0") && !sql.contains("CASE WHEN \"s\" = 0"),
            "Generated SQL must not use alias `s` inside CASE in ORDER BY; got:\n{sql}"
        );
        // The SQL should still contain a CASE expression.
        assert!(
            sql.to_uppercase().contains("CASE"),
            "Expected CASE in ORDER BY; got:\n{sql}"
        );
    }
}
