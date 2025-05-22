use std::{collections::HashSet, sync::Arc};

use datafusion::{
    common::{Column, HashMap, RecursionUnnestOption, UnnestOptions},
    error::{DataFusionError, Result},
    logical_expr::{
        self,
        expr::{
            AggregateFunction, AggregateFunctionParams, Alias, Exists, InList, InSubquery,
            ScalarFunction, Sort, Unnest, WindowFunction, WindowFunctionParams,
        },
        Between, BinaryExpr, Case, Cast, Expr, GroupingSet, Like, Limit, LogicalPlan,
        LogicalPlanBuilder, Projection, Subquery, TryCast,
    },
    sql::TableReference,
};
use datafusion_federation::{get_table_source, table_reference::MultiPartTableReference};

fn collect_known_rewrites_from_plan(
    plan: &LogicalPlan,
    known_rewrites: &mut HashMap<TableReference, MultiPartTableReference>,
) -> Result<()> {
    if let LogicalPlan::TableScan(table_scan) = plan {
        let original_table_name = table_scan.table_name.clone();

        if let Some(federated_source) = get_table_source(&table_scan.source)? {
            if let Some(remote_table_name) = federated_source.remote_table_name() {
                known_rewrites.insert(original_table_name, remote_table_name.clone());
            }
        }
    }

    // Recursively collect from all inputs
    for input in plan.inputs() {
        collect_known_rewrites_from_plan(input, known_rewrites)?;
    }

    for expr in plan.expressions() {
        collect_known_rewrites_from_expr(expr, known_rewrites)?;
    }

    Ok(())
}

fn collect_known_rewrites_from_expr(
    expr: Expr,
    known_rewrites: &mut HashMap<TableReference, MultiPartTableReference>,
) -> Result<()> {
    match expr {
        Expr::Column(_) => Ok(()), // Column references don't have any table scans
        Expr::ScalarSubquery(subquery) => {
            collect_known_rewrites_from_plan(&subquery.subquery, known_rewrites)
        }
        Expr::BinaryExpr(binary_expr) => {
            collect_known_rewrites_from_expr(*binary_expr.left, known_rewrites)?;
            collect_known_rewrites_from_expr(*binary_expr.right, known_rewrites)?;
            Ok(())
        }
        Expr::Alias(alias) => collect_known_rewrites_from_expr(*alias.expr, known_rewrites),
        Expr::Like(like) => {
            collect_known_rewrites_from_expr(*like.expr, known_rewrites)?;
            collect_known_rewrites_from_expr(*like.pattern, known_rewrites)?;
            Ok(())
        }
        Expr::SimilarTo(similar_to) => {
            collect_known_rewrites_from_expr(*similar_to.expr, known_rewrites)?;
            collect_known_rewrites_from_expr(*similar_to.pattern, known_rewrites)?;
            Ok(())
        }
        Expr::Not(e) => collect_known_rewrites_from_expr(*e, known_rewrites),
        Expr::IsNotNull(e) => collect_known_rewrites_from_expr(*e, known_rewrites),
        Expr::IsNull(e) => collect_known_rewrites_from_expr(*e, known_rewrites),
        Expr::IsTrue(e) => collect_known_rewrites_from_expr(*e, known_rewrites),
        Expr::IsFalse(e) => collect_known_rewrites_from_expr(*e, known_rewrites),
        Expr::IsUnknown(e) => collect_known_rewrites_from_expr(*e, known_rewrites),
        Expr::IsNotTrue(e) => collect_known_rewrites_from_expr(*e, known_rewrites),
        Expr::IsNotFalse(e) => collect_known_rewrites_from_expr(*e, known_rewrites),
        Expr::IsNotUnknown(e) => collect_known_rewrites_from_expr(*e, known_rewrites),
        Expr::Negative(e) => collect_known_rewrites_from_expr(*e, known_rewrites),
        Expr::Between(between) => {
            collect_known_rewrites_from_expr(*between.expr, known_rewrites)?;
            collect_known_rewrites_from_expr(*between.low, known_rewrites)?;
            collect_known_rewrites_from_expr(*between.high, known_rewrites)?;
            Ok(())
        }
        Expr::Case(case) => {
            if let Some(expr) = case.expr {
                collect_known_rewrites_from_expr(*expr, known_rewrites)?;
            }
            if let Some(else_expr) = case.else_expr {
                collect_known_rewrites_from_expr(*else_expr, known_rewrites)?;
            }
            for (when, then) in case.when_then_expr {
                collect_known_rewrites_from_expr(*when, known_rewrites)?;
                collect_known_rewrites_from_expr(*then, known_rewrites)?;
            }
            Ok(())
        }
        Expr::Cast(cast) => collect_known_rewrites_from_expr(*cast.expr, known_rewrites),
        Expr::TryCast(try_cast) => collect_known_rewrites_from_expr(*try_cast.expr, known_rewrites),
        Expr::ScalarFunction(sf) => {
            for arg in sf.args {
                collect_known_rewrites_from_expr(arg, known_rewrites)?;
            }
            Ok(())
        }
        Expr::AggregateFunction(af) => {
            for arg in af.params.args {
                collect_known_rewrites_from_expr(arg, known_rewrites)?;
            }
            if let Some(filter) = af.params.filter {
                collect_known_rewrites_from_expr(*filter, known_rewrites)?;
            }
            if let Some(order_by) = af.params.order_by {
                for sort in order_by {
                    collect_known_rewrites_from_expr(sort.expr, known_rewrites)?;
                }
            }
            Ok(())
        }
        Expr::WindowFunction(wf) => {
            for arg in wf.params.args {
                collect_known_rewrites_from_expr(arg, known_rewrites)?;
            }
            for expr in wf.params.partition_by {
                collect_known_rewrites_from_expr(expr, known_rewrites)?;
            }
            for sort in wf.params.order_by {
                collect_known_rewrites_from_expr(sort.expr, known_rewrites)?;
            }
            Ok(())
        }
        Expr::InList(il) => {
            collect_known_rewrites_from_expr(*il.expr, known_rewrites)?;
            for expr in il.list {
                collect_known_rewrites_from_expr(expr, known_rewrites)?;
            }
            Ok(())
        }
        Expr::Exists(exists) => {
            collect_known_rewrites_from_plan(&exists.subquery.subquery, known_rewrites)?;
            for expr in exists.subquery.outer_ref_columns {
                collect_known_rewrites_from_expr(expr, known_rewrites)?;
            }
            Ok(())
        }
        Expr::InSubquery(is) => {
            collect_known_rewrites_from_expr(*is.expr, known_rewrites)?;
            collect_known_rewrites_from_plan(&is.subquery.subquery, known_rewrites)?;
            for expr in is.subquery.outer_ref_columns {
                collect_known_rewrites_from_expr(expr, known_rewrites)?;
            }
            Ok(())
        }
        #[allow(deprecated, reason = "Needed to exhaustively match all variants")]
        Expr::Wildcard { .. } => Ok(()), // Wildcard expressions don't have any table scans
        Expr::GroupingSet(gs) => match gs {
            GroupingSet::Rollup(exprs) | GroupingSet::Cube(exprs) => {
                for expr in exprs {
                    collect_known_rewrites_from_expr(expr, known_rewrites)?;
                }
                Ok(())
            }
            GroupingSet::GroupingSets(vec_exprs) => {
                for exprs in vec_exprs {
                    for expr in exprs {
                        collect_known_rewrites_from_expr(expr, known_rewrites)?;
                    }
                }
                Ok(())
            }
        },
        Expr::OuterReferenceColumn(_, _) => Ok(()), // Outer reference columns don't have any table scans
        Expr::Unnest(unnest) => collect_known_rewrites_from_expr(*unnest.expr, known_rewrites),
        Expr::ScalarVariable(_, _) | Expr::Literal(_) | Expr::Placeholder(_) => Ok(()),
    }
}

/// Rewrite table scans to use the original federated table name.
pub(crate) fn rewrite_table_scans(
    plan: &LogicalPlan,
    known_rewrites: &mut HashMap<TableReference, MultiPartTableReference>,
    subquery_uses_partial_path: bool,
    subquery_table_scans: &mut Option<HashSet<TableReference>>,
) -> Result<LogicalPlan> {
    // First pass: collect all known rewrites
    collect_known_rewrites_from_plan(plan, known_rewrites)?;

    // Second pass: do the actual rewriting with complete known_rewrites
    rewrite_plan_with_known_rewrites(
        plan,
        known_rewrites,
        subquery_uses_partial_path,
        subquery_table_scans,
    )
}

fn rewrite_plan_with_known_rewrites(
    plan: &LogicalPlan,
    known_rewrites: &HashMap<TableReference, MultiPartTableReference>,
    subquery_uses_partial_path: bool,
    subquery_table_scans: &mut Option<HashSet<TableReference>>,
) -> Result<LogicalPlan> {
    if plan.inputs().is_empty() {
        if let LogicalPlan::TableScan(table_scan) = plan {
            let original_table_name = table_scan.table_name.clone();
            let mut new_table_scan = table_scan.clone();

            let Some(federated_source) = get_table_source(&table_scan.source)? else {
                // Not a federated source
                return Ok(plan.clone());
            };

            match federated_source.remote_table_name() {
                Some(remote_table_name) => {
                    // If the remote table name is a MultiPartTableReference, we will not rewrite it here, but rewrite it after the final unparsing on the AST directly.
                    let MultiPartTableReference::TableReference(remote_table_name) =
                        remote_table_name
                    else {
                        return Ok(plan.clone());
                    };

                    if let Some(s) = subquery_table_scans {
                        s.insert(original_table_name);
                    }

                    // Rewrite the schema of this node to have the remote table as the qualifier.
                    let new_schema = (*new_table_scan.projected_schema)
                        .clone()
                        .replace_qualifier(remote_table_name.clone());
                    new_table_scan.projected_schema = Arc::new(new_schema);
                    new_table_scan.table_name = remote_table_name.clone();

                    // Rewrite the filter expression in table scan
                    let mut new_filter_expressions = vec![];
                    for expression in &table_scan.filters {
                        let new_expr = rewrite_table_scans_in_expr(
                            expression.clone(),
                            known_rewrites,
                            subquery_uses_partial_path,
                            subquery_table_scans,
                        )?;
                        new_filter_expressions.push(new_expr);
                    }

                    new_table_scan.filters = new_filter_expressions;
                }
                None => {
                    return Ok(plan.clone());
                }
            }

            return Ok(LogicalPlan::TableScan(new_table_scan));
        } else {
            return Ok(plan.clone());
        }
    }

    let rewritten_inputs = plan
        .inputs()
        .into_iter()
        .map(|i| {
            rewrite_plan_with_known_rewrites(
                i,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    match plan {
        LogicalPlan::Unnest(unnest) => {
            // The Union plan cannot be constructed from rewritten expressions. It requires specialized logic to handle
            // the renaming in UNNEST columns and the corresponding column aliases in the underlying projection plan.
            rewrite_unnest_plan(
                unnest,
                rewritten_inputs,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )
        }
        LogicalPlan::Limit(limit) => {
            let rewritten_skip = limit
                .skip
                .as_ref()
                .map(|skip| {
                    rewrite_table_scans_in_expr(
                        *skip.clone(),
                        known_rewrites,
                        subquery_uses_partial_path,
                        subquery_table_scans,
                    )
                    .map(Box::new)
                })
                .transpose()?;
            let rewritten_fetch = limit
                .fetch
                .as_ref()
                .map(|fetch| {
                    rewrite_table_scans_in_expr(
                        *fetch.clone(),
                        known_rewrites,
                        subquery_uses_partial_path,
                        subquery_table_scans,
                    )
                    .map(Box::new)
                })
                .transpose()?;
            // explicitly set fetch and skip
            let new_plan = LogicalPlan::Limit(Limit {
                skip: rewritten_skip,
                fetch: rewritten_fetch,
                input: Arc::new(rewritten_inputs[0].clone()),
            });
            Ok(new_plan)
        }
        LogicalPlan::Join(datafusion::logical_expr::Join { on, filter, .. }) => {
            let mut new_expressions = vec![];
            if on.len() > 0 {
                for (left, right) in on {
                    let left = rewrite_table_scans_in_expr(
                        left.clone(),
                        known_rewrites,
                        subquery_uses_partial_path,
                        subquery_table_scans,
                    )?;
                    let right = rewrite_table_scans_in_expr(
                        right.clone(),
                        known_rewrites,
                        subquery_uses_partial_path,
                        subquery_table_scans,
                    )?;
                    let equal_expr = Expr::BinaryExpr(BinaryExpr::new(
                        Box::new(left),
                        logical_expr::Operator::Eq,
                        Box::new(right),
                    ));
                    new_expressions.push(equal_expr);
                }
            }
            if let Some(filter) = filter {
                let new_filter = rewrite_table_scans_in_expr(
                    filter.clone(),
                    known_rewrites,
                    subquery_uses_partial_path,
                    subquery_table_scans,
                )?;
                new_expressions.push(new_filter);
            }

            let new_plan = plan.with_new_exprs(new_expressions, rewritten_inputs)?;
            Ok(new_plan)
        }
        _ => {
            let mut new_expressions = vec![];
            for expression in plan.expressions() {
                let new_expr = rewrite_table_scans_in_expr(
                    expression.clone(),
                    known_rewrites,
                    subquery_uses_partial_path,
                    subquery_table_scans,
                )?;
                new_expressions.push(new_expr);
            }
            let new_plan = plan.with_new_exprs(new_expressions, rewritten_inputs)?;
            Ok(new_plan)
        }
    }
}

/// Rewrite an unnest plan to use the original federated table name.
/// In a standard unnest plan, column names are typically referenced in projection columns by wrapping them
/// in aliases such as "UNNEST(table_name.column_name)". `rewrite_table_scans_in_expr` does not handle alias
/// rewriting so we manually collect the rewritten unnest column names/aliases and update the projection
/// plan to ensure that the aliases reflect the new names.
fn rewrite_unnest_plan(
    unnest: &logical_expr::Unnest,
    mut rewritten_inputs: Vec<LogicalPlan>,
    known_rewrites: &HashMap<TableReference, MultiPartTableReference>,
    subquery_uses_partial_path: bool,
    subquery_table_scans: &mut Option<HashSet<TableReference>>,
) -> Result<LogicalPlan> {
    // Unnest plan has a single input
    let input = rewritten_inputs.remove(0);

    let mut known_unnest_rewrites: HashMap<String, String> = HashMap::new();

    // `exec_columns` represent columns to run UNNEST on: rewrite them and collect new names
    let unnest_columns = unnest
        .exec_columns
        .iter()
        .map(|c: &Column| {
            match rewrite_table_scans_in_expr(
                Expr::Column(c.clone()),
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )? {
                Expr::Column(column) => {
                    known_unnest_rewrites.insert(c.name.clone(), column.name.clone());
                    Ok(column)
                }
                _ => Err(DataFusionError::Plan(
                    "Rewritten column expression must be a column".to_string(),
                )),
            }
        })
        .collect::<Result<Vec<Column>>>()?;

    let LogicalPlan::Projection(projection) = input else {
        return Err(DataFusionError::Plan(
            "The input to the unnest plan should be a projection plan".to_string(),
        ));
    };

    // rewrite aliases in inner projection; columns were rewritten via `rewrite_table_scans_in_expr`
    let new_expressions = projection
        .expr
        .into_iter()
        .map(|expr| match expr {
            Expr::Alias(alias) => {
                let name = match known_unnest_rewrites.get(&alias.name) {
                    Some(name) => name,
                    None => &alias.name,
                };
                Ok(Expr::Alias(Alias::new(*alias.expr, alias.relation, name)))
            }
            _ => Ok(expr),
        })
        .collect::<Result<Vec<_>>>()?;

    let updated_unnest_inner_projection =
        Projection::try_new(new_expressions, Arc::clone(&projection.input))?;

    let unnest_options =
        rewrite_unnest_options(&unnest.options, known_rewrites, subquery_table_scans);

    // reconstruct the unnest plan with updated projection and rewritten column names
    let new_plan =
        LogicalPlanBuilder::new(LogicalPlan::Projection(updated_unnest_inner_projection))
            .unnest_columns_with_options(unnest_columns, unnest_options)?
            .build()?;

    Ok(new_plan)
}

/// Rewrites columns names in the unnest options to use the original federated table name:
/// "unnest_placeholder(foo.df_table.a,depth=1)"" -> "unnest_placeholder(remote_table.a,depth=1)""
fn rewrite_unnest_options(
    options: &UnnestOptions,
    known_rewrites: &HashMap<TableReference, MultiPartTableReference>,
    subquery_table_scans: &mut Option<HashSet<TableReference>>,
) -> UnnestOptions {
    let mut new_options = options.clone();
    new_options
        .recursions
        .iter_mut()
        .for_each(|x: &mut RecursionUnnestOption| {
            if let Some(new_name) =
                rewrite_column_name(&x.input_column.name, known_rewrites, subquery_table_scans)
            {
                x.input_column.name = new_name;
            }

            if let Some(new_name) =
                rewrite_column_name(&x.output_column.name, known_rewrites, subquery_table_scans)
            {
                x.output_column.name = new_name;
            }
        });
    new_options
}

/// Checks if any of the rewrites match any substring in col_name, and replace that part of the string if so.
/// This handles cases like "MAX(foo.df_table.a)" -> "MAX(remote_table.a)"
/// Returns the rewritten name if any rewrite was applied, otherwise None.
fn rewrite_column_name(
    col_name: &str,
    known_rewrites: &HashMap<TableReference, MultiPartTableReference>,
    subquery_table_scans: &mut Option<HashSet<TableReference>>,
) -> Option<String> {
    let (new_col_name, was_rewritten) = known_rewrites
        .iter()
        .filter_map(|(table_ref, rewrite)| match rewrite {
            MultiPartTableReference::TableReference(rewrite) => Some((table_ref, rewrite)),
            _ => None,
        })
        .fold(
            (col_name.to_string(), false),
            |(col_name, was_rewritten), (table_ref, rewrite)| {
                let mut rewrite_string = rewrite.to_string();
                if let Some(subquery_reference) = subquery_table_scans {
                    if subquery_reference.get(table_ref).is_some() {
                        rewrite_string = get_partial_table_name(rewrite);
                    }
                }
                match rewrite_column_name_in_expr(
                    &col_name,
                    &table_ref.to_string(),
                    &rewrite_string,
                    0,
                ) {
                    Some(new_name) => (new_name, true),
                    None => (col_name, was_rewritten),
                }
            },
        );

    if was_rewritten {
        Some(new_col_name)
    } else {
        None
    }
}

fn get_partial_table_name(full_table_reference: &TableReference) -> String {
    let full_table_path = full_table_reference.table().to_owned();
    let path_parts: Vec<&str> = full_table_path.split('.').collect();
    path_parts[path_parts.len() - 1].to_owned()
}

// The function replaces occurrences of table_ref_str in col_name with the new name defined by rewrite.
// The name to rewrite should NOT be a substring of another name.
// Supports multiple occurrences of table_ref_str in col_name.
fn rewrite_column_name_in_expr(
    col_name: &str,
    table_ref_str: &str,
    rewrite: &str,
    start_pos: usize,
) -> Option<String> {
    if start_pos >= col_name.len() {
        return None;
    }

    // Find the first occurrence of table_ref_str starting from start_pos
    let idx = col_name[start_pos..].find(table_ref_str)?;

    // Calculate the absolute index of the occurrence in string as the index above is relative to start_pos
    let idx = start_pos + idx;

    // Table name same as column name
    // Shouldn't rewrite in this case
    if idx == 0 && table_ref_str.len() == col_name.len() {
        return None;
    }

    if idx > 0 {
        // Check if the previous character is alphabetic, numeric, underscore or period, in which case we
        // should not rewrite as it is a part of another name.
        if let Some(prev_char) = col_name.chars().nth(idx - 1) {
            if prev_char.is_alphabetic()
                || prev_char.is_numeric()
                || prev_char == '_'
                || prev_char == '.'
            {
                return rewrite_column_name_in_expr(
                    col_name,
                    table_ref_str,
                    rewrite,
                    idx + table_ref_str.len(),
                );
            }
        }
    }

    // Check if the next character is alphabetic, numeric or underscore, in which case we
    // should not rewrite as it is a part of another name.
    if let Some(next_char) = col_name.chars().nth(idx + table_ref_str.len()) {
        if next_char.is_alphabetic() || next_char.is_numeric() || next_char == '_' {
            return rewrite_column_name_in_expr(
                col_name,
                table_ref_str,
                rewrite,
                idx + table_ref_str.len(),
            );
        }
    }

    // Found full match, replace table_ref_str occurrence with rewrite
    let rewritten_name = format!(
        "{}{}{}",
        &col_name[..idx],
        rewrite,
        &col_name[idx + table_ref_str.len()..]
    );

    // Check if the rewritten name contains more occurrence of table_ref_str, and rewrite them as well
    // This is done by providing the updated start_pos for search
    match rewrite_column_name_in_expr(&rewritten_name, table_ref_str, rewrite, idx + rewrite.len())
    {
        Some(new_name) => Some(new_name), // more occurrences found
        None => Some(rewritten_name),     // no more occurrences/changes
    }
}

fn rewrite_table_scans_in_expr(
    expr: Expr,
    known_rewrites: &HashMap<TableReference, MultiPartTableReference>,
    subquery_uses_partial_path: bool,
    subquery_table_scans: &mut Option<HashSet<TableReference>>,
) -> Result<Expr> {
    match expr {
        Expr::ScalarSubquery(subquery) => {
            let new_subquery = if subquery_table_scans.is_some() || !subquery_uses_partial_path {
                rewrite_plan_with_known_rewrites(
                    &subquery.subquery,
                    known_rewrites,
                    subquery_uses_partial_path,
                    subquery_table_scans,
                )?
            } else {
                let mut scans = Some(HashSet::new());
                rewrite_plan_with_known_rewrites(
                    &subquery.subquery,
                    known_rewrites,
                    subquery_uses_partial_path,
                    &mut scans,
                )?
            };
            let outer_ref_columns = subquery
                .outer_ref_columns
                .into_iter()
                .map(|e| {
                    rewrite_table_scans_in_expr(
                        e,
                        known_rewrites,
                        subquery_uses_partial_path,
                        subquery_table_scans,
                    )
                })
                .collect::<Result<Vec<Expr>>>()?;
            Ok(Expr::ScalarSubquery(Subquery {
                subquery: Arc::new(new_subquery),
                outer_ref_columns,
            }))
        }
        Expr::BinaryExpr(binary_expr) => {
            let left = rewrite_table_scans_in_expr(
                *binary_expr.left,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            let right = rewrite_table_scans_in_expr(
                *binary_expr.right,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            Ok(Expr::BinaryExpr(BinaryExpr::new(
                Box::new(left),
                binary_expr.op,
                Box::new(right),
            )))
        }
        Expr::Column(mut col) => {
            if let Some(rewrite) = col
                .relation
                .as_ref()
                .and_then(|r| known_rewrites.get(r))
                .and_then(|rewrite| match rewrite {
                    MultiPartTableReference::TableReference(rewrite) => Some(rewrite),
                    _ => None,
                })
            {
                if let Some(subquery_reference) = subquery_table_scans {
                    if col
                        .relation
                        .as_ref()
                        .and_then(|r| subquery_reference.get(r))
                        .is_some()
                    {
                        // Use the partial table path from source for rewrite
                        // e.g. If the fully qualified name is foo_db.foo_schema.foo
                        // Use foo as partial path
                        let partial_path = get_partial_table_name(rewrite);
                        let partial_table_reference = TableReference::from(partial_path);
                        return Ok(Expr::Column(Column::new(
                            Some(partial_table_reference),
                            &col.name,
                        )));
                    }
                }
                Ok(Expr::Column(Column::new(Some(rewrite.clone()), &col.name)))
            } else {
                // This prevent over-eager rewrite and only pass the column into below rewritten
                // rule like MAX(...)
                if col.relation.is_some() {
                    return Ok(Expr::Column(col));
                }

                // Check if any of the rewrites match any substring in col.name, and replace that part of the string if so.
                // This will handles cases like "MAX(foo.df_table.a)" -> "MAX(remote_table.a)"
                if let Some(new_name) =
                    rewrite_column_name(&col.name, known_rewrites, subquery_table_scans)
                {
                    Ok(Expr::Column(Column::new(col.relation.take(), new_name)))
                } else {
                    Ok(Expr::Column(col))
                }
            }
        }
        Expr::Alias(alias) => {
            let expr = rewrite_table_scans_in_expr(
                *alias.expr,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            if let Some(relation) = &alias.relation {
                if let Some(rewrite) =
                    known_rewrites
                        .get(relation)
                        .and_then(|rewrite| match rewrite {
                            MultiPartTableReference::TableReference(rewrite) => Some(rewrite),
                            _ => None,
                        })
                {
                    return Ok(Expr::Alias(Alias::new(
                        expr,
                        Some(rewrite.clone()),
                        alias.name,
                    )));
                }
            }
            Ok(Expr::Alias(Alias::new(expr, alias.relation, alias.name)))
        }
        Expr::Like(like) => {
            let expr = rewrite_table_scans_in_expr(
                *like.expr,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            let pattern = rewrite_table_scans_in_expr(
                *like.pattern,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            Ok(Expr::Like(Like::new(
                like.negated,
                Box::new(expr),
                Box::new(pattern),
                like.escape_char,
                like.case_insensitive,
            )))
        }
        Expr::SimilarTo(similar_to) => {
            let expr = rewrite_table_scans_in_expr(
                *similar_to.expr,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            let pattern = rewrite_table_scans_in_expr(
                *similar_to.pattern,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            Ok(Expr::SimilarTo(Like::new(
                similar_to.negated,
                Box::new(expr),
                Box::new(pattern),
                similar_to.escape_char,
                similar_to.case_insensitive,
            )))
        }
        Expr::Not(e) => {
            let expr = rewrite_table_scans_in_expr(
                *e,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            Ok(Expr::Not(Box::new(expr)))
        }
        Expr::IsNotNull(e) => {
            let expr = rewrite_table_scans_in_expr(
                *e,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            Ok(Expr::IsNotNull(Box::new(expr)))
        }
        Expr::IsNull(e) => {
            let expr = rewrite_table_scans_in_expr(
                *e,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            Ok(Expr::IsNull(Box::new(expr)))
        }
        Expr::IsTrue(e) => {
            let expr = rewrite_table_scans_in_expr(
                *e,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            Ok(Expr::IsTrue(Box::new(expr)))
        }
        Expr::IsFalse(e) => {
            let expr = rewrite_table_scans_in_expr(
                *e,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            Ok(Expr::IsFalse(Box::new(expr)))
        }
        Expr::IsUnknown(e) => {
            let expr = rewrite_table_scans_in_expr(
                *e,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            Ok(Expr::IsUnknown(Box::new(expr)))
        }
        Expr::IsNotTrue(e) => {
            let expr = rewrite_table_scans_in_expr(
                *e,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            Ok(Expr::IsNotTrue(Box::new(expr)))
        }
        Expr::IsNotFalse(e) => {
            let expr = rewrite_table_scans_in_expr(
                *e,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            Ok(Expr::IsNotFalse(Box::new(expr)))
        }
        Expr::IsNotUnknown(e) => {
            let expr = rewrite_table_scans_in_expr(
                *e,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            Ok(Expr::IsNotUnknown(Box::new(expr)))
        }
        Expr::Negative(e) => {
            let expr = rewrite_table_scans_in_expr(
                *e,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            Ok(Expr::Negative(Box::new(expr)))
        }
        Expr::Between(between) => {
            let expr = rewrite_table_scans_in_expr(
                *between.expr,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            let low = rewrite_table_scans_in_expr(
                *between.low,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            let high = rewrite_table_scans_in_expr(
                *between.high,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            Ok(Expr::Between(Between::new(
                Box::new(expr),
                between.negated,
                Box::new(low),
                Box::new(high),
            )))
        }
        Expr::Case(case) => {
            let expr = case
                .expr
                .map(|e| {
                    rewrite_table_scans_in_expr(
                        *e,
                        known_rewrites,
                        subquery_uses_partial_path,
                        subquery_table_scans,
                    )
                })
                .transpose()?
                .map(Box::new);
            let else_expr = case
                .else_expr
                .map(|e| {
                    rewrite_table_scans_in_expr(
                        *e,
                        known_rewrites,
                        subquery_uses_partial_path,
                        subquery_table_scans,
                    )
                })
                .transpose()?
                .map(Box::new);
            let when_expr = case
                .when_then_expr
                .into_iter()
                .map(|(when, then)| {
                    let when = rewrite_table_scans_in_expr(
                        *when,
                        known_rewrites,
                        subquery_uses_partial_path,
                        subquery_table_scans,
                    );
                    let then = rewrite_table_scans_in_expr(
                        *then,
                        known_rewrites,
                        subquery_uses_partial_path,
                        subquery_table_scans,
                    );

                    match (when, then) {
                        (Ok(when), Ok(then)) => Ok((Box::new(when), Box::new(then))),
                        (Err(e), _) | (_, Err(e)) => Err(e),
                    }
                })
                .collect::<Result<Vec<(Box<Expr>, Box<Expr>)>>>()?;
            Ok(Expr::Case(Case::new(expr, when_expr, else_expr)))
        }
        Expr::Cast(cast) => {
            let expr = rewrite_table_scans_in_expr(
                *cast.expr,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            Ok(Expr::Cast(Cast::new(Box::new(expr), cast.data_type)))
        }
        Expr::TryCast(try_cast) => {
            let expr = rewrite_table_scans_in_expr(
                *try_cast.expr,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            Ok(Expr::TryCast(TryCast::new(
                Box::new(expr),
                try_cast.data_type,
            )))
        }
        Expr::ScalarFunction(sf) => {
            let args = sf
                .args
                .into_iter()
                .map(|e| {
                    rewrite_table_scans_in_expr(
                        e,
                        known_rewrites,
                        subquery_uses_partial_path,
                        subquery_table_scans,
                    )
                })
                .collect::<Result<Vec<Expr>>>()?;
            Ok(Expr::ScalarFunction(ScalarFunction {
                func: sf.func,
                args,
            }))
        }
        Expr::AggregateFunction(af) => {
            let args = af
                .params
                .args
                .into_iter()
                .map(|e| {
                    rewrite_table_scans_in_expr(
                        e,
                        known_rewrites,
                        subquery_uses_partial_path,
                        subquery_table_scans,
                    )
                })
                .collect::<Result<Vec<Expr>>>()?;
            let filter = af
                .params
                .filter
                .map(|e| {
                    rewrite_table_scans_in_expr(
                        *e,
                        known_rewrites,
                        subquery_uses_partial_path,
                        subquery_table_scans,
                    )
                })
                .transpose()?
                .map(Box::new);
            let order_by = af
                .params
                .order_by
                .map(|e| {
                    e.into_iter()
                        .map(|s| {
                            rewrite_table_scans_in_expr(
                                s.expr,
                                known_rewrites,
                                subquery_uses_partial_path,
                                subquery_table_scans,
                            )
                            .map(|e| Sort::new(e, s.asc, s.nulls_first))
                        })
                        .collect::<Result<Vec<Sort>>>()
                })
                .transpose()?;
            Ok(Expr::AggregateFunction(AggregateFunction {
                func: af.func,
                params: AggregateFunctionParams {
                    args,
                    distinct: af.params.distinct,
                    filter,
                    order_by,
                    null_treatment: af.params.null_treatment,
                },
            }))
        }
        Expr::WindowFunction(wf) => {
            let args = wf
                .params
                .args
                .into_iter()
                .map(|e| {
                    rewrite_table_scans_in_expr(
                        e,
                        known_rewrites,
                        subquery_uses_partial_path,
                        subquery_table_scans,
                    )
                })
                .collect::<Result<Vec<Expr>>>()?;
            let partition_by = wf
                .params
                .partition_by
                .into_iter()
                .map(|e| {
                    rewrite_table_scans_in_expr(
                        e,
                        known_rewrites,
                        subquery_uses_partial_path,
                        subquery_table_scans,
                    )
                })
                .collect::<Result<Vec<Expr>>>()?;
            let order_by = wf
                .params
                .order_by
                .into_iter()
                .map(|s| {
                    rewrite_table_scans_in_expr(
                        s.expr,
                        known_rewrites,
                        subquery_uses_partial_path,
                        subquery_table_scans,
                    )
                    .map(|e| Sort::new(e, s.asc, s.nulls_first))
                })
                .collect::<Result<Vec<Sort>>>()?;
            Ok(Expr::WindowFunction(WindowFunction {
                fun: wf.fun,
                params: WindowFunctionParams {
                    args,
                    partition_by,
                    order_by,
                    window_frame: wf.params.window_frame,
                    null_treatment: wf.params.null_treatment,
                },
            }))
        }
        Expr::InList(il) => {
            let expr = rewrite_table_scans_in_expr(
                *il.expr,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            let list = il
                .list
                .into_iter()
                .map(|e| {
                    rewrite_table_scans_in_expr(
                        e,
                        known_rewrites,
                        subquery_uses_partial_path,
                        subquery_table_scans,
                    )
                })
                .collect::<Result<Vec<Expr>>>()?;
            Ok(Expr::InList(InList::new(Box::new(expr), list, il.negated)))
        }
        Expr::Exists(exists) => {
            let subquery_plan = if subquery_table_scans.is_some() || !subquery_uses_partial_path {
                rewrite_plan_with_known_rewrites(
                    &exists.subquery.subquery,
                    known_rewrites,
                    subquery_uses_partial_path,
                    subquery_table_scans,
                )?
            } else {
                let mut scans = Some(HashSet::new());
                rewrite_plan_with_known_rewrites(
                    &exists.subquery.subquery,
                    known_rewrites,
                    subquery_uses_partial_path,
                    &mut scans,
                )?
            };
            let outer_ref_columns = exists
                .subquery
                .outer_ref_columns
                .into_iter()
                .map(|e| {
                    rewrite_table_scans_in_expr(
                        e,
                        known_rewrites,
                        subquery_uses_partial_path,
                        subquery_table_scans,
                    )
                })
                .collect::<Result<Vec<Expr>>>()?;
            let subquery = Subquery {
                subquery: Arc::new(subquery_plan),
                outer_ref_columns,
            };
            Ok(Expr::Exists(Exists::new(subquery, exists.negated)))
        }
        Expr::InSubquery(is) => {
            let expr = rewrite_table_scans_in_expr(
                *is.expr,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            let subquery_plan = if subquery_table_scans.is_some() || !subquery_uses_partial_path {
                rewrite_plan_with_known_rewrites(
                    &is.subquery.subquery,
                    known_rewrites,
                    subquery_uses_partial_path,
                    subquery_table_scans,
                )?
            } else {
                let mut scans = Some(HashSet::new());
                rewrite_plan_with_known_rewrites(
                    &is.subquery.subquery,
                    known_rewrites,
                    subquery_uses_partial_path,
                    &mut scans,
                )?
            };
            let outer_ref_columns = is
                .subquery
                .outer_ref_columns
                .into_iter()
                .map(|e| {
                    rewrite_table_scans_in_expr(
                        e,
                        known_rewrites,
                        subquery_uses_partial_path,
                        subquery_table_scans,
                    )
                })
                .collect::<Result<Vec<Expr>>>()?;
            let subquery = Subquery {
                subquery: Arc::new(subquery_plan),
                outer_ref_columns,
            };
            Ok(Expr::InSubquery(InSubquery::new(
                Box::new(expr),
                subquery,
                is.negated,
            )))
        }
        #[allow(deprecated, reason = "Needed to exhaustively match all variants")]
        Expr::Wildcard { qualifier, options } => {
            if let Some(rewrite) = qualifier
                .as_ref()
                .and_then(|q| known_rewrites.get(q))
                .and_then(|rewrite| match rewrite {
                    MultiPartTableReference::TableReference(rewrite) => Some(rewrite),
                    _ => None,
                })
            {
                Ok(Expr::Wildcard {
                    qualifier: Some(rewrite.clone()),
                    options,
                })
            } else {
                Ok(Expr::Wildcard { qualifier, options })
            }
        }
        Expr::GroupingSet(gs) => match gs {
            GroupingSet::Rollup(exprs) => {
                let exprs = exprs
                    .into_iter()
                    .map(|e| {
                        rewrite_table_scans_in_expr(
                            e,
                            known_rewrites,
                            subquery_uses_partial_path,
                            subquery_table_scans,
                        )
                    })
                    .collect::<Result<Vec<Expr>>>()?;
                Ok(Expr::GroupingSet(GroupingSet::Rollup(exprs)))
            }
            GroupingSet::Cube(exprs) => {
                let exprs = exprs
                    .into_iter()
                    .map(|e| {
                        rewrite_table_scans_in_expr(
                            e,
                            known_rewrites,
                            subquery_uses_partial_path,
                            subquery_table_scans,
                        )
                    })
                    .collect::<Result<Vec<Expr>>>()?;
                Ok(Expr::GroupingSet(GroupingSet::Cube(exprs)))
            }
            GroupingSet::GroupingSets(vec_exprs) => {
                let vec_exprs = vec_exprs
                    .into_iter()
                    .map(|exprs| {
                        exprs
                            .into_iter()
                            .map(|e| {
                                rewrite_table_scans_in_expr(
                                    e,
                                    known_rewrites,
                                    subquery_uses_partial_path,
                                    subquery_table_scans,
                                )
                            })
                            .collect::<Result<Vec<Expr>>>()
                    })
                    .collect::<Result<Vec<Vec<Expr>>>>()?;
                Ok(Expr::GroupingSet(GroupingSet::GroupingSets(vec_exprs)))
            }
        },
        Expr::OuterReferenceColumn(dt, col) => {
            if let Some(rewrite) = col
                .relation
                .as_ref()
                .and_then(|r| known_rewrites.get(r))
                .and_then(|rewrite| match rewrite {
                    MultiPartTableReference::TableReference(rewrite) => Some(rewrite),
                    _ => None,
                })
            {
                Ok(Expr::OuterReferenceColumn(
                    dt,
                    Column::new(Some(rewrite.clone()), &col.name),
                ))
            } else {
                Ok(Expr::OuterReferenceColumn(dt, col))
            }
        }
        Expr::Unnest(unnest) => {
            let expr = rewrite_table_scans_in_expr(
                *unnest.expr,
                known_rewrites,
                subquery_uses_partial_path,
                subquery_table_scans,
            )?;
            Ok(Expr::Unnest(Unnest::new(expr)))
        }
        Expr::ScalarVariable(_, _) | Expr::Literal(_) | Expr::Placeholder(_) => Ok(expr),
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use datafusion::{
        arrow::datatypes::{DataType, Field, Schema, SchemaRef},
        catalog::{MemorySchemaProvider, SchemaProvider},
        common::Column,
        datasource::{DefaultTableSource, TableProvider},
        error::DataFusionError,
        execution::{context::SessionContext, SendableRecordBatchStream},
        logical_expr::LogicalPlanBuilder,
        sql::unparser::{
            dialect::{DefaultDialect, Dialect},
            plan_to_sql,
        },
    };
    use datafusion_federation::FederatedTableProviderAdaptor;

    use crate::{SQLExecutor, SQLFederationProvider, SQLTableSource};

    use super::*;

    struct TestSQLExecutor {}

    #[async_trait]
    impl SQLExecutor for TestSQLExecutor {
        fn name(&self) -> &str {
            "test_sql_table_source"
        }

        fn compute_context(&self) -> Option<String> {
            None
        }

        fn dialect(&self) -> Arc<dyn Dialect> {
            Arc::new(DefaultDialect {})
        }

        fn execute(&self, _query: &str, _schema: SchemaRef) -> Result<SendableRecordBatchStream> {
            Err(DataFusionError::NotImplemented(
                "execute not implemented".to_string(),
            ))
        }

        async fn table_names(&self) -> Result<Vec<String>> {
            Err(DataFusionError::NotImplemented(
                "table inference not implemented".to_string(),
            ))
        }

        async fn get_table_schema(&self, _table_name: &str) -> Result<SchemaRef> {
            Err(DataFusionError::NotImplemented(
                "table inference not implemented".to_string(),
            ))
        }
    }

    fn get_test_table_provider() -> Arc<dyn TableProvider> {
        let sql_federation_provider =
            Arc::new(SQLFederationProvider::new(Arc::new(TestSQLExecutor {})));

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
            Field::new("c", DataType::Date32, false),
            Field::new(
                "d",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                false,
            ),
        ]));
        let table_source = Arc::new(
            SQLTableSource::new_with_schema(
                sql_federation_provider,
                "remote_table".to_string(),
                schema,
            )
            .expect("to have a valid SQLTableSource"),
        );
        Arc::new(FederatedTableProviderAdaptor::new(table_source))
    }

    fn get_test_table_provider_with_full_path() -> Arc<dyn TableProvider> {
        let sql_federation_provider =
            Arc::new(SQLFederationProvider::new(Arc::new(TestSQLExecutor {})));
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
            Field::new("c", DataType::Date32, false),
            Field::new(
                "d",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                false,
            ),
        ]));
        let table_source = Arc::new(
            SQLTableSource::new_with_schema(
                sql_federation_provider,
                "remote_db.remote_schema.remote_table".to_string(),
                schema,
            )
            .expect("to have a valid SQLTableSource"),
        );
        Arc::new(FederatedTableProviderAdaptor::new(table_source))
    }

    fn get_test_table_source() -> Arc<DefaultTableSource> {
        Arc::new(DefaultTableSource::new(get_test_table_provider()))
    }

    fn get_test_df_context() -> SessionContext {
        let ctx = SessionContext::new();
        let catalog = ctx
            .catalog("datafusion")
            .expect("default catalog is datafusion");
        let foo_schema = Arc::new(MemorySchemaProvider::new()) as Arc<dyn SchemaProvider>;
        catalog
            .register_schema("foo", Arc::clone(&foo_schema))
            .expect("to register schema");
        foo_schema
            .register_table("df_table".to_string(), get_test_table_provider())
            .expect("to register table");

        let public_schema = catalog
            .schema("public")
            .expect("public schema should exist");
        public_schema
            .register_table("app_table".to_string(), get_test_table_provider())
            .expect("to register table");
        public_schema
            .register_table("bar".to_string(), get_test_table_provider_with_full_path())
            .expect("to register table");

        ctx
    }

    #[test]
    fn test_rewrite_table_scans_basic() -> Result<()> {
        let default_table_source = get_test_table_source();
        let plan =
            LogicalPlanBuilder::scan("foo.df_table", default_table_source, None)?.project(vec![
                Expr::Column(Column::from_qualified_name("foo.df_table.a")),
                Expr::Column(Column::from_qualified_name("foo.df_table.b")),
                Expr::Column(Column::from_qualified_name("foo.df_table.c")),
            ])?;

        let mut known_rewrites = HashMap::new();
        let rewritten_plan =
            rewrite_table_scans(&plan.build()?, &mut known_rewrites, false, &mut None)?;

        println!("rewritten_plan: \n{:#?}", rewritten_plan);

        let unparsed_sql = plan_to_sql(&rewritten_plan)?;

        println!("unparsed_sql: \n{unparsed_sql}");

        assert_eq!(
            format!("{unparsed_sql}"),
            r#"SELECT remote_table.a, remote_table.b, remote_table.c FROM remote_table"#
        );

        Ok(())
    }

    fn init_tracing() {
        let subscriber = tracing_subscriber::FmtSubscriber::builder()
            .with_env_filter("debug")
            .with_ansi(true)
            .finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
    }

    #[tokio::test]
    async fn test_rewrite_table_scans_agg() -> Result<()> {
        init_tracing();
        let ctx = get_test_df_context();

        let agg_tests = vec![
            (
                "SELECT MAX(a) FROM foo.df_table",
                r#"SELECT max(remote_table.a) FROM remote_table"#,
            ),
            (
                "SELECT foo.df_table.a FROM foo.df_table",
                r#"SELECT remote_table.a FROM remote_table"#,
            ),
            (
                "SELECT MIN(a) FROM foo.df_table",
                r#"SELECT min(remote_table.a) FROM remote_table"#,
            ),
            (
                "SELECT AVG(a) FROM foo.df_table",
                r#"SELECT avg(remote_table.a) FROM remote_table"#,
            ),
            (
                "SELECT SUM(a) FROM foo.df_table",
                r#"SELECT sum(remote_table.a) FROM remote_table"#,
            ),
            (
                "SELECT COUNT(a) FROM foo.df_table",
                r#"SELECT count(remote_table.a) FROM remote_table"#,
            ),
            (
                "SELECT COUNT(a) as cnt FROM foo.df_table",
                r#"SELECT count(remote_table.a) AS cnt FROM remote_table"#,
            ),
            (
                "SELECT COUNT(a) as cnt FROM foo.df_table",
                r#"SELECT count(remote_table.a) AS cnt FROM remote_table"#,
            ),
            (
                "SELECT app_table from (SELECT a as app_table FROM app_table) b",
                r#"SELECT b.app_table FROM (SELECT remote_table.a AS app_table FROM remote_table) AS b"#,
            ),
            (
                "SELECT MAX(app_table) from (SELECT a as app_table FROM app_table) b",
                r#"SELECT max(b.app_table) FROM (SELECT remote_table.a AS app_table FROM remote_table) AS b"#,
            ),
            // multiple occurrences of the same table in single aggregation expression
            (
                "SELECT COUNT(CASE WHEN a > 0 THEN a ELSE 0 END) FROM app_table",
                r#"SELECT count(CASE WHEN (remote_table.a > 0) THEN remote_table.a ELSE 0 END) FROM remote_table"#,
            ),
            // different tables in single aggregation expression
            (
                "SELECT COUNT(CASE WHEN appt.a > 0 THEN appt.a ELSE dft.a END) FROM app_table as appt, foo.df_table as dft",
                "SELECT count(CASE WHEN (appt.a > 0) THEN appt.a ELSE dft.a END) FROM remote_table AS appt JOIN remote_table AS dft"
            ),
        ];

        for test in agg_tests {
            test_sql(&ctx, test.0, test.1, false).await?;
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_rewrite_table_scans_alias() -> Result<()> {
        init_tracing();
        let ctx = get_test_df_context();

        let tests = vec![
            (
                "SELECT COUNT(app_table_a) FROM (SELECT a as app_table_a FROM app_table)",
                r#"SELECT count(app_table_a) FROM (SELECT remote_table.a AS app_table_a FROM remote_table)"#,
            ),
            (
                "SELECT app_table_a FROM (SELECT a as app_table_a FROM app_table)",
                r#"SELECT app_table_a FROM (SELECT remote_table.a AS app_table_a FROM remote_table)"#,
            ),
            (
                "SELECT aapp_table FROM (SELECT a as aapp_table FROM app_table)",
                r#"SELECT aapp_table FROM (SELECT remote_table.a AS aapp_table FROM remote_table)"#,
            ),
        ];

        for test in tests {
            test_sql(&ctx, test.0, test.1, false).await?;
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_rewrite_table_scans_unnest() -> Result<()> {
        init_tracing();
        let ctx = get_test_df_context();

        let tests = vec![
            (
                "SELECT UNNEST([1, 2, 2, 5, NULL]), b, c from app_table where a > 10 order by b limit 10;",
                r#"SELECT UNNEST(make_array(1, 2, 2, 5, NULL)) AS "UNNEST(make_array(Int64(1),Int64(2),Int64(2),Int64(5),NULL))", remote_table.b, remote_table.c FROM remote_table WHERE (remote_table.a > 10) ORDER BY remote_table.b ASC NULLS LAST LIMIT 10"#,
            ),
            (
                "SELECT UNNEST(app_table.d), b, c from app_table where a > 10 order by b limit 10;",
                r#"SELECT UNNEST(remote_table.d) AS "UNNEST(app_table.d)", remote_table.b, remote_table.c FROM remote_table WHERE (remote_table.a > 10) ORDER BY remote_table.b ASC NULLS LAST LIMIT 10"#,
            ),
            (
                "SELECT sum(b.x) AS total FROM (SELECT UNNEST(d) AS x from app_table where a > 0) AS b;",
                r#"SELECT sum(b.x) AS total FROM (SELECT UNNEST(remote_table.d) AS x FROM remote_table WHERE (remote_table.a > 0)) AS b"#,
            ),
        ];

        for test in tests {
            test_sql(&ctx, test.0, test.1, false).await?;
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_subquery_requires_partial_path() -> Result<()> {
        init_tracing();
        let ctx = get_test_df_context();
        let tests = vec![
            (
                "SELECT a FROM bar where a IN (SELECT a FROM bar)",
                r#"SELECT remote_db.remote_schema.remote_table.a FROM remote_db.remote_schema.remote_table WHERE remote_db.remote_schema.remote_table.a IN (SELECT a FROM remote_db.remote_schema.remote_table)"#,
                true,
            ),
            (
                "SELECT a FROM bar where a IN (SELECT a FROM bar)",
                r#"SELECT remote_db.remote_schema.remote_table.a FROM remote_db.remote_schema.remote_table WHERE remote_db.remote_schema.remote_table.a IN (SELECT remote_db.remote_schema.remote_table.a FROM remote_db.remote_schema.remote_table)"#,
                false,
            ),
        ];
        for test in tests {
            test_sql(&ctx, test.0, test.1, test.2).await?;
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_rewrite_outer_ref_columns() -> Result<()> {
        init_tracing();
        let ctx = get_test_df_context();
        let tests = vec![(
            "SELECT foo.df_table.a FROM bar JOIN foo.df_table ON foo.df_table.a = (SELECT bar.a FROM bar WHERE bar.a > foo.df_table.a)",
            r#"SELECT remote_table.a FROM remote_db.remote_schema.remote_table JOIN remote_table ON (remote_table.a = (SELECT a FROM remote_db.remote_schema.remote_table WHERE (remote_table.a > remote_table.a)))"#,
            true,
        )];
        for test in tests {
            test_sql(&ctx, test.0, test.1, test.2).await?;
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_rewrite_column_name_in_expr() -> Result<()> {
        init_tracing();
        let ctx = get_test_df_context();

        let tests = vec![
            (
                // Column alias name same as table name
                "SELECT app_table FROM (SELECT a app_table from app_table limit 100);",
                r#"SELECT app_table FROM (SELECT remote_table.a AS app_table FROM remote_table LIMIT 100)"#,
            ),
            (
                "SELECT a - 1, COUNT(*) AS c FROM app_table GROUP BY a - 1;",
                r#"SELECT (remote_table.a - 1), count(*) AS c FROM remote_table GROUP BY (remote_table.a - 1)"#,
            ),
        ];

        for test in tests {
            test_sql(&ctx, test.0, test.1, false).await?;
        }

        Ok(())
    }

    async fn test_sql(
        ctx: &SessionContext,
        sql_query: &str,
        expected_sql: &str,
        subquery_uses_partial_path: bool,
    ) -> Result<(), datafusion::error::DataFusionError> {
        let data_frame = ctx.sql(sql_query).await?;

        let mut known_rewrites = HashMap::new();
        let rewritten_plan = rewrite_table_scans(
            data_frame.logical_plan(),
            &mut known_rewrites,
            subquery_uses_partial_path,
            &mut None,
        )?;

        let unparsed_sql = plan_to_sql(&rewritten_plan)?;

        assert_eq!(
            format!("{unparsed_sql}"),
            expected_sql,
            "SQL under test: {}",
            sql_query
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_rewrite_table_scans_limit_offset() -> Result<()> {
        init_tracing();
        let ctx = get_test_df_context();
        let tests = vec![
            // Basic LIMIT
            (
                "SELECT a FROM foo.df_table LIMIT 5",
                r#"SELECT remote_table.a FROM remote_table LIMIT 5"#,
            ),
            // Basic OFFSET
            (
                "SELECT a FROM foo.df_table OFFSET 5",
                r#"SELECT remote_table.a FROM remote_table OFFSET 5"#,
            ),
            // OFFSET after LIMIT
            (
                "SELECT a FROM foo.df_table LIMIT 10 OFFSET 5",
                r#"SELECT remote_table.a FROM remote_table LIMIT 10 OFFSET 5"#,
            ),
            // LIMIT after OFFSET
            (
                "SELECT a FROM foo.df_table OFFSET 5 LIMIT 10",
                r#"SELECT remote_table.a FROM remote_table LIMIT 10 OFFSET 5"#,
            ),
            // Zero OFFSET
            (
                "SELECT a FROM foo.df_table OFFSET 0",
                r#"SELECT remote_table.a FROM remote_table OFFSET 0"#,
            ),
            // Zero LIMIT
            (
                "SELECT a FROM foo.df_table LIMIT 0",
                r#"SELECT remote_table.a FROM remote_table LIMIT 0"#,
            ),
            // Zero LIMIT and OFFSET
            (
                "SELECT a FROM foo.df_table LIMIT 0 OFFSET 0",
                r#"SELECT remote_table.a FROM remote_table LIMIT 0 OFFSET 0"#,
            ),
        ];
        for test in tests {
            test_sql(&ctx, test.0, test.1, false).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod collect_rewrites_tests {
    use crate::{SQLExecutor, SQLFederationProvider, SQLTableSource};

    use super::*;
    use async_trait::async_trait;
    use datafusion::{
        arrow::datatypes::{DataType, Field, Schema, SchemaRef},
        common::DFSchema,
        datasource::DefaultTableSource,
        execution::SendableRecordBatchStream,
        sql::unparser::dialect::{DefaultDialect, Dialect},
    };
    use datafusion_federation::FederatedTableProviderAdaptor;

    struct TestSQLExecutor {}

    #[async_trait]
    impl SQLExecutor for TestSQLExecutor {
        fn name(&self) -> &str {
            "test_sql_table_source"
        }

        fn compute_context(&self) -> Option<String> {
            None
        }

        fn dialect(&self) -> Arc<dyn Dialect> {
            Arc::new(DefaultDialect {})
        }

        fn execute(&self, _query: &str, _schema: SchemaRef) -> Result<SendableRecordBatchStream> {
            Err(DataFusionError::NotImplemented(
                "execute not implemented".to_string(),
            ))
        }

        async fn table_names(&self) -> Result<Vec<String>> {
            Err(DataFusionError::NotImplemented(
                "table inference not implemented".to_string(),
            ))
        }

        async fn get_table_schema(&self, _table_name: &str) -> Result<SchemaRef> {
            Err(DataFusionError::NotImplemented(
                "table inference not implemented".to_string(),
            ))
        }
    }

    fn create_test_table_scan() -> LogicalPlan {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
        ]));

        let sql_federation_provider =
            Arc::new(SQLFederationProvider::new(Arc::new(TestSQLExecutor {})));
        let table_source = Arc::new(
            SQLTableSource::new_with_schema(
                sql_federation_provider,
                "remote_table".to_string(),
                schema.clone(),
            )
            .expect("to have a valid SQLTableSource"),
        );
        let source = Arc::new(DefaultTableSource::new(Arc::new(
            FederatedTableProviderAdaptor::new(table_source),
        )));

        let df_schema =
            DFSchema::try_from(schema.as_ref().clone()).expect("to have a valid DFSchema");

        LogicalPlan::TableScan(logical_expr::TableScan {
            table_name: TableReference::from("foo.df_table"),
            source,
            projection: None,
            projected_schema: df_schema.into(),
            filters: vec![],
            fetch: None,
        })
    }

    #[test]
    fn test_collect_from_table_scan() -> Result<()> {
        let plan = create_test_table_scan();
        let mut known_rewrites = HashMap::new();

        collect_known_rewrites_from_plan(&plan, &mut known_rewrites)?;

        assert_eq!(known_rewrites.len(), 1);
        assert_eq!(
            known_rewrites.get(&TableReference::from("foo.df_table")),
            Some(&MultiPartTableReference::TableReference(
                TableReference::from("remote_table")
            ))
        );
        Ok(())
    }

    #[test]
    fn test_collect_from_scalar_subquery() -> Result<()> {
        let table_scan = create_test_table_scan();
        let subquery = Expr::ScalarSubquery(Subquery {
            subquery: Arc::new(table_scan),
            outer_ref_columns: vec![],
        });

        let mut known_rewrites = HashMap::new();
        collect_known_rewrites_from_expr(subquery, &mut known_rewrites)?;

        assert_eq!(known_rewrites.len(), 1);
        assert_eq!(
            known_rewrites.get(&TableReference::from("foo.df_table")),
            Some(&MultiPartTableReference::TableReference(
                TableReference::from("remote_table")
            ))
        );
        Ok(())
    }

    #[test]
    fn test_collect_from_binary_expr() -> Result<()> {
        let left = Expr::Column(Column::from_qualified_name("foo.df_table.a"));
        let right = Expr::Column(Column::from_qualified_name("foo.df_table.b"));
        let binary = Expr::BinaryExpr(BinaryExpr::new(
            Box::new(left),
            datafusion::logical_expr::Operator::Eq,
            Box::new(right),
        ));

        let mut known_rewrites = HashMap::new();
        collect_known_rewrites_from_expr(binary, &mut known_rewrites)?;

        // Column expressions don't generate rewrites on their own
        assert_eq!(known_rewrites.len(), 0);
        Ok(())
    }

    #[test]
    fn test_collect_from_case_expression() -> Result<()> {
        let col = Expr::Column(Column::from_qualified_name("foo.df_table.a"));
        let case = Expr::Case(Case::new(
            Some(Box::new(col.clone())),
            vec![(
                Box::new(Expr::Literal(datafusion::scalar::ScalarValue::Int64(Some(
                    1,
                )))),
                Box::new(col.clone()),
            )],
            Some(Box::new(col)),
        ));

        let mut known_rewrites = HashMap::new();
        collect_known_rewrites_from_expr(case, &mut known_rewrites)?;

        // Column expressions don't generate rewrites on their own
        assert_eq!(known_rewrites.len(), 0);
        Ok(())
    }

    #[test]
    fn test_collect_from_exists_subquery() -> Result<()> {
        let table_scan = create_test_table_scan();
        let exists = Expr::Exists(Exists::new(
            Subquery {
                subquery: Arc::new(table_scan),
                outer_ref_columns: vec![],
            },
            false,
        ));

        let mut known_rewrites = HashMap::new();
        collect_known_rewrites_from_expr(exists, &mut known_rewrites)?;

        assert_eq!(known_rewrites.len(), 1);
        assert_eq!(
            known_rewrites.get(&TableReference::from("foo.df_table")),
            Some(&MultiPartTableReference::TableReference(
                TableReference::from("remote_table")
            ))
        );
        Ok(())
    }
}
