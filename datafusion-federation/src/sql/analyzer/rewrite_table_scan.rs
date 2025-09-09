use std::sync::Arc;

use datafusion::{
    common::{
        tree_node::{Transformed, TreeNode},
        Column, HashMap, RecursionUnnestOption, UnnestOptions,
    },
    error::DataFusionError,
    logical_expr::{
        self, build_join_schema,
        expr::{Alias, Exists, InSubquery},
        Aggregate, Expr, Join, LogicalPlan, LogicalPlanBuilder, Projection, Subquery,
        SubqueryAlias, Union, Window,
    },
    sql::TableReference,
};

use crate::get_table_source;

use crate::sql::{table_reference::MultiPartTableReference, SQLTableSource};

type Result<T> = std::result::Result<T, datafusion::error::DataFusionError>;

/// Rewrite LogicalPlan's table scans and expressions to use the federated table name.
#[derive(Debug)]
pub struct RewriteTableScanAnalyzer;

impl RewriteTableScanAnalyzer {
    pub fn rewrite(
        plan: LogicalPlan,
        known_rewrites: &HashMap<TableReference, MultiPartTableReference>,
    ) -> Result<LogicalPlan> {
        // In f_down, rewrite the table scans in all LogicalPlan nodes (including subqueries).
        let rewrite_table_scans = |plan: LogicalPlan| {
            match plan {
                LogicalPlan::TableScan(mut table_scan) => {
                    let Some(federated_source) = get_table_source(&table_scan.source)? else {
                        // Not a federated source
                        return Ok(Transformed::no(LogicalPlan::TableScan(table_scan)));
                    };

                    let Some(sql_table_source) =
                        federated_source.as_any().downcast_ref::<SQLTableSource>()
                    else {
                        // Not a SQLTableSource (is this possible?)
                        return Ok(Transformed::no(LogicalPlan::TableScan(table_scan)));
                    };

                    let MultiPartTableReference::TableReference(remote_table_name) =
                        sql_table_source.table_reference()
                    else {
                        // If the remote table name is a MultiPartTableReference we will not rewrite it here, but rewrite it after the final unparsing on the AST directly.
                        return Ok(Transformed::no(LogicalPlan::TableScan(table_scan)));
                    };

                    // Rewrite the schema of this node to have the remote table as the qualifier.
                    let new_schema = Arc::unwrap_or_clone(table_scan.projected_schema)
                        .replace_qualifier(remote_table_name.clone());
                    table_scan.projected_schema = Arc::new(new_schema);
                    table_scan.table_name = remote_table_name.clone();

                    Ok(Transformed::yes(LogicalPlan::TableScan(table_scan)))
                }
                _ => Ok(Transformed::no(plan)),
            }
        };

        // In f_up, rewrite the column names in the expressions.
        let rewrite_column_names_in_expressions =
            |plan: LogicalPlan| -> Result<Transformed<LogicalPlan>> {
                let plan = match plan {
                    LogicalPlan::Unnest(unnest) => rewrite_unnest_plan(unnest, known_rewrites)?,
                    _ => plan,
                };

                let plan = plan.map_expressions(|expr| {
                    expr.transform_up(|expr| {
                        #[expect(deprecated)]
                        match expr {
                            Expr::Column(col) => rewrite_column(col, known_rewrites)
                                .map(|t| t.update_data(Expr::Column)),
                            Expr::Alias(alias) => match &alias.relation {
                                Some(relation) => {
                                    let Some(rewrite) =
                                        known_rewrites.get(relation).and_then(|rewrite| {
                                            match rewrite {
                                                MultiPartTableReference::TableReference(
                                                    rewrite,
                                                ) => Some(rewrite),
                                                _ => None,
                                            }
                                        })
                                    else {
                                        return Ok(Transformed::no(Expr::Alias(alias)));
                                    };

                                    Ok(Transformed::yes(Expr::Alias(Alias::new(
                                        *alias.expr,
                                        Some(rewrite.clone()),
                                        alias.name,
                                    ))))
                                }
                                None => Ok(Transformed::no(Expr::Alias(alias))),
                            },
                            Expr::Wildcard { qualifier, options } => {
                                if let Some(rewrite) = qualifier
                                    .as_ref()
                                    .and_then(|q| known_rewrites.get(q))
                                    .and_then(|rewrite| match rewrite {
                                        MultiPartTableReference::TableReference(rewrite) => {
                                            Some(rewrite)
                                        }
                                        _ => None,
                                    })
                                {
                                    Ok(Transformed::yes(Expr::Wildcard {
                                        qualifier: Some(rewrite.clone()),
                                        options,
                                    }))
                                } else {
                                    Ok(Transformed::no(Expr::Wildcard { qualifier, options }))
                                }
                            }
                            // We can't match directly on the outer ref columns until https://github.com/apache/datafusion/issues/16147 is fixed.
                            Expr::ScalarSubquery(Subquery {
                                outer_ref_columns,
                                subquery,
                                spans,
                            }) => {
                                let outer_ref_columns = rewrite_outer_reference_columns(
                                    outer_ref_columns,
                                    known_rewrites,
                                )?;

                                Ok(Transformed::yes(Expr::ScalarSubquery(Subquery {
                                    outer_ref_columns,
                                    subquery,
                                    spans,
                                })))
                            }
                            Expr::Exists(Exists {
                                subquery:
                                    Subquery {
                                        subquery,
                                        outer_ref_columns,
                                        spans,
                                    },
                                negated,
                            }) => {
                                let outer_ref_columns = rewrite_outer_reference_columns(
                                    outer_ref_columns,
                                    known_rewrites,
                                )?;

                                Ok(Transformed::yes(Expr::Exists(Exists {
                                    subquery: Subquery {
                                        outer_ref_columns,
                                        subquery,
                                        spans,
                                    },
                                    negated,
                                })))
                            }
                            Expr::InSubquery(InSubquery {
                                subquery:
                                    Subquery {
                                        outer_ref_columns,
                                        subquery,
                                        spans,
                                    },
                                expr,
                                negated,
                            }) => {
                                let outer_ref_columns = rewrite_outer_reference_columns(
                                    outer_ref_columns,
                                    known_rewrites,
                                )?;

                                Ok(Transformed::yes(Expr::InSubquery(InSubquery {
                                    expr,
                                    subquery: Subquery {
                                        outer_ref_columns,
                                        subquery,
                                        spans,
                                    },
                                    negated,
                                })))
                            }
                            _ => Ok(Transformed::no(expr)),
                        }
                    })
                })?;

                // Recalculate the schemas now that all of the inner expressions have been rewritten.
                plan.map_data(|plan| match plan {
                    LogicalPlan::Aggregate(aggr) => Ok(LogicalPlan::Aggregate(Aggregate::try_new(
                        aggr.input,
                        aggr.group_expr,
                        aggr.aggr_expr,
                    )?)),
                    LogicalPlan::Window(window) => Ok(LogicalPlan::Window(Window::try_new(
                        window.window_expr,
                        window.input,
                    )?)),
                    LogicalPlan::Projection(projection) => Ok(LogicalPlan::Projection(
                        Projection::try_new(projection.expr, projection.input)?,
                    )),
                    LogicalPlan::Join(join) => {
                        let join_schema = build_join_schema(
                            join.left.schema(),
                            join.right.schema(),
                            &join.join_type,
                        )?;

                        Ok(LogicalPlan::Join(Join {
                            left: join.left,
                            right: join.right,
                            on: join.on,
                            filter: join.filter,
                            join_type: join.join_type,
                            join_constraint: join.join_constraint,
                            schema: Arc::new(join_schema),
                            null_equality: join.null_equality,
                        }))
                    }
                    LogicalPlan::SubqueryAlias(subquery_alias) => Ok(LogicalPlan::SubqueryAlias(
                        SubqueryAlias::try_new(subquery_alias.input, subquery_alias.alias)?,
                    )),
                    LogicalPlan::Union(union) => Ok(LogicalPlan::Union(
                        Union::try_new_with_loose_types(union.inputs)?,
                    )),
                    plan => Ok(plan),
                })
            };

        plan.transform_down_up_with_subqueries(
            rewrite_table_scans,
            rewrite_column_names_in_expressions,
        )
        .map(|t| t.data)
    }
}

fn rewrite_outer_reference_columns(
    outer_ref_columns: Vec<Expr>,
    known_rewrites: &HashMap<TableReference, MultiPartTableReference>,
) -> Result<Vec<Expr>> {
    let outer_ref_columns = outer_ref_columns
        .into_iter()
        .map(|e| {
            let Expr::OuterReferenceColumn(dt, col) = e else {
                return Ok(e);
            };

            let rewritten_col = rewrite_column(col, known_rewrites)?.data;

            Ok(Expr::OuterReferenceColumn(dt, rewritten_col))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(outer_ref_columns)
}

fn rewrite_column(
    mut col: Column,
    known_rewrites: &HashMap<TableReference, MultiPartTableReference>,
) -> Result<Transformed<Column>> {
    if let Some(rewrite) = col
        .relation
        .as_ref()
        .and_then(|r| known_rewrites.get(r))
        .and_then(|rewrite| match rewrite {
            MultiPartTableReference::TableReference(rewrite) => Some(rewrite),
            _ => None,
        })
    {
        Ok(Transformed::yes(Column::new(
            Some(rewrite.clone()),
            &col.name,
        )))
    } else {
        // This prevent over-eager rewrite and only pass the column into below rewritten
        // rule like MAX(...)
        if col.relation.is_some() {
            return Ok(Transformed::no(col));
        }

        // Check if any of the rewrites match any substring in col.name, and replace that part of the string if so.
        // This will handles cases like "MAX(foo.df_table.a)" -> "MAX(remote_table.a)"
        if let Some(new_name) = rewrite_column_name(&col.name, known_rewrites) {
            Ok(Transformed::yes(Column::new(col.relation.take(), new_name)))
        } else {
            Ok(Transformed::no(col))
        }
    }
}

/// Rewrite an unnest plan to use the original federated table name.
/// In a standard unnest plan, column names are typically referenced in projection columns by wrapping them
/// in aliases such as "UNNEST(table_name.column_name)". `rewrite_table_scans_in_expr` does not handle alias
/// rewriting so we manually collect the rewritten unnest column names/aliases and update the projection
/// plan to ensure that the aliases reflect the new names.
fn rewrite_unnest_plan(
    unnest: logical_expr::Unnest,
    known_rewrites: &HashMap<TableReference, MultiPartTableReference>,
) -> Result<LogicalPlan> {
    let input = unnest.input;

    let mut known_unnest_rewrites: HashMap<String, String> = HashMap::new();

    // `exec_columns` represent columns to run UNNEST on: rewrite them and collect new names
    let unnest_columns = unnest
        .exec_columns
        .into_iter()
        .map(|c: Column| {
            let original_column_name = c.name.clone();
            let column = rewrite_column(c, known_rewrites)?.data;
            known_unnest_rewrites.insert(original_column_name, column.name.clone());
            Ok(column)
        })
        .collect::<Result<Vec<Column>>>()?;

    let LogicalPlan::Projection(projection) = Arc::unwrap_or_clone(input) else {
        return Err(DataFusionError::Plan(
            "The input to the unnest plan should be a projection plan".to_string(),
        ));
    };

    // rewrite aliases in inner projection; columns were rewritten via `rewrite_column`
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

    let unnest_options = rewrite_unnest_options(unnest.options, known_rewrites);

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
    mut options: UnnestOptions,
    known_rewrites: &HashMap<TableReference, MultiPartTableReference>,
) -> UnnestOptions {
    options
        .recursions
        .iter_mut()
        .for_each(|x: &mut RecursionUnnestOption| {
            if let Some(new_name) = rewrite_column_name(&x.input_column.name, known_rewrites) {
                x.input_column.name = new_name;
            }

            if let Some(new_name) = rewrite_column_name(&x.output_column.name, known_rewrites) {
                x.output_column.name = new_name;
            }
        });
    options
}

/// Checks if any of the rewrites match any substring in col_name, and replace that part of the string if so.
/// This handles cases like "MAX(foo.df_table.a)" -> "MAX(remote_table.a)"
/// Returns the rewritten name if any rewrite was applied, otherwise None.
fn rewrite_column_name(
    col_name: &str,
    known_rewrites: &HashMap<TableReference, MultiPartTableReference>,
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
                let rewrite_string = rewrite.to_string();
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

// The function replaces occurrences of table_ref_str in col_name with the new name defined by rewrite.
// The name to rewrite should NOT be a substring of another name.
// Supports multiple occurrences of table_ref_str in col_name.
pub fn rewrite_column_name_in_expr(
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

#[cfg(test)]
mod tests {
    use crate::sql::analyzer::collect_known_rewrites;
    use crate::sql::table::SQLTable;
    use crate::sql::table_reference::MultiPartTableReference;
    use crate::sql::{RemoteTableRef, SQLExecutor, SQLFederationProvider, SQLTableSource};
    use crate::FederatedTableProviderAdaptor;
    use async_trait::async_trait;
    use datafusion::arrow::datatypes::{Schema, SchemaRef};
    use datafusion::execution::SendableRecordBatchStream;
    use datafusion::sql::unparser::dialect::Dialect;
    use datafusion::sql::unparser::plan_to_sql;
    use datafusion::{
        arrow::datatypes::{DataType, Field},
        catalog::{MemorySchemaProvider, SchemaProvider},
        common::Column,
        datasource::{DefaultTableSource, TableProvider},
        execution::context::SessionContext,
        logical_expr::LogicalPlanBuilder,
        prelude::Expr,
    };

    use super::*;

    struct TestExecutor;

    #[async_trait]
    impl SQLExecutor for TestExecutor {
        fn name(&self) -> &str {
            "TestExecutor"
        }

        fn compute_context(&self) -> Option<String> {
            None
        }

        fn dialect(&self) -> Arc<dyn Dialect> {
            unimplemented!()
        }

        fn execute(&self, _query: &str, _schema: SchemaRef) -> Result<SendableRecordBatchStream> {
            unimplemented!()
        }

        async fn table_names(&self) -> Result<Vec<String>> {
            unimplemented!()
        }

        async fn get_table_schema(&self, _table_name: &str) -> Result<SchemaRef> {
            unimplemented!()
        }
    }

    #[derive(Debug)]
    struct TestTable {
        name: RemoteTableRef,
        schema: SchemaRef,
    }

    impl TestTable {
        fn new(name: String, schema: SchemaRef) -> Self {
            TestTable {
                name: name.try_into().unwrap(),
                schema,
            }
        }
    }

    impl SQLTable for TestTable {
        fn table_reference(&self) -> MultiPartTableReference {
            MultiPartTableReference::from(&self.name)
        }

        fn schema(&self) -> datafusion::arrow::datatypes::SchemaRef {
            self.schema.clone()
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn get_test_table_provider() -> Arc<dyn TableProvider> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
            Field::new("c", DataType::Date32, false),
        ]));
        let table = Arc::new(TestTable::new("remote_table".to_string(), schema));
        let provider = Arc::new(SQLFederationProvider::new(Arc::new(TestExecutor)));
        let table_source = Arc::new(SQLTableSource { provider, table });
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

        ctx
    }

    #[test]
    fn test_rewrite_table_scans_basic() -> Result<()> {
        let plan = LogicalPlanBuilder::scan("foo.df_table", get_test_table_source(), None)?
            .project(vec![
                Expr::Column(Column::from_qualified_name("foo.df_table.a")),
                Expr::Column(Column::from_qualified_name("foo.df_table.b")),
                Expr::Column(Column::from_qualified_name("foo.df_table.c")),
            ])?
            .build()?;

        let known_rewrites = collect_known_rewrites(&plan)?;
        let rewritten_plan = RewriteTableScanAnalyzer::rewrite(plan, &known_rewrites)?;

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
                "SELECT count(CASE WHEN (appt.a > 0) THEN appt.a ELSE dft.a END) FROM remote_table AS appt CROSS JOIN remote_table AS dft"
            ),
        ];

        for test in agg_tests {
            test_sql(&ctx, test.0, test.1).await?;
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
            test_sql(&ctx, test.0, test.1).await?;
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_rewrite_table_scans_preserve_existing_alias() -> Result<()> {
        init_tracing();
        let ctx = get_test_df_context();

        let tests = vec![
            (
                "SELECT b.a AS app_table_a FROM app_table AS b",
                r#"SELECT b.a AS app_table_a FROM remote_table AS b"#,
            ),
            (
                "SELECT app_table_a FROM (SELECT a as app_table_a FROM app_table AS b)",
                r#"SELECT app_table_a FROM (SELECT b.a AS app_table_a FROM remote_table AS b)"#,
            ),
            (
                "SELECT COUNT(b.a) FROM app_table AS b",
                r#"SELECT count(b.a) FROM remote_table AS b"#,
            ),
        ];

        for test in tests {
            test_sql(&ctx, test.0, test.1).await?;
        }

        Ok(())
    }

    async fn test_sql(ctx: &SessionContext, sql_query: &str, expected_sql: &str) -> Result<()> {
        let data_frame = ctx.sql(sql_query).await?;

        println!("before optimization: \n{:#?}", data_frame.logical_plan());

        let plan = data_frame.logical_plan().clone();
        let known_rewrites = collect_known_rewrites(&plan)?;
        let rewritten_plan = RewriteTableScanAnalyzer::rewrite(plan, &known_rewrites)?;

        println!("rewritten_plan: \n{:#?}", rewritten_plan);

        let unparsed_sql = plan_to_sql(&rewritten_plan)?;

        println!("unparsed_sql: \n{unparsed_sql}");

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
            test_sql(&ctx, test.0, test.1).await?;
        }

        Ok(())
    }

    fn get_multipart_test_table_provider() -> Arc<dyn TableProvider> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
            Field::new("c", DataType::Date32, false),
        ]));
        let table = Arc::new(TestTable::new("default.remote_table".to_string(), schema));
        let provider = Arc::new(SQLFederationProvider::new(Arc::new(TestExecutor)));
        let table_source = Arc::new(SQLTableSource { provider, table });
        Arc::new(FederatedTableProviderAdaptor::new(table_source))
    }

    fn get_multipart_test_df_context() -> SessionContext {
        let ctx = SessionContext::new();
        let catalog = ctx
            .catalog("datafusion")
            .expect("default catalog is datafusion");
        let foo_schema = Arc::new(MemorySchemaProvider::new()) as Arc<dyn SchemaProvider>;
        catalog
            .register_schema("foo", Arc::clone(&foo_schema))
            .expect("to register schema");
        foo_schema
            .register_table("df_table".to_string(), get_multipart_test_table_provider())
            .expect("to register table");

        let public_schema = catalog
            .schema("public")
            .expect("public schema should exist");
        public_schema
            .register_table("app_table".to_string(), get_multipart_test_table_provider())
            .expect("to register table");

        ctx
    }

    #[tokio::test]
    async fn test_rewrite_multipart_table() -> Result<()> {
        init_tracing();
        let ctx = get_multipart_test_df_context();

        let tests = vec![
            (
                "SELECT MAX(a) FROM foo.df_table",
                r#"SELECT max(remote_table.a) FROM "default".remote_table"#,
            ),
            (
                "SELECT foo.df_table.a FROM foo.df_table",
                r#"SELECT a FROM "default".remote_table"#,
            ),
            (
                "SELECT MIN(a) FROM foo.df_table",
                r#"SELECT min(remote_table.a) FROM "default".remote_table"#,
            ),
            (
                "SELECT AVG(a) FROM foo.df_table",
                r#"SELECT avg(remote_table.a) FROM "default".remote_table"#,
            ),
            (
                "SELECT COUNT(a) as cnt FROM foo.df_table",
                r#"SELECT count(remote_table.a) AS cnt FROM "default".remote_table"#,
            ),
            (
                "SELECT app_table from (SELECT a as app_table FROM app_table) b",
                r#"SELECT b.app_table FROM (SELECT remote_table.a AS app_table FROM "default".remote_table) AS b"#,
            ),
            (
                "SELECT MAX(app_table) from (SELECT a as app_table FROM app_table) b",
                r#"SELECT max(b.app_table) FROM (SELECT remote_table.a AS app_table FROM "default".remote_table) AS b"#,
            ),
            (
                "SELECT COUNT(app_table_a) FROM (SELECT a as app_table_a FROM app_table)",
                r#"SELECT count(app_table_a) FROM (SELECT remote_table.a AS app_table_a FROM "default".remote_table)"#,
            ),
            (
                "SELECT app_table_a FROM (SELECT a as app_table_a FROM app_table)",
                r#"SELECT app_table_a FROM (SELECT remote_table.a AS app_table_a FROM "default".remote_table)"#,
            ),
            (
                "SELECT aapp_table FROM (SELECT a as aapp_table FROM app_table)",
                r#"SELECT aapp_table FROM (SELECT remote_table.a AS aapp_table FROM "default".remote_table)"#,
            ),
        ];

        for test in tests {
            test_sql(&ctx, test.0, test.1).await?;
        }

        Ok(())
    }
}
