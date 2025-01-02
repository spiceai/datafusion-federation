use std::{collections::HashMap, vec};

use datafusion::sql::{
    sqlparser::ast::{
        self, Ident, ObjectName, Query, SelectItem, SetExpr, TableFactor, TableWithJoins,
    },
    TableReference,
};
use datafusion_federation::table_reference::MultiTableReference;

/// Rewrites table references in a SQL AST to use the original federated table names.
/// This is similar to rewrite_table_scans but operates on the sqlparser AST instead
/// of DataFusion logical plans.
pub(crate) fn rewrite_multi_part_statement(
    statement: &mut ast::Statement,
    known_rewrites: &HashMap<TableReference, MultiTableReference>,
) {
    let known_rewrites = known_rewrites
        .iter()
        .map(|(k, v)| (table_reference_to_object_name(k), v.clone()))
        .collect();
    if let ast::Statement::Query(query) = statement {
        rewrite_multi_part_table_reference_in_query(&mut *query, &known_rewrites);
    }
}

fn rewrite_multi_part_table_with_joins(
    table_with_joins: &mut Vec<TableWithJoins>,
    known_rewrites: &HashMap<ObjectName, MultiTableReference>,
) {
    for table in table_with_joins {
        match &mut table.relation {
            TableFactor::Table { name, .. } => {
                if let Some(rewrite) = known_rewrites.get(name) {
                    // Create new object name from the rewritten table reference
                    let new_name = ObjectName(
                        rewrite
                            .parts
                            .iter()
                            .map(|p| Ident::new(p.to_string()))
                            .collect(),
                    );
                    *name = new_name;
                }
            }
            TableFactor::Derived { subquery, .. } => {
                // Recursively rewrite any table references in subqueries
                rewrite_multi_part_table_reference_in_query(subquery, known_rewrites);
            }
            TableFactor::TableFunction { .. } => {
                // Table functions don't have table references to rewrite
            }
            TableFactor::UNNEST { .. } => {
                // UNNEST doesn't have table references to rewrite
            }
            TableFactor::NestedJoin { .. }
            | TableFactor::Function { .. }
            | TableFactor::JsonTable { .. }
            | TableFactor::Pivot { .. }
            | TableFactor::Unpivot { .. }
            | TableFactor::MatchRecognize { .. } => {
                // TODO: Handle these table factors if needed
            }
        }
    }
}

/// Rewrites table references within a query expression
fn rewrite_multi_part_table_reference_in_query(
    query: &mut Query,
    known_rewrites: &HashMap<ObjectName, MultiTableReference>,
) {
    rewrite_multi_part_table_reference_in_set_expr(&mut query.body, known_rewrites);

    // Handle WITH clause if present
    if let Some(with) = &mut query.with {
        for cte in &mut with.cte_tables {
            rewrite_multi_part_table_reference_in_query(&mut cte.query, known_rewrites);
        }
    }
}

fn rewrite_multi_part_table_reference_in_set_expr(
    set_expr: &mut SetExpr,
    known_rewrites: &HashMap<ObjectName, MultiTableReference>,
) {
    match set_expr {
        SetExpr::Select(select) => {
            // Rewrite table references in the FROM clause
            rewrite_multi_part_table_with_joins(&mut select.from, known_rewrites);

            // Rewrite any subqueries in WHERE clause
            if let Some(selection) = &mut select.selection {
                rewrite_multi_part_table_reference_in_expr(selection, known_rewrites);
            }

            // Rewrite any subqueries in the projection list
            for item in &mut select.projection {
                match item {
                    SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. } => {
                        rewrite_multi_part_table_reference_in_expr(expr, known_rewrites);
                    }
                    _ => {}
                }
            }
        }
        SetExpr::Query(subquery) => {
            rewrite_multi_part_table_reference_in_query(&mut *subquery, known_rewrites);
        }
        SetExpr::SetOperation { left, right, .. } => {
            rewrite_multi_part_table_reference_in_set_expr(left, known_rewrites);
            rewrite_multi_part_table_reference_in_set_expr(right, known_rewrites);
        }
        SetExpr::Values(_) | SetExpr::Insert(_) | SetExpr::Update(_) | SetExpr::Table(_) => (),
    }
}

/// Rewrites table references within expressions
fn rewrite_multi_part_table_reference_in_expr(
    expr: &mut ast::Expr,
    known_rewrites: &HashMap<ObjectName, MultiTableReference>,
) {
    match expr {
        ast::Expr::CompoundIdentifier(idents) => {
            // This should be impossible, but handle it defensively
            if idents.len() < 2 {
                return;
            }

            // Get the column name (last identifier) and table name (all other identifiers)
            let column_name = idents.last().cloned();
            let obj_name = ObjectName(idents[..idents.len() - 1].to_vec());

            if let Some(rewrite) = known_rewrites.get(&obj_name) {
                // Rewrite the table parts
                let mut new_idents: Vec<Ident> = rewrite
                    .parts
                    .iter()
                    .map(|p| Ident::new(p.to_string()))
                    .collect();

                // Add back the column name
                if let Some(col) = column_name {
                    new_idents.push(col);
                }

                *idents = new_idents;
            }
        }
        ast::Expr::Subquery(query) => {
            rewrite_multi_part_table_reference_in_query(query, known_rewrites);
        }
        ast::Expr::BinaryOp { left, right, .. } => {
            rewrite_multi_part_table_reference_in_expr(left, known_rewrites);
            rewrite_multi_part_table_reference_in_expr(right, known_rewrites);
        }
        ast::Expr::UnaryOp { expr, .. } => {
            rewrite_multi_part_table_reference_in_expr(expr, known_rewrites);
        }
        ast::Expr::Function(_func) => {
            // TODO: Implement this
            // for arg in &mut func.args {
            //     rewrite_multi_part_table_reference_in_expr(arg, known_rewrites);
            // }
        }
        ast::Expr::Case {
            operand,
            conditions,
            results,
            else_result,
            ..
        } => {
            if let Some(op) = operand {
                rewrite_multi_part_table_reference_in_expr(op, known_rewrites);
            }
            for condition in conditions {
                rewrite_multi_part_table_reference_in_expr(condition, known_rewrites);
            }
            for result in results {
                rewrite_multi_part_table_reference_in_expr(result, known_rewrites);
            }
            if let Some(else_res) = else_result {
                rewrite_multi_part_table_reference_in_expr(else_res, known_rewrites);
            }
        }
        _ => {}
    }
}

fn table_reference_to_object_name(table_reference: &TableReference) -> ObjectName {
    match table_reference {
        TableReference::Bare { table } => ObjectName(vec![Ident::new(table.to_string())]),
        TableReference::Partial { schema, table } => ObjectName(vec![
            Ident::new(schema.to_string()),
            Ident::new(table.to_string()),
        ]),
        TableReference::Full {
            catalog,
            schema,
            table,
        } => ObjectName(vec![
            Ident::new(catalog.to_string()),
            Ident::new(schema.to_string()),
            Ident::new(table.to_string()),
        ]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::sql::sqlparser::dialect::GenericDialect;
    use datafusion::sql::sqlparser::parser::Parser;
    use std::collections::HashMap;

    fn parse_sql(sql: &str) -> ast::Statement {
        let dialect = GenericDialect {};
        let mut parser = Parser::new(&dialect)
            .try_with_sql(sql)
            .expect("Failed to parse SQL");
        parser.parse_statement().expect("Failed to parse statement")
    }

    fn create_test_rewrites() -> HashMap<TableReference, MultiTableReference> {
        let mut rewrites = HashMap::new();

        rewrites.insert(
            TableReference::Bare {
                table: "test_table".into(),
            },
            MultiTableReference {
                parts: vec!["catalog".into(), "schema".into(), "real_table".into()],
            },
        );

        rewrites.insert(
            TableReference::Partial {
                schema: "test_schema".into(),
                table: "test_table2".into(),
            },
            MultiTableReference {
                parts: vec![
                    "other_catalog".into(),
                    "other_schema".into(),
                    "real_table2".into(),
                ],
            },
        );

        rewrites
    }

    #[test]
    fn test_rewrite_simple_query() {
        let mut stmt = parse_sql("SELECT * FROM test_table");
        let rewrites = create_test_rewrites();

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        assert_eq!(stmt.to_string(), "SELECT * FROM catalog.schema.real_table");
    }

    #[test]
    fn test_rewrite_compound_identifier() {
        let mut stmt =
            parse_sql("SELECT test_schema.test_table2.column FROM test_schema.test_table2");
        let rewrites = create_test_rewrites();

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT other_catalog.other_schema.real_table2.column FROM other_catalog.other_schema.real_table2"
        );
    }

    #[test]
    fn test_rewrite_deep_compound_identifier() {
        let mut stmt = parse_sql("SELECT level1.level2.level3.column FROM level1.level2.level3");

        let mut rewrites = HashMap::new();
        rewrites.insert(
            TableReference::Full {
                catalog: "level1".into(),
                schema: "level2".into(),
                table: "level3".into(),
            },
            MultiTableReference {
                parts: vec![
                    "new_level1".into(),
                    "new_level2".into(),
                    "new_level3".into(),
                    "new_level4".into(),
                    "new_level5".into(),
                ],
            },
        );

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT new_level1.new_level2.new_level3.new_level4.new_level5.column FROM new_level1.new_level2.new_level3.new_level4.new_level5"
        );
    }

    #[test]
    fn test_rewrite_query_with_deep_table_reference() {
        let mut stmt = parse_sql("SELECT * FROM part1.part2.part3");

        let mut rewrites = HashMap::new();
        rewrites.insert(
            TableReference::Full {
                catalog: "part1".into(),
                schema: "part2".into(),
                table: "part3".into(),
            },
            MultiTableReference {
                parts: vec![
                    "new1".into(),
                    "new2".into(),
                    "new3".into(),
                    "new4".into(),
                    "new5".into(),
                ],
            },
        );

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        assert_eq!(stmt.to_string(), "SELECT * FROM new1.new2.new3.new4.new5");
    }

    #[test]
    fn test_table_reference_to_object_name() {
        // Test full table reference
        let table_ref = TableReference::Full {
            catalog: "cat".into(),
            schema: "sch".into(),
            table: "tbl".into(),
        };
        let obj_name = table_reference_to_object_name(&table_ref);
        assert_eq!(obj_name.to_string(), "cat.sch.tbl");

        // Test partial table reference
        let partial_ref = TableReference::Partial {
            schema: "sch".into(),
            table: "tbl".into(),
        };
        let obj_name = table_reference_to_object_name(&partial_ref);
        assert_eq!(obj_name.to_string(), "sch.tbl");

        // Test bare table reference
        let bare_ref = TableReference::Bare {
            table: "tbl".into(),
        };
        let obj_name = table_reference_to_object_name(&bare_ref);
        assert_eq!(obj_name.to_string(), "tbl");
    }
}
