use std::vec;

use crate::table_reference::MultiTableReference;
use datafusion::{
    common::HashMap,
    sql::{
        sqlparser::ast::{
            self, Ident, ObjectName, Query, SelectItem, SetExpr, TableFactor, TableWithJoins,
        },
        TableReference,
    },
};

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
        rewrite_multi_part_table_factor(&mut table.relation, known_rewrites);

        for join in &mut table.joins {
            rewrite_multi_part_table_factor(&mut join.relation, known_rewrites);

            match &mut join.join_operator {
                ast::JoinOperator::RightAnti(join_constraint)
                | ast::JoinOperator::LeftAnti(join_constraint)
                | ast::JoinOperator::RightSemi(join_constraint)
                | ast::JoinOperator::LeftSemi(join_constraint)
                | ast::JoinOperator::FullOuter(join_constraint)
                | ast::JoinOperator::RightOuter(join_constraint)
                | ast::JoinOperator::Inner(join_constraint)
                | ast::JoinOperator::LeftOuter(join_constraint)
                | ast::JoinOperator::Semi(join_constraint)
                | ast::JoinOperator::Anti(join_constraint) => {
                    if let ast::JoinConstraint::On(expr) = join_constraint {
                        rewrite_multi_part_table_reference_in_expr(expr, known_rewrites);
                    }
                }
                ast::JoinOperator::OuterApply
                | ast::JoinOperator::CrossApply
                | ast::JoinOperator::CrossJoin => {}
                ast::JoinOperator::AsOf {
                    match_condition,
                    constraint,
                } => {
                    rewrite_multi_part_table_reference_in_expr(match_condition, known_rewrites);
                    if let ast::JoinConstraint::On(expr) = constraint {
                        rewrite_multi_part_table_reference_in_expr(expr, known_rewrites);
                    }
                }
            }
        }
    }
}

fn rewrite_object_name(
    object_name: &mut ObjectName,
    known_rewrites: &HashMap<ObjectName, MultiTableReference>,
) {
    if let Some(rewrite) = known_rewrites.get(object_name) {
        // Create new object name from the rewritten table reference
        let new_name = ObjectName(
            rewrite
                .parts
                .iter()
                .map(|p| Ident::new(p.to_string()))
                .collect(),
        );
        *object_name = new_name;
    }
}

fn rewrite_multi_part_table_factor(
    table_factor: &mut TableFactor,
    known_rewrites: &HashMap<ObjectName, MultiTableReference>,
) {
    match table_factor {
        TableFactor::Table { name, .. } => {
            rewrite_object_name(name, known_rewrites);
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
        | TableFactor::MatchRecognize { .. }
        | TableFactor::OpenJsonTable { .. } => {
            // TODO: Handle these table factors if needed
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
        ast::Expr::Function(func) => {
            if let Some(filter) = &mut func.filter {
                rewrite_multi_part_table_reference_in_expr(filter, known_rewrites);
            }
            match &mut func.args {
                ast::FunctionArguments::None => (),
                ast::FunctionArguments::Subquery(query) => {
                    rewrite_multi_part_table_reference_in_query(query, known_rewrites);
                }
                ast::FunctionArguments::List(function_argument_list) => {
                    for arg in function_argument_list.args.iter_mut() {
                        match arg {
                            ast::FunctionArg::Named {
                                arg: ast::FunctionArgExpr::Expr(arg),
                                ..
                            } => {
                                rewrite_multi_part_table_reference_in_expr(arg, known_rewrites);
                            }
                            ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(arg)) => {
                                rewrite_multi_part_table_reference_in_expr(arg, known_rewrites);
                            }
                            _ => {}
                        }
                    }
                }
            }
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
        ast::Expr::Nested(nested) => {
            rewrite_multi_part_table_reference_in_expr(&mut *nested, known_rewrites);
        }
        ast::Expr::Identifier(..) => {}
        ast::Expr::JsonAccess { value, .. } => {
            rewrite_multi_part_table_reference_in_expr(&mut *value, known_rewrites);
        }
        ast::Expr::CompositeAccess { expr, .. } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
        }
        ast::Expr::IsFalse(expr) => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
        }
        ast::Expr::IsNotFalse(expr) => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
        }
        ast::Expr::IsTrue(expr) => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
        }
        ast::Expr::IsNotTrue(expr) => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
        }
        ast::Expr::IsNull(expr) => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
        }
        ast::Expr::IsNotNull(expr) => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
        }
        ast::Expr::IsUnknown(expr) => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
        }
        ast::Expr::IsNotUnknown(expr) => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
        }
        ast::Expr::IsDistinctFrom(expr, expr1) => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
            rewrite_multi_part_table_reference_in_expr(&mut *expr1, known_rewrites);
        }
        ast::Expr::IsNotDistinctFrom(expr, expr1) => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
            rewrite_multi_part_table_reference_in_expr(&mut *expr1, known_rewrites);
        }
        ast::Expr::InList { expr, list, .. } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
            for item in list {
                rewrite_multi_part_table_reference_in_expr(&mut *item, known_rewrites);
            }
        }
        ast::Expr::InSubquery { expr, subquery, .. } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
            rewrite_multi_part_table_reference_in_query(&mut *subquery, known_rewrites);
        }
        ast::Expr::InUnnest {
            expr, array_expr, ..
        } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
            rewrite_multi_part_table_reference_in_expr(&mut *array_expr, known_rewrites);
        }
        ast::Expr::Between {
            expr, low, high, ..
        } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
            rewrite_multi_part_table_reference_in_expr(&mut *low, known_rewrites);
            rewrite_multi_part_table_reference_in_expr(&mut *high, known_rewrites);
        }
        ast::Expr::Like { expr, pattern, .. } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
            rewrite_multi_part_table_reference_in_expr(&mut *pattern, known_rewrites);
        }
        ast::Expr::ILike { expr, pattern, .. } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
            rewrite_multi_part_table_reference_in_expr(&mut *pattern, known_rewrites);
        }
        ast::Expr::SimilarTo { expr, pattern, .. } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
            rewrite_multi_part_table_reference_in_expr(&mut *pattern, known_rewrites);
        }
        ast::Expr::RLike { expr, pattern, .. } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
            rewrite_multi_part_table_reference_in_expr(&mut *pattern, known_rewrites);
        }
        ast::Expr::AnyOp { left, right, .. } => {
            rewrite_multi_part_table_reference_in_expr(&mut *left, known_rewrites);
            rewrite_multi_part_table_reference_in_expr(&mut *right, known_rewrites);
        }
        ast::Expr::AllOp { left, right, .. } => {
            rewrite_multi_part_table_reference_in_expr(&mut *left, known_rewrites);
            rewrite_multi_part_table_reference_in_expr(&mut *right, known_rewrites);
        }
        ast::Expr::Convert {
            expr,
            charset,
            styles,
            ..
        } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
            if let Some(charset) = charset {
                rewrite_object_name(charset, known_rewrites);
            }
            for style in styles {
                rewrite_multi_part_table_reference_in_expr(style, known_rewrites);
            }
        }
        ast::Expr::Cast { expr, .. } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
        }
        ast::Expr::AtTimeZone {
            timestamp,
            time_zone,
        } => {
            rewrite_multi_part_table_reference_in_expr(&mut *timestamp, known_rewrites);
            rewrite_multi_part_table_reference_in_expr(&mut *time_zone, known_rewrites);
        }
        ast::Expr::Extract { expr, .. } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
        }
        ast::Expr::Ceil { expr, .. } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
        }
        ast::Expr::Floor { expr, .. } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
        }
        ast::Expr::Position { expr, r#in } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
            rewrite_multi_part_table_reference_in_expr(&mut *r#in, known_rewrites);
        }
        ast::Expr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
            if let Some(substring_from) = substring_from {
                rewrite_multi_part_table_reference_in_expr(substring_from, known_rewrites);
            }
            if let Some(substring_for) = substring_for {
                rewrite_multi_part_table_reference_in_expr(substring_for, known_rewrites);
            }
        }
        ast::Expr::Trim {
            expr,
            trim_what,
            trim_characters,
            ..
        } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
            if let Some(trim_what) = trim_what {
                rewrite_multi_part_table_reference_in_expr(&mut *trim_what, known_rewrites);
            }
            if let Some(trim_characters) = trim_characters {
                for trim_character in trim_characters {
                    rewrite_multi_part_table_reference_in_expr(trim_character, known_rewrites);
                }
            }
        }
        ast::Expr::Overlay {
            expr,
            overlay_what,
            overlay_from,
            overlay_for,
        } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
            rewrite_multi_part_table_reference_in_expr(&mut *overlay_what, known_rewrites);
            rewrite_multi_part_table_reference_in_expr(&mut *overlay_from, known_rewrites);
            if let Some(overlay_for) = overlay_for {
                rewrite_multi_part_table_reference_in_expr(&mut *overlay_for, known_rewrites);
            }
        }
        ast::Expr::Collate { expr, collation } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
            rewrite_object_name(collation, known_rewrites);
        }
        ast::Expr::Value(..) => {}
        ast::Expr::IntroducedString { .. } => {}
        ast::Expr::TypedString { .. } => {}
        ast::Expr::MapAccess { column, keys } => {
            rewrite_multi_part_table_reference_in_expr(&mut *column, known_rewrites);

            for key in keys {
                rewrite_multi_part_table_reference_in_expr(&mut key.key, known_rewrites);
            }
        }
        ast::Expr::Exists { subquery, .. } => {
            rewrite_multi_part_table_reference_in_query(&mut *subquery, known_rewrites);
        }
        ast::Expr::GroupingSets(vec) => {
            for expr in vec.iter_mut().flatten() {
                rewrite_multi_part_table_reference_in_expr(expr, known_rewrites);
            }
        }
        ast::Expr::Cube(vec) => {
            for expr in vec.iter_mut().flatten() {
                rewrite_multi_part_table_reference_in_expr(expr, known_rewrites);
            }
        }
        ast::Expr::Rollup(vec) => {
            for expr in vec.iter_mut().flatten() {
                rewrite_multi_part_table_reference_in_expr(expr, known_rewrites);
            }
        }
        ast::Expr::Tuple(vec) => {
            for expr in vec {
                rewrite_multi_part_table_reference_in_expr(expr, known_rewrites);
            }
        }
        ast::Expr::Struct { values, .. } => {
            for expr in values {
                rewrite_multi_part_table_reference_in_expr(expr, known_rewrites);
            }
        }
        ast::Expr::Named { expr, .. } => {
            rewrite_multi_part_table_reference_in_expr(expr, known_rewrites);
        }
        ast::Expr::Dictionary(vec) => {
            for expr in vec {
                rewrite_multi_part_table_reference_in_expr(&mut expr.value, known_rewrites);
            }
        }
        ast::Expr::Map(map) => {
            for entry in map.entries.iter_mut() {
                rewrite_multi_part_table_reference_in_expr(&mut entry.key, known_rewrites);
                rewrite_multi_part_table_reference_in_expr(&mut entry.value, known_rewrites);
            }
        }
        ast::Expr::Subscript { expr, subscript } => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
            match &mut **subscript {
                ast::Subscript::Index { index } => {
                    rewrite_multi_part_table_reference_in_expr(index, known_rewrites);
                }
                ast::Subscript::Slice {
                    lower_bound,
                    upper_bound,
                    stride,
                } => {
                    if let Some(lower_bound) = lower_bound {
                        rewrite_multi_part_table_reference_in_expr(lower_bound, known_rewrites);
                    }
                    if let Some(upper_bound) = upper_bound {
                        rewrite_multi_part_table_reference_in_expr(upper_bound, known_rewrites);
                    }
                    if let Some(stride) = stride {
                        rewrite_multi_part_table_reference_in_expr(stride, known_rewrites);
                    }
                }
            }
        }
        ast::Expr::Array(array) => {
            for expr in array.elem.iter_mut() {
                rewrite_multi_part_table_reference_in_expr(expr, known_rewrites);
            }
        }
        ast::Expr::Interval(interval) => {
            rewrite_multi_part_table_reference_in_expr(&mut interval.value, known_rewrites);
        }
        ast::Expr::MatchAgainst { .. } => {}
        ast::Expr::Wildcard(_) => {}
        ast::Expr::QualifiedWildcard(object_name, _) => {
            rewrite_object_name(object_name, known_rewrites);
        }
        ast::Expr::OuterJoin(expr) => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
        }
        ast::Expr::Prior(expr) => {
            rewrite_multi_part_table_reference_in_expr(&mut *expr, known_rewrites);
        }
        ast::Expr::Lambda(lambda_function) => {
            rewrite_multi_part_table_reference_in_expr(&mut lambda_function.body, known_rewrites);
        }
        ast::Expr::Method(method) => {
            rewrite_multi_part_table_reference_in_expr(&mut method.expr, known_rewrites);
        }
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
    use datafusion::common::HashMap;
    use datafusion::sql::sqlparser::dialect::GenericDialect;
    use datafusion::sql::sqlparser::parser::Parser;

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
    fn test_rewrite_max_query() {
        let mut stmt = parse_sql("SELECT MAX(test_table.a) FROM test_table");
        let rewrites = create_test_rewrites();

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT MAX(catalog.schema.real_table.a) FROM catalog.schema.real_table"
        );
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
    fn test_rewrite_alias_table() {
        tracing_subscriber::fmt::init();
        let mut stmt = parse_sql("SELECT * FROM test_table as t1");
        let rewrites = create_test_rewrites();

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT * FROM catalog.schema.real_table AS t1"
        );
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

    #[test]
    fn test_rewrite_subquery() {
        let mut stmt = parse_sql(
            "SELECT * FROM test_table WHERE a IN (SELECT b FROM test_schema.test_table2)",
        );
        let rewrites = create_test_rewrites();

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT * FROM catalog.schema.real_table WHERE a IN (SELECT b FROM other_catalog.other_schema.real_table2)"
        );
    }

    #[test]
    fn test_rewrite_case_expression() {
        let mut stmt = parse_sql(
            "SELECT CASE WHEN test_table.a > 0 THEN test_schema.test_table2.b ELSE test_table.c END FROM test_table",
        );
        let rewrites = create_test_rewrites();

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT CASE WHEN catalog.schema.real_table.a > 0 THEN other_catalog.other_schema.real_table2.b ELSE catalog.schema.real_table.c END FROM catalog.schema.real_table"
        );
    }

    #[test]
    fn test_rewrite_join_conditions() {
        let mut stmt = parse_sql(
            "SELECT * FROM test_table JOIN test_schema.test_table2 \
             ON test_table.id = test_schema.test_table2.id \
             AND test_table.a > test_schema.test_table2.b",
        );
        let rewrites = create_test_rewrites();

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT * FROM catalog.schema.real_table JOIN other_catalog.other_schema.real_table2 \
             ON catalog.schema.real_table.id = other_catalog.other_schema.real_table2.id \
             AND catalog.schema.real_table.a > other_catalog.other_schema.real_table2.b"
        );
    }

    #[test]
    fn test_rewrite_nested_expressions() {
        let mut stmt = parse_sql(
            "SELECT * FROM test_table WHERE \
             EXISTS (SELECT 1 FROM test_schema.test_table2 WHERE test_schema.test_table2.id = test_table.id) \
             AND test_table.a IN (SELECT b FROM test_schema.test_table2)",
        );
        let rewrites = create_test_rewrites();

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT * FROM catalog.schema.real_table WHERE \
             EXISTS (SELECT 1 FROM other_catalog.other_schema.real_table2 \
             WHERE other_catalog.other_schema.real_table2.id = catalog.schema.real_table.id) \
             AND catalog.schema.real_table.a IN (SELECT b FROM other_catalog.other_schema.real_table2)"
        );
    }

    #[test]
    fn test_rewrite_with_cte() {
        let mut stmt = parse_sql(
            "WITH cte AS (SELECT a FROM test_table) \
             SELECT * FROM cte JOIN test_schema.test_table2 ON cte.a = test_schema.test_table2.b",
        );
        let rewrites = create_test_rewrites();

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        assert_eq!(
            stmt.to_string(),
            "WITH cte AS (SELECT a FROM catalog.schema.real_table) \
             SELECT * FROM cte JOIN other_catalog.other_schema.real_table2 ON cte.a = other_catalog.other_schema.real_table2.b"
        );
    }

    #[test]
    fn test_rewrite_union() {
        let mut stmt =
            parse_sql("SELECT a FROM test_table UNION SELECT b FROM test_schema.test_table2");
        let rewrites = create_test_rewrites();

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT a FROM catalog.schema.real_table UNION SELECT b FROM other_catalog.other_schema.real_table2"
        );
    }

    #[test]
    fn test_rewrite_derived_table() {
        let mut stmt = parse_sql(
            "SELECT * FROM (SELECT a FROM test_table) t1 JOIN test_schema.test_table2 ON t1.a = test_schema.test_table2.b",
        );
        let rewrites = create_test_rewrites();

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT * FROM (SELECT a FROM catalog.schema.real_table) AS t1 JOIN other_catalog.other_schema.real_table2 ON t1.a = other_catalog.other_schema.real_table2.b"
        );
    }

    #[test]
    fn test_rewrite_correlated_subquery() {
        let mut stmt = parse_sql(
            "SELECT * FROM test_table t1 WHERE EXISTS \
             (SELECT 1 FROM test_schema.test_table2 t2 WHERE t2.id = t1.id \
             AND t2.b > (SELECT MAX(a) FROM test_table WHERE test_table.group = t2.group))",
        );
        let rewrites = create_test_rewrites();

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT * FROM catalog.schema.real_table AS t1 WHERE EXISTS (SELECT 1 FROM other_catalog.other_schema.real_table2 AS t2 WHERE t2.id = t1.id AND t2.b > (SELECT MAX(a) FROM catalog.schema.real_table WHERE catalog.schema.real_table.group = t2.group))"
        );
    }

    #[test]
    fn test_no_rewrite_for_unknown_table() {
        let mut stmt = parse_sql("SELECT * FROM unknown_table");
        let rewrites = create_test_rewrites();

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        // Should remain unchanged
        assert_eq!(stmt.to_string(), "SELECT * FROM unknown_table");
    }

    #[test]
    fn test_rewrite_multiple_ctes() {
        let mut stmt = parse_sql(
            "WITH cte1 AS (SELECT a FROM test_table), \
             cte2 AS (SELECT b FROM test_schema.test_table2) \
             SELECT * FROM cte1 JOIN cte2 ON cte1.a = cte2.b",
        );
        let rewrites = create_test_rewrites();

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        assert_eq!(
            stmt.to_string(),
            "WITH cte1 AS (SELECT a FROM catalog.schema.real_table), \
             cte2 AS (SELECT b FROM other_catalog.other_schema.real_table2) \
             SELECT * FROM cte1 JOIN cte2 ON cte1.a = cte2.b"
        );
    }

    #[test]
    fn test_rewrite_between_expression() {
        let mut stmt = parse_sql(
            "SELECT * FROM test_table WHERE test_table.a BETWEEN test_schema.test_table2.b AND test_schema.test_table2.c",
        );
        let rewrites = create_test_rewrites();

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT * FROM catalog.schema.real_table WHERE catalog.schema.real_table.a BETWEEN other_catalog.other_schema.real_table2.b AND other_catalog.other_schema.real_table2.c"
        );
    }

    #[test]
    fn test_rewrite_nested_functions() {
        let mut stmt = parse_sql(
            "SELECT * FROM test_table WHERE EXISTS(SELECT 1 FROM test_schema.test_table2 WHERE MAX(test_table.a) > MIN(test_schema.test_table2.b))",
        );
        let rewrites = create_test_rewrites();

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT * FROM catalog.schema.real_table WHERE EXISTS (SELECT 1 FROM other_catalog.other_schema.real_table2 WHERE MAX(catalog.schema.real_table.a) > MIN(other_catalog.other_schema.real_table2.b))"
        );
    }

    #[test]
    fn test_rewrite_group_by_having() {
        let mut stmt = parse_sql(
            "SELECT test_table.a, COUNT(*) FROM test_table \
             GROUP BY test_table.a \
             HAVING COUNT(*) > (SELECT AVG(b) FROM test_schema.test_table2)",
        );
        let rewrites = create_test_rewrites();

        rewrite_multi_part_statement(&mut stmt, &rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT catalog.schema.real_table.a, COUNT(*) FROM catalog.schema.real_table \
            GROUP BY test_table.a HAVING COUNT(*) > (SELECT AVG(b) FROM test_schema.test_table2)"
        );
    }
}
