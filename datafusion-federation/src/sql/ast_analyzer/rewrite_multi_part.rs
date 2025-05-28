use std::ops::ControlFlow;

use datafusion::{
    common::HashMap,
    sql::{
        sqlparser::ast::{self, Ident, ObjectName, VisitMut, VisitorMut},
        TableReference,
    },
};

use crate::sql::table_reference::{MultiPartTableReference, MultiTableReference};

/// Rewrites sqlparser AST statements to use the original table name if any are MultiTableReferences (i.e. "a"."b"."c"."d")
///
/// Does nothing if there are no MultiTableReferences.
#[derive(Debug)]
pub struct RewriteMultiTableReference;

impl RewriteMultiTableReference {
    pub fn rewrite(
        statement: &mut ast::Statement,
        known_rewrites: HashMap<TableReference, MultiPartTableReference>,
    ) {
        let mut known_rewrites_multi = HashMap::new();

        // Iterate over the known rewrites and collect only the multi table references that need rewriting.
        for (table_reference, multi_table_reference) in known_rewrites {
            if let MultiPartTableReference::Multi(multi_table_ref) = multi_table_reference {
                known_rewrites_multi.insert(
                    table_reference_to_object_name(&table_reference),
                    multi_table_ref,
                );
            }
        }

        if known_rewrites_multi.is_empty() {
            return;
        }

        let mut visitor = RewriteMultiTableVisitor {
            known_rewrites: known_rewrites_multi,
        };

        let _ = VisitMut::visit(statement, &mut visitor);
    }
}

struct RewriteMultiTableVisitor {
    known_rewrites: HashMap<ObjectName, MultiTableReference>,
}

impl VisitorMut for RewriteMultiTableVisitor {
    type Break = ();

    fn pre_visit_relation(&mut self, table_factor: &mut ObjectName) -> ControlFlow<Self::Break> {
        if let Some(rewrite) = self.known_rewrites.get(table_factor) {
            // Create new object name from the rewritten table reference
            let new_name = ObjectName(
                rewrite
                    .parts()
                    .iter()
                    .map(|p| Ident::new(p.to_string()))
                    .collect(),
            );
            *table_factor = new_name;
        }

        ControlFlow::Continue(())
    }

    fn pre_visit_expr(&mut self, expr: &mut ast::Expr) -> ControlFlow<Self::Break> {
        if let ast::Expr::CompoundIdentifier(idents) = expr {
            // This should be impossible, but handle it defensively
            if idents.len() < 2 {
                return ControlFlow::Continue(());
            }

            // Get the column name (last identifier) and table name (all other identifiers)
            let column_name = idents.last().cloned();
            let obj_name = ObjectName(idents[..idents.len() - 1].to_vec());

            if let Some(rewrite) = self.known_rewrites.get(&obj_name) {
                // Rewrite the table parts
                let mut new_idents: Vec<Ident> = rewrite
                    .parts()
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

        ControlFlow::Continue(())
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

    fn create_test_rewrites() -> HashMap<TableReference, MultiPartTableReference> {
        let mut rewrites = HashMap::new();

        rewrites.insert(
            TableReference::Bare {
                table: "test_table".into(),
            },
            MultiPartTableReference::Multi(MultiTableReference::new(vec![
                "catalog".into(),
                "schema1".into(),
                "schema2".into(),
                "real_table".into(),
            ])),
        );

        rewrites.insert(
            TableReference::Partial {
                schema: "test_schema".into(),
                table: "test_table2".into(),
            },
            MultiPartTableReference::Multi(MultiTableReference::new(vec![
                "other_catalog".into(),
                "other_schema1".into(),
                "other_schema2".into(),
                "real_table2".into(),
            ])),
        );

        rewrites
    }

    #[test]
    fn test_rewrite_simple_query() {
        let mut stmt = parse_sql("SELECT * FROM test_table");
        let rewrites = create_test_rewrites();

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT * FROM catalog.schema1.schema2.real_table"
        );
    }

    #[test]
    fn test_rewrite_max_query() {
        let mut stmt = parse_sql("SELECT MAX(test_table.a) FROM test_table");
        let rewrites = create_test_rewrites();

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT MAX(catalog.schema1.schema2.real_table.a) FROM catalog.schema1.schema2.real_table"
        );
    }

    #[test]
    fn test_rewrite_count_query() {
        let mut stmt = parse_sql("SELECT count(*) FROM test_table");
        let rewrites = create_test_rewrites();

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT count(*) FROM catalog.schema1.schema2.real_table"
        );
    }

    #[test]
    fn test_rewrite_compound_identifier() {
        let mut stmt =
            parse_sql("SELECT test_schema.test_table2.column FROM test_schema.test_table2");
        let rewrites = create_test_rewrites();

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT other_catalog.other_schema1.other_schema2.real_table2.column FROM other_catalog.other_schema1.other_schema2.real_table2"
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
            MultiPartTableReference::Multi(MultiTableReference::new(vec![
                "new_level1".into(),
                "new_level2".into(),
                "new_level3".into(),
                "new_level4".into(),
                "new_level5".into(),
            ])),
        );

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

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
            MultiPartTableReference::Multi(MultiTableReference::new(vec![
                "new1".into(),
                "new2".into(),
                "new3".into(),
                "new4".into(),
                "new5".into(),
            ])),
        );

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

        assert_eq!(stmt.to_string(), "SELECT * FROM new1.new2.new3.new4.new5");
    }

    #[test]
    fn test_rewrite_alias_table() {
        let _ = tracing_subscriber::fmt::try_init();
        let mut stmt = parse_sql("SELECT * FROM test_table as t1");
        let rewrites = create_test_rewrites();

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT * FROM catalog.schema1.schema2.real_table AS t1"
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

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT * FROM catalog.schema1.schema2.real_table WHERE a IN (SELECT b FROM other_catalog.other_schema1.other_schema2.real_table2)"
        );
    }

    #[test]
    fn test_rewrite_case_expression() {
        let mut stmt = parse_sql(
            "SELECT CASE WHEN test_table.a > 0 THEN test_schema.test_table2.b ELSE test_table.c END FROM test_table",
        );
        let rewrites = create_test_rewrites();

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT CASE WHEN catalog.schema1.schema2.real_table.a > 0 THEN other_catalog.other_schema1.other_schema2.real_table2.b ELSE catalog.schema1.schema2.real_table.c END FROM catalog.schema1.schema2.real_table"
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

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT * FROM catalog.schema1.schema2.real_table JOIN other_catalog.other_schema1.other_schema2.real_table2 \
             ON catalog.schema1.schema2.real_table.id = other_catalog.other_schema1.other_schema2.real_table2.id \
             AND catalog.schema1.schema2.real_table.a > other_catalog.other_schema1.other_schema2.real_table2.b"
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

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT * FROM catalog.schema1.schema2.real_table WHERE \
             EXISTS (SELECT 1 FROM other_catalog.other_schema1.other_schema2.real_table2 \
             WHERE other_catalog.other_schema1.other_schema2.real_table2.id = catalog.schema1.schema2.real_table.id) \
             AND catalog.schema1.schema2.real_table.a IN (SELECT b FROM other_catalog.other_schema1.other_schema2.real_table2)"
        );
    }

    #[test]
    fn test_rewrite_with_cte() {
        let mut stmt = parse_sql(
            "WITH cte AS (SELECT a FROM test_table) \
             SELECT * FROM cte JOIN test_schema.test_table2 ON cte.a = test_schema.test_table2.b",
        );
        let rewrites = create_test_rewrites();

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

        assert_eq!(
            stmt.to_string(),
            "WITH cte AS (SELECT a FROM catalog.schema1.schema2.real_table) \
             SELECT * FROM cte JOIN other_catalog.other_schema1.other_schema2.real_table2 ON cte.a = other_catalog.other_schema1.other_schema2.real_table2.b"
        );
    }

    #[test]
    fn test_rewrite_union() {
        let mut stmt =
            parse_sql("SELECT a FROM test_table UNION SELECT b FROM test_schema.test_table2");
        let rewrites = create_test_rewrites();

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT a FROM catalog.schema1.schema2.real_table UNION SELECT b FROM other_catalog.other_schema1.other_schema2.real_table2"
        );
    }

    #[test]
    fn test_rewrite_derived_table() {
        let mut stmt = parse_sql(
            "SELECT * FROM (SELECT a FROM test_table) t1 JOIN test_schema.test_table2 ON t1.a = test_schema.test_table2.b",
        );
        let rewrites = create_test_rewrites();

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT * FROM (SELECT a FROM catalog.schema1.schema2.real_table) AS t1 JOIN other_catalog.other_schema1.other_schema2.real_table2 ON t1.a = other_catalog.other_schema1.other_schema2.real_table2.b"
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

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT * FROM catalog.schema1.schema2.real_table AS t1 WHERE EXISTS \
            (SELECT 1 FROM other_catalog.other_schema1.other_schema2.real_table2 AS t2 WHERE t2.id = t1.id \
            AND t2.b > (SELECT MAX(a) FROM catalog.schema1.schema2.real_table WHERE catalog.schema1.schema2.real_table.group = t2.group))"
        );
    }

    #[test]
    fn test_no_rewrite_for_unknown_table() {
        let mut stmt = parse_sql("SELECT * FROM unknown_table");
        let rewrites = create_test_rewrites();

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

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

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

        assert_eq!(
            stmt.to_string(),
            "WITH cte1 AS (SELECT a FROM catalog.schema1.schema2.real_table), \
             cte2 AS (SELECT b FROM other_catalog.other_schema1.other_schema2.real_table2) \
             SELECT * FROM cte1 JOIN cte2 ON cte1.a = cte2.b"
        );
    }

    #[test]
    fn test_rewrite_between_expression() {
        let mut stmt = parse_sql(
            "SELECT * FROM test_table WHERE test_table.a BETWEEN test_schema.test_table2.b AND test_schema.test_table2.c",
        );
        let rewrites = create_test_rewrites();

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT * FROM catalog.schema1.schema2.real_table WHERE catalog.schema1.schema2.real_table.a BETWEEN other_catalog.other_schema1.other_schema2.real_table2.b AND other_catalog.other_schema1.other_schema2.real_table2.c"
        );
    }

    #[test]
    fn test_rewrite_nested_functions() {
        let mut stmt = parse_sql(
            "SELECT * FROM test_table WHERE EXISTS(SELECT 1 FROM test_schema.test_table2 WHERE MAX(test_table.a) > MIN(test_schema.test_table2.b))",
        );
        let rewrites = create_test_rewrites();

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT * FROM catalog.schema1.schema2.real_table WHERE EXISTS (SELECT 1 FROM other_catalog.other_schema1.other_schema2.real_table2 WHERE MAX(catalog.schema1.schema2.real_table.a) > MIN(other_catalog.other_schema1.other_schema2.real_table2.b))"
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

        RewriteMultiTableReference::rewrite(&mut stmt, rewrites);

        assert_eq!(
            stmt.to_string(),
            "SELECT catalog.schema1.schema2.real_table.a, COUNT(*) FROM catalog.schema1.schema2.real_table \
            GROUP BY catalog.schema1.schema2.real_table.a HAVING COUNT(*) > (SELECT AVG(b) FROM other_catalog.other_schema1.other_schema2.real_table2)"
        );
    }
}
