use datafusion::{
    common::{error::Result, tree_node::TreeNodeRecursion, HashMap},
    logical_expr::LogicalPlan,
    sql::TableReference,
};

use crate::{
    get_table_source,
    sql::{table_reference::MultiPartTableReference, SQLTableSource},
};

/// Walks the LogicalPlan and collects a map of rewrites that we need to apply.
pub fn collect_known_rewrites(
    plan: &LogicalPlan,
) -> Result<HashMap<TableReference, MultiPartTableReference>> {
    let mut known_rewrites = HashMap::new();

    plan.apply_with_subqueries(|plan| {
        if let LogicalPlan::TableScan(table_scan) = plan {
            let original_table_name = table_scan.table_name.clone();
            if let Some(federated_source) = get_table_source(&table_scan.source)? {
                if let Some(sql_table_source) =
                    federated_source.as_any().downcast_ref::<SQLTableSource>()
                {
                    known_rewrites.insert(original_table_name, sql_table_source.table_reference());
                }
            }
        }

        Ok(TreeNodeRecursion::Continue)
    })?;

    Ok(known_rewrites)
}
