use std::ops::ControlFlow;

use datafusion::sql::sqlparser::ast::{
    FunctionArg, Ident, Statement, TableAlias, TableFactor, TableFunctionArgs, VisitMut, VisitorMut,
};

use super::{table_reference::MultiPartTableReference, AstAnalyzer};

pub fn replace_table_args_analyzer(mut visitor: TableArgReplace) -> AstAnalyzer {
    let x = move |mut statement: Statement| {
        VisitMut::visit(&mut statement, &mut visitor);
        Ok(statement)
    };
    Box::new(x)
}

/// Used to construct a AstAnalyzer that can replace table arguments.
///
/// ```rust
/// use datafusion::sql::sqlparser::ast::{FunctionArg, Expr, Value};
/// use datafusion::sql::TableReference;
/// use datafusion_federation::sql::ast_analyzer::TableArgReplace;
///
/// let mut analyzer = TableArgReplace::default().with(
///     TableReference::parse_str("table1"),
///     vec![FunctionArg::Unnamed(
///         Expr::value(
///             Value::Number("1".to_string(), false),
///         )
///         .into(),
///     )],
/// );
/// let analyzer = analyzer.into_analyzer();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TableArgReplace {
    pub tables: Vec<(MultiPartTableReference, TableFunctionArgs)>,
}

impl TableArgReplace {
    /// Constructs a new `TableArgReplace` instance.
    pub fn new(tables: Vec<(MultiPartTableReference, Vec<FunctionArg>)>) -> Self {
        Self {
            tables: tables
                .into_iter()
                .map(|(table, args)| {
                    (
                        table,
                        TableFunctionArgs {
                            args,
                            settings: None,
                        },
                    )
                })
                .collect(),
        }
    }

    /// Adds a new table argument replacement.
    pub fn with(mut self, table: MultiPartTableReference, args: Vec<FunctionArg>) -> Self {
        self.tables.push((
            table,
            TableFunctionArgs {
                args,
                settings: None,
            },
        ));
        self
    }

    /// Converts the `TableArgReplace` instance into an `AstAnalyzer`.
    pub fn into_analyzer(self) -> AstAnalyzer {
        replace_table_args_analyzer(self)
    }
}

impl VisitorMut for TableArgReplace {
    type Break = ();
    fn pre_visit_table_factor(
        &mut self,
        table_factor: &mut TableFactor,
    ) -> ControlFlow<Self::Break> {
        if let TableFactor::Table {
            name, args, alias, ..
        } = table_factor
        {
            let name = &*name;
            let name_as_tableref = name.into();
            if let Some((table, arg)) = self
                .tables
                .iter()
                .find(|(t, _)| t.resolved_eq(&name_as_tableref))
            {
                *args = Some(arg.clone());
                if alias.is_none() {
                    *alias = Some(TableAlias {
                        name: Ident::new(table.table()),
                        columns: vec![],
                    })
                }
            }
        }
        ControlFlow::Continue(())
    }
}
