use std::sync::Arc;

use datafusion::{
    error::DataFusionError,
    sql::{
        sqlparser::{
            self,
            ast::{FunctionArg, ObjectName, ObjectNamePart},
            dialect::{Dialect, GenericDialect},
            tokenizer::Token,
        },
        TableReference,
    },
};

macro_rules! ident_match {
    ($part:expr) => {
        match $part {
            ObjectNamePart::Identifier(ident) => ident.value.into(),
            v => {
                return Err(DataFusionError::NotImplemented(format!(
                    "Unsupported ObjectNamePart variant: {:?}",
                    v
                )));
            }
        }
    };
    ($part:expr, true) => {
        match $part {
            ObjectNamePart::Identifier(ident) => Ok(ident.value.into()),
            v => {
                return Err(DataFusionError::NotImplemented(format!(
                    "Unsupported ObjectNamePart variant: {:?}",
                    v
                )));
            }
        }
    };
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MultiTableReference {
    parts: Vec<Arc<str>>,
}

impl MultiTableReference {
    pub fn new(parts: Vec<Arc<str>>) -> Self {
        assert!(parts.len() > 3, "Use TableReference, not MultiTableReference, for table references with less than 3 parts.");
        Self { parts }
    }

    pub fn parts(&self) -> &Vec<Arc<str>> {
        &self.parts
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MultiPartTableReference {
    TableReference(TableReference),
    Multi(MultiTableReference),
}

impl MultiPartTableReference {
    /// Convert a `MultiPartTableReference` to a quoted string.
    ///
    /// Example:
    ///
    /// ```ignore
    /// let parts = vec![Arc::from("a"), Arc::from("b"), Arc::from("c"), Arc::from("d")];
    /// let multi_part_table_reference = MultiPartTableReference::encode_multi_part_table_reference(&parts);
    /// assert_eq!(multi_part_table_reference.to_quoted_string(), r#""a"."b"."c"."d""#);
    /// ```
    #[must_use]
    pub fn to_quoted_string(&self) -> String {
        match self {
            MultiPartTableReference::TableReference(table_reference) => match table_reference {
                // The `TableReference` will sometimes not quote the table name, even if we ask it to because it detects that it would be safe (within DataFusion).
                // Unfortunately, some systems have reserved keywords that will error if we don't quote them.
                // Err on the safe side and always quote the table name.
                TableReference::Bare { table } => quote_identifier(table),
                TableReference::Partial { schema, table } => {
                    format!("{}.{}", quote_identifier(schema), quote_identifier(table))
                }
                TableReference::Full {
                    catalog,
                    schema,
                    table,
                } => format!(
                    "{}.{}.{}",
                    quote_identifier(catalog),
                    quote_identifier(schema),
                    quote_identifier(table)
                ),
            },
            MultiPartTableReference::Multi(parts) => parts
                .iter()
                .map(|p| quote_identifier(p))
                .collect::<Vec<_>>()
                .join("."),
        }
    }

    /// Compare with another [`MultiPartTableReference`] as if both are resolved.
    /// This allows comparing across variants. If a field is not present
    /// in both variants being compared then it is ignored in the comparison.
    ///
    /// e.g. this allows a [`TableReference::Bare`] to be considered equal to a
    /// fully qualified [`TableReference::Full`] if the table names match.
    pub fn resolved_eq(&self, other: &Self) -> bool {
        match self {
            MultiPartTableReference::TableReference(table_reference) => match other {
                MultiPartTableReference::TableReference(other_table_reference) => {
                    table_reference.resolved_eq(other_table_reference)
                }
                MultiPartTableReference::Multi(_) => false,
            },
            MultiPartTableReference::Multi(parts) => match other {
                MultiPartTableReference::Multi(other_parts) => {
                    parts.iter().zip(other_parts.iter()).all(|(a, b)| a == b)
                }
                MultiPartTableReference::TableReference(_) => false,
            },
        }
    }

    pub fn table(&self) -> &str {
        match self {
            MultiPartTableReference::TableReference(table_reference) => table_reference.table(),
            MultiPartTableReference::Multi(parts) => parts.last().unwrap(),
        }
    }
}

impl std::fmt::Display for MultiPartTableReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MultiPartTableReference::TableReference(table_reference) => {
                write!(f, "{table_reference}")
            }
            MultiPartTableReference::Multi(parts) => {
                write!(f, "{}", parts.join("."))
            }
        }
    }
}

impl std::ops::Deref for MultiTableReference {
    type Target = Vec<Arc<str>>;

    fn deref(&self) -> &Self::Target {
        &self.parts
    }
}

impl TryFrom<ObjectName> for MultiPartTableReference {
    type Error = DataFusionError;

    fn try_from(name: ObjectName) -> Result<Self, Self::Error> {
        let mut parts = name.0;
        let multi_ref = match parts.len() {
            1 => MultiPartTableReference::TableReference(TableReference::Bare {
                table: ident_match!(parts.remove(0)),
            }),
            2 => MultiPartTableReference::TableReference(TableReference::Partial {
                schema: ident_match!(parts.remove(0)),
                table: ident_match!(parts.remove(0)),
            }),
            3 => MultiPartTableReference::TableReference(TableReference::Full {
                catalog: ident_match!(parts.remove(0)),
                schema: ident_match!(parts.remove(0)),
                table: ident_match!(parts.remove(0)),
            }),
            _ => MultiPartTableReference::Multi(MultiTableReference {
                parts: parts
                    .clone()
                    .into_iter()
                    .map(|p| ident_match!(p, true))
                    .collect::<Result<Vec<Arc<str>>, Self::Error>>()?,
            }),
        };

        Ok(multi_ref)
    }
}

impl TryFrom<&ObjectName> for MultiPartTableReference {
    type Error = DataFusionError;

    fn try_from(name: &ObjectName) -> Result<Self, Self::Error> {
        MultiPartTableReference::try_from(name.clone())
    }
}

impl From<&RemoteTableRef> for MultiPartTableReference {
    fn from(table_ref: &RemoteTableRef) -> Self {
        table_ref.table_ref().clone()
    }
}

impl From<TableReference> for MultiPartTableReference {
    fn from(table_ref: TableReference) -> Self {
        MultiPartTableReference::TableReference(table_ref)
    }
}

impl From<&TableReference> for MultiPartTableReference {
    fn from(table_ref: &TableReference) -> Self {
        MultiPartTableReference::TableReference(table_ref.clone())
    }
}

/// A multipart identifier to a remote table, view or parameterized view.
///
/// RemoteTableRef can be created by parsing from a string representing a table object with optional
/// ```rust
/// use datafusion_federation::sql::RemoteTableRef;
/// use datafusion::sql::sqlparser::dialect::PostgreSqlDialect;
///
/// RemoteTableRef::try_from("myschema.table");
/// RemoteTableRef::try_from(r#"myschema."Table""#);
/// RemoteTableRef::try_from("myschema.view('obj')");
///
/// RemoteTableRef::parse_with_dialect("myschema.view(name = 'obj')", &PostgreSqlDialect {});
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RemoteTableRef {
    pub table_ref: MultiPartTableReference,
    pub args: Option<Arc<[FunctionArg]>>,
}

impl RemoteTableRef {
    /// Get quoted_string representation for the table it is referencing, this is same as calling to_quoted_string on the inner table reference.
    pub fn to_quoted_string(&self) -> String {
        self.table_ref.to_quoted_string()
    }

    /// Create new using general purpose dialect. Prefer [`Self::parse_with_dialect`] if the dialect is known beforehand
    pub fn parse_with_default_dialect(s: &str) -> Result<Self, DataFusionError> {
        Self::parse_with_dialect(s, &GenericDialect {})
    }

    /// Create new using a specific instance of dialect.
    pub fn parse_with_dialect(s: &str, dialect: &dyn Dialect) -> Result<Self, DataFusionError> {
        let mut parser = sqlparser::parser::Parser::new(dialect).try_with_sql(s)?;
        let name = parser.parse_object_name(true)?;
        let args = if parser.consume_token(&Token::LParen) {
            parser.parse_optional_args()?
        } else {
            vec![]
        };

        let mut parts = name.0;
        let table_ref = match parts.len() {
            1 => MultiPartTableReference::TableReference(TableReference::Bare {
                table: ident_match!(parts.remove(0)),
            }),
            2 => MultiPartTableReference::TableReference(TableReference::Partial {
                schema: ident_match!(parts.remove(0)),
                table: ident_match!(parts.remove(0)),
            }),
            3 => MultiPartTableReference::TableReference(TableReference::Full {
                catalog: ident_match!(parts.remove(0)),
                schema: ident_match!(parts.remove(0)),
                table: ident_match!(parts.remove(0)),
            }),
            _ => MultiPartTableReference::Multi(MultiTableReference {
                parts: parts
                    .into_iter()
                    .map(|p| ident_match!(p, true))
                    .collect::<Result<_, DataFusionError>>()?,
            }),
        };

        if !args.is_empty() {
            Ok(RemoteTableRef {
                table_ref,
                args: Some(args.into()),
            })
        } else {
            Ok(RemoteTableRef {
                table_ref,
                args: None,
            })
        }
    }

    pub fn table_ref(&self) -> &MultiPartTableReference {
        &self.table_ref
    }

    pub fn args(&self) -> Option<&[FunctionArg]> {
        self.args.as_deref()
    }
}

impl From<TableReference> for RemoteTableRef {
    fn from(table_ref: TableReference) -> Self {
        RemoteTableRef {
            table_ref: MultiPartTableReference::TableReference(table_ref),
            args: None,
        }
    }
}

impl TryFrom<MultiPartTableReference> for TableReference {
    type Error = DataFusionError;

    fn try_from(value: MultiPartTableReference) -> Result<Self, Self::Error> {
        match value {
            MultiPartTableReference::TableReference(table_reference) => Ok(table_reference),
            MultiPartTableReference::Multi(_) => Err(DataFusionError::External(
                "MultiPartTableReference cannot be converted to TableReference".into(),
            )),
        }
    }
}

impl TryFrom<&MultiPartTableReference> for TableReference {
    type Error = DataFusionError;

    fn try_from(value: &MultiPartTableReference) -> Result<Self, Self::Error> {
        match value {
            MultiPartTableReference::TableReference(table_reference) => Ok(table_reference.clone()),
            MultiPartTableReference::Multi(_) => Err(DataFusionError::External(
                "MultiPartTableReference cannot be converted to TableReference".into(),
            )),
        }
    }
}

impl From<MultiPartTableReference> for RemoteTableRef {
    fn from(value: MultiPartTableReference) -> Self {
        RemoteTableRef {
            table_ref: value,
            args: None,
        }
    }
}

impl From<(TableReference, Vec<FunctionArg>)> for RemoteTableRef {
    fn from((table_ref, args): (TableReference, Vec<FunctionArg>)) -> Self {
        RemoteTableRef {
            table_ref: MultiPartTableReference::TableReference(table_ref),
            args: Some(args.into()),
        }
    }
}

impl TryFrom<&str> for RemoteTableRef {
    type Error = DataFusionError;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::parse_with_default_dialect(s)
    }
}

impl TryFrom<String> for RemoteTableRef {
    type Error = DataFusionError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::parse_with_default_dialect(&s)
    }
}

impl TryFrom<&String> for RemoteTableRef {
    type Error = DataFusionError;
    fn try_from(s: &String) -> Result<Self, Self::Error> {
        Self::parse_with_default_dialect(s)
    }
}

/// Wraps identifier string in double quotes, escaping any double quotes in
/// the identifier by replacing it with two double quotes
///
/// e.g. identifier `tab.le"name` becomes `"tab.le""name"`
#[must_use]
pub fn quote_identifier(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::sql::sqlparser::{ast::ValueWithSpan, tokenizer::Span};
    use sqlparser::{
        ast::{self, Expr, FunctionArgOperator, Ident, Value},
        dialect,
    };

    #[test]
    fn bare_table_reference() {
        let table_ref = RemoteTableRef::parse_with_default_dialect("table").unwrap();
        let expected = RemoteTableRef::from(TableReference::bare("table"));
        assert_eq!(table_ref, expected);

        let table_ref = RemoteTableRef::parse_with_default_dialect("Table").unwrap();
        let expected = RemoteTableRef::from(TableReference::bare("Table"));
        assert_eq!(table_ref, expected);
    }

    #[test]
    fn bare_table_reference_with_args() {
        let table_ref = RemoteTableRef::parse_with_default_dialect("table(1, 2)").unwrap();
        let expected = RemoteTableRef::from((
            TableReference::bare("table"),
            vec![
                FunctionArg::Unnamed(
                    Expr::Value(ValueWithSpan {
                        value: Value::Number("1".to_string(), false),
                        span: Span::empty(),
                    })
                    .into(),
                ),
                FunctionArg::Unnamed(
                    Expr::Value(ValueWithSpan {
                        value: Value::Number("2".to_string(), false),
                        span: Span::empty(),
                    })
                    .into(),
                ),
            ],
        ));
        assert_eq!(table_ref, expected);

        let table_ref = RemoteTableRef::parse_with_default_dialect("Table(1, 2)").unwrap();
        let expected = RemoteTableRef::from((
            TableReference::bare("Table"),
            vec![
                FunctionArg::Unnamed(
                    Expr::Value(ValueWithSpan {
                        value: Value::Number("1".to_string(), false),
                        span: Span::empty(),
                    })
                    .into(),
                ),
                FunctionArg::Unnamed(
                    Expr::Value(ValueWithSpan {
                        value: Value::Number("2".to_string(), false),
                        span: Span::empty(),
                    })
                    .into(),
                ),
            ],
        ));
        assert_eq!(table_ref, expected);
    }

    #[test]
    fn bare_table_reference_with_args_and_whitespace() {
        let table_ref = RemoteTableRef::parse_with_default_dialect("table (1, 2)").unwrap();
        let expected = RemoteTableRef::from((
            TableReference::bare("table"),
            vec![
                FunctionArg::Unnamed(
                    Expr::Value(ValueWithSpan {
                        value: Value::Number("1".to_string(), false),
                        span: Span::empty(),
                    })
                    .into(),
                ),
                FunctionArg::Unnamed(
                    Expr::Value(ValueWithSpan {
                        value: Value::Number("2".to_string(), false),
                        span: Span::empty(),
                    })
                    .into(),
                ),
            ],
        ));
        assert_eq!(table_ref, expected);

        let table_ref = RemoteTableRef::parse_with_default_dialect("Table (1, 2)").unwrap();
        let expected = RemoteTableRef::from((
            TableReference::bare("Table"),
            vec![
                FunctionArg::Unnamed(
                    Expr::Value(ValueWithSpan {
                        value: Value::Number("1".to_string(), false),
                        span: Span::empty(),
                    })
                    .into(),
                ),
                FunctionArg::Unnamed(
                    Expr::Value(ValueWithSpan {
                        value: Value::Number("2".to_string(), false),
                        span: Span::empty(),
                    })
                    .into(),
                ),
            ],
        ));
        assert_eq!(table_ref, expected);
    }

    #[test]
    fn multi_table_reference_with_no_args() {
        let table_ref = RemoteTableRef::parse_with_default_dialect("schema.table").unwrap();
        let expected = RemoteTableRef::from(TableReference::partial("schema", "table"));
        assert_eq!(table_ref, expected);

        let table_ref = RemoteTableRef::parse_with_default_dialect("schema.Table").unwrap();
        let expected = RemoteTableRef::from(TableReference::partial("schema", "Table"));
        assert_eq!(table_ref, expected);
    }

    #[test]
    fn multi_table_reference_with_args() {
        let table_ref = RemoteTableRef::parse_with_default_dialect("schema.table(1, 2)").unwrap();
        let expected = RemoteTableRef::from((
            TableReference::partial("schema", "table"),
            vec![
                FunctionArg::Unnamed(
                    Expr::Value(ValueWithSpan {
                        value: Value::Number("1".to_string(), false),
                        span: Span::empty(),
                    })
                    .into(),
                ),
                FunctionArg::Unnamed(
                    Expr::Value(ValueWithSpan {
                        value: Value::Number("2".to_string(), false),
                        span: Span::empty(),
                    })
                    .into(),
                ),
            ],
        ));
        assert_eq!(table_ref, expected);

        let table_ref = RemoteTableRef::parse_with_default_dialect("schema.Table(1, 2)").unwrap();
        let expected = RemoteTableRef::from((
            TableReference::partial("schema", "Table"),
            vec![
                FunctionArg::Unnamed(
                    Expr::Value(ValueWithSpan {
                        value: Value::Number("1".to_string(), false),
                        span: Span::empty(),
                    })
                    .into(),
                ),
                FunctionArg::Unnamed(
                    Expr::Value(ValueWithSpan {
                        value: Value::Number("2".to_string(), false),
                        span: Span::empty(),
                    })
                    .into(),
                ),
            ],
        ));
        assert_eq!(table_ref, expected);
    }

    #[test]
    fn multi_table_reference_with_args_and_whitespace() {
        let table_ref = RemoteTableRef::parse_with_default_dialect("schema.table (1, 2)").unwrap();
        let expected = RemoteTableRef::from((
            TableReference::partial("schema", "table"),
            vec![
                FunctionArg::Unnamed(
                    Expr::Value(ValueWithSpan {
                        value: Value::Number("1".to_string(), false),
                        span: Span::empty(),
                    })
                    .into(),
                ),
                FunctionArg::Unnamed(
                    Expr::Value(ValueWithSpan {
                        value: Value::Number("2".to_string(), false),
                        span: Span::empty(),
                    })
                    .into(),
                ),
            ],
        ));
        assert_eq!(table_ref, expected);
    }

    #[test]
    fn bare_reference_with_named_args() {
        let table_ref = RemoteTableRef::parse_with_dialect(
            "Table (user_id => 1, age => 2)",
            &dialect::PostgreSqlDialect {},
        )
        .unwrap();
        let expected = RemoteTableRef::from((
            TableReference::bare("Table"),
            vec![
                FunctionArg::ExprNamed {
                    name: ast::Expr::Identifier(Ident::new("user_id")),
                    arg: Expr::Value(ValueWithSpan {
                        value: Value::Number("1".to_string(), false),
                        span: Span::empty(),
                    })
                    .into(),
                    operator: FunctionArgOperator::RightArrow,
                },
                FunctionArg::ExprNamed {
                    name: ast::Expr::Identifier(Ident::new("age")),
                    arg: Expr::Value(ValueWithSpan {
                        value: Value::Number("2".to_string(), false),
                        span: Span::empty(),
                    })
                    .into(),
                    operator: FunctionArgOperator::RightArrow,
                },
            ],
        ));
        assert_eq!(table_ref, expected);
    }
}
