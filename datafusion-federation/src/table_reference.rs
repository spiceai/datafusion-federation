//! A MultiPartTableReference is an extension of the DataFusion provided TableReference
//! that allows for referencing tables that are nested deeper than the 3-part
//! catalog.schema.table.
//!
//! This is useful for federated queries where the target system supports
//! arbitrarily nested tables, i.e. Dremio/Iceberg.

use std::sync::Arc;

use datafusion::{
    error::{DataFusionError, Result as DataFusionResult},
    sql::{
        sqlparser::{ast::Ident, dialect::GenericDialect, parser::Parser},
        TableReference,
    },
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiTableReference {
    pub parts: Vec<Arc<str>>,
}

impl MultiTableReference {
    pub fn new(parts: Vec<Arc<str>>) -> Self {
        Self { parts }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

impl From<&str> for MultiPartTableReference {
    fn from(s: &str) -> Self {
        parse_multi_part_table_reference(s)
    }
}

impl From<String> for MultiPartTableReference {
    fn from(s: String) -> Self {
        parse_multi_part_table_reference(&s)
    }
}

impl From<&String> for MultiPartTableReference {
    fn from(s: &String) -> Self {
        parse_multi_part_table_reference(s)
    }
}

impl From<TableReference> for MultiPartTableReference {
    fn from(table_reference: TableReference) -> Self {
        MultiPartTableReference::TableReference(table_reference)
    }
}

impl From<Vec<Arc<str>>> for MultiPartTableReference {
    fn from(parts: Vec<Arc<str>>) -> Self {
        MultiPartTableReference::Multi(MultiTableReference { parts })
    }
}

impl From<Vec<String>> for MultiPartTableReference {
    fn from(parts: Vec<String>) -> Self {
        MultiPartTableReference::Multi(MultiTableReference {
            parts: parts.into_iter().map(Arc::from).collect(),
        })
    }
}

impl From<Vec<&str>> for MultiPartTableReference {
    fn from(parts: Vec<&str>) -> Self {
        MultiPartTableReference::Multi(MultiTableReference {
            parts: parts.into_iter().map(Arc::from).collect(),
        })
    }
}

impl From<Vec<&String>> for MultiPartTableReference {
    fn from(parts: Vec<&String>) -> Self {
        MultiPartTableReference::Multi(MultiTableReference {
            parts: parts.into_iter().map(|s| Arc::from(s.as_str())).collect(),
        })
    }
}

impl From<&[String]> for MultiPartTableReference {
    fn from(parts: &[String]) -> Self {
        MultiPartTableReference::Multi(MultiTableReference {
            parts: parts.iter().map(|s| Arc::from(s.as_str())).collect(),
        })
    }
}

impl From<&[&str]> for MultiPartTableReference {
    fn from(parts: &[&str]) -> Self {
        MultiPartTableReference::Multi(MultiTableReference {
            parts: parts.iter().map(|&s| Arc::from(s)).collect(),
        })
    }
}

impl<const N: usize> From<[String; N]> for MultiPartTableReference {
    fn from(parts: [String; N]) -> Self {
        MultiPartTableReference::Multi(MultiTableReference {
            parts: parts.into_iter().map(Arc::from).collect(),
        })
    }
}

impl<const N: usize> From<[&str; N]> for MultiPartTableReference {
    fn from(parts: [&str; N]) -> Self {
        MultiPartTableReference::Multi(MultiTableReference {
            parts: parts.into_iter().map(Arc::from).collect(),
        })
    }
}

impl PartialEq<TableReference> for MultiPartTableReference {
    fn eq(&self, other: &TableReference) -> bool {
        match self {
            MultiPartTableReference::TableReference(table_ref) => table_ref == other,
            MultiPartTableReference::Multi(_) => false,
        }
    }
}

impl PartialEq<MultiPartTableReference> for TableReference {
    fn eq(&self, other: &MultiPartTableReference) -> bool {
        other == self
    }
}

/// Parses a dataset path string into a `MultiPartTableReference`, handling quoted identifiers and multi-part paths.
/// Parts can be quoted with double quotes to include periods or other special characters.
#[must_use]
pub fn parse_multi_part_table_reference(s: &str) -> MultiPartTableReference {
    let mut parts = parse_identifiers_normalized(s, false);

    match parts.len() {
        1 => MultiPartTableReference::TableReference(TableReference::Bare {
            table: parts.remove(0).into(),
        }),
        2 => MultiPartTableReference::TableReference(TableReference::Partial {
            schema: parts.remove(0).into(),
            table: parts.remove(0).into(),
        }),
        3 => MultiPartTableReference::TableReference(TableReference::Full {
            catalog: parts.remove(0).into(),
            schema: parts.remove(0).into(),
            table: parts.remove(0).into(),
        }),
        _ => MultiPartTableReference::Multi(MultiTableReference {
            parts: parts.into_iter().map(Arc::from).collect(),
        }),
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

pub(crate) fn parse_identifiers(s: &str) -> DataFusionResult<Vec<Ident>> {
    let dialect = GenericDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(s)?;
    let idents = parser.parse_multipart_identifier()?;
    Ok(idents)
}

pub(crate) fn parse_identifiers_normalized(s: &str, ignore_case: bool) -> Vec<String> {
    parse_identifiers(s)
        .unwrap_or_default()
        .into_iter()
        .map(|id| match id.quote_style {
            Some(_) => id.value,
            None if ignore_case => id.value,
            _ => id.value.to_ascii_lowercase(),
        })
        .collect::<Vec<_>>()
}

impl std::ops::Deref for MultiTableReference {
    type Target = Vec<Arc<str>>;

    fn deref(&self) -> &Self::Target {
        &self.parts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_quoted_string_table_reference() {
        let table_ref = TableReference::Bare {
            table: "simple".into(),
        };
        let multi = MultiPartTableReference::TableReference(table_ref);
        assert_eq!(multi.to_quoted_string(), r#""simple""#);
    }

    #[test]
    fn test_to_quoted_string_multi() {
        let parts = MultiTableReference {
            parts: vec![
                Arc::from("a"),
                Arc::from("b"),
                Arc::from("c"),
                Arc::from("d"),
            ],
        };
        let multi = MultiPartTableReference::Multi(parts);
        assert_eq!(multi.to_quoted_string(), r#""a"."b"."c"."d""#);
    }

    #[test]
    fn test_quote_identifier() {
        // Test basic identifier
        assert_eq!(quote_identifier("simple"), r#""simple""#);

        // Test identifier with dots
        assert_eq!(quote_identifier("table.name"), r#""table.name""#);

        // Test identifier with quotes
        assert_eq!(quote_identifier(r#"table"name"#), r#""table""name""#);

        // Test identifier with both dots and quotes
        assert_eq!(quote_identifier(r#"my.table"name"#), r#""my.table""name""#);
    }

    #[test]
    fn test_to_quoted_string_partial_reference() {
        let table_ref = TableReference::Partial {
            schema: "my.schema".into(),
            table: "table.name".into(),
        };
        let multi = MultiPartTableReference::TableReference(table_ref);
        assert_eq!(multi.to_quoted_string(), r#""my.schema"."table.name""#);
    }

    #[test]
    fn test_to_quoted_string_full_reference() {
        let table_ref = TableReference::Full {
            catalog: "my.catalog".into(),
            schema: "my.schema".into(),
            table: "table.name".into(),
        };
        let multi = MultiPartTableReference::TableReference(table_ref);
        assert_eq!(
            multi.to_quoted_string(),
            r#""my.catalog"."my.schema"."table.name""#
        );
    }

    #[test]
    fn test_to_quoted_string_with_quotes() {
        let table_ref = TableReference::Bare {
            table: r#"my"table"#.into(),
        };
        let multi = MultiPartTableReference::TableReference(table_ref);
        assert_eq!(multi.to_quoted_string(), r#""my""table""#);
    }

    #[test]
    fn test_to_quoted_string_multi_with_special_chars() {
        let parts = vec![
            Arc::from("my.catalog"),
            Arc::from(r#"special"schema"#),
            Arc::from("table.name"),
            Arc::from(r#"part"4"#),
        ];
        let multi = MultiPartTableReference::Multi(MultiTableReference {
            parts: parts.into_iter().map(Arc::from).collect(),
        });
        assert_eq!(
            multi.to_quoted_string(),
            r#""my.catalog"."special""schema"."table.name"."part""4""#
        );
    }

    #[test]
    fn test_parse_identifiers() -> DataFusionResult<()> {
        // Test simple identifiers
        let idents = parse_identifiers("table")?;
        assert_eq!(idents.len(), 1);
        assert_eq!(idents[0].value, "table");
        assert!(idents[0].quote_style.is_none());

        // Test multi-part identifiers
        let idents = parse_identifiers("schema.table")?;
        assert_eq!(idents.len(), 2);
        assert_eq!(idents[0].value, "schema");
        assert_eq!(idents[1].value, "table");

        // Test quoted identifiers
        let idents = parse_identifiers(r#""My.Schema"."Table.Name""#)?;
        assert_eq!(idents.len(), 2);
        assert_eq!(idents[0].value, "My.Schema");
        assert_eq!(idents[1].value, "Table.Name");
        assert!(idents[0].quote_style.is_some());
        assert!(idents[1].quote_style.is_some());

        // Test mixed quoted and unquoted
        let idents = parse_identifiers(r#"catalog."schema.name".table"#)?;
        assert_eq!(idents.len(), 3);
        assert_eq!(idents[0].value, "catalog");
        assert_eq!(idents[1].value, "schema.name");
        assert_eq!(idents[2].value, "table");
        assert!(idents[0].quote_style.is_none());
        assert!(idents[1].quote_style.is_some());
        assert!(idents[2].quote_style.is_none());

        Ok(())
    }

    #[test]
    fn test_parse_identifiers_normalized() {
        // Test case-sensitive (ignore_case = false)
        let parts = parse_identifiers_normalized("MyTable", false);
        assert_eq!(parts, vec!["mytable"]);

        let parts = parse_identifiers_normalized(r#""MyTable""#, false);
        assert_eq!(parts, vec!["MyTable"]);

        // Test case-insensitive (ignore_case = true)
        let parts = parse_identifiers_normalized("MyTable", true);
        assert_eq!(parts, vec!["MyTable"]);

        // Test multi-part identifiers
        let parts = parse_identifiers_normalized("Schema.MyTable", false);
        assert_eq!(parts, vec!["schema", "mytable"]);

        // Test quoted identifiers with special characters
        let parts = parse_identifiers_normalized(r#""My.Schema"."Table.Name""#, false);
        assert_eq!(parts, vec!["My.Schema", "Table.Name"]);

        // Test invalid SQL (should return empty vec)
        let parts = parse_identifiers_normalized("invalid..sql", false);
        assert!(parts.is_empty());
    }

    #[test]
    fn test_parse_multi_part_table_reference() {
        // Test single part
        let table_ref = parse_multi_part_table_reference("table");
        assert!(matches!(
            table_ref,
            MultiPartTableReference::TableReference(TableReference::Bare { table })
            if table == "table".into()
        ));

        // Test two parts
        let table_ref = parse_multi_part_table_reference("schema.table");
        assert!(matches!(
            table_ref,
            MultiPartTableReference::TableReference(TableReference::Partial { schema, table })
            if schema == "schema".into() && table == "table".into()
        ));

        // Test three parts
        let table_ref = parse_multi_part_table_reference("catalog.schema.table");
        assert!(matches!(
            table_ref,
            MultiPartTableReference::TableReference(TableReference::Full { catalog, schema, table })
            if catalog == "catalog".into() && schema == "schema".into() && table == "table".into()
        ));

        // Test quoted identifiers
        let table_ref = parse_multi_part_table_reference(r#""My.Catalog"."Schema"."Table""#);
        assert!(matches!(
            table_ref,
            MultiPartTableReference::TableReference(TableReference::Full { catalog, schema, table })
            if catalog == "My.Catalog".into() && schema == "Schema".into() && table == "Table".into()
        ));

        // Test more than three parts (should join with UNIT_SEPARATOR)
        let table_ref = parse_multi_part_table_reference("a.b.c.d");
        assert!(matches!(
            table_ref,
            MultiPartTableReference::Multi(parts)
            if parts == MultiTableReference {
                parts: vec![Arc::from("a"), Arc::from("b"), Arc::from("c"), Arc::from("d")],
            }
        ));
    }

    #[test]
    fn test_from_slice_implementations() {
        let string_slice = &["a", "b", "c"][..];
        let ref_string_slice = &[String::from("a"), String::from("b")][..];

        let from_str_slice = MultiPartTableReference::from(string_slice);
        let from_string_slice = MultiPartTableReference::from(ref_string_slice);

        assert!(matches!(from_str_slice,
            MultiPartTableReference::Multi(parts)
            if parts == MultiTableReference {
                parts: vec![Arc::from("a"), Arc::from("b"), Arc::from("c")],
            }
        ));

        assert!(matches!(from_string_slice,
            MultiPartTableReference::Multi(parts)
            if parts == MultiTableReference {
                parts: vec![Arc::from("a"), Arc::from("b")],
            }
        ));
    }

    #[test]
    fn test_from_array_implementations() {
        let str_array = ["a", "b", "c"];
        let string_array = [String::from("a"), String::from("b")];

        let from_str_array = MultiPartTableReference::from(str_array);
        let from_string_array = MultiPartTableReference::from(string_array);

        assert!(matches!(from_str_array,
            MultiPartTableReference::Multi(parts)
            if parts == MultiTableReference {
                parts: vec![Arc::from("a"), Arc::from("b"), Arc::from("c")],
            }
        ));

        assert!(matches!(from_string_array,
            MultiPartTableReference::Multi(parts)
            if parts == MultiTableReference {
                parts: vec![Arc::from("a"), Arc::from("b")],
            }
        ));
    }

    #[test]
    fn test_table_reference_equality() {
        let table_ref = TableReference::Bare {
            table: "mytable".into(),
        };
        let multi_ref = MultiPartTableReference::TableReference(table_ref.clone());
        let multi_parts = MultiPartTableReference::Multi(MultiTableReference {
            parts: vec![Arc::from("a"), Arc::from("b")],
        });

        // Test equality between MultiPartTableReference and TableReference
        assert_eq!(multi_ref, table_ref);
        assert_eq!(table_ref, multi_ref);

        // Test inequality
        assert_ne!(multi_parts, table_ref);
        assert_ne!(table_ref, multi_parts);

        // Test with different TableReference variants
        let different_ref = TableReference::Partial {
            schema: "schema".into(),
            table: "mytable".into(),
        };
        assert_ne!(multi_ref, different_ref);
        assert_ne!(different_ref, multi_ref);
    }
}
