use datafusion::arrow::{
    array::{Array, RecordBatch, RecordBatchOptions},
    compute::{cast_with_options, CastOptions},
    datatypes::{DataType, IntervalUnit, SchemaRef},
};
use std::sync::Arc;

use super::{
    intervals_cast::{
        cast_interval_monthdaynano_to_daytime, cast_interval_monthdaynano_to_yearmonth,
    },
    lists_cast::{cast_string_to_fixed_size_list, cast_string_to_large_list, cast_string_to_list},
    struct_cast::cast_string_to_struct,
};

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    UnableToConvertRecordBatch {
        source: datafusion::arrow::error::ArrowError,
    },

    UnableToCastColumn {
        source: datafusion::arrow::error::ArrowError,
        column_index: usize,
        column_name: String,
        from_type: DataType,
        to_type: DataType,
    },

    UnexpectedNumberOfColumns {
        expected: usize,
        found: usize,
    },
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::UnableToConvertRecordBatch { source } => {
                write!(f, "Unable to convert record batch: {source}")
            }
            Error::UnableToCastColumn {
                source,
                column_index,
                column_name,
                from_type,
                to_type,
            } => {
                write!(
                    f,
                    "Unable to cast column {column_index} '{column_name}' from {from_type} to {to_type}: {source}"
                )
            }
            Error::UnexpectedNumberOfColumns { expected, found } => {
                write!(
                    f,
                    "Unexpected number of columns. Expected: {expected}, Found: {found}",
                )
            }
        }
    }
}

/// Cast a given record batch into a new record batch with the given schema.
/// It assumes the record batch columns are correctly ordered.
#[allow(clippy::needless_pass_by_value)]
pub fn try_cast_to(record_batch: RecordBatch, expected_schema: SchemaRef) -> Result<RecordBatch> {
    let actual_schema = record_batch.schema();

    if actual_schema.fields().len() != expected_schema.fields().len() {
        tracing::debug!(
            actual_schema = ?actual_schema,
            expected_schema = ?expected_schema,
            "Schema mismatch in try_cast_to"
        );
        tracing::trace!(record_batch = ?record_batch, "Record batch contents");
        return Err(Error::UnexpectedNumberOfColumns {
            expected: expected_schema.fields().len(),
            found: actual_schema.fields().len(),
        });
    }

    let cast_options = CastOptions {
        safe: false,
        ..CastOptions::default()
    };

    let cols = expected_schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, expected_field)| {
            let record_batch_col = record_batch.column(i);
            let from_type = record_batch_col.data_type().clone();
            let to_type = expected_field.data_type().clone();
            let make_err = |e| Error::UnableToCastColumn {
                source: e,
                column_index: i,
                column_name: expected_field.name().clone(),
                from_type: from_type.clone(),
                to_type: to_type.clone(),
            };

            match (record_batch_col.data_type(), expected_field.data_type()) {
                (DataType::Utf8, DataType::List(item_type)) => {
                    cast_string_to_list::<i32>(record_batch_col, item_type).map_err(make_err)
                }
                (DataType::Utf8, DataType::LargeList(item_type)) => {
                    cast_string_to_large_list::<i32>(record_batch_col, item_type).map_err(make_err)
                }
                (DataType::Utf8, DataType::FixedSizeList(item_type, value_length)) => {
                    cast_string_to_fixed_size_list::<i32>(
                        record_batch_col,
                        item_type,
                        *value_length,
                    )
                    .map_err(make_err)
                }
                (DataType::Utf8, DataType::Struct(_)) => {
                    cast_string_to_struct::<i32>(record_batch_col, expected_field.clone())
                        .map_err(make_err)
                }
                (DataType::LargeUtf8, DataType::List(item_type)) => {
                    cast_string_to_list::<i64>(record_batch_col, item_type).map_err(make_err)
                }
                (DataType::LargeUtf8, DataType::LargeList(item_type)) => {
                    cast_string_to_large_list::<i64>(record_batch_col, item_type).map_err(make_err)
                }
                (DataType::LargeUtf8, DataType::FixedSizeList(item_type, value_length)) => {
                    cast_string_to_fixed_size_list::<i64>(
                        record_batch_col,
                        item_type,
                        *value_length,
                    )
                    .map_err(make_err)
                }
                (DataType::LargeUtf8, DataType::Struct(_)) => {
                    cast_string_to_struct::<i64>(record_batch_col, expected_field.clone())
                        .map_err(make_err)
                }
                (
                    DataType::Interval(IntervalUnit::MonthDayNano),
                    DataType::Interval(IntervalUnit::YearMonth),
                ) => cast_interval_monthdaynano_to_yearmonth(record_batch_col).map_err(make_err),
                (
                    DataType::Interval(IntervalUnit::MonthDayNano),
                    DataType::Interval(IntervalUnit::DayTime),
                ) => cast_interval_monthdaynano_to_daytime(record_batch_col).map_err(make_err),
                _ => cast_with_options(
                    record_batch_col.as_ref(),
                    expected_field.data_type(),
                    &cast_options,
                )
                .map_err(make_err),
            }
        })
        .collect::<Result<Vec<Arc<dyn Array>>>>()
        .inspect_err(|_| {
            tracing::debug!(
                actual_schema = ?actual_schema,
                expected_schema = ?expected_schema,
                "Cast error in try_cast_to"
            );
            tracing::trace!(record_batch = ?record_batch, "Record batch contents");
        })?;

    let options = RecordBatchOptions::new().with_row_count(Some(record_batch.num_rows()));
    RecordBatch::try_new_with_options(expected_schema.clone(), cols, &options).map_err(|e| {
        tracing::debug!(
            actual_schema = ?actual_schema,
            expected_schema = ?expected_schema,
            "RecordBatch creation error in try_cast_to"
        );
        tracing::trace!(record_batch = ?record_batch, "Record batch contents");
        Error::UnableToConvertRecordBatch { source: e }
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use datafusion::arrow::array::{Decimal128Array, LargeStringArray, RecordBatchOptions};
    use datafusion::arrow::{
        array::{Int32Array, StringArray},
        datatypes::{DataType, Field, Schema, TimeUnit},
    };
    use datafusion::assert_batches_eq;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
            Field::new("c", DataType::Utf8, false),
        ]))
    }

    fn to_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::LargeUtf8, false),
            Field::new("c", DataType::Timestamp(TimeUnit::Microsecond, None), false),
        ]))
    }

    fn batch_input() -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["foo", "bar", "baz"])),
                Arc::new(StringArray::from(vec![
                    "2024-01-13 03:18:09.000000",
                    "2024-01-13 03:18:09",
                    "2024-01-13 03:18:09.000",
                ])),
            ],
        )
        .expect("record batch should not panic")
    }

    #[test]
    fn test_string_to_timestamp_conversion() {
        let result = try_cast_to(batch_input(), to_schema()).expect("converted");
        let expected = [
            "+---+-----+---------------------+",
            "| a | b   | c                   |",
            "+---+-----+---------------------+",
            "| 1 | foo | 2024-01-13T03:18:09 |",
            "| 2 | bar | 2024-01-13T03:18:09 |",
            "| 3 | baz | 2024-01-13T03:18:09 |",
            "+---+-----+---------------------+",
        ];

        assert_batches_eq!(expected, &[result]);
    }

    fn large_string_from_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::LargeUtf8, false),
            Field::new("c", DataType::LargeUtf8, false),
        ]))
    }

    fn large_string_to_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::LargeUtf8, false),
            Field::new("c", DataType::Timestamp(TimeUnit::Microsecond, None), false),
        ]))
    }

    fn large_string_batch_input() -> RecordBatch {
        RecordBatch::try_new(
            large_string_from_schema(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(LargeStringArray::from(vec!["foo", "bar", "baz"])),
                Arc::new(LargeStringArray::from(vec![
                    "2024-01-13 03:18:09.000000",
                    "2024-01-13 03:18:09",
                    "2024-01-13 03:18:09.000",
                ])),
            ],
        )
        .expect("record batch should not panic")
    }

    #[test]
    fn test_large_string_to_timestamp_conversion() {
        let result =
            try_cast_to(large_string_batch_input(), large_string_to_schema()).expect("converted");
        let expected = [
            "+---+-----+---------------------+",
            "| a | b   | c                   |",
            "+---+-----+---------------------+",
            "| 1 | foo | 2024-01-13T03:18:09 |",
            "| 2 | bar | 2024-01-13T03:18:09 |",
            "| 3 | baz | 2024-01-13T03:18:09 |",
            "+---+-----+---------------------+",
        ];
        assert_batches_eq!(expected, &[result]);
    }

    #[test]
    fn test_convert_empty_batch() {
        let schema = SchemaRef::new(Schema::empty());
        let options = RecordBatchOptions::new().with_row_count(Some(10));
        let batch = RecordBatch::try_new_with_options(schema.clone(), vec![], &options)
            .expect("failed to create empty batch");
        let result = try_cast_to(batch, schema).expect("converted");
        let expected = ["++", "++", "++"];
        assert_batches_eq!(expected, &[result]);
    }

    /// Casting Decimal128(38,9) → Decimal128(38,27) must return an error when
    /// the upscale would overflow, instead of silently producing NULL.
    #[test]
    fn test_try_cast_to_decimal_overflow_returns_error() {
        // Value with 12 integer digits: 110_367_043_872.497010000
        // Internal i128 at scale 9 = 110367043872497010000
        let value_i128: i128 = 110_367_043_872_497_010_000;

        let source_schema = Arc::new(Schema::new(vec![Field::new(
            "sum_charge",
            DataType::Decimal128(38, 9),
            true,
        )]));

        let source_array = Decimal128Array::from(vec![Some(value_i128)])
            .with_precision_and_scale(38, 9)
            .expect("valid Decimal128(38,9)");

        let batch =
            RecordBatch::try_new(source_schema, vec![Arc::new(source_array)]).expect("valid batch");

        // Target schema with wider scale (38,27) — only allows 11 integer digits
        let target_schema = Arc::new(Schema::new(vec![Field::new(
            "sum_charge",
            DataType::Decimal128(38, 27),
            true,
        )]));

        let err =
            try_cast_to(batch, target_schema).expect_err("Decimal overflow should return an error");
        assert!(
            matches!(err, Error::UnableToCastColumn { .. }),
            "Expected UnableToCastColumn, got: {err:?}"
        );
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("is too large to store in a Decimal128"),
            "Expected overflow message, got: {err_msg}"
        );
    }

    /// Casting Decimal128 with values that fit should succeed.
    #[test]
    fn test_try_cast_to_decimal_no_overflow_succeeds() {
        // Value with 11 integer digits: 99_999_999_999.000000000 (fits in 38-27=11 digits)
        let value_i128: i128 = 99_999_999_999_000_000_000;

        let source_schema = Arc::new(Schema::new(vec![Field::new(
            "amount",
            DataType::Decimal128(38, 9),
            true,
        )]));

        let source_array = Decimal128Array::from(vec![Some(value_i128)])
            .with_precision_and_scale(38, 9)
            .expect("valid Decimal128(38,9)");

        let batch =
            RecordBatch::try_new(source_schema, vec![Arc::new(source_array)]).expect("valid batch");

        let target_schema = Arc::new(Schema::new(vec![Field::new(
            "amount",
            DataType::Decimal128(38, 27),
            true,
        )]));

        let result = try_cast_to(batch, target_schema);
        assert!(
            result.is_ok(),
            "Decimal cast should succeed when value fits: {result:?}"
        );
    }
}
