use crate::bench::Query;
use anyhow::{anyhow, Result};
use chrono::{DateTime, NaiveDate};
use datafusion::{
    arrow::array::{
        Array, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array, Int16Array,
        Int32Array, Int64Array, Int8Array, LargeStringArray, RecordBatch, StringArray,
        StringViewArray, TimestampMicrosecondArray, TimestampMillisecondArray,
        TimestampNanosecondArray, TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array,
        UInt8Array,
    },
    arrow::csv::{reader::Format, ReaderBuilder},
    arrow::datatypes::{DataType, SchemaRef, TimeUnit},
};
use std::{
    collections::BTreeMap,
    io::Seek,
    sync::{Arc, LazyLock},
};

macro_rules! generate_tpch_answers {
    ( $( $i:tt ),* ) => {
        vec![
            $(
                (
                    concat!("tpch_q", stringify!($i)),
                    include_str!(concat!("./tpch/q", stringify!($i), ".csv"))
                )
            ),*
        ]
    }
}

static TPCH_ANSWERS: LazyLock<BTreeMap<Arc<str>, Vec<RecordBatch>>> = LazyLock::new(|| {
    #[allow(clippy::expect_used)]
    {
        let mut map = BTreeMap::new();
        let answers = generate_tpch_answers!(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
        );

        for (query_name, csv_contents) in answers {
            let mut string_reader = std::io::Cursor::new(csv_contents);
            let format = Format::default().with_delimiter(b'|').with_header(true);
            let (schema, _) = format
                .infer_schema(&mut string_reader, None)
                .expect("Should infer schema");
            string_reader.rewind().expect("Should rewind file");

            // create a builder
            let reader = ReaderBuilder::new(Arc::new(schema))
                .with_format(format.clone())
                .build(string_reader)
                .expect("Should build reader");

            // read the batches
            let mut batches = Vec::new();
            for batch in reader {
                let batch = batch.expect("Should read batch");
                batches.push(batch);
            }

            // Store the batches in the map
            map.insert(query_name.into(), batches.clone());
            map.insert(
                query_name.replace("tpch_", "tpch[parameterized]_").into(),
                batches,
            );
        }

        map
    }
});

macro_rules! downcast_and_stringify {
    ($array:expr, $index:expr, $t:ty) => {{
        Ok(Some(
            $array
                .as_any()
                .downcast_ref::<$t>()
                .ok_or_else(|| anyhow!("Failed to downcast array"))?
                .value($index)
                .to_string(),
        ))
    }};
}

macro_rules! downcast_and_stringify_ts {
    ($array:expr, $index:expr, $t:ty, $scale:expr, $format:expr) => {{
        let ts = $array
            .as_any()
            .downcast_ref::<$t>()
            .ok_or_else(|| anyhow!("Failed to downcast timestamp array"))?
            .value($index);
        let secs = ts / $scale;
        let sub = ts.rem_euclid($scale);
        let sub_u32 = u32::try_from(sub)
            .map_err(|_| anyhow!("Subsecond value out of range for u32: {}", sub))?;
        let nanos = sub_u32 * (1_000_000_000u32 / $scale as u32);
        let dt = DateTime::from_timestamp(secs, nanos)
            .ok_or_else(|| anyhow!("Invalid timestamp from seconds={} nanos={}", secs, nanos))?;
        Ok(Some(dt.format($format).to_string()))
    }};
}

pub fn validate_tpch_query(query: &Query, batches: &[RecordBatch]) -> Result<()> {
    let Some(expected_batches) = TPCH_ANSWERS.get(&query.name) else {
        return Err(anyhow!("No expected answer for query {}", query.name));
    };

    match (expected_batches.is_empty(), batches.is_empty()) {
        (true, true) | (false, false) => {}
        (true, false) => {
            return Err(anyhow!("No expected answer for query {}", query.name));
        }
        _ => {
            return Err(anyhow!("No answer for query {}", query.name));
        }
    }

    let Some(expected_schema) = expected_batches
        .first()
        .map(datafusion::arrow::array::RecordBatch::schema)
    else {
        return Err(anyhow!("No answer for query {}", query.name));
    };
    let Some(actual_schema) = batches
        .first()
        .map(datafusion::arrow::array::RecordBatch::schema)
    else {
        return Err(anyhow!("No answer for query {}", query.name));
    };

    if !equivalent_schemas(&expected_schema, &actual_schema) {
        println!("expected_schema: {expected_schema:?}");
        println!("actual_schema: {actual_schema:?}");

        return Err(anyhow!("Schema mismatch for query {}", query.name));
    }

    let expected_batches =
        datafusion::arrow::compute::concat_batches(&expected_schema, expected_batches)?;
    let actual_batches = datafusion::arrow::compute::concat_batches(&actual_schema, batches)?;

    if expected_batches.num_rows() != actual_batches.num_rows() {
        return Err(anyhow!("Row count mismatch for query {}", query.name));
    }

    validate_batches_as_strings(&expected_batches, &actual_batches)?;
    println!("Query {} passed validation", query.name);

    Ok(())
}

fn datatype_equivalent(expected_type: DataType, actual_type: DataType) -> bool {
    if expected_type == actual_type {
        return true;
    }

    // Check for logical equivalence, with a lenient set of rules
    // E.g. a number could be returned as a string, number, or float.
    matches!(
        (expected_type, actual_type),
        (DataType::Float32, DataType::Float64)
            | (
                DataType::Float64 | DataType::Int64,
                DataType::Decimal128(_, _)
            )
            | (DataType::Int32, DataType::Int64)
            | (
                DataType::Int64,
                DataType::Int32
                    | DataType::Float64
                    | DataType::Utf8
                    | DataType::LargeUtf8
                    | DataType::Utf8View
            )
            | (DataType::Utf8, DataType::LargeUtf8)
            | (DataType::LargeUtf8, DataType::Utf8)
    )
}

fn equivalent_schemas(expected_schema: &SchemaRef, actual_schema: &SchemaRef) -> bool {
    if expected_schema.fields().len() != actual_schema.fields().len() {
        return false;
    }

    expected_schema
        .fields()
        .iter()
        .zip(actual_schema.fields().iter())
        .all(|(f1, f2)| datatype_equivalent(f1.data_type().clone(), f2.data_type().clone()))
}

pub fn validate_batches_as_strings(expected: &RecordBatch, actual: &RecordBatch) -> Result<()> {
    let schema = expected.schema();

    for (i, field) in schema.fields().iter().enumerate() {
        let column_name = field.name().clone();
        let data_type = field.data_type();
        let expected_array = expected.column(i).as_ref();
        let actual_array = actual.column(i).as_ref();

        if expected_array.len() != actual_array.len() {
            return Err(anyhow!(
                "Column {} has different lengths: expected {}, actual {}",
                column_name,
                expected_array.len(),
                actual_array.len()
            ));
        }

        for row in 0..expected_array.len() {
            let expected_val = array_value_to_string(expected_array, row)?;
            let actual_val = array_value_to_string(actual_array, row)?;

            match (expected_val, actual_val) {
                (None, None) => {}
                (Some(val), None) => {
                    return Err(anyhow!(
                        "Column {} has different values: expected {}, actual None",
                        column_name,
                        val
                    ));
                }
                (None, Some(val)) => {
                    return Err(anyhow!(
                        "Column {} has different values: expected None, actual {}",
                        column_name,
                        val
                    ));
                }
                (Some(expected_val), Some(actual_val)) => {
                    if expected_val != actual_val {
                        if data_type.is_numeric() {
                            let delta = 0.05;

                            if let (Ok(expected_num), Ok(actual_num)) =
                                (expected_val.parse::<f64>(), actual_val.parse::<f64>())
                            {
                                let diff = (expected_num - actual_num).abs();
                                let tolerance = (expected_num.abs() * delta).max(1e-12); // avoid zero-multiplied tolerance
                                if diff <= tolerance {
                                    continue; // numeric match within tolerance
                                }
                            }
                        }

                        return Err(anyhow!(
                            "Column {} has different values: expected {}, actual {}",
                            column_name,
                            expected_val,
                            actual_val
                        ));
                    }
                }
            }
        }
    }

    Ok(())
}

pub fn array_value_to_string(array: &dyn Array, index: usize) -> Result<Option<String>> {
    if array.len() <= index {
        return Err(anyhow!("Index out of bounds: {index} >= {}", array.len()));
    }

    if array.is_null(index) {
        return Ok(None);
    }

    match array.data_type() {
        DataType::Int64 => downcast_and_stringify!(array, index, Int64Array),
        DataType::Int32 => downcast_and_stringify!(array, index, Int32Array),
        DataType::Int16 => downcast_and_stringify!(array, index, Int16Array),
        DataType::Int8 => downcast_and_stringify!(array, index, Int8Array),
        DataType::UInt64 => downcast_and_stringify!(array, index, UInt64Array),
        DataType::UInt32 => downcast_and_stringify!(array, index, UInt32Array),
        DataType::UInt16 => downcast_and_stringify!(array, index, UInt16Array),
        DataType::UInt8 => downcast_and_stringify!(array, index, UInt8Array),
        DataType::Float32 => downcast_and_stringify!(array, index, Float32Array),
        DataType::Float64 => downcast_and_stringify!(array, index, Float64Array),
        DataType::Utf8 => downcast_and_stringify!(array, index, StringArray),
        DataType::LargeUtf8 => downcast_and_stringify!(array, index, LargeStringArray),
        DataType::Utf8View => downcast_and_stringify!(array, index, StringViewArray),
        DataType::Boolean => downcast_and_stringify!(array, index, BooleanArray),

        DataType::Date32 => {
            let days = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .ok_or_else(|| anyhow!("Failed to downcast Date32 array"))?
                .value(index);
            let date = NaiveDate::from_ymd_opt(1970, 1, 1)
                .ok_or_else(|| anyhow!("Invalid base date"))?
                .checked_add_signed(chrono::Duration::days(i64::from(days)))
                .ok_or_else(|| anyhow!("Date out of range"))?;
            Ok(Some(date.format("%Y-%m-%d").to_string()))
        }

        DataType::Decimal128(_, scale) => {
            let val = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| anyhow!("Failed to downcast Decimal128 array"))?
                .value(index);

            let sign = if val < 0 { "-" } else { "" };
            let abs_val = val.abs();
            let scale = usize::try_from(*scale)?; // Convert scale to usize

            let str_val = abs_val.to_string(); // Convert the absolute value to a string

            // Split the string into integer and fractional parts
            let len = str_val.len();
            let (int_part, frac_part) = if len > scale {
                let (a, b) = str_val.split_at(len - scale);
                (a.to_string(), b.to_string())
            } else {
                ("0".to_string(), format!("{str_val:0>scale$}"))
            };

            if frac_part.is_empty() {
                Ok(Some(format!("{sign}{int_part}")))
            } else {
                Ok(Some(format!("{sign}{int_part}.{frac_part}")))
            }
        }

        DataType::Timestamp(unit, _) => match unit {
            TimeUnit::Second => {
                let ts = array
                    .as_any()
                    .downcast_ref::<TimestampSecondArray>()
                    .ok_or_else(|| anyhow!("Failed to downcast TimestampSecondArray"))?
                    .value(index);
                let dt = DateTime::from_timestamp(ts, 0)
                    .ok_or_else(|| anyhow!("Invalid timestamp for seconds={}", ts))?;
                Ok(Some(dt.format("%Y-%m-%d %H:%M:%S").to_string()))
            }
            TimeUnit::Millisecond => {
                let ts = array
                    .as_any()
                    .downcast_ref::<TimestampMillisecondArray>()
                    .ok_or_else(|| anyhow!("Failed to downcast TimestampMillisecondArray"))?
                    .value(index);
                let secs = ts / 1000;
                let sub_ms = ts.rem_euclid(1000);
                let sub_u32 = u32::try_from(sub_ms)?;
                let nanos = sub_u32 * 1_000_000;
                let dt = DateTime::from_timestamp(secs, nanos)
                    .ok_or_else(|| anyhow!("Invalid timestamp"))?;
                Ok(Some(dt.format("%Y-%m-%d %H:%M:%S%.3f").to_string()))
            }
            TimeUnit::Microsecond => {
                downcast_and_stringify_ts!(
                    array,
                    index,
                    TimestampMicrosecondArray,
                    1_000_000,
                    "%Y-%m-%d %H:%M:%S%.6f"
                )
            }
            TimeUnit::Nanosecond => {
                downcast_and_stringify_ts!(
                    array,
                    index,
                    TimestampNanosecondArray,
                    1_000_000_000,
                    "%Y-%m-%d %H:%M:%S%.9f"
                )
            }
        },

        dt => Err(anyhow::anyhow!(
            "Unsupported data type for validation: {dt:?}",
        )),
    }
}
