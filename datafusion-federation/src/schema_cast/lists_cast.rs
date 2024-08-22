use datafusion::arrow::{
    array::{
        Array, ArrayRef, BooleanBuilder, FixedSizeListBuilder, Float32Builder, Float64Builder,
        Int16Builder, Int32Builder, Int64Builder, Int8Builder, LargeListBuilder, ListBuilder,
        StringArray, StringBuilder,
    },
    datatypes::{DataType, Field, FieldRef},
    error::ArrowError,
};
use std::sync::Arc;

pub type Result<T, E = crate::schema_cast::record_convert::Error> = std::result::Result<T, E>;

macro_rules! cast_string_to_list_array {
    ($string_array:expr, $field_name:expr, $data_type:expr, $builder_type:expr, $item_type:ty) => {{
        let item_field = Field::new($field_name, $data_type, true);
        let mut list_builder = ListBuilder::with_capacity($builder_type, $string_array.len())
            .with_field(Arc::new(item_field));

        for value in $string_array {
            match value {
                None => list_builder.append_null(),
                Some(string_value) => {
                    let items = serde_json::from_str::<Vec<Option<$item_type>>>(string_value)
                        .map_err(|e| {
                            ArrowError::CastError(format!("Failed to parse value: {e}"))
                        })?;
                    list_builder.append_value(items);
                }
            }
        }
        Ok(Arc::new(list_builder.finish()))
    }};
}

macro_rules! cast_string_to_large_list_array {
    ($string_array:expr, $field_name:expr, $data_type:expr, $builder_type:expr, $item_type:ty) => {{
        let item_field = Field::new($field_name, $data_type, true);
        let mut list_builder = LargeListBuilder::with_capacity($builder_type, $string_array.len())
            .with_field(Arc::new(item_field));

        for value in $string_array {
            match value {
                None => list_builder.append_null(),
                Some(string_value) => {
                    let items = serde_json::from_str::<Vec<Option<$item_type>>>(string_value)
                        .map_err(|e| {
                            ArrowError::CastError(format!("Failed to parse value: {e}"))
                        })?;
                    list_builder.append_value(items);
                }
            }
        }
        Ok(Arc::new(list_builder.finish()))
    }};
}

macro_rules! cast_string_to_fixed_size_list_array {
    ($string_array:expr, $field_name:expr, $data_type:expr, $builder_type:expr, $item_type:ty, $value_length:expr) => {{
        let item_field = Field::new($field_name, $data_type, true);
        let mut list_builder =
            FixedSizeListBuilder::with_capacity($builder_type, $value_length, $string_array.len())
                .with_field(Arc::new(item_field));

        for value in $string_array {
            match value {
                None => {
                    for _ in 0..$value_length {
                        list_builder.values().append_null()
                    }
                    list_builder.append(true)
                }
                Some(string_value) => {
                    let items = serde_json::from_str::<Vec<Option<$item_type>>>(string_value)
                        .map_err(|e| {
                            ArrowError::CastError(format!("Failed to parse value: {e}"))
                        })?;
                    for item in items {
                        match item {
                            Some(val) => list_builder.values().append_value(val),
                            None => list_builder.values().append_null(),
                        }
                    }
                    list_builder.append(true);
                }
            }
        }
        Ok(Arc::new(list_builder.finish()))
    }};
}

pub(crate) fn cast_string_to_list(
    array: &dyn Array,
    list_item_field: &FieldRef,
) -> Result<ArrayRef, ArrowError> {
    let string_array = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| ArrowError::CastError("Failed to downcast to StringArray".to_string()))?;

    let field_name = list_item_field.name();

    match list_item_field.data_type() {
        DataType::Utf8 => {
            cast_string_to_list_array!(
                string_array,
                field_name,
                DataType::Utf8,
                StringBuilder::new(),
                String
            )
        }
        DataType::Boolean => {
            cast_string_to_list_array!(
                string_array,
                field_name,
                DataType::Boolean,
                BooleanBuilder::new(),
                bool
            )
        }
        DataType::Int8 => {
            cast_string_to_list_array!(
                string_array,
                field_name,
                DataType::Int8,
                Int8Builder::new(),
                i8
            )
        }
        DataType::Int16 => {
            cast_string_to_list_array!(
                string_array,
                field_name,
                DataType::Int16,
                Int16Builder::new(),
                i16
            )
        }
        DataType::Int32 => {
            cast_string_to_list_array!(
                string_array,
                field_name,
                DataType::Int32,
                Int32Builder::new(),
                i32
            )
        }
        DataType::Int64 => {
            cast_string_to_list_array!(
                string_array,
                field_name,
                DataType::Int64,
                Int64Builder::new(),
                i64
            )
        }
        DataType::Float32 => {
            cast_string_to_list_array!(
                string_array,
                field_name,
                DataType::Float32,
                Float32Builder::new(),
                f32
            )
        }
        DataType::Float64 => {
            cast_string_to_list_array!(
                string_array,
                field_name,
                DataType::Float64,
                Float64Builder::new(),
                f64
            )
        }
        _ => Err(ArrowError::CastError(format!(
            "Unsupported list item type: {}",
            list_item_field.data_type()
        ))),
    }
}

pub(crate) fn cast_string_to_large_list(
    array: &dyn Array,
    list_item_field: &FieldRef,
) -> Result<ArrayRef, ArrowError> {
    let string_array = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| ArrowError::CastError("Failed to downcast to StringArray".to_string()))?;

    let field_name = list_item_field.name();

    match list_item_field.data_type() {
        DataType::Utf8 => {
            cast_string_to_large_list_array!(
                string_array,
                field_name,
                DataType::Utf8,
                StringBuilder::new(),
                String
            )
        }
        DataType::Boolean => {
            cast_string_to_large_list_array!(
                string_array,
                field_name,
                DataType::Boolean,
                BooleanBuilder::new(),
                bool
            )
        }
        DataType::Int8 => {
            cast_string_to_large_list_array!(
                string_array,
                field_name,
                DataType::Int8,
                Int8Builder::new(),
                i8
            )
        }
        DataType::Int16 => {
            cast_string_to_large_list_array!(
                string_array,
                field_name,
                DataType::Int16,
                Int16Builder::new(),
                i16
            )
        }
        DataType::Int32 => {
            cast_string_to_large_list_array!(
                string_array,
                field_name,
                DataType::Int32,
                Int32Builder::new(),
                i32
            )
        }
        DataType::Int64 => {
            cast_string_to_large_list_array!(
                string_array,
                field_name,
                DataType::Int64,
                Int64Builder::new(),
                i64
            )
        }
        DataType::Float32 => {
            cast_string_to_large_list_array!(
                string_array,
                field_name,
                DataType::Float32,
                Float32Builder::new(),
                f32
            )
        }
        DataType::Float64 => {
            cast_string_to_large_list_array!(
                string_array,
                field_name,
                DataType::Float64,
                Float64Builder::new(),
                f64
            )
        }
        _ => Err(ArrowError::CastError(format!(
            "Unsupported list item type: {}",
            list_item_field.data_type()
        ))),
    }
}

pub(crate) fn cast_string_to_fixed_size_list(
    array: &dyn Array,
    list_item_field: &FieldRef,
    value_length: i32,
) -> Result<ArrayRef, ArrowError> {
    let string_array = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| ArrowError::CastError("Failed to downcast to StringArray".to_string()))?;

    let field_name = list_item_field.name();

    match list_item_field.data_type() {
        DataType::Utf8 => {
            cast_string_to_fixed_size_list_array!(
                string_array,
                field_name,
                DataType::Utf8,
                StringBuilder::new(),
                String,
                value_length
            )
        }
        DataType::Boolean => {
            cast_string_to_fixed_size_list_array!(
                string_array,
                field_name,
                DataType::Boolean,
                BooleanBuilder::new(),
                bool,
                value_length
            )
        }
        DataType::Int8 => {
            cast_string_to_fixed_size_list_array!(
                string_array,
                field_name,
                DataType::Int8,
                Int8Builder::new(),
                i8,
                value_length
            )
        }
        DataType::Int16 => {
            cast_string_to_fixed_size_list_array!(
                string_array,
                field_name,
                DataType::Int16,
                Int16Builder::new(),
                i16,
                value_length
            )
        }
        DataType::Int32 => {
            cast_string_to_fixed_size_list_array!(
                string_array,
                field_name,
                DataType::Int32,
                Int32Builder::new(),
                i32,
                value_length
            )
        }
        DataType::Int64 => {
            cast_string_to_fixed_size_list_array!(
                string_array,
                field_name,
                DataType::Int64,
                Int64Builder::new(),
                i64,
                value_length
            )
        }
        DataType::Float32 => {
            cast_string_to_fixed_size_list_array!(
                string_array,
                field_name,
                DataType::Float32,
                Float32Builder::new(),
                f32,
                value_length
            )
        }
        DataType::Float64 => {
            cast_string_to_fixed_size_list_array!(
                string_array,
                field_name,
                DataType::Float64,
                Float64Builder::new(),
                f64,
                value_length
            )
        }
        _ => Err(ArrowError::CastError(format!(
            "Unsupported list item type: {}",
            list_item_field.data_type()
        ))),
    }
}

#[cfg(test)]
mod test {
    use datafusion::arrow::{
        array::{RecordBatch, StringArray},
        datatypes::{DataType, Field, Schema, SchemaRef},
    };

    use crate::schema_cast::record_convert::try_cast_to;

    use super::*;

    fn input_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Utf8, false),
            Field::new("b", DataType::Utf8, false),
            Field::new("c", DataType::Utf8, false),
        ]))
    }

    fn output_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new(
                "a",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                false,
            ),
            Field::new(
                "b",
                DataType::LargeList(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
            Field::new(
                "c",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Boolean, true)), 3),
                false,
            ),
        ]))
    }

    fn batch_input() -> RecordBatch {
        RecordBatch::try_new(
            input_schema(),
            vec![
                Arc::new(StringArray::from(vec![
                    Some("[1, 2, 3]"),
                    Some("[4, 5, 6]"),
                ])),
                Arc::new(StringArray::from(vec![
                    Some("[\"foo\", \"bar\"]"),
                    Some("[\"baz\", \"qux\"]"),
                ])),
                Arc::new(StringArray::from(vec![
                    Some("[true, false, true]"),
                    Some("[false, true, false]"),
                ])),
            ],
        )
        .expect("record batch should not panic")
    }

    fn batch_expected() -> RecordBatch {
        let mut list_builder = ListBuilder::new(Int32Builder::new());
        list_builder.append_value([Some(1), Some(2), Some(3)]);
        list_builder.append_value([Some(4), Some(5), Some(6)]);
        let list_array = list_builder.finish();

        let mut large_list_builder = LargeListBuilder::new(StringBuilder::new());
        large_list_builder.append_value([Some("foo"), Some("bar")]);
        large_list_builder.append_value([Some("baz"), Some("qux")]);
        let large_list_array = large_list_builder.finish();

        let mut fixed_size_list_builder = FixedSizeListBuilder::new(BooleanBuilder::new(), 3);
        fixed_size_list_builder.values().append_value(true);
        fixed_size_list_builder.values().append_value(false);
        fixed_size_list_builder.values().append_value(true);
        fixed_size_list_builder.append(true);
        fixed_size_list_builder.values().append_value(false);
        fixed_size_list_builder.values().append_value(true);
        fixed_size_list_builder.values().append_value(false);
        fixed_size_list_builder.append(true);
        let fixed_size_list_array = fixed_size_list_builder.finish();

        RecordBatch::try_new(
            output_schema(),
            vec![
                Arc::new(list_array),
                Arc::new(large_list_array),
                Arc::new(fixed_size_list_array),
            ],
        )
        .expect("Failed to create expected RecordBatch")
    }

    #[test]
    fn test_cast_to_list_largelist_fixedsizelist() {
        let input_batch = batch_input();
        let expected = batch_expected();
        let actual = try_cast_to(input_batch, output_schema()).expect("cast should succeed");

        assert_eq!(actual, expected);
    }
}
