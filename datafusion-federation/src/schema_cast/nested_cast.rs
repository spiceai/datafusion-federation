use arrow_json::ReaderBuilder;
use datafusion::arrow::array::{
    new_empty_array, Array, ArrayRef, GenericStringArray, OffsetSizeTrait,
};
use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::arrow::error::ArrowError;
use std::sync::Arc;

/// Decodes a string column that holds one JSON value per row into the nested Arrow type
/// `expected`: a `List`, `LargeList`, `FixedSizeList` or `Struct` whose items may themselves be
/// nested (a list of structs, a struct holding a list, a list of lists), with any leaf type the
/// JSON reader accepts.
///
/// A null string and the JSON literal `null` both decode to a null value. Every row must hold
/// exactly one JSON value; a row that holds none or several would shift the rows after it, so
/// the cast fails instead.
pub(crate) fn cast_string_to_nested<O: OffsetSizeTrait>(
    array: &dyn Array,
    expected: &DataType,
) -> Result<ArrayRef, ArrowError> {
    let strings = array
        .as_any()
        .downcast_ref::<GenericStringArray<O>>()
        .ok_or_else(|| {
            ArrowError::CastError(format!(
                "Failed to decode value: unable to downcast {} to a string array",
                array.data_type()
            ))
        })?;

    if strings.is_empty() {
        return Ok(new_empty_array(expected));
    }

    // One decoder pass over the whole column. The batch size is the column length so a
    // single flush yields every row; the reader's default (1024) would stop decoding there.
    let field = Arc::new(Field::new("value", expected.clone(), true));
    let mut decoder = ReaderBuilder::new_with_field(field)
        .with_batch_size(strings.len())
        .build_decoder()
        .map_err(|e| ArrowError::CastError(format!("Failed to create JSON decoder: {e}")))?;

    for (row, value) in strings.iter().enumerate() {
        let bytes = value.map_or(b"null".as_slice(), str::as_bytes);
        let consumed = decoder.decode(bytes).map_err(|e| {
            ArrowError::CastError(format!(
                "Failed to decode {expected} from JSON value {}: {e}",
                shown(bytes)
            ))
        })?;
        // Each row must end at a value boundary: a value left open here would be completed
        // by the next row's text, and every row after it would shift by one.
        if decoder.has_partial_record() {
            return Err(ArrowError::CastError(format!(
                "Failed to decode {expected} from JSON: value {} is incomplete",
                shown(bytes)
            )));
        }
        // The decoder has buffered one value per earlier row, so this row must have added
        // exactly one: none would pull the next row's value up, two would push it down.
        let values = decoder.len() - row;
        if values != 1 {
            return Err(ArrowError::CastError(format!(
                "Failed to decode {expected} from JSON: value {} holds {values} JSON values, not one",
                shown(bytes)
            )));
        }
        // The decoder stops decoding once it holds as many rows as the column has, so a
        // surplus value in the last row is left in the buffer rather than counted.
        if consumed != bytes.len() {
            return Err(ArrowError::CastError(format!(
                "Failed to decode {expected} from JSON: value {} was not fully consumed ({consumed} of {} bytes)",
                shown(bytes),
                bytes.len()
            )));
        }
    }

    let batch = decoder
        .flush()
        .map_err(|e| ArrowError::CastError(format!("Failed to decode {expected} from JSON: {e}")))?
        .ok_or_else(|| {
            ArrowError::CastError(format!(
                "Failed to decode {expected} from JSON: {} strings decoded to no values",
                strings.len()
            ))
        })?;

    if batch.num_rows() != strings.len() {
        return Err(ArrowError::CastError(format!(
            "Failed to decode {expected} from JSON: {} strings decoded to {} values, so a string holds more or less than one JSON value",
            strings.len(),
            batch.num_rows()
        )));
    }

    Ok(Arc::clone(batch.column(0)))
}

/// The row text as it appears in an error: quoted, lossily decoded, and cut after
/// `SHOWN_BYTES` so a multi-megabyte value does not become a multi-megabyte message.
fn shown(bytes: &[u8]) -> String {
    const SHOWN_BYTES: usize = 256;
    if bytes.len() <= SHOWN_BYTES {
        format!("{:?}", String::from_utf8_lossy(bytes))
    } else {
        format!(
            "{:?}\u{2026} ({} bytes)",
            String::from_utf8_lossy(&bytes[..SHOWN_BYTES]),
            bytes.len()
        )
    }
}

#[cfg(test)]
mod test {
    use arrow_json::{writer::JsonArray, WriterBuilder};
    use datafusion::arrow::{
        array::{
            BooleanBuilder, Date32Array, Decimal128Array, FixedSizeListBuilder, Int32Array,
            Int32Builder, LargeListBuilder, LargeStringArray, ListArray, ListBuilder, RecordBatch,
            StringArray, StringBuilder, StructArray, StructBuilder, TimestampMicrosecondArray,
        },
        buffer::OffsetBuffer,
        datatypes::{DataType, Field, Fields, Schema, SchemaRef},
    };

    use crate::schema_cast::record_convert::try_cast_to;
    use datafusion::arrow::buffer::NullBuffer;

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
            Field::new("a", DataType::new_list(DataType::Int32, true), false),
            Field::new("b", DataType::new_large_list(DataType::Utf8, true), false),
            Field::new(
                "c",
                DataType::new_fixed_size_list(DataType::Boolean, 3, true),
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
        fixed_size_list_builder
            .values()
            .append_slice(&[true, false, true]);
        fixed_size_list_builder.append(true);
        fixed_size_list_builder
            .values()
            .append_slice(&[false, true, false]);
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

    fn author_fields() -> Fields {
        Fields::from(vec![Field::new("login", DataType::Utf8, true)])
    }

    /// The `comments` column of a GitHub issues dataset: a list of structs, one of which
    /// holds a nested struct, with a null leaf, an empty list and a null list.
    fn comment_fields() -> Fields {
        Fields::from(vec![
            Field::new("author", DataType::Struct(author_fields()), true),
            Field::new("body", DataType::Utf8, true),
        ])
    }

    fn comments_type() -> DataType {
        DataType::new_list(DataType::Struct(comment_fields()), true)
    }

    /// Appends one valid `{"author":{"login":…},"body":…}` comment to a comments item builder.
    fn append_comment(item: &mut StructBuilder, login: &str, body: Option<&str>) {
        let author = item.field_builder::<StructBuilder>(0).expect("author");
        author
            .field_builder::<StringBuilder>(0)
            .expect("login")
            .append_value(login);
        author.append(true);
        item.field_builder::<StringBuilder>(1)
            .expect("body")
            .append_option(body);
        item.append(true);
    }

    fn comments_expected() -> ListArray {
        let item_builder = StructBuilder::new(
            comment_fields(),
            vec![
                Box::new(StructBuilder::new(
                    author_fields(),
                    vec![Box::new(StringBuilder::new())],
                )),
                Box::new(StringBuilder::new()),
            ],
        );
        let mut list_builder = ListBuilder::new(item_builder);

        append_comment(list_builder.values(), "alice", Some("hello"));
        append_comment(list_builder.values(), "bob", None);
        list_builder.append(true);
        list_builder.append(true); // []
        list_builder.append_null(); // null
        append_comment(
            list_builder.values(),
            "carol",
            Some("a \"quoted\" body, with ünïcödé"),
        );
        list_builder.append(true);
        list_builder.finish()
    }

    #[test]
    fn a_list_of_structs_decodes_with_its_nested_struct_nulls_and_empty_list() {
        let strings = StringArray::from(vec![
            Some(
                r#"[{"author":{"login":"alice"},"body":"hello"},{"author":{"login":"bob"},"body":null}]"#,
            ),
            Some("[]"),
            None,
            Some(r#"[{"author":{"login":"carol"},"body":"a \"quoted\" body, with ünïcödé"}]"#),
        ]);
        let actual = cast_string_to_nested::<i32>(&strings, &comments_type()).expect("cast");
        let expected: ArrayRef = Arc::new(comments_expected());
        assert_eq!(&actual, &expected);
    }

    fn comment_list_builder() -> ListBuilder<StructBuilder> {
        ListBuilder::new(StructBuilder::new(
            comment_fields(),
            vec![
                Box::new(StructBuilder::new(
                    author_fields(),
                    vec![Box::new(StringBuilder::new())],
                )),
                Box::new(StringBuilder::new()),
            ],
        ))
    }

    #[test]
    fn the_json_literal_null_decodes_like_a_null_string() {
        let strings = StringArray::from(vec![Some("null"), None, Some("[]")]);
        let actual = cast_string_to_nested::<i32>(&strings, &comments_type()).expect("cast");

        let mut builder = comment_list_builder();
        builder.append_null();
        builder.append_null();
        builder.append(true);
        let expected: ArrayRef = Arc::new(builder.finish());
        assert_eq!(&actual, &expected);
    }

    #[test]
    fn a_large_string_column_decodes_a_large_list_of_structs() {
        let strings = LargeStringArray::from(vec![
            Some(r#"[{"author":{"login":"alice"},"body":"hello"}]"#),
            None,
        ]);
        let expected_type = DataType::new_large_list(DataType::Struct(comment_fields()), true);
        let actual = cast_string_to_nested::<i64>(&strings, &expected_type).expect("cast");

        let mut builder = LargeListBuilder::new(StructBuilder::new(
            comment_fields(),
            vec![
                Box::new(StructBuilder::new(
                    author_fields(),
                    vec![Box::new(StringBuilder::new())],
                )),
                Box::new(StringBuilder::new()),
            ],
        ));
        append_comment(builder.values(), "alice", Some("hello"));
        builder.append(true);
        builder.append_null();
        let expected: ArrayRef = Arc::new(builder.finish());
        assert_eq!(actual.data_type(), &expected_type);
        assert_eq!(&actual, &expected);
    }

    #[test]
    fn a_list_of_lists_decodes_each_level() {
        let strings = StringArray::from(vec![Some("[[1,2],[],null,[3]]"), Some("[]")]);
        let expected_type = DataType::new_list(DataType::new_list(DataType::Int32, true), true);

        let mut builder = ListBuilder::new(ListBuilder::new(Int32Builder::new()));
        builder.values().append_value([Some(1), Some(2)]);
        builder.values().append(true);
        builder.values().append_null();
        builder.values().append_value([Some(3)]);
        builder.append(true);
        builder.append(true);
        let expected: ArrayRef = Arc::new(builder.finish());

        let actual = cast_string_to_nested::<i32>(&strings, &expected_type).expect("cast");
        assert_eq!(actual.data_type(), &expected_type);
        assert_eq!(&actual, &expected);
    }

    #[test]
    fn a_struct_holding_a_list_of_structs_decodes_through_try_cast_to() {
        let struct_type = DataType::Struct(Fields::from(vec![
            Field::new("n", DataType::Int32, true),
            Field::new("comments", comments_type(), true),
        ]));
        let input = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec![
                Some(r#"{"n":1,"comments":[{"author":{"login":"alice"},"body":"hello"}]}"#),
                Some(r#"{"n":null,"comments":[]}"#),
                None,
            ]))],
        )
        .expect("input batch");
        let expected_schema = Arc::new(Schema::new(vec![Field::new(
            "s",
            struct_type.clone(),
            true,
        )]));

        let actual = try_cast_to(input, Arc::clone(&expected_schema)).expect("cast");
        assert_eq!(actual.schema(), expected_schema);

        let mut comments = comment_list_builder();
        append_comment(comments.values(), "alice", Some("hello"));
        comments.append(true);
        comments.append(true);
        comments.append_null();
        let expected = StructArray::try_new(
            match &struct_type {
                DataType::Struct(fields) => fields.clone(),
                other => unreachable!("{other} is a struct"),
            },
            vec![
                Arc::new(Int32Array::from(vec![Some(1), None, None])),
                Arc::new(comments.finish()),
            ],
            Some(NullBuffer::from(vec![true, true, false])),
        )
        .expect("expected struct");
        let expected: ArrayRef = Arc::new(expected);
        assert_eq!(actual.column(0), &expected);
    }

    /// Encodes `array` with Arrow's own JSON writer, one object per row, and returns the
    /// per-row JSON of its single column: the oracle shares nothing with the decoder.
    fn json_strings_of(array: ArrayRef) -> StringArray {
        let field = Field::new("v", array.data_type().clone(), true);
        let batch =
            RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![array]).expect("batch");
        let mut writer = WriterBuilder::new()
            .with_explicit_nulls(true)
            .build::<_, JsonArray>(Vec::new());
        writer.write(&batch).expect("write");
        writer.finish().expect("finish");
        let rows: Vec<serde_json::Value> =
            serde_json::from_slice(&writer.into_inner()).expect("json rows");
        StringArray::from(
            rows.iter()
                .map(|row| row["v"].to_string())
                .collect::<Vec<_>>(),
        )
    }

    /// Splits `items` into a `List` whose rows have the given `lengths`, encodes it with
    /// Arrow's JSON writer and asserts that decoding that JSON gives back the same list.
    fn assert_list_round_trips(items: ArrayRef, lengths: impl IntoIterator<Item = usize>) {
        let list: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new_list_field(items.data_type().clone(), true)),
            OffsetBuffer::from_lengths(lengths),
            items,
            None,
        ));
        let decoded =
            cast_string_to_nested::<i32>(&json_strings_of(Arc::clone(&list)), list.data_type())
                .unwrap_or_else(|e| panic!("decode {}: {e}", list.data_type()));
        assert_eq!(&decoded, &list);
    }

    #[test]
    fn temporal_and_decimal_items_round_trip_through_arrows_json_writer() {
        assert_list_round_trips(
            Arc::new(Date32Array::from(vec![Some(0), Some(19_723), None])),
            [2, 0, 1],
        );
        assert_list_round_trips(
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(1_700_000_000_123_456),
                None,
            ])),
            [2],
        );
        assert_list_round_trips(
            Arc::new(
                Decimal128Array::from(vec![Some(123_456), Some(-5), None])
                    .with_precision_and_scale(10, 3)
                    .expect("decimal"),
            ),
            [3],
        );
    }

    #[test]
    fn a_column_longer_than_the_json_readers_default_batch_decodes_every_row() {
        let n: usize = 3_000;
        let strings = StringArray::from((0..n).map(|i| Some(format!("[{i}]"))).collect::<Vec<_>>());
        let expected_type = DataType::new_list(DataType::Int32, true);
        let actual = cast_string_to_nested::<i32>(&strings, &expected_type).expect("cast");
        let mut builder = ListBuilder::new(Int32Builder::new());
        for i in 0..n {
            builder.append_value([Some(i32::try_from(i).expect("fits"))]);
        }
        let expected: ArrayRef = Arc::new(builder.finish());
        assert_eq!(&actual, &expected);

        let structs = StringArray::from(
            (0..n)
                .map(|i| Some(format!(r#"{{"n":{i}}}"#)))
                .collect::<Vec<_>>(),
        );
        let fields = Fields::from(vec![Field::new("n", DataType::Int32, true)]);
        let actual = cast_string_to_nested::<i32>(&structs, &DataType::Struct(fields.clone()))
            .expect("cast");
        let expected = StructArray::try_new(
            fields,
            vec![Arc::new(Int32Array::from_iter_values(
                (0..n).map(|i| i32::try_from(i).expect("fits")),
            ))],
            None,
        )
        .expect("expected structs");
        let expected: ArrayRef = Arc::new(expected);
        assert_eq!(&actual, &expected);
    }

    #[test]
    fn an_empty_column_decodes_to_an_empty_array_of_the_expected_type() {
        let strings = StringArray::from(Vec::<Option<&str>>::new());
        let actual = cast_string_to_nested::<i32>(&strings, &comments_type()).expect("cast");
        assert_eq!(actual.len(), 0);
        assert_eq!(actual.data_type(), &comments_type());
    }

    #[test]
    fn a_row_holding_two_json_values_none_or_malformed_json_is_an_error() {
        let expected_type = DataType::new_list(DataType::Int32, true);

        // Two values in one row are refused at that row, whether or not a later row with no
        // value would bring the total back to one per row.
        let two_values = StringArray::from(vec![Some("[1] [2]"), Some("[3]")]);
        let err =
            cast_string_to_nested::<i32>(&two_values, &expected_type).expect_err("two values");
        assert_eq!(
            err.to_string(),
            "Cast error: Failed to decode List(Int32) from JSON: value \"[1] [2]\" holds 2 JSON values, not one"
        );
        let compensated = StringArray::from(vec![Some("[1] [2]"), Some("")]);
        let err =
            cast_string_to_nested::<i32>(&compensated, &expected_type).expect_err("compensated");
        assert_eq!(
            err.to_string(),
            "Cast error: Failed to decode List(Int32) from JSON: value \"[1] [2]\" holds 2 JSON values, not one"
        );
        // In the last row the decoder stops at the column's row count, so the surplus value
        // is left unconsumed rather than counted.
        let two_values_last = StringArray::from(vec![Some("[3]"), Some("[1] [2]")]);
        let err =
            cast_string_to_nested::<i32>(&two_values_last, &expected_type).expect_err("two values");
        assert_eq!(
            err.to_string(),
            "Cast error: Failed to decode List(Int32) from JSON: value \"[1] [2]\" was not fully consumed (4 of 7 bytes)"
        );

        // No value in a row, empty or whitespace only.
        let no_value = StringArray::from(vec![Some(""), Some("[3]")]);
        let err = cast_string_to_nested::<i32>(&no_value, &expected_type).expect_err("no value");
        assert_eq!(
            err.to_string(),
            "Cast error: Failed to decode List(Int32) from JSON: value \"\" holds 0 JSON values, not one"
        );
        let blank = StringArray::from(vec![Some("[3]"), Some("  \n ")]);
        let err = cast_string_to_nested::<i32>(&blank, &expected_type).expect_err("blank");
        assert_eq!(
            err.to_string(),
            "Cast error: Failed to decode List(Int32) from JSON: value \"  \\n \" holds 0 JSON values, not one"
        );

        // A row that stops inside a value: the next row's text would complete it and every
        // row after that would shift by one, so the row is refused on its own.
        let incomplete = StringArray::from(vec![Some("[1, 2"), Some("[3]")]);
        let err =
            cast_string_to_nested::<i32>(&incomplete, &expected_type).expect_err("incomplete");
        assert_eq!(
            err.to_string(),
            "Cast error: Failed to decode List(Int32) from JSON: value \"[1, 2\" is incomplete"
        );
        let spanning = StringArray::from(vec![Some("[1,2"), Some("]"), Some("[7][8]")]);
        let err = cast_string_to_nested::<i32>(&spanning, &expected_type).expect_err("spanning");
        assert_eq!(
            err.to_string(),
            "Cast error: Failed to decode List(Int32) from JSON: value \"[1,2\" is incomplete"
        );

        let malformed = StringArray::from(vec![Some("[1, x]"), Some("[3]")]);
        let err = cast_string_to_nested::<i32>(&malformed, &expected_type).expect_err("malformed");
        assert!(
            err.to_string().starts_with(
                "Cast error: Failed to decode List(Int32) from JSON value \"[1, x]\": "
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn a_long_row_is_cut_in_the_error_message() {
        let expected_type = DataType::new_list(DataType::Int32, true);
        let long = format!("[{}", "1,".repeat(2_000));
        let strings = StringArray::from(vec![Some(long.as_str())]);
        let err = cast_string_to_nested::<i32>(&strings, &expected_type).expect_err("incomplete");
        let message = err.to_string();
        assert!(
            message.len() < 400,
            "message is {} bytes: {message}",
            message.len()
        );
        assert!(
            message.ends_with("\u{2026} (4001 bytes) is incomplete"),
            "unexpected error: {message}"
        );
    }
}
