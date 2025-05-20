use std::sync::Arc;

pub trait Benchmark {
    /// Name of the benchmark (e.g., "tpch", "tpcds")
    fn name(&self) -> &str;

    /// Database file name (e.g., "tpch_sf1.db", "tpcds_sf1.db")
    fn db_file_name(&self) -> String;

    /// Get all table names for this benchmark
    fn table_names(&self) -> Vec<String>;

    /// Get all queries for this benchmark
    fn queries(&self) -> Vec<Query>;
}

pub struct Query {
    pub name: Arc<str>,
    pub sql: Arc<str>,
}

macro_rules! generate_tpch_queries {
     ( $( $i:tt ),* ) => {
         vec![
             $(
                 Query {
                     name: concat!("tpch_", stringify!($i)).into(),
                     sql: include_str!(concat!("../../queries/tpch/", stringify!($i), ".sql")).into(),
                 }
             ),*
         ]
     }
}

pub struct TpchBenchmark;

impl Benchmark for TpchBenchmark {
    fn name(&self) -> &str {
        "tpch"
    }

    fn db_file_name(&self) -> String {
        "tpch_sf1.db".to_string()
    }

    fn table_names(&self) -> Vec<String> {
        vec![
            "customer".to_string(),
            "orders".to_string(),
            "lineitem".to_string(),
            "part".to_string(),
            "supplier".to_string(),
            "partsupp".to_string(),
            "nation".to_string(),
            "region".to_string(),
        ]
    }

    fn queries(&self) -> Vec<Query> {
        generate_tpch_queries!(
            q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q16, q17, q18, q19, q20,
            q21, q22
        )
    }
}
