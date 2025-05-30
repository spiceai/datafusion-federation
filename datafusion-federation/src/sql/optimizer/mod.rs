//! A collection of optimizer rules that benefit SQL federation.

mod push_down_filter;
pub use push_down_filter::PushDownFilterFederation;

mod optimize_projections;
pub use optimize_projections::OptimizeProjectionsFederation;
