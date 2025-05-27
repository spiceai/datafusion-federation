mod rewrite_multi_part;
mod table_arg;

use datafusion::{common::error::Result, sql::sqlparser::ast};
pub use rewrite_multi_part::RewriteMultiTableReference;
pub use table_arg::TableArgReplace;

pub type AstAnalyzerRule = Box<dyn FnMut(ast::Statement) -> Result<ast::Statement>>;

#[derive(Default)]
pub struct AstAnalyzer {
    rules: Vec<AstAnalyzerRule>,
}

impl AstAnalyzer {
    pub fn new(rules: Vec<AstAnalyzerRule>) -> Self {
        Self { rules }
    }

    pub fn add_rule(&mut self, rule: AstAnalyzerRule) {
        self.rules.push(rule)
    }

    pub fn analyze(&mut self, mut statement: ast::Statement) -> Result<ast::Statement> {
        for rule in self.rules.iter_mut() {
            statement = rule(statement)?;
        }

        Ok(statement)
    }
}
