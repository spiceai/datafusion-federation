use core::fmt;
use std::{
    fmt::Debug,
    hash::{Hash, Hasher},
    sync::Arc,
};

use async_trait::async_trait;
use datafusion::{
    common::DFSchemaRef,
    error::{DataFusionError, Result},
    execution::context::{QueryPlanner, SessionState},
    logical_expr::{
        Expr, Extension, LogicalPlan, UserDefinedLogicalNode, UserDefinedLogicalNodeCore,
    },
    physical_plan::ExecutionPlan,
    physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FederatedQueryType {
    Explain,
    Analyze,
}

impl FederatedQueryType {
    pub fn prefix(self) -> &'static str {
        match self {
            Self::Explain => "EXPLAIN",
            Self::Analyze => "EXPLAIN ANALYZE",
        }
    }
}

pub struct FederatedPlanNode {
    pub(crate) plan: LogicalPlan,
    pub(crate) planner: Arc<dyn FederationPlanner>,
    pub(crate) query_type: Option<FederatedQueryType>,
}

impl FederatedPlanNode {
    pub fn new(plan: LogicalPlan, planner: Arc<dyn FederationPlanner>) -> Self {
        Self::new_with_query_type(plan, planner, None)
    }

    pub fn new_with_query_type(
        plan: LogicalPlan,
        planner: Arc<dyn FederationPlanner>,
        query_type: Option<FederatedQueryType>,
    ) -> Self {
        Self {
            plan,
            planner,
            query_type,
        }
    }

    pub fn plan(&self) -> &LogicalPlan {
        &self.plan
    }

    pub fn planner(&self) -> &Arc<dyn FederationPlanner> {
        &self.planner
    }

    pub fn query_type(&self) -> Option<FederatedQueryType> {
        self.query_type
    }
}

impl Debug for FederatedPlanNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        UserDefinedLogicalNodeCore::fmt_for_explain(self, f)
    }
}

impl UserDefinedLogicalNodeCore for FederatedPlanNode {
    fn name(&self) -> &str {
        "Federated"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        Vec::new()
    }

    fn schema(&self) -> &DFSchemaRef {
        self.plan.schema()
    }

    fn expressions(&self) -> Vec<Expr> {
        Vec::new()
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Federated\n {}", self.plan)
    }

    fn with_exprs_and_inputs(&self, exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> Result<Self> {
        if !inputs.is_empty() {
            return Err(DataFusionError::Plan("input size inconsistent".into()));
        }
        if !exprs.is_empty() {
            return Err(DataFusionError::Plan("expression size inconsistent".into()));
        }

        Ok(Self {
            plan: self.plan.clone(),
            planner: Arc::clone(&self.planner),
            query_type: self.query_type,
        })
    }
}

#[derive(Default, Debug)]
pub struct FederatedQueryPlanner {}

impl FederatedQueryPlanner {
    pub fn new() -> Self {
        Self::default()
    }

    fn annotate_query_type(
        plan: &LogicalPlan,
        query_type: FederatedQueryType,
    ) -> Result<LogicalPlan> {
        let new_inputs = plan
            .inputs()
            .into_iter()
            .map(|input| Self::annotate_query_type(input, query_type))
            .collect::<Result<Vec<_>>>()?;
        let plan = if new_inputs.is_empty() {
            plan.clone()
        } else {
            plan.with_new_exprs(plan.expressions(), new_inputs)?
        };

        if let LogicalPlan::Extension(Extension { node }) = &plan {
            if let Some(federated_node) = node.as_any().downcast_ref::<FederatedPlanNode>() {
                return Ok(LogicalPlan::Extension(Extension {
                    node: Arc::new(FederatedPlanNode::new_with_query_type(
                        federated_node.plan.clone(),
                        Arc::clone(&federated_node.planner),
                        Some(federated_node.query_type.unwrap_or(query_type)),
                    )),
                }));
            }
        }

        Ok(plan)
    }

    pub(crate) fn annotate_query_directives(plan: &LogicalPlan) -> Result<LogicalPlan> {
        match plan {
            LogicalPlan::Explain(_) => {
                let inputs = plan.inputs();
                let [input] = inputs.as_slice() else {
                    return Err(DataFusionError::Plan(
                        "Explain plan must have exactly one input".into(),
                    ));
                };
                let annotated_input =
                    Self::annotate_query_type(input, FederatedQueryType::Explain)?;
                plan.with_new_exprs(plan.expressions(), vec![annotated_input])
            }
            LogicalPlan::Analyze(_) => {
                let inputs = plan.inputs();
                let [input] = inputs.as_slice() else {
                    return Err(DataFusionError::Plan(
                        "Analyze plan must have exactly one input".into(),
                    ));
                };
                let annotated_input =
                    Self::annotate_query_type(input, FederatedQueryType::Analyze)?;
                plan.with_new_exprs(plan.expressions(), vec![annotated_input])
            }
            _ => Ok(plan.clone()),
        }
    }
}

#[async_trait]
impl QueryPlanner for FederatedQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let logical_plan = Self::annotate_query_directives(logical_plan)?;

        let physical_planner =
            DefaultPhysicalPlanner::with_extension_planners(vec![
                Arc::new(FederatedPlanner::new()),
            ]);
        physical_planner
            .create_physical_plan(&logical_plan, session_state)
            .await
    }
}

#[async_trait]
pub trait FederationPlanner: Send + Sync {
    async fn plan_federation(
        &self,
        node: &FederatedPlanNode,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>>;
}

impl std::fmt::Debug for dyn FederationPlanner {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FederationPlanner")
    }
}

impl PartialEq<FederatedPlanNode> for FederatedPlanNode {
    /// Comparing name, args and return_type
    fn eq(&self, other: &FederatedPlanNode) -> bool {
        self.plan == other.plan && self.query_type == other.query_type
    }
}

impl PartialOrd<FederatedPlanNode> for FederatedPlanNode {
    fn partial_cmp(&self, other: &FederatedPlanNode) -> Option<std::cmp::Ordering> {
        match self.plan.partial_cmp(&other.plan) {
            Some(std::cmp::Ordering::Equal) => self.query_type.partial_cmp(&other.query_type),
            ordering => ordering,
        }
    }
}

impl Eq for FederatedPlanNode {}

impl Hash for FederatedPlanNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.plan.hash(state);
        self.query_type.hash(state);
    }
}

#[derive(Default)]
pub struct FederatedPlanner {}

impl FederatedPlanner {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ExtensionPlanner for FederatedPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let dc_node = node.as_any().downcast_ref::<FederatedPlanNode>();
        if let Some(fed_node) = dc_node {
            if !logical_inputs.is_empty() || !physical_inputs.is_empty() {
                return Err(DataFusionError::Plan(
                    "Inconsistent number of inputs".into(),
                ));
            }

            let fed_planner = Arc::clone(&fed_node.planner);
            let exec_plan = fed_planner.plan_federation(fed_node, session_state).await?;
            return Ok(Some(exec_plan));
        }
        Ok(None)
    }
}
