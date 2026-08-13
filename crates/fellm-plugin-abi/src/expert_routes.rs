//! Thread-local routed-expert observations produced by host MoE kernels.

use std::cell::{Cell, RefCell};

thread_local! {
    static EXECUTION_OP: Cell<u64> = const { Cell::new(u64::MAX) };
    static ROUTES: RefCell<Vec<(u64, Vec<u32>)>> = const { RefCell::new(Vec::new()) };
}

pub fn set_current_execution_op(operation: u64) {
    EXECUTION_OP.set(operation);
}

pub fn record_expert_route(experts: impl IntoIterator<Item = u32>) {
    let experts = experts.into_iter().collect::<Vec<_>>();
    if experts.is_empty() {
        return;
    }
    let operation = EXECUTION_OP.get();
    ROUTES.with_borrow_mut(|routes| routes.push((operation, experts)));
}

#[must_use]
pub fn take_expert_routes() -> Vec<(u64, Vec<u32>)> {
    ROUTES.with_borrow_mut(std::mem::take)
}
