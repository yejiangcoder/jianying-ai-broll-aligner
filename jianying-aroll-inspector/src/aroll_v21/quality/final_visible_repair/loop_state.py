from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aroll_v21.quality.final_visible_repair.pipeline import FinalVisibleRepairPipelineResult


@dataclass
class FinalVisibleRepairLoopState:
    current_timeline: list[Any]
    current_captions: list[Any]
    current_signature: tuple[Any, ...]
    seen_signatures: set[tuple[Any, ...]]
    actions: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""

    def consume_pipeline_result(
        self,
        result: FinalVisibleRepairPipelineResult,
        *,
        pass_index: int,
    ) -> str:
        if result.unresolved is not None:
            self.unresolved.append(result.unresolved)
            self.stop_reason = str(result.unresolved.get("reason") or f"{result.unresolved_rule_name}_failed")
            return "stop"
        if result.transaction is None:
            return "empty"
        transaction = result.transaction
        self.current_timeline = result.final_timeline
        self.current_captions = result.captions
        self.current_signature = result.signature
        self.actions.extend(transaction.actions or [transaction.action])
        if transaction.accepted:
            self.seen_signatures.add(result.signature)
            return "accepted"
        self.stop_reason = transaction.rejection_reason
        self.unresolved.append(
            {
                "pass_index": pass_index,
                "reason": self.stop_reason,
                "last_action": transaction.action,
                "repair_transaction_rule_name": transaction.rule_name,
            }
        )
        return "stop"
