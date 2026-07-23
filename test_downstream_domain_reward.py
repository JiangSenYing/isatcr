import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
import pytest


def load_method(
    method_name,
    source_name="SatelliteNetworkSimulator_Computing.py",
    class_name="Propagator_Computing",
):
    source_path = Path(__file__).parent / source_name
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    propagator_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method_node = next(
        node
        for node in propagator_class.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    method_node.decorator_list = []
    namespace = {"Dict": Dict, "Optional": Optional, "np": np}
    exec(compile(ast.Module(body=[method_node], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace[method_name]


class FakeEnvironment:
    def __init__(self, now):
        self.now = now

    def timeout(self, delay):
        self.now += delay
        return None


class RewardRecorder:
    def __init__(self):
        self.calls = []

    def reach_reward(self, delay):
        self.calls.append(("computed", delay))
        return 1.0

    def reach_reward_abnormal(self, delay):
        self.calls.append(("uncomputed", delay))
        return -1.0


class FormulaReward:
    def __init__(self, reach_factor=1.0, delay_factor=0.05):
        self.reach_factor = reach_factor
        self.delay_factor = delay_factor

    def reach_reward(self, delay):
        return self.reach_factor - self.delay_factor * delay

    def reach_reward_abnormal(self, delay):
        return -self.delay_factor * delay


def run_downstream(*, is_computed, task_type, segment_time):
    propagator = SimpleNamespace()
    propagator.env = FakeEnvironment(now=30.0)
    propagator.downstream_delays = 2.0
    propagator.is_separated_mode = lambda: False
    propagator.leo_domain_delay = lambda packet, now: load_method("leo_domain_delay")(packet, now)
    propagator.reward_function = RewardRecorder()
    propagator.satellites = {
        "destination": SimpleNamespace(
            max_size=100.0,
            computing_ability=100.0,
            _get_state_for_decision=lambda *args: ("current_obs", None),
        )
    }
    propagator.statics_data = {
        "Reached_after_computed_0": 0,
        "Reached_after_computed_1": 0,
        "Reached_0": 0,
        "Reached_1": 0,
        "Total_hops_0": 0,
        "Total_hops_1": 0,
        "Total_delay_0": 0.0,
        "Total_delay_1": 0.0,
        "Computing_waiting_time": 0.0,
    }
    propagator.logger = SimpleNamespace(log=lambda *args, **kwargs: None)
    propagator.finish_meo_decision = lambda *args, **kwargs: None
    propagator.final_rewards = []

    packet = SimpleNamespace(
        source="source",
        destination="destination",
        hops=3,
        creation_time=5.0,
        size=10.0,
        information=[is_computed, task_type, 1.0, 1.0, 5.0, None, None],
        leo_domain_entry_time=segment_time,
        computing_waiting_time=0.0,
        computing_node=None,
    )

    downstream = load_method("downstream")(propagator, "destination", packet)
    next(downstream)
    with pytest.raises(StopIteration):
        next(downstream)
    return propagator


@pytest.mark.parametrize(
    ("is_computed", "expected_kind"),
    [(True, "computed"), (False, "uncomputed")],
)
def test_downstream_reward_uses_current_domain_delay(is_computed, expected_kind):
    propagator = run_downstream(
        is_computed=is_computed,
        task_type=0,
        segment_time=20.0,
    )

    assert propagator.reward_function.calls == [(expected_kind, 12.0)]
    assert propagator.statics_data["Total_delay_0"] == 27.0


@pytest.mark.parametrize(
    ("is_computed", "expected_kind"),
    [(True, "computed"), (False, "uncomputed")],
)
def test_downstream_reward_falls_back_to_packet_creation_time(is_computed, expected_kind):
    propagator = run_downstream(
        is_computed=is_computed,
        task_type=1,
        segment_time=None,
    )

    assert propagator.reward_function.calls == [(expected_kind, 27.0)]
    assert propagator.statics_data["Total_delay_1"] == 27.0


def test_cross_domain_receive_advances_leo_domain_timestamps():
    propagator = SimpleNamespace(
        env=SimpleNamespace(now=12.5),
        satellites={
            "source": SimpleNamespace(masterMeo="MEO-A"),
            "target": SimpleNamespace(masterMeo="MEO-B"),
        },
    )
    packet = SimpleNamespace(
        creation_time=1.0,
        leo_domain_entry_time=4.0,
        leo_previous_domain_entry_time=None,
        meo_segment_time=None,
    )

    load_method("_mark_cross_domain_entry")(propagator, "source", "target", packet)

    assert packet.leo_previous_domain_entry_time == 4.0
    assert packet.leo_domain_entry_time == 12.5
    assert packet.meo_segment_time == 12.5


def test_completed_domain_delay_uses_previous_entry_and_exact_crossing_time():
    packet = SimpleNamespace(
        creation_time=1.0,
        leo_previous_domain_entry_time=4.0,
        meo_segment_time=12.5,
    )

    delay = load_method("completed_leo_domain_delay")(packet, current_time=20.0)

    assert delay == 8.5


def test_same_domain_receive_does_not_advance_domain_timestamps():
    propagator = SimpleNamespace(
        env=SimpleNamespace(now=12.5),
        satellites={
            "source": SimpleNamespace(masterMeo="MEO-A"),
            "target": SimpleNamespace(masterMeo="MEO-A"),
        },
    )
    packet = SimpleNamespace(
        creation_time=1.0,
        leo_domain_entry_time=4.0,
        leo_previous_domain_entry_time=None,
        meo_segment_time=None,
    )

    load_method("_mark_cross_domain_entry")(propagator, "source", "target", packet)

    assert packet.leo_previous_domain_entry_time is None
    assert packet.leo_domain_entry_time == 4.0
    assert packet.meo_segment_time is None


def test_domain_entry_reward_uses_packet_compute_state():
    propagator = SimpleNamespace(reward_function=RewardRecorder())
    entry_reward = load_method("leo_domain_entry_reward")

    assert entry_reward(propagator, True, 2.5) == 1.0
    assert entry_reward(propagator, False, 3.5) == -1.0
    assert propagator.reward_function.calls == [
        ("computed", 2.5),
        ("uncomputed", 3.5),
    ]


def test_computed_domain_entry_bonus_equals_reach_factor_for_same_segment_delay():
    reward_function = FormulaReward(reach_factor=1.0, delay_factor=0.05)
    propagator = SimpleNamespace(reward_function=reward_function)
    entry_reward = load_method("leo_domain_entry_reward")
    segment_delay = 8.5

    computed_reward = entry_reward(propagator, True, segment_delay)
    uncomputed_reward = entry_reward(propagator, False, segment_delay)

    assert computed_reward == pytest.approx(1.0 - 0.05 * segment_delay)
    assert uncomputed_reward == pytest.approx(-0.05 * segment_delay)
    assert computed_reward - uncomputed_reward == pytest.approx(
        reward_function.reach_factor
    )


def test_intra_domain_normal_reward_does_not_depend_on_compute_state():
    normal_reward = load_method(
        "normal_reward",
        class_name="Reward_Function",
    )
    reward_function = SimpleNamespace(
        delay_factor=0.05,
        memory_factor=0.25,
        memory_threshold=0.1,
    )

    assert list(inspect.signature(normal_reward).parameters) == [
        "self",
        "delay",
        "memory_remain",
    ]
    assert normal_reward(reward_function, 3.0, 0.5) == pytest.approx(-0.15)


def test_domain_entry_experience_preserves_previous_action_and_compute_mark():
    source_path = Path(__file__).with_name("SatelliteNetworkSimulator_Computing.py")
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    satellite_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "Satellite_with_Computing"
    )
    forward_packet = next(
        node
        for node in satellite_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward_packet"
    )
    entry_block = next(
        node
        for node in ast.walk(forward_packet)
        if isinstance(node, ast.If)
        and "reached_meo_entry and last_obs is not None" in ast.unparse(node.test)
    )
    experience_calls = [
        node
        for node in ast.walk(ast.Module(body=entry_block.body, type_ignores=[]))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_experience"
    ]

    assert len(experience_calls) == 1
    assert [ast.unparse(arg) for arg in experience_calls[0].args[:6]] == [
        "last_obs",
        "is_computed",
        "last_action",
        "leo_entry_reward",
        "current_obs",
        "leo_entry_done",
    ]
    assert any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "last_obs" for target in node.targets)
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
        for node in entry_block.body
    )


def test_meo_segment_reward_adds_weighted_leo_entry_reward():
    segment_reward = load_method(
        "segment_reward",
        source_name="Transformer_module/meo_router.py",
        class_name="MEODomainRewardFunction",
    )
    reward_function = SimpleNamespace(
        segment_success_reward=0.5,
        terminal_reward_weight=0.25,
        progress_reward_weight=0.0,
        gamma=0.97,
        domain_hop_penalty=0.0,
        path_hop_penalty=0.0,
        boundary_load_penalty=0.0,
        boundary_delay_penalty=0.0,
        _segment_hops=lambda trace: max(
            0,
            int(trace["segment_hops"])
            if trace.get("segment_hops") is not None
            else len(trace.get("path", []) or []) - 1,
        ),
        _boundary_costs=lambda transitions: (0.0, 0.0),
    )
    trace = {"decision_time": 4.0}

    original_reward = segment_reward(reward_function, trace, current_time=7.0)
    weighted_reward = segment_reward(
        reward_function,
        trace,
        current_time=7.0,
        leo_reward=-2.0,
    )

    assert weighted_reward == pytest.approx(original_reward + 0.25 * -2.0)


def test_intermediate_domain_entry_bootstraps_without_normal_duplicate():
    source = Path(__file__).with_name("SatelliteNetworkSimulator_Computing.py").read_text(encoding="utf-8")

    entry_done = load_method("leo_domain_entry_done")

    assert entry_done("entry", "final") == 0
    assert "leo_entry_reward, current_obs, leo_entry_done" in source
    assert "if reached_meo_entry and last_obs is not None:" in source
    assert "last_obs = None" in source
    assert "leo_reward=leo_entry_reward" in source


def test_domain_entry_at_final_destination_is_terminal():
    entry_done = load_method("leo_domain_entry_done")

    assert entry_done("final", "final") == 1


def test_domain_entry_rewards_are_separate_from_final_rewards():
    propagator = SimpleNamespace(
        experiences=[1],
        experiences_2hop=[1],
        experiences_graph=[1],
        final_rewards=[1.0],
        domain_entry_rewards=[0.5],
    )

    load_method("reset_parameters")(propagator)

    assert propagator.final_rewards == []
    assert propagator.domain_entry_rewards == []

    environment_source = Path(__file__).with_name("RL_environment_for_computing.py").read_text(encoding="utf-8")
    assert "Average domain-entry reward:" in environment_source


def test_in_transit_failure_is_included_in_domain_entry_rewards():
    propagator = SimpleNamespace(
        satellites={
            "current": SimpleNamespace(masterMeo="MEO-A"),
            "destination": SimpleNamespace(masterMeo="MEO-C"),
        },
        domain_entry_rewards=[],
    )
    packet = SimpleNamespace(destination="destination")
    record_failure = load_method("record_domain_entry_failure")

    included = record_failure(propagator, packet, -1.2, current_node="current")

    assert included is True
    assert propagator.domain_entry_rewards == [-1.2]


@pytest.mark.parametrize("current_node", ["destination", "destination-peer"])
def test_failure_after_entering_destination_domain_is_not_a_domain_entry_reward(
    current_node,
):
    propagator = SimpleNamespace(
        satellites={
            "destination": SimpleNamespace(masterMeo="MEO-C"),
            "destination-peer": SimpleNamespace(masterMeo="MEO-C"),
        },
        domain_entry_rewards=[],
    )
    packet = SimpleNamespace(destination="destination")
    record_failure = load_method("record_domain_entry_failure")

    included = record_failure(propagator, packet, -1.2, current_node=current_node)

    assert included is False
    assert propagator.domain_entry_rewards == []


def test_unknown_destination_domain_is_still_treated_as_in_transit_failure():
    propagator = SimpleNamespace(
        satellites={"current": SimpleNamespace(masterMeo="MEO-A")},
        domain_entry_rewards=[],
    )
    packet = SimpleNamespace(destination="missing-destination")
    record_failure = load_method("record_domain_entry_failure")

    included = record_failure(propagator, packet, -1.2, current_node="current")

    assert included is True
    assert propagator.domain_entry_rewards == [-1.2]


def test_all_leo_loss_paths_classify_the_domain_entry_outcome():
    source_path = Path(__file__).with_name("SatelliteNetworkSimulator_Computing.py")
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    calls = [node for node in ast.walk(module) if isinstance(node, ast.Call)]
    loss_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "loss_reward"
    ]
    classification_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "record_domain_entry_failure"
    ]

    assert loss_calls
    assert len(classification_calls) == len(loss_calls)


def test_packet_starts_first_leo_subtask_at_creation_time():
    packet = SimpleNamespace()
    packet_init = load_method(
        "__init__",
        source_name="SatelliteNetworkSimulator_Beta.py",
        class_name="Packet",
    )

    packet_init(packet, "source", "destination", 3.5, 100.0)

    assert packet.leo_domain_entry_time == 3.5
    assert packet.leo_previous_domain_entry_time is None


def test_adjacent_inter_domain_traffic_is_not_skipped_by_meo_planning():
    source = Path(__file__).with_name("SatelliteNetworkSimulator_Computing.py").read_text(encoding="utf-8")

    assert "if dest_meo not in self.satellites[src_meo].neighbors:" not in source
