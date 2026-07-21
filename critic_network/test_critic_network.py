from types import SimpleNamespace

import numpy as np
import torch

from critic_network import GlobalActionCritic, GlobalCriticTrainer, JointAction
from Transformer_module.global_trainer import GlobalTransformerTrainer
from Transformer_module.transformer_forecaster import GlobalNetworkSnapshot


NODES = ["MEO_A", "MEO_B", "MEO_C", "LEO_1", "LEO_2"]
EDGES = [("MEO_A", "MEO_B"), ("LEO_1", "LEO_2")]


def snapshot(sim_time, queue_shift=0.0, link_shift=0.0, compute_shift=0.0):
    return GlobalNetworkSnapshot(
        node_names=list(NODES),
        edge_names=list(EDGES),
        queue_load=np.asarray([[0.1 + queue_shift, 0.0]] * len(NODES), dtype=np.float32),
        link_load=np.asarray([[0.2 + link_shift, 0.0]] * len(EDGES), dtype=np.float32),
        compute_queue=np.asarray([[0.05 + compute_shift, 0.0]] * len(NODES), dtype=np.float32),
        business_time=np.asarray([[0.3]] * len(NODES), dtype=np.float32),
        adjacency={name: [] for name in NODES},
        propagation_delays={edge: 0.01 for edge in EDGES},
        node_mask=np.ones((len(NODES), 1), dtype=np.float32),
        link_mask=np.ones((len(EDGES), 1), dtype=np.float32),
        sim_time=float(sim_time),
    )


def action(action_type=0, decision_id="decision-1"):
    return JointAction(
        decision_id=decision_id,
        current_meo="MEO_A",
        next_meo="MEO_B",
        target_meo="MEO_C",
        current_leo="LEO_1",
        target_leo="LEO_2" if action_type == 0 else "LEO_1",
        action_type=action_type,
        meo_features=np.asarray([0.0, 0.25, 0.5, 0.25], dtype=np.float32),
        task_features=np.asarray([1.0, 0.2, 0.3, 0.1, 0.05, 0.0], dtype=np.float32),
    )


def test_model_output_shapes_masks_and_gradients():
    model = GlobalActionCritic(node_feature_dim=5, edge_feature_dim=3, hidden_dim=16, dropout=0.0)
    batch = {
        "node_features": torch.rand(2, 5, 5),
        "edge_features": torch.rand(2, 2, 3),
        "node_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool),
        "edge_mask": torch.tensor([[1, 1], [0, 0]], dtype=torch.bool),
        "edge_index": torch.tensor([[[0, 1], [3, 4]], [[0, 1], [0, 0]]]),
        "action_indices": torch.tensor([[0, 1, 2, 3, 4], [0, 1, 0, 0, 0]]),
        "action_node_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.float32),
        "action_type": torch.tensor([0, 1]),
        "meo_features": torch.rand(2, 4),
        "task_features": torch.rand(2, 6),
    }
    output = model(batch)
    assert output.delta_queue.shape == (2, 5)
    assert output.delta_compute.shape == (2, 5)
    assert output.delta_link.shape == (2, 2)
    assert output.success_logit.shape == output.delay.shape == output.impact.shape == (2,)
    assert torch.all(output.delta_queue[1, 2:] == 0)
    assert torch.all(output.delta_link[1] == 0)
    loss = (
        output.delta_queue.sum() + output.delta_compute.sum() + output.delta_link.sum()
        + output.success_logit.sum() + output.delay.sum() + output.impact.sum()
    )
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_pending_event_accepts_snapshot_and_terminal_in_either_order():
    trainer = GlobalCriticTrainer({"enabled": True, "batch_size": 2, "warmup_samples": 2}, "cpu")
    packet_a = SimpleNamespace(global_critic_event_ids=[])
    packet_b = SimpleNamespace(global_critic_event_ids=[])
    trainer.start_event(packet_a, snapshot(0), action(0, "a"), action_time=0)
    trainer.start_event(packet_b, snapshot(0), action(1, "b"), action_time=0)

    trainer.observe_snapshot(snapshot(1, queue_shift=0.1))
    trainer.finish_packet(packet_a, terminal_reward=1.0, success=True, delay=2.0)
    trainer.finish_packet(packet_b, terminal_reward=-1.0, success=False, delay=3.0)
    assert len(trainer.replay_buffer) == 2

    packet_c = SimpleNamespace(global_critic_event_ids=[])
    trainer.start_event(packet_c, snapshot(1), action(0, "c"), action_time=1)
    trainer.finish_packet(packet_c, terminal_reward=0.5, success=True, delay=1.0)
    assert len(trainer.replay_buffer) == 2
    trainer.observe_snapshot(snapshot(2, link_shift=0.1))
    assert len(trainer.replay_buffer) == 3
    assert trainer.update_if_ready() is not None
    log = trainer.format_training_log(step=2, total_steps=10, round_idx=1)
    assert "[GlobalCritic]" in log
    assert "phase=training" in log
    assert "impact_mae=" in log


def test_impact_penalizes_global_load_growth():
    trainer = GlobalCriticTrainer({
        "enabled": True,
        "impact_weights": {"queue": 1.0, "link": 1.0, "compute": 1.0, "imbalance": 1.0},
    })
    trainer.initialize(snapshot(0))
    neutral = trainer.compute_impact_target(snapshot(0), snapshot(1), terminal_reward=1.0)
    congested = trainer.compute_impact_target(
        snapshot(0), snapshot(1, queue_shift=0.2, link_shift=0.2, compute_shift=0.2), terminal_reward=1.0
    )
    assert congested < neutral


def test_checkpoint_round_trip(tmp_path):
    path = tmp_path / "critic.pth"
    cfg = {"enabled": True, "model_path": str(path), "hidden_dim": 16}
    original = GlobalCriticTrainer(cfg)
    original.initialize(snapshot(0))
    original.train_steps = 7
    original.selection_uses = 3
    original.selection_disagreements = 2
    original.selection_fallbacks = 1
    assert original.save()

    restored = GlobalCriticTrainer(cfg)
    restored.initialize(snapshot(0))
    assert restored.train_steps == 7
    assert restored.selection_uses == 3
    assert restored.selection_disagreements == 2
    assert restored.selection_fallbacks == 1
    for left, right in zip(original.model.parameters(), restored.model.parameters()):
        assert torch.equal(left, right)


def test_selection_gate_requires_samples_and_train_steps(capsys):
    trainer = GlobalCriticTrainer({
        "enabled": True,
        "selection_enabled": True,
        "selection_min_samples": 2,
        "selection_min_train_steps": 3,
    })
    current = snapshot(0)
    trainer.initialize(current)
    trainer.replay_buffer.extend([object(), object()])
    trainer.train_steps = 2
    assert not trainer.selection_ready(current)
    trainer.train_steps = 3
    assert trainer.selection_ready(current)
    output = capsys.readouterr().out
    assert "LEO selection gate enabled" in output
    assert "buffer=2/2" in output
    assert "train_steps=3/3" in output


def test_selection_is_disabled_when_new_config_is_omitted():
    trainer = GlobalCriticTrainer({"enabled": True})
    current = snapshot(0)
    trainer.initialize(current)
    trainer.replay_buffer.extend([object()] * 512)
    trainer.train_steps = 100
    assert not trainer.selection_ready(current)


def test_candidate_ranking_uses_negative_impact_as_risk():
    trainer = GlobalCriticTrainer({
        "enabled": True,
        "selection_enabled": True,
        "selection_min_samples": 0,
        "selection_min_train_steps": 0,
        "selection_weight": 0.2,
    })
    current = snapshot(0)
    trainer.initialize(current)
    trainer.predict_many = lambda _snapshot, _actions: [
        {"impact": -1.0},
        {"impact": 2.0},
    ]
    ranked = trainer.rank_actions(current, [action(0, "a"), action(1, "b")], [1.0, 1.0])
    assert ranked["selected_index"] == 1
    assert ranked["risks"].tolist() == [1.0, -2.0]
    assert ranked["combined_scores"][1] > ranked["combined_scores"][0]
    assert trainer.selection_uses == 1
    assert trainer.selection_disagreements == 1


def test_candidate_ranking_falls_back_on_non_finite_critic_output():
    trainer = GlobalCriticTrainer({
        "enabled": True,
        "selection_enabled": True,
        "selection_min_samples": 0,
        "selection_min_train_steps": 0,
    })
    current = snapshot(0)
    trainer.initialize(current)
    trainer.predict_many = lambda _snapshot, _actions: [{"impact": 0.0}, {"impact": float("nan")}]
    assert trainer.rank_actions(current, [action(0, "a"), action(1, "b")], [0.0, 1.0]) is None
    assert trainer.selection_fallbacks == 1


def test_predict_many_preserves_candidate_order_and_batch_shape():
    trainer = GlobalCriticTrainer({"enabled": True, "hidden_dim": 16})
    predictions = trainer.predict_many(snapshot(0), [action(0, "a"), action(1, "b")])
    assert len(predictions) == 2
    assert all(np.isfinite(item["impact"]) for item in predictions)
    assert all(0.0 <= item["success_probability"] <= 1.0 for item in predictions)


def test_global_trainer_records_only_active_cross_domain_trace():
    manager = GlobalTransformerTrainer({
        "enabled": False,
        "meo_exit_enabled": False,
        "critic": {"enabled": True, "batch_size": 2, "warmup_samples": 2},
    }, torch.device("cpu"))
    manager.snapshots = [snapshot(0)]
    leo = SimpleNamespace(
        name="LEO_1",
        max_size=100.0,
        computing_ability=100.0,
        max_hop=10,
        env=SimpleNamespace(now=0.5),
    )
    packet = SimpleNamespace()
    assert not manager.record_critic_action(packet, leo, next_hop_target="LEO_2")

    packet.meo_decision_trace = {
        "decision_id": "joint-1",
        "current_domain": "MEO_A",
        "next_domain": "MEO_B",
        "target_domain": "MEO_C",
    }
    assert manager.record_critic_action(
        packet,
        leo,
        next_hop_target="LEO_2",
        packet_size=20.0,
        computing_demand=30.0,
        size_after_computing=10.0,
    )
    assert len(manager.critic.pending) == 1
    prediction = packet.meo_decision_trace["global_critic_predictions"][0]
    assert prediction["leo_action"] == "forward"
    assert 0.0 <= prediction["success_probability"] <= 1.0


def test_global_trainer_reranks_only_active_trace_and_encodes_valid_actions():
    manager = GlobalTransformerTrainer({
        "enabled": False,
        "meo_exit_enabled": False,
        "critic": {
            "enabled": True,
            "selection_enabled": True,
            "selection_min_samples": 0,
            "selection_min_train_steps": 0,
            "selection_weight": 0.2,
            "hidden_dim": 16,
        },
    }, torch.device("cpu"))
    manager.snapshots = [snapshot(0)]
    leo = SimpleNamespace(
        name="LEO_1",
        neighbors=["LEO_2"],
        max_size=100.0,
        computing_ability=100.0,
        max_hop=10,
    )
    packet = SimpleNamespace()
    assert manager.select_leo_action_with_critic(packet, leo, [1, 9, 9, 9, 1], [0, 4]) is None

    packet.meo_decision_trace = {
        "decision_id": "joint-select-1",
        "current_domain": "MEO_A",
        "next_domain": "MEO_B",
        "target_domain": "MEO_C",
    }
    captured = []

    def fake_predict(_snapshot, actions):
        captured.extend(actions)
        return [{"impact": -1.0}, {"impact": 2.0}]

    manager.critic.predict_many = fake_predict
    selected = manager.select_leo_action_with_critic(
        packet,
        leo,
        q_values=[1, 9, 9, 9, 1],
        valid_actions=[0, 4],
        packet_size=20,
        computing_demand=30,
        size_after_computing=10,
    )
    assert selected["action"] == 4
    assert selected["original_action"] == 0
    assert [candidate.action_type for candidate in captured] == [0, 1]
    assert [candidate.target_leo for candidate in captured] == ["LEO_2", "LEO_1"]


def test_leo_dqn_uses_critic_only_in_greedy_branch():
    from SatelliteNetworkSimulator_Computing import Satellite_with_Computing

    class FakeTransformer:
        def __init__(self):
            self.calls = 0

        def select_leo_action_with_critic(self, **kwargs):
            self.calls += 1
            assert kwargs["valid_actions"] == [0, 4]
            return {"action": 4, "score": 0.25}

    transformer = FakeTransformer()
    leo = object.__new__(Satellite_with_Computing)
    leo.mode = "DQN"
    leo.epsilon = -1.0
    leo.device = torch.device("cpu")
    leo.q_net = lambda _obs: torch.tensor([[3.0, 2.0, 1.0, 0.0, 1.0]])
    leo.leo_action_mask_enabled = True
    leo.neighbors = ["LEO_2"]
    leo.neighbor_hops = {"LEO_2": {"LEO_2": 0}}
    leo.max_hop = 10
    leo.propagator = SimpleNamespace(obs_type="flat", transformer_module=transformer)

    # No critic context: the original Q argmax remains active during warmup/non-MEO decisions.
    assert leo.get_next_hop([0.0], "LEO_2", False) == 0
    assert transformer.calls == 0

    context = {"packet": SimpleNamespace(), "task_type": 1, "packet_size": 10.0}
    assert leo.get_next_hop([0.0], "LEO_2", False, critic_context=context) == 4
    assert transformer.calls == 1

    # Epsilon exploration does not invoke critic reranking.
    leo.epsilon = 1.0
    leo.get_next_hop([0.0], "LEO_2", False, critic_context=context)
    assert transformer.calls == 1
