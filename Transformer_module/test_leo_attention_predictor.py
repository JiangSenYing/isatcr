import networkx as nx
import torch as th

try:
    from Transformer_module import LEOAttentionDecisionPredictor
    from Transformer_module.leo_attention_predictor import batch_from_networkx, from_networkx
except ModuleNotFoundError:
    from leo_attention_predictor import LEOAttentionDecisionPredictor, batch_from_networkx, from_networkx


def _toy_graph():
    graph = nx.DiGraph()
    graph.add_node(
        "A",
        remaining_memory=9.0,
        remaining_computing=8.0,
        memory_occupancy_rate=0.1,
        computing_occupancy_rate=0.2,
        is_producing=1,
        current_computing_queue_size=0.2,
        business_time=0.0,
    )
    graph.add_node(
        "B",
        remaining_memory=6.0,
        remaining_computing=7.0,
        memory_occupancy_rate=0.4,
        computing_occupancy_rate=0.3,
        is_producing=0,
        compute_queue=0.1,
        task_state=0.5,
    )
    graph.add_node(
        "C",
        remaining_memory=4.0,
        remaining_computing=5.0,
        memory_occupancy_rate=0.6,
        computing_occupancy_rate=0.5,
        is_producing=0,
        compute_queue=0.3,
        business_time=0.2,
    )
    graph.add_edge("A", "B", delay=1.0, link_load=0.2, link_queue=0.1, weight=0.3, target_compute_remain=7.0)
    graph.add_edge("B", "C", propagation_delay=2.0, link_load=0.4, queue_occupancy=0.2, weight=0.5, target_compute_remain=5.0)
    graph.add_edge("A", "C", delay=3.0, link_load=0.9, link_queue=0.5, weight=0.8, target_compute_remain=5.0)
    return graph


def test_from_networkx_builds_batch():
    batch = from_networkx(
        _toy_graph(),
        source="A",
        task_context={
            "task_type": 1,
            "packet_size": 0.2,
            "computing_demand": 0.3,
            "size_after_computing": 0.1,
            "hops": 0.4,
            "is_computed": 1,
        },
    )

    assert batch.node_features.shape == (1, 3, 3)
    assert batch.edge_features.shape == (1, 3, 6)
    assert batch.task_features.shape == (1, 6)
    assert th.allclose(batch.node_features[0, 0], th.tensor([0.9, 0.8, 1.0]))
    assert th.allclose(batch.task_features[0], th.tensor([1.0, 0.2, 0.3, 0.1, 0.4, 1.0]))
    assert batch.edge_features[0, :, -1].tolist() == [2.0, 2.0, 2.0]
    assert batch.source_index.tolist() == [0]
    assert batch.neighbor_mask[0].tolist() == [False, True, True]
    assert batch.compute_mask[0].tolist() == [True, True, True]


def test_predictor_forward_shapes_and_masks():
    graph = _toy_graph()
    batch = from_networkx(graph, source="A")
    batch.compute_mask[0, 2] = False
    model = LEOAttentionDecisionPredictor(hidden_dim=32, num_layers=1, dropout=0.0)
    assert model.node_proj[0].in_features == 3
    assert model.edge_proj[0].in_features == 6

    outputs = model(batch)

    assert outputs["next_hop_logits"].shape == (1, 3)
    assert outputs["compute_node_logits"].shape == (1, 3)
    assert outputs["node_embeddings"].shape == (1, 3, 32)
    assert outputs["next_hop_logits"][0, 0].item() < -1e8
    assert outputs["compute_node_logits"][0, 2].item() < -1e8


def test_training_loss_backpropagates():
    batch = from_networkx(_toy_graph(), source="A")
    model = LEOAttentionDecisionPredictor(hidden_dim=32, num_layers=1, dropout=0.0)

    loss, parts = model.training_loss(batch, next_hop_target=["B"], compute_node_target=["C"])
    loss.backward()

    assert loss.item() > 0.0
    assert parts["next_hop_loss"].item() > 0.0
    assert parts["compute_node_loss"].item() > 0.0
    assert any(param.grad is not None for param in model.parameters())


def test_training_loss_can_train_next_hop_only():
    batch = from_networkx(_toy_graph(), source="A")
    model = LEOAttentionDecisionPredictor(hidden_dim=32, num_layers=1, dropout=0.0)

    loss, parts = model.training_loss(batch, next_hop_target=["B"], compute_node_target=None)
    loss.backward()

    assert loss.item() > 0.0
    assert parts["next_hop_loss"].item() > 0.0
    assert parts["compute_node_loss"].item() == 0.0


def test_training_loss_can_train_compute_only():
    batch = from_networkx(_toy_graph(), source="A")
    model = LEOAttentionDecisionPredictor(hidden_dim=32, num_layers=1, dropout=0.0)

    loss, parts = model.training_loss(batch, next_hop_target=None, compute_node_target=["A"])
    loss.backward()

    assert loss.item() > 0.0
    assert parts["next_hop_loss"].item() == 0.0
    assert parts["compute_node_loss"].item() > 0.0


def test_training_loss_supports_mixed_missing_targets():
    batch = batch_from_networkx(
        [_toy_graph(), _toy_graph()],
        sources=["A", "A"],
        task_contexts=[None, None],
    )
    model = LEOAttentionDecisionPredictor(hidden_dim=32, num_layers=1, dropout=0.0)

    loss, parts = model.training_loss(
        batch,
        next_hop_target=["B", None],
        compute_node_target=[None, "A"],
    )
    loss.backward()

    assert loss.item() > 0.0
    assert parts["next_hop_loss"].item() > 0.0
    assert parts["compute_node_loss"].item() > 0.0


def test_predict_returns_node_names():
    graph = _toy_graph()
    model = LEOAttentionDecisionPredictor(hidden_dim=32, num_layers=1, dropout=0.0)

    result = model.predict(graph, source="A")

    assert result["next_hop"] in {"B", "C"}
    assert result["compute_node"] in {"A", "B", "C"}
    assert result["next_hop_probabilities"].shape == (1, 3)
    assert result["compute_node_probabilities"].shape == (1, 3)


def test_batch_from_networkx_pads_graphs():
    graph1 = _toy_graph()
    graph2 = nx.DiGraph()
    graph2.add_node("X", remaining_memory=1.0, remaining_computing=1.0)
    graph2.add_node("Y", remaining_memory=1.0, remaining_computing=1.0)
    graph2.add_edge("X", "Y", delay=1.0)

    batch = batch_from_networkx(
        [graph1, graph2],
        sources=["A", "X"],
        task_contexts=[th.ones(6), [0, 1, 2, 3, 0, 4]],
    )

    assert batch.node_features.shape == (2, 3, 3)
    assert batch.edge_features.shape == (2, 3, 6)
    assert batch.node_mask[1].tolist() == [True, True, False]
    assert batch.edge_mask[1].tolist() == [True, False, False]


def test_edge_destination_distances_are_sample_specific_and_do_not_mutate_graph():
    graph = _toy_graph()
    batch = batch_from_networkx(
        [graph, graph],
        sources=["A", "A"],
        edge_target_distances=[
            {("A", "B"): 0.2, ("A", "C"): 0.0, ("B", "C"): 0.0},
            {("A", "B"): 0.8},
        ],
    )

    assert th.allclose(batch.edge_features[0, :, -1], th.tensor([0.2, 0.0, 0.0]))
    assert th.allclose(batch.edge_features[1, :, -1], th.tensor([0.8, 2.0, 2.0]))
    assert all("destination_distance" not in attrs for _, _, attrs in graph.edges(data=True))
