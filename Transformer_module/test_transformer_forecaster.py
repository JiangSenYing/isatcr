import torch as th

try:
    from Transformer_module import GlobalNetworkSnapshot, SatelliteLoadTransformer, TransformerPathPlanner
except ModuleNotFoundError:
    from transformer_forecaster import GlobalNetworkSnapshot, SatelliteLoadTransformer, TransformerPathPlanner


def test_satellite_load_transformer_shapes():
    batch_size = 2
    history_len = 6
    horizon = 3
    n_satellites = 4
    n_links = 8

    model = SatelliteLoadTransformer(
        queue_input_dim=2,
        link_input_dim=3,
        compute_input_dim=2,
        forecast_horizon=horizon,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
    )
    queue_history = th.randn(batch_size, history_len, n_satellites, 2)
    link_history = th.randn(batch_size, history_len, n_links, 3)
    compute_history = th.randn(batch_size, history_len, n_satellites, 2)

    outputs = model(queue_history, link_history, compute_history)

    assert outputs["queue_forecast"].shape == (batch_size, horizon, n_satellites, 1)
    assert outputs["link_forecast"].shape == (batch_size, horizon, n_links, 1)
    assert outputs["compute_queue_forecast"].shape == (batch_size, horizon, n_satellites, 1)
    assert outputs["business_time_forecast"].shape == (batch_size, horizon, n_satellites, 1)


def test_transformer_path_planner_returns_path():
    model = SatelliteLoadTransformer(
        queue_input_dim=1,
        link_input_dim=1,
        compute_input_dim=1,
        forecast_horizon=2,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    snapshot = GlobalNetworkSnapshot(
        node_names=["A", "B", "C"],
        edge_names=[("A", "B"), ("B", "C"), ("A", "C")],
        queue_load=th.tensor([[0.1], [0.2], [0.3]]).numpy(),
        link_load=th.tensor([[0.2], [0.2], [0.8]]).numpy(),
        compute_queue=th.tensor([[0.1], [0.4], [0.2]]).numpy(),
        adjacency={"A": ["B", "C"], "B": ["C"], "C": []},
        propagation_delays={("A", "B"): 1.0, ("B", "C"): 1.0, ("A", "C"): 1.0},
        memory_capacity={"A": 10.0, "B": 10.0, "C": 10.0},
        computing_capacity={"A": 10.0, "B": 10.0, "C": 10.0},
    )
    planner = TransformerPathPlanner.from_snapshot(model, snapshot, history_len=2)
    planner.add_snapshot(snapshot)

    plan = planner.plan("A", "C", top_k=2, max_hops=2)

    assert plan.path[0] == "A"
    assert plan.path[-1] == "C"
    assert len(plan.compute_flags) == len(plan.path)
    assert sum(plan.compute_flags) in (0, 1)
    assert plan.score >= 0.0


def test_compute_planner_uses_single_node_and_future_links():
    import networkx as nx

    model = SatelliteLoadTransformer(
        queue_input_dim=1,
        link_input_dim=1,
        compute_input_dim=1,
        forecast_horizon=2,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    snapshot = GlobalNetworkSnapshot(
        node_names=["A", "B", "C"],
        edge_names=[("A", "B"), ("B", "C")],
        queue_load=th.tensor([[0.1], [0.1], [0.1]]).numpy(),
        link_load=th.tensor([[0.1], [0.1]]).numpy(),
        compute_queue=th.tensor([[0.0], [0.0], [0.0]]).numpy(),
        adjacency={"A": ["B"], "B": ["C"], "C": []},
        propagation_delays={("A", "B"): 1.0, ("B", "C"): 1.0},
        computing_capacity={"A": 1.0, "B": 10.0, "C": 10.0},
        sim_time=0.0,
    )
    planner = TransformerPathPlanner.from_snapshot(model, snapshot, history_len=1)
    graphs = []
    for sim_time, has_bc in [(0.5, True), (1.0, True), (1.1, True), (3.0, False)]:
        graph = nx.DiGraph()
        graph.add_nodes_from(["A", "B", "C"])
        graph.add_edge("A", "B", propagation_delay=1.0, predicted_link_load=0.1)
        if has_bc:
            graph.add_edge("B", "C", propagation_delay=1.0, predicted_link_load=0.1)
        for node in graph.nodes:
            graph.nodes[node]["predicted_compute_queue"] = 0.0
            graph.nodes[node]["predicted_queue_load"] = 0.1
        graph.graph["sim_time"] = sim_time
        graphs.append(graph)

    nodes, shares, _, _, risk = planner._best_compute_plan(
        path=["A", "B", "C"],
        predicted_compute={"A": 0.0, "B": 0.0},
        computing_demand=1.0,
        packet_size=1.0,
        size_after_computing=0.5,
        graphs=graphs,
        business_duration=1.1,
    )

    assert nodes == ["B"]
    assert shares == {"B": 1.0}
    assert risk == 0.0

    graph_by_time = {float(graph.graph["sim_time"]): graph for graph in graphs}
    plan = planner._score_path(
        path=["A", "B", "C"],
        preds=planner._current_as_prediction(),
        graphs=graph_by_time,
        packet_size=1.0,
        computing_demand=0.0,
        size_after_computing=None,
        business_duration=None,
        need_compute=False,
    )

    assert plan.details["future_disappearing_link_count"] == 1.0
    assert plan.disappearing_links == [("B", "C")]
    assert plan.disappearing_link_times == {("B", "C"): [3.0]}


def test_path_scoring_uses_arrival_time_loads():
    import networkx as nx

    model = SatelliteLoadTransformer(
        queue_input_dim=1,
        link_input_dim=1,
        compute_input_dim=1,
        forecast_horizon=2,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    snapshot = GlobalNetworkSnapshot(
        node_names=["A", "B", "C"],
        edge_names=[("A", "B"), ("B", "C")],
        queue_load=th.tensor([[0.1], [0.1], [0.1]]).numpy(),
        link_load=th.tensor([[0.1], [0.1]]).numpy(),
        compute_queue=th.tensor([[0.1], [0.1], [0.1]]).numpy(),
        adjacency={"A": ["B"], "B": ["C"], "C": []},
        propagation_delays={("A", "B"): 2.0, ("B", "C"): 1.0},
        memory_capacity={"A": 10.0, "B": 10.0, "C": 10.0},
        computing_capacity={"A": 10.0, "B": 10.0, "C": 10.0},
        sim_time=0.0,
    )
    planner = TransformerPathPlanner.from_snapshot(model, snapshot, history_len=1)

    graphs = {}
    for sim_time, b_queue, bc_load in [(1.0, 0.98, 0.98), (2.0, 0.1, 0.1)]:
        graph = nx.DiGraph()
        graph.add_nodes_from(["A", "B", "C"])
        graph.add_edge("A", "B", propagation_delay=2.0, predicted_link_load=0.1)
        graph.add_edge("B", "C", propagation_delay=1.0, predicted_link_load=bc_load)
        for node in graph.nodes:
            graph.nodes[node]["predicted_queue_load"] = b_queue if node == "B" else 0.1
            graph.nodes[node]["predicted_compute_queue"] = 0.1
        graph.graph["sim_time"] = sim_time
        graphs[sim_time] = graph

    plan = planner._score_path(
        path=["A", "B", "C"],
        preds=planner._current_as_prediction(),
        graphs=graphs,
        packet_size=1.0,
        computing_demand=0.0,
        size_after_computing=None,
        business_duration=None,
        need_compute=False,
    )

    assert plan.max_queue_load == 0.1
    assert plan.max_link_load == 0.1
    assert plan.details["packet_capacity_risk"] == 0.0


def test_predict_future_can_return_prediction_graphs():
    model = SatelliteLoadTransformer(
        queue_input_dim=1,
        link_input_dim=1,
        compute_input_dim=1,
        forecast_horizon=2,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    snapshot = GlobalNetworkSnapshot(
        node_names=["A", "B"],
        edge_names=[("A", "B")],
        queue_load=th.tensor([[0.1], [0.2]]).numpy(),
        link_load=th.tensor([[0.3]]).numpy(),
        compute_queue=th.tensor([[0.4], [0.5]]).numpy(),
        adjacency={"A": ["B"], "B": []},
        propagation_delays={("A", "B"): 1.5},
        sim_time=10.0,
    )
    next_snapshot = GlobalNetworkSnapshot(
        node_names=snapshot.node_names,
        edge_names=snapshot.edge_names,
        queue_load=snapshot.queue_load,
        link_load=snapshot.link_load,
        compute_queue=snapshot.compute_queue,
        adjacency=snapshot.adjacency,
        propagation_delays=snapshot.propagation_delays,
        sim_time=11.0,
    )
    planner = TransformerPathPlanner.from_snapshot(model, snapshot, history_len=2)
    planner.add_snapshot(next_snapshot)

    preds, graph_preds = planner.predict_future(return_graphs=True)

    assert preds["queue_forecast"].shape == (1, 2, 2, 1)
    assert list(graph_preds.keys()) == [12.0, 13.0]
    first_graph = graph_preds[12.0]
    assert first_graph.nodes["A"]["predicted_queue_load"] == float(preds["queue_forecast"][0, 0, 0, 0])
    assert first_graph.nodes["A"]["predicted_compute_queue"] == float(preds["compute_queue_forecast"][0, 0, 0, 0])
    assert first_graph.nodes["A"]["predicted_business_time"] == float(preds["business_time_forecast"][0, 0, 0, 0])
    assert first_graph.edges[("A", "B")]["predicted_link_load"] == float(preds["link_forecast"][0, 0, 0, 0])
    assert first_graph.edges[("A", "B")]["propagation_delay"] == 1.5


def test_transformer_accepts_and_returns_directed_graphs():
    import networkx as nx

    model = SatelliteLoadTransformer(
        queue_input_dim=1,
        link_input_dim=1,
        compute_input_dim=1,
        forecast_horizon=2,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    history = []
    for sim_time in [1.0, 2.0]:
        graph = nx.DiGraph()
        graph.add_node("A", queue_load=0.1, compute_queue=0.2, business_time=0.3)
        graph.add_node("B", queue_load=0.2, compute_queue=0.1, business_time=0.4)
        graph.add_edge("A", "B", link_load=0.5, propagation_delay=1.5)
        graph.graph["sim_time"] = sim_time
        history.append(graph)

    graphs = model.predict_graphs(history)

    assert list(graphs.keys()) == [3.0, 4.0]
    first_graph = graphs[3.0]
    assert isinstance(first_graph, nx.DiGraph)
    assert "predicted_queue_load" in first_graph.nodes["A"]
    assert "predicted_compute_queue" in first_graph.nodes["A"]
    assert "predicted_business_time" in first_graph.nodes["A"]
    assert "predicted_link_load" in first_graph.edges["A", "B"]


def test_predict_future_writes_only_future_graph_edges():
    import networkx as nx

    model = SatelliteLoadTransformer(
        queue_input_dim=1,
        link_input_dim=1,
        compute_input_dim=1,
        forecast_horizon=2,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    snapshot = GlobalNetworkSnapshot(
        node_names=["A", "B", "C"],
        edge_names=[("A", "B"), ("B", "C")],
        queue_load=th.tensor([[0.1], [0.2], [0.3]]).numpy(),
        link_load=th.tensor([[0.4], [0.5]]).numpy(),
        compute_queue=th.tensor([[0.6], [0.7], [0.8]]).numpy(),
        adjacency={"A": ["B"], "B": ["C"], "C": []},
        sim_time=1.0,
    )
    next_snapshot = GlobalNetworkSnapshot(
        node_names=snapshot.node_names,
        edge_names=snapshot.edge_names,
        queue_load=snapshot.queue_load,
        link_load=snapshot.link_load,
        compute_queue=snapshot.compute_queue,
        adjacency=snapshot.adjacency,
        sim_time=2.0,
    )
    planner = TransformerPathPlanner.from_snapshot(model, snapshot, history_len=2)
    planner.add_snapshot(next_snapshot)
    calls = []

    def future_builder(forecast_times, latest_sim_time):
        graphs = {}
        for sim_time in forecast_times:
            calls.append((sim_time, latest_sim_time))
            graph = nx.Graph()
            graph.add_nodes_from(["A", "B", "C"])
            graph.add_edge("A", "B")
            graphs[float(sim_time)] = graph
        return graphs

    planner.set_future_graph_builder(future_builder)
    _, graph_preds = planner.predict_future(return_graphs=True)

    assert len(calls) == 2
    for graph in graph_preds.values():
        assert ("A", "B") in graph.edges
        assert ("B", "C") not in graph.edges
        assert "predicted_link_load" in graph.edges[("A", "B")]


def test_candidate_paths_use_temporal_future_edge_union():
    import networkx as nx

    model = SatelliteLoadTransformer(
        queue_input_dim=1,
        link_input_dim=1,
        compute_input_dim=1,
        forecast_horizon=2,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    snapshot = GlobalNetworkSnapshot(
        node_names=["A", "B", "C", "D"],
        edge_names=[("A", "B"), ("B", "C"), ("C", "D")],
        queue_load=th.tensor([[0.1], [0.1], [0.1], [0.1]]).numpy(),
        link_load=th.tensor([[0.1], [0.1], [0.1]]).numpy(),
        compute_queue=th.tensor([[0.1], [0.1], [0.1], [0.1]]).numpy(),
        adjacency={"A": ["B"], "B": ["C"], "C": [], "D": []},
        propagation_delays={("A", "B"): 1.0, ("B", "C"): 1.0, ("C", "D"): 1.0},
        sim_time=0.0,
    )
    planner = TransformerPathPlanner.from_snapshot(model, snapshot, history_len=1)
    graphs = {}
    for sim_time, has_cd in [(1.0, False), (2.0, False), (3.0, True)]:
        graph = nx.DiGraph()
        graph.add_nodes_from(["A", "B", "C", "D"])
        graph.add_edge("A", "B", propagation_delay=1.0, predicted_link_load=0.1)
        graph.add_edge("B", "C", propagation_delay=1.0, predicted_link_load=0.1)
        if has_cd:
            graph.add_edge("C", "D", propagation_delay=1.0, predicted_link_load=0.1)
        for node in graph.nodes:
            graph.nodes[node]["predicted_queue_load"] = 0.1
            graph.nodes[node]["predicted_compute_queue"] = 0.1
        graph.graph["sim_time"] = sim_time
        graphs[sim_time] = graph

    candidates = planner._candidate_paths(
        "A",
        "D",
        top_k=1,
        max_hops=3,
        delay_top_k=1,
        load_top_k=0,
        preds=planner._current_as_prediction(),
        graphs=graphs,
    )

    assert ["A", "B", "C", "D"] in candidates


def test_path_scoring_waits_for_future_edge_window():
    import networkx as nx

    model = SatelliteLoadTransformer(
        queue_input_dim=1,
        link_input_dim=1,
        compute_input_dim=1,
        forecast_horizon=2,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    snapshot = GlobalNetworkSnapshot(
        node_names=["A", "B", "C", "D"],
        edge_names=[("A", "B"), ("B", "C"), ("C", "D")],
        queue_load=th.tensor([[0.1], [0.1], [0.1], [0.1]]).numpy(),
        link_load=th.tensor([[0.1], [0.1], [0.1]]).numpy(),
        compute_queue=th.tensor([[0.1], [0.1], [0.1], [0.1]]).numpy(),
        adjacency={"A": ["B"], "B": ["C"], "C": [], "D": []},
        propagation_delays={("A", "B"): 1.0, ("B", "C"): 1.0, ("C", "D"): 1.0},
        memory_capacity={"A": 10.0, "B": 10.0, "C": 10.0, "D": 10.0},
        computing_capacity={"A": 10.0, "B": 10.0, "C": 10.0, "D": 10.0},
        sim_time=0.0,
    )
    planner = TransformerPathPlanner.from_snapshot(model, snapshot, history_len=1)
    graphs = {}
    for sim_time, has_cd in [(1.0, False), (2.0, False), (3.0, True)]:
        graph = nx.DiGraph()
        graph.add_nodes_from(["A", "B", "C", "D"])
        graph.add_edge("A", "B", propagation_delay=1.0, predicted_link_load=0.1)
        graph.add_edge("B", "C", propagation_delay=1.0, predicted_link_load=0.1)
        if has_cd:
            graph.add_edge("C", "D", propagation_delay=1.0, predicted_link_load=0.1)
        for node in graph.nodes:
            graph.nodes[node]["predicted_queue_load"] = 0.1
            graph.nodes[node]["predicted_compute_queue"] = 0.1
        graph.graph["sim_time"] = sim_time
        graphs[sim_time] = graph

    plan = planner._score_path(
        path=["A", "B", "C", "D"],
        preds=planner._current_as_prediction(),
        graphs=graphs,
        packet_size=1.0,
        computing_demand=0.0,
        size_after_computing=None,
        business_duration=None,
        need_compute=False,
    )

    assert plan.details["topology_risk"] == 0.0
    assert plan.predicted_delay == 4.0


if __name__ == "__main__":
    test_satellite_load_transformer_shapes()
    test_transformer_path_planner_returns_path()
    test_compute_planner_uses_single_node_and_future_links()
    test_path_scoring_uses_arrival_time_loads()
    test_predict_future_can_return_prediction_graphs()
    test_transformer_accepts_and_returns_directed_graphs()
    test_predict_future_writes_only_future_graph_edges()
    test_candidate_paths_use_temporal_future_edge_union()
    test_path_scoring_waits_for_future_edge_window()
    print("Transformer_module smoke test passed.")
