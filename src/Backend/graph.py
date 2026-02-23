import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import degree
from typing import Optional
import pandas as pd

"""
    things to add:
    make everything more modular 
	and make efficient and add more functions
	
"""

class node:
	def __init__(self, IPaddress: str, node_id: int, features: torch.Tensor) -> None:
		self.IPaddress = IPaddress
		self.node_id = node_id
		self.features = features
		self.out_neighbors: list[int] = []
		self.in_neighbors: list[int] = []


class network:
	def __init__(
		self,
		num_nodes: int,
		device: torch.device | str | None = None,
		nodes: list[node] | None = None,
	) -> None:
		self.num_nodes = num_nodes
		self.device = torch.device(device) if device is not None else torch.device("cpu")
		self._degree_cache: dict[str, torch.Tensor] = {}
		self._edge_index_cache: torch.Tensor | None = None
		self.nodes: dict[int, node] = {}
		if nodes is not None:
			for n in nodes:
				self.add_node(n)

	def out_degree(self) -> torch.Tensor:
		cached = self._degree_cache.get("out")
		if cached is not None:
			return cached
		edge_index = self.build_edge_index()
		src = edge_index[0]
		out = degree(src, num_nodes=self.num_nodes, dtype=torch.long)
		self._degree_cache["out"] = out
		return out

	def in_degree(self) -> torch.Tensor:
		cached = self._degree_cache.get("in")
		if cached is not None:
			return cached
		edge_index = self.build_edge_index()
		dst = edge_index[1]
		ind = degree(dst, num_nodes=self.num_nodes, dtype=torch.long)
		self._degree_cache["in"] = ind
		return ind

	def add_node(self, n: node) -> None:
		if n.node_id in self.nodes:
			raise ValueError(f"node_id {n.node_id} already exists")
		self.nodes[n.node_id] = n
		self.num_nodes = max(self.num_nodes, n.node_id + 1)
		self._degree_cache.clear()
		self._edge_index_cache = None

	def add_edge(self, src_id: int, dst_id: int) -> None:
		if src_id not in self.nodes or dst_id not in self.nodes:
			raise ValueError("src_id and dst_id must exist in nodes")
		self.nodes[src_id].out_neighbors.append(dst_id)
		self.nodes[dst_id].in_neighbors.append(src_id)
		self._degree_cache.clear()
		self._edge_index_cache = None

	def set_node_features(self) -> None:
		if not self.nodes:
			return
		ordered = [self.nodes[i].features for i in sorted(self.nodes.keys())]
		self.x = torch.stack(ordered, dim=0).to(self.device)

	def build_edge_index(self) -> torch.Tensor:
		if self._edge_index_cache is not None:
			return self._edge_index_cache
		# Vectorised: pre-allocate numpy arrays instead of growing Python lists
		total = sum(len(n.out_neighbors) for n in self.nodes.values())
		if total == 0:
			edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
		else:
			src_arr = np.empty(total, dtype=np.int64)
			dst_arr = np.empty(total, dtype=np.int64)
			pos = 0
			for n in self.nodes.values():
				k = len(n.out_neighbors)
				if k:
					src_arr[pos:pos + k] = n.node_id
					dst_arr[pos:pos + k] = n.out_neighbors
					pos += k
			edge_index = torch.from_numpy(np.stack([src_arr, dst_arr])).to(self.device)
		self._edge_index_cache = edge_index
		return edge_index

    # For larger graphs this is inefficient; consider sparse operations
	def build_sparse_adjacency(self) -> torch.Tensor:
		edge_index = self.build_edge_index()
		values = torch.ones(edge_index.size(1), device=self.device)
		adj = torch.sparse_coo_tensor(
			edge_index,
			values,
			size=(self.num_nodes, self.num_nodes),
		)
		return adj.coalesce()

	@classmethod
	def from_edge_list(
		cls,
		num_nodes: int,
		edges: list[tuple[int, int]],
		device: torch.device | str | None = None,
	) -> "network":
		g = cls(num_nodes=num_nodes, device=device)
		for i in range(num_nodes):
			g.add_node(node(IPaddress=f"node_{i}", node_id=i, features=torch.zeros(1)))
		for src_id, dst_id in edges:
			g.add_edge(src_id, dst_id)
		return g

	def to_pyg_data(self) -> Data:
		edge_index = self.build_edge_index()
		x = getattr(self, "x", None)
		return Data(edge_index=edge_index, x=x, num_nodes=self.num_nodes)
        
# Base temporal graph pipeline (snapshot graphs)
def add_window_id(df, ts_col: str, delta_t: float, t0: Optional[float] = None):
	

	if t0 is None:
		t0 = float(df[ts_col].min())
	window_id = ((df[ts_col] - t0) / delta_t).astype(int)
	df = df.copy()
	df["window_id"] = window_id
	df["window_start"] = t0 + window_id * delta_t
	return df


def build_node_map(df, src_col: str, dst_col: str) -> dict[str, int]:

	ips = pd.concat([df[src_col], df[dst_col]], ignore_index=True).dropna().unique()
	return {ip: idx for idx, ip in enumerate(ips)}


def build_flow_table(
	df,
	ts_col: str = "timestamp",
	src_col: str = "src_ip",
	dst_col: str = "dst_ip",
	proto_col: str = "protocol",
	bytes_col: str = "packet_length",
	payload_col: str = "payload_length",
	udp_len_col: Optional[str] = "udp_length",
	tcp_flags_col: Optional[str] = "tcp_flags",
	label_col: Optional[str] = "label",
):

	keys = ["window_id", src_col, dst_col, proto_col]
	df = df.sort_values(keys + [ts_col]).copy()
	df["iat"] = df.groupby(keys)[ts_col].diff().fillna(0.0)

	agg = {
		ts_col: "count",
		bytes_col: "sum",
		payload_col: "mean",
		"iat": ["mean", "std"],
	}

	if udp_len_col and udp_len_col in df.columns:
		agg[udp_len_col] = "mean"

	if tcp_flags_col and tcp_flags_col in df.columns:
		df["tcp_syn"] = df[tcp_flags_col].astype(str).str.contains("S").astype(int)
		agg["tcp_syn"] = "mean"

	if label_col and label_col in df.columns:
		agg[label_col] = "max"

	grouped = df.groupby(keys).agg(agg)
	grouped.columns = ["_".join([c for c in col if c]) for col in grouped.columns.to_flat_index()]
	grouped = grouped.reset_index()

	grouped = grouped.rename(
		columns={
			f"{ts_col}_count": "packet_count",
			f"{bytes_col}_sum": "total_bytes",
			f"{payload_col}_mean": "mean_payload_length",
			"iat_mean": "mean_inter_arrival",
			"iat_std": "std_inter_arrival",
			"udp_length_mean": "udp_length_mean",
			"tcp_syn_mean": "tcp_flag_syn_rate",
		}
	)
	if label_col and f"{label_col}_max" in grouped.columns:
		grouped = grouped.rename(columns={f"{label_col}_max": "edge_label"})

	if "std_inter_arrival" in grouped.columns:
		grouped["std_inter_arrival"] = grouped["std_inter_arrival"].fillna(0.0)

	if "edge_label" not in grouped.columns:
		grouped["edge_label"] = 0

	for col in ["udp_length_mean", "tcp_flag_syn_rate"]:
		if col not in grouped.columns:
			grouped[col] = 0.0

	return grouped


def build_window_data(
	flows_df,
	node_map: dict[str, int],
	src_col: str = "src_ip",
	dst_col: str = "dst_ip",
	proto_col: str = "protocol",
):
	feature_cols = [
		"packet_count",
		"total_bytes",
		"mean_payload_length",
		"mean_inter_arrival",
		"std_inter_arrival",
	]

	n_nodes = len(node_map)

	# Pre-map IPs → integer codes ONCE (avoids per-window .map() overhead)
	src_codes = flows_df[src_col].map(node_map).astype(np.int64).values
	dst_codes = flows_df[dst_col].map(node_map).astype(np.int64).values
	feat_arr = flows_df[feature_cols].to_numpy(dtype=np.float32)
	label_arr = flows_df["edge_label"].to_numpy(dtype=np.float32)
	wid_arr = flows_df["window_id"].values

	# Vectorised split: sort by window_id + searchsorted boundaries
	order = np.argsort(wid_arr, kind="mergesort")
	wid_sorted = wid_arr[order]
	unique_wids, first_idx = np.unique(wid_sorted, return_index=True)
	splits = np.append(first_idx, len(wid_sorted))

	data_list = []
	for i, wid in enumerate(unique_wids):
		lo, hi = splits[i], splits[i + 1]
		idx = order[lo:hi]
		edge_index = torch.from_numpy(np.stack([src_codes[idx], dst_codes[idx]])).long()
		edge_attr = torch.from_numpy(feat_arr[idx])
		y_edge = torch.from_numpy(label_arr[idx])

		data = Data(
			edge_index=edge_index,
			edge_attr=edge_attr,
			y=y_edge,
			num_nodes=n_nodes,
		)
		data.window_id = int(wid)
		data.window_start = float(wid)  # temporal ordering key
		data_list.append(data)

	return data_list


def build_snapshot_dataset(
	packets_df,
	delta_t: float = 5.0,
	ts_col: str = "timestamp",
	src_col: str = "src_ip",
	dst_col: str = "dst_ip",
	proto_col: str = "protocol",
):
	packets_df = add_window_id(packets_df, ts_col=ts_col, delta_t=delta_t)
	flows_df = build_flow_table(
		packets_df,
		ts_col=ts_col,
		src_col=src_col,
		dst_col=dst_col,
		proto_col=proto_col,
	)
	node_map = build_node_map(packets_df, src_col=src_col, dst_col=dst_col)
	data_list = build_window_data(
		flows_df,
		node_map=node_map,
		src_col=src_col,
		dst_col=dst_col,
		proto_col=proto_col,
	)
	return data_list, node_map, flows_df


#testing code sample graaph 
def build_sample_graph() -> network:
	# Directed edges: 0->1, 0->2, 1->2, 2->3, 3->1
	g = network(num_nodes=4)
	for i in range(4):
		g.add_node(node(IPaddress=f"10.0.0.{i}", node_id=i, features=torch.zeros(1)))
	g.add_edge(0, 1)
	g.add_edge(0, 2)
	g.add_edge(1, 2)
	g.add_edge(2, 3)
	g.add_edge(3, 1)
	return g


if __name__ == "__main__":
   
	graph = build_sample_graph()
	print("edge_index:\n", graph.build_edge_index())
	print("out_degree:", graph.out_degree().tolist())
	print("in_degree:", graph.in_degree().tolist())
