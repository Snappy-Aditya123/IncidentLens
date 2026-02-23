# from graph_data_wrapper import load_graph_dataset
# import torch

# packet_csv = r"C:\Users\adity\Desktop\AI\heavy\elasticsearch\data\ssdp_packets_rich.csv"
# label_csv = r"C:\Users\adity\Desktop\AI\heavy\elasticsearch\data\SSDP_Flood_labels.csv\SSDP_Flood_labels.csv"

# data_list = load_graph_dataset(
#     packet_csv=packet_csv,
#     label_csv=label_csv,
#     delta_t=1.0
# )

# print("Number of graph snapshots:", len(data_list))

# # sanity check
# print("First snapshot:")
# print(data_list[0])
# print("Unique labels in first snapshot:", data_list[0].y.unique())

# # collect all edge labels from all snapshots
# all_edge_labels = torch.cat([data.y for data in data_list])

# num_edges = len(all_edge_labels)
# num_malicious = (all_edge_labels == 1).sum().item()
# num_normal = (all_edge_labels == 0).sum().item()

# print("\nAggregated edge-level stats:")
# print("Total edges:", num_edges)
# print("Malicious edges:", num_malicious)
# print("Normal edges:", num_normal)
# print("Malicious %:", 100 * num_malicious / num_edges)


# ──────────────────────────────────────────────
# IncidentLens CLI — delegates to testingentry
# Run: python main.py health | ingest | investigate | serve | convert
# ──────────────────────────────────────────────

from src.Backend.tests.testingentry import main

if __name__ == "__main__":
    main()
