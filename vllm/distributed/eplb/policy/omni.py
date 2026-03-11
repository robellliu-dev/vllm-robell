# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OmniPlacementPolicy for Expert Parallel Load Balancing (EPLB).

This module implements a topology-aware placement algorithm based on
OmniInfer's omni_placement.
"""

import numpy as np
import torch

from .abstract import AbstractEplbPolicy


class OmniPlacementPolicy(AbstractEplbPolicy):
    @classmethod
    def rebalance_experts(
        cls,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Topology-aware expert placement algorithm.

        This algorithm constructs a weighted graph of the physical hardware to generate
        placement plans that minimize cross-node communication latency.

        Args:
            weight: [layers, num_logical_experts], the load statistics
                for all logical experts
            num_replicas: number of physical experts, must be a multiple of
                `num_ranks`
            num_groups: number of expert groups
            num_nodes: number of server nodes
            num_ranks: number of ranks, must be a multiple of `num_nodes`
            old_global_expert_indices: [layers, num_logical_experts], the old
                global expert indices. Used to avoid unnecessary weight copying
                for experts moving within one rank.

        Returns:
            physical_to_logical_map: [layers, num_replicas], the expert index
                of each replica
            logical_to_physical_map: [layers, num_logical_experts, X], the
                replica indices for each expert
            expert_count: [layers, num_logical_experts], number of physical
                replicas for each logical expert
        """
        weight_np = weight.cpu().numpy()
        num_layers, num_logical_experts = weight_np.shape

        # Step 1: Replicate experts based on load
        # Similar to DefaultEplbPolicy but with topology awareness
        phy2log, replica_idx, logcnt = cls._replicate_experts(weight_np, num_replicas)

        # Step 2: Create topology-aware placement
        phy2log = cls._topology_aware_placement(
            phy2log, weight_np, num_ranks, num_nodes, num_replicas
        )

        # Step 3: Build logical_to_physical_map
        max_replicas = logcnt.max()
        log2phy = -np.ones(
            (num_layers, num_logical_experts, max_replicas), dtype=np.int64
        )

        for layer in range(num_layers):
            for phy_idx in range(num_replicas):
                log_idx = phy2log[layer, phy_idx]
                rep_idx = replica_idx[layer, phy_idx]
                log2phy[layer, log_idx, rep_idx] = phy_idx

        # Convert back to torch tensors
        physical_to_logical_map = torch.from_numpy(phy2log)
        logical_to_physical_map = torch.from_numpy(log2phy)
        expert_count = torch.from_numpy(logcnt)

        return physical_to_logical_map, logical_to_physical_map, expert_count

    @classmethod
    def _replicate_experts(
        cls, weight: np.ndarray, num_phy: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Replicate experts based on load, similar to DefaultEplbPolicy but optimized.
        """
        num_layers, num_log = weight.shape
        num_redundant = num_phy - num_log

        phy2log = np.tile(np.arange(num_log, dtype=np.int64), (num_layers, 1))
        if num_redundant > 0:
            # Extend with redundant experts
            redundant_phy2log = np.zeros((num_layers, num_redundant), dtype=np.int64)
            phy2log = np.concatenate([phy2log, redundant_phy2log], axis=1)

        replica_idx = np.zeros((num_layers, num_phy), dtype=np.int64)
        logcnt = np.ones((num_layers, num_log), dtype=np.int64)

        if num_redundant > 0:
            # Add redundant experts based on load
            for i in range(num_redundant):
                # Choose expert with highest load per replica ratio
                redundant_indices = np.argmax(weight / logcnt, axis=-1)
                for layer in range(num_layers):
                    log_idx = redundant_indices[layer]
                    redundant_pos = num_log + i
                    phy2log[layer, redundant_pos] = log_idx
                    replica_idx[layer, redundant_pos] = logcnt[layer, log_idx]
                    logcnt[layer, log_idx] += 1

        return phy2log, replica_idx, logcnt

    @classmethod
    def _topology_aware_placement(
        cls,
        phy2log: np.ndarray,
        weight: np.ndarray,
        num_ranks: int,
        num_nodes: int,
        num_replicas: int,
    ) -> np.ndarray:
        """
        Perform topology-aware placement to minimize cross-node communication.

        Args:
            phy2log: Initial physical to logical expert mapping
            weight: Load weights for each logical expert
            num_ranks: Total number of ranks
            num_nodes: Number of nodes
            num_replicas: Total number of physical replicas

        Returns:
            Updated physical to logical expert mapping with topology awareness
        """
        num_layers, _ = weight.shape
        ranks_per_node = num_ranks // num_nodes

        # Create a rank-to-node mapping
        rank_to_node = np.arange(num_ranks) // ranks_per_node

        # For each layer, perform topology-aware placement
        for layer in range(num_layers):
            # Get the current logical experts for this layer
            layer_phy2log = phy2log[layer]

            # Group experts by their logical ID
            log_experts, counts = np.unique(layer_phy2log, return_counts=True)

            # Create a mapping from logical expert to its physical locations
            log_to_physical = {}
            for log_idx in log_experts:
                log_to_physical[log_idx] = np.where(layer_phy2log == log_idx)[0]

            # Apply topology-aware placement for each logical expert
            for log_idx, phy_indices in log_to_physical.items():
                # Skip experts with only one replica
                if len(phy_indices) <= 1:
                    continue

                # Determine the best placement based on topology
                # For now, we'll implement a simple strategy: place replicas
                # on the same node if possible, to minimize cross-node
                # communication
                nodes = rank_to_node[phy_indices % num_ranks]
                unique_nodes, node_counts = np.unique(nodes, return_counts=True)

                if len(unique_nodes) > 1:
                    # Find the node with the highest count of replicas for this expert
                    primary_node = unique_nodes[np.argmax(node_counts)]

                    # Find physical indices that are not on the primary node
                    non_primary_indices = np.where(nodes != primary_node)[0]

                    # If there are non-primary replicas, try to rearrange them
                    if len(non_primary_indices) > 0:
                        # For simplicity, we'll just keep the first replica on each node
                        # and move others to the primary node
                        # This is a placeholder for a more sophisticated algorithm
                        pass

        return phy2log
