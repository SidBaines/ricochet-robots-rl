from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .ricochet_core import Board, apply_action, reached_goal, UP, DOWN, LEFT, RIGHT

Action = Tuple[int, int]  # (robot_id, direction)


def serialize(board: Board) -> Tuple[Tuple[Tuple[int, int], ...]]:
    # Note: Serialization only includes robot positions, assuming walls/goal/target
    # are constant for the current BFS instance. Do not reuse across different boards.
    order = tuple(board.robot_positions[rid] for rid in sorted(board.robot_positions.keys()))
    return (order,)


@dataclass
class SolveResult:
    actions: List[Action]


def neighbors(board: Board) -> List[Tuple[Board, Action]]:
    results: List[Tuple[Board, Action]] = []
    for rid in board.robot_positions.keys():
        for d in (UP, DOWN, LEFT, RIGHT):
            next_board = apply_action(board, rid, d)
            if next_board.robot_positions[rid] != board.robot_positions[rid]:
                results.append((next_board, (rid, d)))
    return results


def solve_bfs(
    start: Board,
    max_depth: int = 30,
    max_nodes: int = 20000,
) -> Optional[List[Action]]:
    """Find a sequence of actions to reach the goal using BFS up to limits."""
    if reached_goal(start):
        return []

    start_key = serialize(start)
    visited = {start_key}
    queue = deque()
    queue.append((start, []))
    nodes_expanded = 0

    while queue:
        board, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for nb, act in neighbors(board):
            nodes_expanded += 1
            if nodes_expanded > max_nodes:
                return None
            key = serialize(nb)
            if key in visited:
                continue
            new_path = path + [act]
            if reached_goal(nb):
                return new_path
            visited.add(key)
            queue.append((nb, new_path))
    return None
