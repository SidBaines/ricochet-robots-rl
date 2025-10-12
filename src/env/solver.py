from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappush, heappop
from typing import Dict, List, Optional, Tuple, Literal

from .ricochet_core import Board, apply_action, reached_goal, UP, DOWN, LEFT, RIGHT

Action = Tuple[int, int]  # (robot_id, direction)


def serialize(board: Board) -> Tuple[Tuple[Tuple[int, int], ...]]:
    """
    Hashable state key for visited sets based on robot positions only.
    Assumes fixed walls/goal/target for this search instance (recompute for other boards).
    If state gains mutable elements in future, extend this key.
    Deterministic ordering via sorted robot ids.
    """
    order = tuple(board.robot_positions[rid] for rid in sorted(board.robot_positions.keys()))
    return (order,)


@dataclass
class SolveResult:
    """Result of a planning call with metadata.

    - actions: sequence of (robot_id, direction)
    - nodes_expanded: number of nodes removed from the frontier
    - depth: length of actions (slides)
    - found: True if a solution was found within limits
    """
    actions: List[Action]
    nodes_expanded: int
    depth: int
    found: bool


def neighbors(board: Board, *,
              prune_repeat_direction: bool = False,
              last_move: Optional[Action] = None) -> List[Tuple[Board, Action]]:
    """
    Generate neighbor states via all robot slides in 4 directions, pruning non-moves.
    Deterministic expansion order (sorted robot ids).
    - prune_repeat_direction: optionally skip repeating the same (robot,direction) as last_move.
    """
    results: List[Tuple[Board, Action]] = []
    for rid in sorted(board.robot_positions.keys()):
        for d in (UP, DOWN, LEFT, RIGHT):
            if prune_repeat_direction and last_move is not None and (rid, d) == last_move:
                continue
            next_board = apply_action(board, rid, d)
            if next_board.robot_positions[rid] != board.robot_positions[rid]:
                results.append((next_board, (rid, d)))
    return results


def solve_bfs(
    start: Board,
    max_depth: int = 30,
    max_nodes: int = 20000,
    *,
    prioritize_target_first: bool = True,
    prune_repeat_direction: bool = False,
    return_metadata: bool = False,
) -> Optional[List[Action]] | SolveResult:
    """
    Breadth-first search minimizing number of slides (each slide = unit cost).
    Returns minimal action list if found, else None if search limits are exceeded
    or no solution exists within the given limits.

    Note: None may indicate cutoff due to max_depth/max_nodes, not true unsolvability.
    """
    if reached_goal(start):
        return SolveResult(actions=[], nodes_expanded=0, depth=0, found=True) if return_metadata else []

    start_key = serialize(start)
    visited = {start_key}
    queue = deque()
    queue.append((start, [], None))  # (board, path, last_move)
    nodes_expanded = 0

    while queue:
        board, path, last_move = queue.popleft()
        if len(path) >= max_depth:
            continue
        nbrs = neighbors(board, prune_repeat_direction=prune_repeat_direction, last_move=last_move)
        if prioritize_target_first:
            target_id = board.target_robot
            # Stable partition: target-first
            nbrs.sort(key=lambda item: 0 if item[1][0] == target_id else 1)

        for nb, act in nbrs:
            nodes_expanded += 1
            if nodes_expanded > max_nodes:
                return SolveResult(actions=[], nodes_expanded=nodes_expanded, depth=-1, found=False) if return_metadata else None
            key = serialize(nb)
            if key in visited:
                continue
            new_path = path + [act]
            if reached_goal(nb):
                return SolveResult(actions=new_path, nodes_expanded=nodes_expanded, depth=len(new_path), found=True) if return_metadata else new_path
            visited.add(key)
            queue.append((nb, new_path, act))
    return SolveResult(actions=[], nodes_expanded=nodes_expanded, depth=-1, found=False) if return_metadata else None


HeuristicMode = Literal["admissible_zero", "admissible_one", "manhattan_cells"]


def heuristic(start: Board, mode: HeuristicMode) -> int:
    """Heuristic in units of slides. admissible_zero and admissible_one are admissible; manhattan_cells is not."""
    tr = start.target_robot
    r, c = start.robot_positions[tr]
    gr, gc = start.goal_position
    manhattan_cells = abs(r - gr) + abs(c - gc)
    if mode == "admissible_zero":
        return 0
    if mode == "admissible_one":
        return 0 if (r == gr and c == gc) else 1
    # Non-admissible but informative
    return manhattan_cells


def solve_astar(
    start: Board,
    max_depth: int = 200,
    max_nodes: int = 100000,
    *,
    prune_repeat_direction: bool = False,
    h_mode: HeuristicMode = "admissible_zero",
    return_metadata: bool = False,
) -> Optional[List[Action]] | SolveResult:
    """
    A* search with selectable heuristic:
    - admissible_zero: h=0 (optimal but may expand more)
    - admissible_one: h∈{0,1} (admissible)
    - manhattan_cells: non-admissible; may be faster but not guaranteed optimal

    Cost unit: one slide = one step.
    Returns minimal action list if found within limits (for admissible modes), else may be suboptimal with manhattan_cells.
    """
    if reached_goal(start):
        return SolveResult(actions=[], nodes_expanded=0, depth=0, found=True) if return_metadata else []

    start_key = serialize(start)
    g_costs: Dict[Tuple, int] = {start_key: 0}
    came_from: Dict[Tuple, Tuple[Tuple, Action]] = {}

    def reconstruct(end_key: Tuple) -> List[Action]:
        actions_rev: List[Action] = []
        key = end_key
        while key in came_from:
            prev_key, act = came_from[key]
            actions_rev.append(act)
            key = prev_key
        actions_rev.reverse()
        return actions_rev

    open_heap: List[Tuple[int, int, Tuple, Board, Optional[Action]]] = []
    h0 = heuristic(start, h_mode)
    heappush(open_heap, (h0, 0, start_key, start, None))

    visited: set = set()
    nodes_expanded = 0

    while open_heap:
        f, g, key, board, last_move = heappop(open_heap)
        if key in visited:
            continue
        visited.add(key)
        if reached_goal(board):
            acts = reconstruct(key)
            return SolveResult(actions=acts, nodes_expanded=nodes_expanded, depth=len(acts), found=True) if return_metadata else acts
        if g >= max_depth:
            continue

        for nb, act in neighbors(board, prune_repeat_direction=prune_repeat_direction, last_move=last_move):
            nodes_expanded += 1
            if nodes_expanded > max_nodes:
                return SolveResult(actions=[], nodes_expanded=nodes_expanded, depth=-1, found=False) if return_metadata else None
            nb_key = serialize(nb)
            tentative_g = g + 1
            if nb_key not in g_costs or tentative_g < g_costs[nb_key]:
                g_costs[nb_key] = tentative_g
                came_from[nb_key] = (key, act)
                h = heuristic(nb, h_mode)
                heappush(open_heap, (tentative_g + h, tentative_g, nb_key, nb, act))

    return SolveResult(actions=[], nodes_expanded=nodes_expanded, depth=-1, found=False) if return_metadata else None


def solve_bfs_with_metadata(start: Board, **kwargs) -> SolveResult:
    actions_or_result = solve_bfs(start, return_metadata=True, **kwargs)
    assert isinstance(actions_or_result, SolveResult)
    return actions_or_result


def solve_astar_with_metadata(start: Board, **kwargs) -> SolveResult:
    actions_or_result = solve_astar(start, return_metadata=True, **kwargs)
    assert isinstance(actions_or_result, SolveResult)
    return actions_or_result


def apply_actions(board: Board, actions: List[Action]) -> Board:
    """Apply a list of actions to a board and return the resulting board."""
    cur = board
    for rid, d in actions:
        cur = apply_action(cur, rid, d)
    return cur
