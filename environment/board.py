import numpy as np
from .utils import NORTH, EAST, SOUTH, WEST, DIRECTIONS

class Board:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        # self.walls[r, c, direction_idx] is True if cell (r,c) has a wall on that side
        self.walls = np.zeros((height, width, 4), dtype=bool)
        self._setup_perimeter_walls()

    def _setup_perimeter_walls(self):
        # North walls for top row
        self.walls[0, :, NORTH] = True
        # South walls for bottom row
        self.walls[self.height - 1, :, SOUTH] = True
        # West walls for first column
        self.walls[:, 0, WEST] = True
        # East walls for last column
        self.walls[:, self.width - 1, EAST] = True

    def add_wall(self, r: int, c: int, direction_idx: int):
        """Adds a wall to cell (r,c) on the specified direction_idx side.
        Also adds the corresponding wall to the neighboring cell.
        """
        if not (0 <= r < self.height and 0 <= c < self.width):
            raise ValueError(f"Cell ({r},{c}) is out of bounds.")

        self.walls[r, c, direction_idx] = True

        # Add corresponding wall to neighbor
        dr, dc = DIRECTIONS[direction_idx]
        neighbor_r, neighbor_c = r + dr, c + dc

        if 0 <= neighbor_r < self.height and 0 <= neighbor_c < self.width:
            opposite_direction_idx = (direction_idx + 2) % 4 # N<->S, E<->W
            self.walls[neighbor_r, neighbor_c, opposite_direction_idx] = True

    def has_wall(self, r: int, c: int, direction_idx: int) -> bool:
        """Checks if cell (r,c) has a wall on the specified direction_idx side."""
        if not (0 <= r < self.height and 0 <= c < self.width):
            # Outside bounds is effectively a wall in that direction
            return True
        return self.walls[r, c, direction_idx]

    def __repr__(self):
        return f"Board(height={self.height}, width={self.width})"

    def get_central_block_coords(self) -> set[tuple[int, int]]:
        """
        Returns a set of (row, col) tuples for the standard 2x2 central block.
        These cells are often impassable or have special rules.
        """
        if self.height < 4 or self.width < 4: # Need space for a central block
            return set()
        # Standard Ricochet central 2x2 square (e.g., (7,7), (7,8), (8,7), (8,8) for 16x16)
        mid_r, mid_c = self.height // 2, self.width // 2
        return {
            (mid_r - 1, mid_c - 1), (mid_r - 1, mid_c),
            (mid_r, mid_c - 1), (mid_r, mid_c)
        }

    def count_cell_walls(self, r: int, c: int) -> int:
        """Counts the number of existing walls for a given cell (r,c)."""
        if not (0 <= r < self.height and 0 <= c < self.width):
            return 4 # Effectively max walls if out of bounds (perimeter counts)
        return np.sum(self.walls[r, c, :])

    def _is_safe_to_add_wall_segment(self, r: int, c: int, direction_idx: int, central_block: set) -> bool:
        """
        Checks if adding a wall segment at (r,c,direction_idx) and its counterpart
        is valid regarding the "max 2 walls" rule for non-central cells.
        Assumes the wall segment does not already exist.
        """
        # Check cell (r,c) itself
        if (r,c) not in central_block:
            # If it already has 2 walls, adding another (this one) would make it 3.
            if self.count_cell_walls(r,c) >= 2:
                return False

        # Check neighbor cell that gets the other side of the wall
        dr_neighbor, dc_neighbor = DIRECTIONS[direction_idx]
        nr, nc = r + dr_neighbor, c + dc_neighbor
        
        if 0 <= nr < self.height and 0 <= nc < self.width: # If neighbor is on board
            if (nr,nc) not in central_block:
                # If neighbor already has 2 walls, adding another makes it 3.
                if self.count_cell_walls(nr,nc) >= 2:
                    return False
        return True

    def add_middle_blocked_walls(self):
        # assert self.height % 2 == 0 and self.width % 2 == 0
        assert self.height == self.width, "Height and width must be equal"
        if (self.height%2)==0:
            cr, cc = self.height // 2, self.width // 2
            # Walls around (cr-2, cc-2) to (cr+1, cc+1) effectively
            # (7,7) (7,8)
            # (8,7) (8,8)
            # Top walls of (7,7) and (7,8)
            self.add_wall(cr - 1, cc - 1, NORTH) # Wall below (6,6) which is N of (7,6)
            self.add_wall(cr - 1, cc, NORTH) # Wall below (6,7) which is N of (7,7)

            # Bottom walls of (8,7) and (8,8)
            self.add_wall(cr, cc - 1, SOUTH) # Wall above (9,6) which is S of (8,6)
            self.add_wall(cr, cc, SOUTH) # Wall above (9,7) which is S of (8,7)

            # Left walls of (7,7) and (8,7)
            self.add_wall(cr - 1, cc - 1, WEST) # Wall right of (7,5) which is W of (7,6)
            self.add_wall(cr, cc - 1, WEST) # Wall right of (8,5) which is W of (8,6)

            # Right walls of (7,8) and (8,8)
            self.add_wall(cr - 1, cc, EAST) # Wall left of (7,9) which is E of (7,8)
            self.add_wall(cr, cc, EAST) # Wall left of (8,9) which is E of (8,8)
        else:
            cr, cc = self.height // 2, self.width // 2
            # Top walls of (2,2) [assuming the height is eg. 5]
            self.add_wall(cr, cc , NORTH) # Wall below (6,6) which is N of (7,6)
            self.add_wall(cr, cc , SOUTH) # Wall below (6,6) which is N of (7,6)
            self.add_wall(cr, cc , EAST) # Wall below (6,6) which is N of (7,6)
            self.add_wall(cr, cc , WEST) # Wall below (6,6) which is N of (7,6)
    
    def add_standard_ricochet_walls(self):
        """Adds some example walls typical of Ricochet Robots boards.
        This is a simplified example. Real boards have specific configurations.
        """
        # Block off the center 2x2 square (usually it's impassable or special)
        # For simplicity, we'll just add walls around it.
        self.add_middle_blocked_walls()
        # Some example outer walls (these are often specific per quadrant)
        if self.height == 16 and self.width == 16: # Standard size
            self.add_wall(2, 2, EAST)
            self.add_wall(2, 2, SOUTH)

            self.add_wall(2, 13, WEST)
            self.add_wall(2, 13, SOUTH)

            self.add_wall(13, 2, EAST)
            self.add_wall(13, 2, NORTH)

            self.add_wall(13, 13, WEST)
            self.add_wall(13, 13, NORTH) 

    def generate_random_walls(self, 
                              num_edge_walls_per_quadrant: int, 
                              num_floating_walls_per_quadrant: int,
                              max_attempts_factor: int = 30,
                              rng=None):
        """
        Generates random walls based on specified rules per quadrant.
        - 'edge walls': Single wall segments parallel to a board edge, originating
                        from a cell adjacent to that edge and extending one unit.
        - 'floating walls': L-shaped pairs not connected to edges or central block.
        - No non-central square gets more than 2 walls.
        - Edge walls along the same board edge have at least one empty square between them.

        It's recommended to call this on a board that already has perimeter walls
        and potentially the standard central block walls.
        """
        if rng is None:
            rng = np.random  # fallback to global RNG, but should always pass in env._np_random

        central_block = self.get_central_block_coords()
        mid_r, mid_c = self.height // 2, self.width // 2
        quadrant_defs = [
            {"name": "NW", "r_lim": (0, mid_r), "c_lim": (0, mid_c)},
            {"name": "NE", "r_lim": (0, mid_r), "c_lim": (mid_c, self.width)},
            {"name": "SW", "r_lim": (mid_r, self.height), "c_lim": (0, mid_c)},
            {"name": "SE", "r_lim": (mid_r, self.height), "c_lim": (mid_c, self.width)},
        ]
        
        for quad_idx, quad in enumerate(quadrant_defs):
            r_start, r_end_exclusive = quad["r_lim"]
            c_start, c_end_exclusive = quad["c_lim"]
            
            # --- Place Edge Walls ---
            placed_edge_count = 0
            # candidate_edge_origins stores: ((cell_r, cell_c_hosting_wall), direction_of_wall_on_this_cell)
            # The cell (cell_r, cell_c) is inside the quadrant, adjacent to the perimeter.
            # The wall is parallel to the perimeter edge it's near.
            candidate_edge_origins = []

            # Top edge of quad is board's top edge (r_start == 0)
            # Wall is vertical, on cells (0, c_idx).
            if r_start == 0: 
                if quad["name"] == "NW" or quad["name"] == "SW": # Jut EAST (wall is EAST of cell (0,c))
                    for c_idx in range(c_start, c_end_exclusive):
                        candidate_edge_origins.append(((0, c_idx), EAST))
                elif quad["name"] == "NE" or quad["name"] == "SE": # Jut WEST (wall is WEST of cell (0,c))
                    for c_idx in range(c_start, c_end_exclusive):
                        candidate_edge_origins.append(((0, c_idx), WEST))
            
            # Bottom edge of quad is board's bottom edge (r_end_exclusive == self.height)
            # Wall is vertical, on cells (self.height - 1, c_idx).
            if r_end_exclusive == self.height:
                r_edge = self.height - 1
                if quad["name"] == "NW" or quad["name"] == "SW": # Jut EAST
                    for c_idx in range(c_start, c_end_exclusive):
                        candidate_edge_origins.append(((r_edge, c_idx), EAST))
                elif quad["name"] == "NE" or quad["name"] == "SE": # Jut WEST
                    for c_idx in range(c_start, c_end_exclusive):
                        candidate_edge_origins.append(((r_edge, c_idx), WEST))

            # Left edge of quad is board's left edge (c_start == 0)
            # Wall is horizontal, on cells (r_idx, 0).
            if c_start == 0:
                if quad["name"] == "NW" or quad["name"] == "NE": # Jut SOUTH (wall is SOUTH of cell (r,0))
                    for r_idx in range(r_start, r_end_exclusive):
                        candidate_edge_origins.append(((r_idx, 0), SOUTH))
                elif quad["name"] == "SW" or quad["name"] == "SE": # Jut NORTH (wall is NORTH of cell (r,0))
                    for r_idx in range(r_start, r_end_exclusive):
                        candidate_edge_origins.append(((r_idx, 0), NORTH))

            # Right edge of quad is board's right edge (c_end_exclusive == self.width)
            # Wall is horizontal, on cells (r_idx, self.width - 1).
            if c_end_exclusive == self.width:
                c_edge = self.width - 1
                if quad["name"] == "NW" or quad["name"] == "NE": # Jut SOUTH
                    for r_idx in range(r_start, r_end_exclusive):
                        candidate_edge_origins.append(((r_idx, c_edge), SOUTH))
                elif quad["name"] == "SW" or quad["name"] == "SE": # Jut NORTH
                    for r_idx in range(r_start, r_end_exclusive):
                        candidate_edge_origins.append(((r_idx, c_edge), NORTH))
            
            rng.shuffle(candidate_edge_origins)
            
            placed_this_quad_edge_walls = set() # Stores ((r,c), wall_direction)

            for (er, ec), wall_direction in candidate_edge_origins: # er,ec is cell hosting wall
                if placed_edge_count >= num_edge_walls_per_quadrant: break

                # Rule 1: Cell hosting the wall must not be in the central block.
                if (er, ec) in central_block: continue
                
                # Rule 2: Wall segment must not already exist.
                if self.walls[er, ec, wall_direction]: continue

                # Rule 3: Spacing - "at least two squares between them" (one empty square)
                # This applies to walls placed along the same board edge.
                is_too_close = False
                for (prev_r, prev_c), prev_wall_dir in placed_this_quad_edge_walls:
                    # Check if current and previous walls are of the same orientation and on the same "line"
                    if wall_direction == EAST or wall_direction == WEST: # Current wall is vertical
                        # Previous wall also vertical and on the same row er (e.g. both on top edge)
                        if (prev_wall_dir == EAST or prev_wall_dir == WEST) and prev_r == er:
                            if abs(prev_c - ec) <= 1: # Adjacent columns or same column
                                is_too_close = True; break
                    elif wall_direction == NORTH or wall_direction == SOUTH: # Current wall is horizontal
                        # Previous wall also horizontal and in the same column ec (e.g. both on left edge)
                        if (prev_wall_dir == NORTH or prev_wall_dir == SOUTH) and prev_c == ec:
                            if abs(prev_r - er) <= 1: # Adjacent rows or same row
                                is_too_close = True; break
                if is_too_close: continue
                
                # Rule 4: Max 2 walls per non-central cell constraint (checked by _is_safe_to_add_wall_segment)
                if self._is_safe_to_add_wall_segment(er, ec, wall_direction, central_block):
                    self.add_wall(er, ec, wall_direction)
                    placed_edge_count += 1
                    placed_this_quad_edge_walls.add(((er,ec), wall_direction))
            
            if placed_edge_count < num_edge_walls_per_quadrant:
                print(f"Warning: Quadrant {quad['name']} ({quad_idx+1}) could only place {placed_edge_count}/{num_edge_walls_per_quadrant} edge walls.")

            # --- Place Floating Walls (L-shapes) ---
            placed_floating_count = 0
            attempts = 0
            max_total_attempts = num_floating_walls_per_quadrant * max_attempts_factor
            
            potential_l_corners = []
            # L-corners must be at least 1 cell away from quadrant boundaries to be "floating" within it
            # and also away from board edges if quadrant edge is board edge.
            for r in range(r_start + 1, r_end_exclusive - 1):
                for c in range(c_start + 1, c_end_exclusive - 1):
                    if (r,c) not in central_block:
                        # Check not adjacent to central block
                        is_adj_to_central = False
                        for dr_adj, dc_adj in DIRECTIONS: # Check all 4 neighbors
                            if (r + dr_adj, c + dc_adj) in central_block:
                                is_adj_to_central = True; break
                        if not is_adj_to_central:
                           potential_l_corners.append((r,c))
            
            rng.shuffle(potential_l_corners)
            
            # L-shape orientations: (wall_direction1_from_corner, wall_direction2_from_corner)
            orientations = [(SOUTH, EAST), (SOUTH, WEST), (NORTH, EAST), (NORTH, WEST)]

            for cr, cc in potential_l_corners:
                if placed_floating_count >= num_floating_walls_per_quadrant: break
                if attempts >= max_total_attempts and placed_floating_count < num_floating_walls_per_quadrant: break

                rng.shuffle(orientations)
                for d1, d2 in orientations:
                    attempts += 1
                    # L-shape walls are (cr, cc, d1) and (cr, cc, d2)
                    
                    # 1. Corner cell (cr, cc) must have 0 existing walls to accept two new ones.
                    if self.count_cell_walls(cr, cc) != 0: continue
                    
                    # 2. Wall (cr,cc,d1) must not exist and be safe to add.
                    if self.walls[cr,cc,d1] or not self._is_safe_to_add_wall_segment(cr, cc, d1, central_block):
                        continue
                        
                    # 3. Wall (cr,cc,d2) must not exist.
                    #    And it must be safe to add *given that d1 might hypothetically be added first*.
                    #    This means (cr,cc) would have 1 wall (d1) when checking d2.
                    #    And neighbor for d2 must be safe.
                    if self.walls[cr,cc,d2]: continue

                    # Simulate adding d1 to check d2's safety accurately
                    self.walls[cr, cc, d1] = True # Hypothetically add d1
                    dr1, dc1 = DIRECTIONS[d1]
                    nr1, nc1 = cr + dr1, cc + dc1
                    if 0 <= nr1 < self.height and 0 <= nc1 < self.width:
                        self.walls[nr1, nc1, (d1+2)%4] = True
                    
                    safe_to_add_d2_after_d1 = self._is_safe_to_add_wall_segment(cr, cc, d2, central_block)
                    
                    # Revert hypothetical d1
                    self.walls[cr, cc, d1] = False 
                    if 0 <= nr1 < self.height and 0 <= nc1 < self.width:
                        self.walls[nr1, nc1, (d1+2)%4] = False

                    if not safe_to_add_d2_after_d1:
                        continue

                    # If all checks passed, add the L-shape
                    self.add_wall(cr, cc, d1)
                    self.add_wall(cr, cc, d2)
                    placed_floating_count += 1
                    break # Found a valid L-shape for this corner, move to next corner if needed
            
            if placed_floating_count < num_floating_walls_per_quadrant:
                 print(f"Warning: Quadrant {quad['name']} ({quad_idx+1}) could only place {placed_floating_count}/{num_floating_walls_per_quadrant} floating L-walls.")

    def is_valid_position(self, r: int, c: int) -> bool:
        """Checks if a position is valid on the board."""
        return 0 <= r < self.height and 0 <= c < self.width
    
