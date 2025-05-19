class Robot:
    def __init__(self, robot_id: int, color: str, initial_pos: tuple[int, int]):
        self.id = robot_id
        self.color_char = color[0].upper() # For rendering
        self.pos = initial_pos # (row, col)

    def __repr__(self):
        return f"Robot(id={self.id}, color_char='{self.color_char}', pos={self.pos})"

    def set_position(self, r: int, c: int):
        self.pos = (r, c) 