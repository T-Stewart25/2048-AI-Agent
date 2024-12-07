import pygame
import random

class Tile:
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

class Tiles:
    def __init__(self, tile_size=90, score_area_height=60):
        self.tile_size = tile_size
        self.score_area_height = score_area_height
        self.tiles = []
        self.score = 0
        self.action_space = ['up', 'down', 'left', 'right']
        self.reset()

    def draw_tiles(self, screen):
        margin = 10
        font = pygame.font.Font(None, 45)

        for tile in self.tiles:
            rect = pygame.Rect(
                tile.x * (self.tile_size + margin) + margin,
                tile.y * (self.tile_size + margin) + margin + self.score_area_height,
                self.tile_size,
                self.tile_size
            )
            color = (238, 228, 218)  # default tile color
            pygame.draw.rect(screen, color, rect)
            text = font.render(str(tile.value), True, (119, 110, 101))
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)

    def reset(self):
        """Resets the board to its initial state."""
        self.tiles = []
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()

    def get_board(self):
        """Returns a 4x4 grid representing the game board with tile values."""
        board = [[0] * 4 for _ in range(4)]
        for tile in self.tiles:
            board[tile.y][tile.x] = tile.value
        return board

    def add_random_tile(self):
        empty_cells = self.get_empty_cells()
        if empty_cells:
            x, y = random.choice(empty_cells)
            value = random.choice([2, 4])
            self.tiles.append(Tile(x, y, value))

    def get_empty_cells(self):
        return [(x, y) for x in range(4) for y in range(4) if not any(t.x == x and t.y == y for t in self.tiles)]

    def move(self, direction):
        moved = False
        if direction == 'up':
            moved = self.move_tiles(0, -1)
        elif direction == 'down':
            moved = self.move_tiles(0, 1)
        elif direction == 'left':
            moved = self.move_tiles(1, -1)
        elif direction == 'right':
            moved = self.move_tiles(1, 1)

        moved = bool(moved)
        if moved:
            self.add_random_tile()
        return moved, self.get_score(), self.check_game_over()

    def move_tiles(self, axis, direction):
        moved = False
        merged_positions = set()
        def key_func(tile):
            return (tile.y, tile.x) if axis == 0 else (tile.x, tile.y)
        sorted_tiles = sorted(self.tiles, key=key_func, reverse=(direction == 1))

        for tile in sorted_tiles:
            while True:
                next_x = tile.x + (direction if axis == 1 else 0)
                next_y = tile.y + (direction if axis == 0 else 0)
                if 0 <= next_x < 4 and 0 <= next_y < 4:
                    neighbor = next((t for t in self.tiles if t.x == next_x and t.y == next_y), None)
                    if neighbor is None:
                        tile.x, tile.y = next_x, next_y
                        moved = True
                    elif neighbor.value == tile.value and (neighbor.x, neighbor.y) not in merged_positions:
                        tile.x, tile.y = next_x, next_y
                        tile.value *= 2
                        self.score += tile.value
                        self.tiles.remove(neighbor)
                        merged_positions.add((tile.x, tile.y))
                        moved = True
                        break
                    else:
                        break
                else:
                    break
        return moved

    def get_score(self):
        return self.score

    def check_game_over(self):
        if len(self.tiles) < 16:
            return False
        for direction in ['up', 'down', 'left', 'right']:
            tiles_copy = [Tile(tile.x, tile.y, tile.value) for tile in self.tiles]
            temp_game = Tiles(self.tile_size, self.score_area_height)
            temp_game.tiles = tiles_copy
            if temp_game.move_tiles(0 if direction in ['up', 'down'] else 1, 1 if direction in ['up', 'left'] else -1):
                return False
        return True
    
    def clone(self):
        cloned_tiles = Tiles(self.tile_size, self.score_area_height)
        cloned_tiles.score = self.score
        cloned_tiles.tiles = [Tile(tile.x, tile.y, tile.value) for tile in self.tiles]
        return cloned_tiles
