import pygame

def initialize_game():
    pygame.init()
    colors = {
        'background': (187, 173, 160),
        'grid': (205, 193, 180),
    }
    grid_size = 4
    tile_size = 100
    margin = 10
    score_area_height = 60
    window_size = grid_size * (tile_size + margin) + margin
    screen = pygame.display.set_mode((window_size, window_size + score_area_height))
    pygame.display.set_caption("2048 Game")
    return screen, grid_size, tile_size, colors, score_area_height

def draw_grid(screen, grid_size, tile_size, colors, score_area_height):
    margin = 10
    for row in range(grid_size):
        for col in range(grid_size):
            rect = pygame.Rect(
                col * (tile_size + margin) + margin,
                row * (tile_size + margin) + margin + score_area_height,
                tile_size,
                tile_size
            )
            pygame.draw.rect(screen, colors['grid'], rect)
