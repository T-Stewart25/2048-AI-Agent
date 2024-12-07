import pygame
from game_2048.setup import initialize_game, draw_grid
from game_2048.tiles import Tiles

def run_2048_game():
    screen, grid_size, tile_size, base_colors, score_area_height = initialize_game()
    tiles = Tiles(tile_size, score_area_height)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    tiles.move('down')
                elif event.key == pygame.K_UP:
                    tiles.move('up')
                elif event.key == pygame.K_LEFT:
                    tiles.move('left')
                elif event.key == pygame.K_RIGHT:
                    tiles.move('right')

        if tiles.check_game_over():
            print("Game Over!")
            running = False

        screen.fill(base_colors['background'])
        draw_grid(screen, grid_size, tile_size, base_colors, score_area_height)
        tiles.draw_tiles(screen)

        score = tiles.score
        font = pygame.font.Font(None, 40)
        score_text = font.render(f"Score: {score}", True, (119, 110, 101))
        screen.blit(score_text, (10, 10))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    run_2048_game()
