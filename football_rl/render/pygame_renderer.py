from __future__ import annotations

import numpy as np


class PygameRenderer:
    def __init__(self, config):
        self.config = config
        self._pygame = None
        self._screen = None
        self._clock = None
        self._window_size = (960, 640)
        self._scale_x = self._window_size[0] / config.physics.field_width
        self._scale_y = self._window_size[1] / config.physics.field_height

    def _ensure_init(self) -> None:
        if self._pygame is not None:
            return
        import pygame

        self._pygame = pygame
        pygame.init()
        self._screen = pygame.display.set_mode(self._window_size)
        pygame.display.set_caption("football-rl")
        self._clock = pygame.time.Clock()

    def _to_screen(self, point) -> tuple[int, int]:
        return int(point[0] * self._scale_x), int(point[1] * self._scale_y)

    def render(self, simulator, mode: str = "human"):
        self._ensure_init()
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
        self._screen.fill(self.config.background_color)
        field_rect = pygame.Rect(0, 0, self._window_size[0], self._window_size[1])
        pygame.draw.rect(self._screen, (230, 230, 230), field_rect, width=4)
        pygame.draw.line(self._screen, (230, 230, 230), (self._window_size[0] // 2, 0), (self._window_size[0] // 2, self._window_size[1]), width=2)
        goal_top = int((self.config.physics.field_height / 2 - self.config.physics.goal_width / 2) * self._scale_y)
        goal_bottom = int((self.config.physics.field_height / 2 + self.config.physics.goal_width / 2) * self._scale_y)
        pygame.draw.line(self._screen, (255, 255, 255), (0, goal_top), (0, goal_bottom), width=6)
        pygame.draw.line(self._screen, (255, 255, 255), (self._window_size[0] - 1, goal_top), (self._window_size[0] - 1, goal_bottom), width=6)
        for obstacle in simulator.obstacles:
            px, py = self._to_screen(obstacle.position)
            hw = int(obstacle.half_extents[0] * self._scale_x)
            hh = int(obstacle.half_extents[1] * self._scale_y)
            rect = pygame.Rect(px - hw, py - hh, 2 * hw, 2 * hh)
            pygame.draw.rect(self._screen, (220, 220, 80), rect)
        for player in simulator.players.values():
            px, py = self._to_screen(player.position)
            radius = int(player.radius * min(self._scale_x, self._scale_y))
            color = self.config.team_colors[player.color_id]
            pygame.draw.circle(self._screen, color, (px, py), radius)
            if player.role == player.role.GOALKEEPER:
                pygame.draw.circle(self._screen, (255, 255, 255), (px, py), max(radius - 4, 2), width=2)
        bx, by = self._to_screen(simulator.ball.position)
        br = int(simulator.ball.radius * min(self._scale_x, self._scale_y))
        pygame.draw.circle(self._screen, (245, 245, 245), (bx, by), br)
        pygame.display.flip()
        self._clock.tick(self.config.render_fps)
        if mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self._screen), (1, 0, 2))
        return None

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
            self._screen = None
            self._clock = None
