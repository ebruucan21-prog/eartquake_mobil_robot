"""
navigation.py
A* Yol Planlama + Potansiyel Alan Takibi

1. A* ile global yol planla (engel haritasını bilir)
2. Robot yolu takip eder
3. LiDAR ile anlık dinamik engel kaçınma
4. Artçı sarsıntı sonrası yolu yeniden planla
"""

import numpy as np
import heapq
from robot import DT, MAX_SPEED, MAX_OMEGA
from environment import GRID_SIZE, FREE

# ──────────────────────────────────────────────
# Parametreler
# ──────────────────────────────────────────────
GOAL_THRESHOLD  = 1.5   # m — hedefe ulaşma mesafesi
WAYPOINT_DIST   = 1.8   # m — bir waypoint'e bu kadar yaklaşınca sonrakine geç
D0              = 1.5   # m — itici kuvvet etkili mesafesi
K_REP           = 2.0   # itici kuvvet kazancı
REPLAN_INTERVAL = 50   # Her bu adımda yolu yeniden planla


# ──────────────────────────────────────────────
# A* Yol Planlayıcı
# ──────────────────────────────────────────────
def astar(grid: np.ndarray, start: tuple, goal: tuple) -> list:
    """
    grid  : GRID_SIZE x GRID_SIZE numpy dizisi (0=serbest, diğer=engel)
    start : (row, col)
    goal  : (row, col)
    Döndürür: [(row,col), ...] yol listesi, bulunamazsa []
    """
    def h(a, b):
        return np.hypot(a[0]-b[0], a[1]-b[1])

    open_set = []
    heapq.heappush(open_set, (h(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    # 8 yönlü hareket
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                 (-1,-1),(-1,1),(1,-1),(1,1)]

    while open_set:
        _, cost, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dr, dc in neighbors:
            nr, nc = current[0]+dr, current[1]+dc
            if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                continue
            if grid[nr, nc] != FREE:
                continue

            step = np.hypot(dr, dc)
            tentative_g = g_score[current] + step

            if tentative_g < g_score.get((nr, nc), float('inf')):
                came_from[(nr, nc)] = current
                g_score[(nr, nc)] = tentative_g
                f = tentative_g + h((nr, nc), goal)
                heapq.heappush(open_set, (f, tentative_g, (nr, nc)))

    return []  # Yol bulunamadı


def smooth_path(path: list, window: int = 5) -> list:
    """Yolu hareketli ortalama ile yumuşat."""
    if len(path) <= window:
        return path
    smoothed = [path[0]]
    for i in range(1, len(path)-1):
        start_i = max(0, i - window//2)
        end_i   = min(len(path), i + window//2 + 1)
        avg_r = np.mean([p[0] for p in path[start_i:end_i]])
        avg_c = np.mean([p[1] for p in path[start_i:end_i]])
        smoothed.append((avg_r, avg_c))
    smoothed.append(path[-1])
    return smoothed


# ──────────────────────────────────────────────
# Ana Navigator
# ──────────────────────────────────────────────
class APFNavigator:
    def __init__(self, goal: tuple):
        self.goal         = np.array(goal, dtype=float)
        self.waypoints    : list = []
        self.wp_index     : int  = 0
        self.planned_path : list = []
        self.step_count   : int  = 0
        self._grid        = None

    # ── Yol Planlama ──────────────────────────
    def plan(self, grid: np.ndarray,
             robot_x: float, robot_y: float):
        """A* ile global yol planla, waypoint listesini güncelle."""
        self._grid = grid.copy()
        start = (int(round(robot_x)), int(round(robot_y)))
        goal  = (int(round(self.goal[0])), int(round(self.goal[1])))

        # Sınır kontrolü
        start = (np.clip(start[0],0,GRID_SIZE-1),
                 np.clip(start[1],0,GRID_SIZE-1))
        goal  = (np.clip(goal[0], 0,GRID_SIZE-1),
                 np.clip(goal[1], 0,GRID_SIZE-1))

        raw_path = astar(grid, start, goal)

        if not raw_path:
            print("[NAVİGASYON] A* yol bulamadı! Doğrudan hedefe yöneliniyor.")
            self.waypoints = [tuple(self.goal)]
        else:
            smoothed = smooth_path(raw_path, window=7)
            # Her 3 noktadan birini waypoint olarak al (çok sık olmasın)
            self.waypoints = [(float(p[0]), float(p[1]))
                              for p in smoothed[::3]]
            if tuple(goal) not in self.waypoints:
                self.waypoints.append((float(goal[0]), float(goal[1])))

        self.wp_index     = 0
        self.planned_path = [(float(p[0]), float(p[1]))
                             for p in (raw_path if raw_path else [])]
        print(f"[NAVİGASYON] Yol planlandı: {len(self.waypoints)} waypoint")

    # ── Ana Kontrol ───────────────────────────
    def compute_control(self, robot_x: float, robot_y: float,
                        robot_theta: float,
                        lidar_points: np.ndarray,
                        lidar_valid: np.ndarray) -> tuple:

        self.step_count += 1

        if not self.waypoints:
            return self._steer_to(robot_x, robot_y, robot_theta,
                                  self.goal[0], self.goal[1],
                                  lidar_points, lidar_valid)

        # Aktif waypoint
        if self.wp_index >= len(self.waypoints):
            self.wp_index = len(self.waypoints) - 1

        wx, wy = self.waypoints[self.wp_index]

        # Waypoint'e ulaştıysa sonrakine geç
        if np.hypot(robot_x - wx, robot_y - wy) < WAYPOINT_DIST:
            self.wp_index = min(self.wp_index + 1, len(self.waypoints)-1)
            wx, wy = self.waypoints[self.wp_index]

        return self._steer_to(robot_x, robot_y, robot_theta,
                              wx, wy, lidar_points, lidar_valid)

    # ── Hedefe Yönelme + İtici Kuvvet ─────────
    def _steer_to(self, rx, ry, rtheta,
                  tx, ty,
                  lidar_points, lidar_valid) -> tuple:

        # Hedef yönü
        desired_angle = np.arctan2(ty - ry, tx - rx)
        angle_error   = self._norm_angle(desired_angle - rtheta)

        # İtici kuvvet (LiDAR)
        f_rep = self._repulsive(np.array([rx, ry]),
                                lidar_points, lidar_valid)
        rep_angle = np.arctan2(f_rep[1], f_rep[0])
        rep_mag   = np.linalg.norm(f_rep)

        # İtici kuvveti açı hatasına ekle
        if rep_mag > 0.1:
            blend      = min(rep_mag / 3.0, 1.0)
            blended    = (1-blend)*desired_angle + blend*rep_angle
            angle_error = self._norm_angle(blended - rtheta)

        # Hız hesapla
        v_scale = np.cos(angle_error) ** 2
        v       = MAX_SPEED * 0.8 * max(0.1, v_scale)

        # Büyük açı hatasında yerinde dön
        if abs(angle_error) > np.pi / 3:
            v = 0.05

        omega = np.clip(3.0 * angle_error, -MAX_OMEGA, MAX_OMEGA)

        return float(v), float(omega)

    # ── İtici Kuvvet ──────────────────────────
    def _repulsive(self, pos, lidar_points, lidar_valid) -> np.ndarray:
        f = np.zeros(2)
        if lidar_points is None or not np.any(lidar_valid):
            return f
        for pt in lidar_points[lidar_valid]:
            diff = pos - pt
            d    = np.linalg.norm(diff)
            if 0.1 < d < D0:
                mag  = K_REP * (1/d - 1/D0) / d**2
                f   += mag * diff / d
        norm = np.linalg.norm(f)
        return f / norm * min(norm, 4.0) if norm > 1e-6 else f

    # ── Yardımcılar ───────────────────────────
    def is_goal_reached(self, rx, ry) -> bool:
        return np.hypot(rx-self.goal[0], ry-self.goal[1]) < GOAL_THRESHOLD

    def distance_to_goal(self, rx, ry) -> float:
        return float(np.hypot(rx-self.goal[0], ry-self.goal[1]))

    @staticmethod
    def _norm_angle(a):
        return (a + np.pi) % (2*np.pi) - np.pi


if __name__ == "__main__":
    print("Navigation modülü hazır (A* + APF).")