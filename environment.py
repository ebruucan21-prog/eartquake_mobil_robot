"""
environment.py — 30x30 m bina planı
Sağ taraf açık, sol tarafta 2 kapalı oda, ortada koridor
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

GRID_SIZE  = 30
CELL_SIZE  = 1.0
FREE       = 0
WALL       = 1
DEBRIS     = 2
DYN_DEBRIS = 3

START = (2, 2)    # Sol alt
GOAL  = (27, 27)  # Sağ üst


class Environment:
    def __init__(self, seed: int = 42):
        self.rng   = np.random.default_rng(seed)
        self.grid  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        self.dynamic_obstacles = []
        self.aftershock_count  = 0
        self._build_walls()
        self._place_static_debris()

    def _build_walls(self):
        g = self.grid

        # Dış çevre
        g[0, :]  = WALL
        g[-1, :] = WALL
        g[:, 0]  = WALL
        g[:, -1] = WALL

        # Yatay duvar 1 — satır 10, SADECE sol parça (sütun 1-5)
        # Geçit: sütun 5-29 tamamen açık (sağ taraf açık)
        g[10, 1:5] = WALL

        # Yatay duvar 2 — satır 20
        # Sol parça: sütun 1-8
        g[20, 1:8] = WALL
        # Sağ parça: sütun 15-29
        g[20, 15:29] = WALL
        # Geçit: sütun 8-15 (7 hücre, geniş)

        # Dikey duvar — sütun 15, sadece satır 10-20
        g[10:20, 15] = WALL

        # Başlangıç ve hedef çevresini temizle (3 hücre)
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                for (pr, pc) in [START, GOAL]:
                    r, c = pr+dr, pc+dc
                    if 0 < r < GRID_SIZE-1 and 0 < c < GRID_SIZE-1:
                        g[r, c] = FREE

    def _place_static_debris(self):
        # Sabit enkaz — duvara ya da geçide yakın olmasın
        fixed = [
            (4,  4),  (4, 14),  (4, 22),
            (7,  7),  (7, 20),
            (14, 4),  (14, 21),
            (25, 5),  (25, 19),
            (22, 10),
        ]
        for r, c in fixed:
            if self.grid[r, c] == FREE:
                self.grid[r, c] = DEBRIS

    def trigger_aftershock(self, n_new=3):
        self.aftershock_count += 1
        added, attempts = [], 0
        while len(added) < n_new and attempts < 5_000:
            r = int(self.rng.integers(2, GRID_SIZE-2))
            c = int(self.rng.integers(2, GRID_SIZE-2))
            if (self.grid[r, c] == FREE
                    and abs(r-START[0]) > 3 and abs(c-START[1]) > 3
                    and abs(r-GOAL[0])  > 3 and abs(c-GOAL[1])  > 3):
                self.grid[r, c] = DYN_DEBRIS
                self.dynamic_obstacles.append((r, c))
                added.append((r, c))
            attempts += 1
        print(f"[Artçı #{self.aftershock_count}] {len(added)} yeni engel: {added}")
        return added

    def is_free(self, r, c):
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            return self.grid[r, c] == FREE
        return False

    def is_obstacle(self, r, c):
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            return self.grid[r, c] != FREE
        return True

    def get_grid(self):
        return self.grid.copy()

    def render(self, robot_pos=None, robot_path=None,
               ekf_path=None, planned_path=None,
               title="Ortam Haritası"):
        cmap = ListedColormap(["#F5F0E8","#2C2C2C","#C0392B","#E67E22"])
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.grid, cmap=cmap, origin="lower",
                  vmin=0, vmax=3, interpolation="nearest")
        if planned_path and len(planned_path) > 1:
            pp = np.array(planned_path)
            ax.plot(pp[:,1], pp[:,0], "w--", lw=1, alpha=0.5, label="Planlanan")
        if robot_path and len(robot_path) > 1:
            rp = np.array(robot_path)
            ax.plot(rp[:,1], rp[:,0], color="#F39C12", lw=2, label="Gerçek Yol")
        if ekf_path and len(ekf_path) > 1:
            ep = np.array(ekf_path)
            ax.plot(ep[:,1], ep[:,0], color="#3498DB", lw=1.5,
                    linestyle="--", alpha=0.8, label="EKF Tahmini")
        if robot_pos is not None:
            ax.plot(robot_pos[1], robot_pos[0], "D",
                    color="#8E44AD", ms=10, label="Robot")
        ax.plot(START[1], START[0], "go", ms=12, label="Başlangıç")
        ax.plot(GOAL[1],  GOAL[0],  "b*", ms=16, label="Çıkış (Hedef)")
        legend_patches = [
            mpatches.Patch(color="#F5F0E8", label="Serbest"),
            mpatches.Patch(color="#2C2C2C", label="Duvar"),
            mpatches.Patch(color="#C0392B", label="Sabit Enkaz"),
            mpatches.Patch(color="#E67E22", label="Dinamik Engel"),
        ]
        h, _ = ax.get_legend_handles_labels()
        ax.legend(handles=h+legend_patches, loc="upper left",
                  fontsize=8, framealpha=0.9)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Y (m)"); ax.set_ylabel("X (m)")
        plt.tight_layout()
        return fig, ax


if __name__ == "__main__":
    env = Environment(seed=42)
    print(f"Başlangıç serbest: {env.is_free(*START)}")
    print(f"Hedef serbest    : {env.is_free(*GOAL)}")
    fig, ax = env.render(title="Bina Planı — Deprem Senaryosu")
    plt.savefig("outputs/environment_map.png", dpi=150, bbox_inches="tight")
    plt.show()