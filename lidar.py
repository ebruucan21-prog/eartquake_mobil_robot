"""
lidar.py
2B LiDAR Simülatörü + Engel Kümeleme
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import Environment, GRID_SIZE, FREE

NUM_RAYS   = 36
MAX_RANGE  = 10.0
ANGLE_STEP = 2 * np.pi / NUM_RAYS
NOISE_STD  = 0.15
MISS_PROB  = 0.05


class LiDAR:
    def __init__(self, env: Environment, seed: int = 1):
        self.env  = env
        self.rng  = np.random.default_rng(seed)
        self.grid = env.get_grid()

        self.last_ranges  = None
        self.last_angles  = None
        self.last_points  = None
        self.last_clusters = []

    # ──────────────────────────────────────────
    # Ana tarama
    # ──────────────────────────────────────────
    def scan(self, robot_x, robot_y, robot_theta) -> dict:
        self.grid = self.env.get_grid()

        ranges = np.full(NUM_RAYS, MAX_RANGE)
        valid  = np.ones(NUM_RAYS, dtype=bool)
        angles = np.array([robot_theta + i * ANGLE_STEP
                           for i in range(NUM_RAYS)])

        for i, angle in enumerate(angles):
            if self.rng.random() < MISS_PROB:
                ranges[i] = MAX_RANGE
                valid[i]  = False
                continue
            dist = self._cast_ray(robot_x, robot_y, angle)
            dist += self.rng.normal(0, NOISE_STD)
            dist  = np.clip(dist, 0.0, MAX_RANGE)
            ranges[i] = dist

        points = np.column_stack([
            robot_x + ranges * np.cos(angles),
            robot_y + ranges * np.sin(angles),
        ])

        self.last_ranges = ranges
        self.last_angles = angles - robot_theta
        self.last_points = points

        # Engel kümeleme
        self.last_clusters = self.cluster_obstacles(points, valid, ranges)

        return {
            "ranges"  : ranges,
            "angles"  : self.last_angles,
            "points"  : points,
            "valid"   : valid,
            "clusters": self.last_clusters,
        }

    # ──────────────────────────────────────────
    # DDA Işın Döküm
    # ──────────────────────────────────────────
    def _cast_ray(self, ox, oy, angle) -> float:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        step  = 0.1
        dist  = 0.0
        while dist < MAX_RANGE:
            dist += step
            cx = ox + dist * cos_a
            cy = oy + dist * sin_a
            r, c = int(cx), int(cy)
            if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE):
                return dist
            if self.grid[r, c] != FREE:
                return dist
        return MAX_RANGE

    # ──────────────────────────────────────────
    # Engel Kümeleme (basit mesafe tabanlı)
    # ──────────────────────────────────────────
    def cluster_obstacles(self, points, valid, ranges,
                          cluster_dist=1.5, min_points=2) -> list:
        """
        Geçerli ve MAX_RANGE'den kısa mesafeli LiDAR noktalarını
        mesafe bazlı kümelere ayırır.

        Döndürür: [ {"center": (x,y), "points": [...], "size": n}, ... ]
        """
        # Sadece engele çarpan (kısa mesafeli) geçerli noktalar
        mask = valid & (ranges < MAX_RANGE - 0.5)
        if not np.any(mask):
            return []

        pts = points[mask]
        n   = len(pts)
        labels     = -np.ones(n, dtype=int)
        cluster_id = 0

        for i in range(n):
            if labels[i] >= 0:
                continue
            # Komşuları bul
            dists = np.linalg.norm(pts - pts[i], axis=1)
            neighbors = np.where(dists < cluster_dist)[0]
            if len(neighbors) < min_points:
                continue
            # Yeni küme
            labels[neighbors] = cluster_id
            # Genişlet
            queue = list(neighbors)
            while queue:
                idx = queue.pop()
                d2  = np.linalg.norm(pts - pts[idx], axis=1)
                new = np.where((d2 < cluster_dist) & (labels == -1))[0]
                labels[new] = cluster_id
                queue.extend(new.tolist())
            cluster_id += 1

        clusters = []
        for cid in range(cluster_id):
            cpts = pts[labels == cid]
            if len(cpts) == 0:
                continue
            center = cpts.mean(axis=0)
            clusters.append({
                "center": (float(center[0]), float(center[1])),
                "points": cpts,
                "size"  : len(cpts),
            })

        return clusters

    # ──────────────────────────────────────────
    # Konum tahmini (Kalman için ölçüm)
    # ──────────────────────────────────────────
    def estimate_position(self, robot_x, robot_y, robot_theta):
        scan_data = self.scan(robot_x, robot_y, robot_theta)
        rng  = scan_data["ranges"]
        pts  = scan_data["points"]
        valid = scan_data["valid"]
        weights = np.where(valid, 1.0 / (rng + 0.1), 0.0)
        if weights.sum() < 1e-9:
            return np.array([robot_x, robot_y])
        min_idx = np.argmin(rng)
        return np.array([robot_x, robot_y,
                         pts[min_idx, 0], pts[min_idx, 1]])

    # ──────────────────────────────────────────
    # Görselleştirme
    # ──────────────────────────────────────────
    def render(self, robot_x, robot_y, robot_theta,
               ax=None, show=True):
        scan_data = self.scan(robot_x, robot_y, robot_theta)
        angles = scan_data["angles"]
        ranges = scan_data["ranges"]
        valid  = scan_data["valid"]

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(7, 7),
                                   subplot_kw={"projection": "polar"})

        ax.scatter(angles[valid],  ranges[valid],
                   s=8, color="#E74C3C", alpha=0.8, label="Algılanan")
        ax.scatter(angles[~valid], ranges[~valid],
                   s=4, color="#95A5A6", alpha=0.4, label="Kayıp ışın")
        ax.set_rmax(MAX_RANGE)
        ax.set_title(f"LiDAR ({NUM_RAYS} ışın, σ={NOISE_STD}m)",
                     va="bottom", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)

        if standalone and show:
            plt.tight_layout()
            plt.savefig("outputs/lidar_scan.png", dpi=150,
                        bbox_inches="tight")
            plt.show()
            print("LiDAR taraması kaydedildi.")


if __name__ == "__main__":
    env   = Environment(seed=42)
    lidar = LiDAR(env, seed=3)

    rx, ry, rtheta = 5.0, 5.0, 0.0
    data = lidar.scan(rx, ry, rtheta)

    print(f"Işın sayısı      : {len(data['ranges'])}")
    print(f"Min mesafe       : {data['ranges'].min():.2f} m")
    print(f"Küme sayısı      : {len(data['clusters'])}")
    for i, cl in enumerate(data["clusters"]):
        print(f"  Küme {i+1}: merkez={cl['center']}, "
              f"nokta sayısı={cl['size']}")

    lidar.render(rx, ry, rtheta)