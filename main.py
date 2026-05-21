"""
main.py — Deprem Sonrası Bina İçi Otonom Navigasyon
Dead reckoning + EKF + LiDAR kümeleme
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from environment import Environment, GRID_SIZE, FREE, START, GOAL
from robot       import DifferentialRobot, DT
from lidar       import LiDAR, NUM_RAYS, MAX_RANGE
from kalman      import ExtendedKalmanFilter
from navigation  import APFNavigator, REPLAN_INTERVAL

MAX_STEPS        = 3000
AFTERSHOCK_STEPS = [500, 1000, 1500]
LIDAR_FREQ       = 3
SEED             = 42
OUT              = "outputs/"


def median_filter(ranges, k=3):
    out = np.zeros_like(ranges)
    n   = len(ranges)
    for i in range(n):
        nb = [ranges[(i+j) % n] for j in range(-k, k+1)]
        out[i] = np.median(nb)
    return out


def run_simulation():
    print("="*55)
    print("  Deprem Sonrası Bina — Otonom Navigasyon")
    print("="*55)

    env   = Environment(seed=SEED)
    robot = DifferentialRobot((float(START[0]), float(START[1])),
                               start_theta=np.pi/4, seed=SEED)
    lidar = LiDAR(env, seed=SEED+1)
    ekf   = ExtendedKalmanFilter(
                np.array([float(START[0]), float(START[1]), np.pi/4]))
    nav   = APFNavigator(goal=(float(GOAL[0]), float(GOAL[1])))
    nav.plan(env.get_grid(), robot.x, robot.y)

    # Dead reckoning başlangıç
    dr_x, dr_y, dr_theta = float(START[0]), float(START[1]), np.pi/4

    true_path, ekf_path, dr_path = [(robot.x, robot.y)], [(robot.x, robot.y)], [(dr_x, dr_y)]
    dist_log, err_ekf, err_dr    = [], [], []
    lidar_raw_saved = lidar_filt_saved = None
    cluster_log = []

    v, omega     = 0.5, 0.0
    goal_reached = False

    print(f"Başlangıç: {START}  →  Hedef: {GOAL}\n")

    for step in range(MAX_STEPS):

        # 1. Gerçek hareket
        true_pose = robot.step(v, omega)

        # 2. Artçı sarsıntı
        if step in AFTERSHOCK_STEPS:
            env.trigger_aftershock(n_new=3)
            nav.plan(env.get_grid(), true_pose[0], true_pose[1])

        # 3. Periyodik yeniden planlama
        if step > 0 and step % REPLAN_INTERVAL == 0:
            nav.plan(env.get_grid(), true_pose[0], true_pose[1])

        # 4. Sensörler
        enc = robot.read_encoder()
        imu = robot.read_imu()

        lidar_data = None
        if step % LIDAR_FREQ == 0:
            lidar_data  = lidar.scan(*true_pose)
            filt_ranges = median_filter(lidar_data["ranges"])
            if lidar_raw_saved is None:
                lidar_raw_saved  = lidar_data
                lidar_filt_saved = filt_ranges
            # Küme sayısını kaydet
            cluster_log.append(len(lidar_data["clusters"]))

        # 5. Dead Reckoning (sadece enkoder ile, EKF yok)
        dr_x     += enc["v"] * np.cos(dr_theta) * DT
        dr_y     += enc["v"] * np.sin(dr_theta) * DT
        dr_theta += enc["omega"] * DT
        dr_theta  = (dr_theta + np.pi) % (2*np.pi) - np.pi

        # 6. EKF
        ekf.predict(enc["v"], enc["omega"])
        ekf.update_imu(imu["theta_est"])
        if lidar_data is not None:
            lx = true_pose[0] + np.random.normal(0, 0.20)
            ly = true_pose[1] + np.random.normal(0, 0.20)
            ekf.update_lidar(lx, ly)
        ekf.record_true(true_pose)
        est = ekf.get_estimate()

        # 7. Kayıtlar
        true_path.append((true_pose[0], true_pose[1]))
        ekf_path.append((est[0], est[1]))
        dr_path.append((dr_x, dr_y))
        dist_log.append(nav.distance_to_goal(*true_pose[:2]))
        err_ekf.append(float(np.linalg.norm(true_pose[:2] - est[:2])))
        err_dr.append(float(np.hypot(true_pose[0]-dr_x,
                                     true_pose[1]-dr_y)))

        # 8. Hedef
        if nav.is_goal_reached(*true_pose[:2]):
            goal_reached = True
            print(f"\n✅ Hedefe ulaşıldı! Adım: {step+1}")
            break

        # 9. Kontrol
        if lidar_data is not None:
            v, omega = nav.compute_control(
                est[0], est[1], est[2],
                lidar_data["points"], lidar_data["valid"])
        else:
            gd = np.array([GOAL[0]-est[0], GOAL[1]-est[1]])
            da = np.arctan2(gd[1], gd[0])
            ae = (da - est[2] + np.pi) % (2*np.pi) - np.pi
            v, omega = 0.4, float(np.clip(2.0*ae, -2.0, 2.0))

        if step % 200 == 0:
            print(f"Adım {step:4d} | ({true_pose[0]:.1f},{true_pose[1]:.1f})"
                  f" | hedefe={dist_log[-1]:.1f}m"
                  f" | EKF_hata={err_ekf[-1]:.3f}m"
                  f" | DR_hata={err_dr[-1]:.3f}m")

    if not goal_reached:
        print(f"\n⚠ Maks adıma ulaşıldı. En yakın: {min(dist_log):.2f}m")

    errors = ekf.compute_errors()
    _plot_all(env, true_path, ekf_path, dr_path,
              dist_log, err_ekf, err_dr, errors,
              lidar_raw_saved, lidar_filt_saved,
              cluster_log, nav.planned_path)
    return true_path, ekf_path, errors


def _plot_all(env, true_path, ekf_path, dr_path,
              dist_log, err_ekf, err_dr, errors,
              lidar_raw, lidar_filt, cluster_log, planned_path):

    cmap = ListedColormap(["#F5F0E8","#2C2C2C","#C0392B","#E67E22"])

    # ── 1. Ortam Haritası + Yol Planı ──────────
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(env.grid, cmap=cmap, origin="lower",
              vmin=0, vmax=3, interpolation="nearest")

    if planned_path:
        pp = np.array(planned_path)
        ax.plot(pp[:,1], pp[:,0], "w--", lw=1,
                alpha=0.5, label="Planlanan Yol")

    tp = np.array(true_path)
    ep = np.array(ekf_path)
    dp = np.array(dr_path)

    ax.plot(tp[:,1], tp[:,0], color="#F39C12", lw=2,   label="Gerçek Yol")
    ax.plot(ep[:,1], ep[:,0], color="#3498DB", lw=1.5,
            linestyle="--", alpha=0.8, label="EKF Tahmini")
    ax.plot(dp[:,1], dp[:,0], color="#9B59B6", lw=1.2,
            linestyle=":", alpha=0.7, label="Dead Reckoning")

    ax.plot(START[1], START[0], "go", ms=12, label="Başlangıç")
    ax.plot(GOAL[1],  GOAL[0],  "b*", ms=16, label="Çıkış (Hedef)")

    patches = [
        mpatches.Patch(color="#F5F0E8", label="Serbest Alan"),
        mpatches.Patch(color="#2C2C2C", label="Duvar"),
        mpatches.Patch(color="#C0392B", label="Sabit Enkaz"),
        mpatches.Patch(color="#E67E22", label="Dinamik Engel"),
    ]
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h+patches, loc="upper left",
              fontsize=8, framealpha=0.9)
    ax.set_title("Ortam Haritası ve Robot Yol Planı",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Y (m)"); ax.set_ylabel("X (m)")
    plt.tight_layout()
    plt.savefig(OUT+"map_and_path.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Harita + Yol → map_and_path.png")

    # ── 2. Lokalizasyon: Gerçek vs EKF vs DR ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(tp[:,0], tp[:,1], "g-",  lw=2,   label="Gerçek Yol")
    ax.plot(ep[:,0], ep[:,1], "b--", lw=1.5, label="EKF Tahmini")
    ax.plot(dp[:,0], dp[:,1], color="#9B59B6", lw=1.2,
            linestyle=":", label="Dead Reckoning")
    ax.plot(*START, "go", ms=10)
    ax.plot(*GOAL,  "b*", ms=14)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("2B Yol: Gerçek vs EKF vs Dead Reckoning")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax2 = axes[1]
    t = np.arange(len(tp))
    n = min(len(t), len(ep), len(dp))
    ax2.plot(t[:n], tp[:n,0], "g-",   lw=1.5, label="Gerçek x(t)")
    ax2.plot(t[:n], ep[:n,0], "b--",  lw=1.2, label="EKF x(t)")
    ax2.plot(t[:n], dp[:n,0], color="#9B59B6",
             lw=1.0, linestyle=":", label="DR x(t)")
    ax2.plot(t[:n], tp[:n,1], "lime", lw=1.5, label="Gerçek y(t)")
    ax2.plot(t[:n], ep[:n,1], "cyan", lw=1.2,
             linestyle="--", label="EKF y(t)")
    ax2.set_xlabel("Adım (t)"); ax2.set_ylabel("Konum (m)")
    ax2.set_title("x(t) ve y(t): Gerçek vs EKF vs DR")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.suptitle("Lokalizasyon Sonuçları", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT+"localization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Lokalizasyon → localization.png")

    # ── 3. Hata Analizi ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(err_ekf, color="#E74C3C", lw=1.2, label="EKF Konum Hatası")
    ax.plot(err_dr,  color="#9B59B6", lw=1.0,
            linestyle="--", alpha=0.8, label="Dead Reckoning Hatası")
    if errors:
        ax.axhline(errors["rmse"], color="#3498DB", ls="--",
                   label=f"EKF RMSE={errors['rmse']:.4f} m")
        ax.axhline(errors["mae"],  color="#2ECC71", ls=":",
                   label=f"EKF MAE={errors['mae']:.4f} m")
    ax.set_xlabel("Adım"); ax.set_ylabel("Hata (m)")
    ax.set_title("Zaman Boyunca Konum Hatası\n(EKF vs Dead Reckoning)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(dist_log, color="#8E44AD", lw=1.5)
    ax2.axhline(1.5, color="#E74C3C", ls="--",
                lw=1.2, label="Hedef eşiği (1.5 m)")
    for s in AFTERSHOCK_STEPS:
        if s < len(dist_log):
            ax2.axvline(s, color="orange", ls=":", lw=1.2,
                        label=f"Artçı @{s}")
    ax2.set_xlabel("Adım"); ax2.set_ylabel("Hedefe Mesafe (m)")
    ax2.set_title("Hedefe Mesafe (Artçı Sarsıntı Anları İşaretli)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.suptitle("Hata Analizi", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT+"error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Hata analizi → error_analysis.png")

    # ── 4. LiDAR Ham vs Filtrelenmiş ──────────
    if lidar_raw is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                                  subplot_kw={"projection": "polar"})
        angles = lidar_raw["angles"]
        axes[0].scatter(angles, lidar_raw["ranges"],
                        s=8, c="#E74C3C", alpha=0.7)
        axes[0].set_title("Ham LiDAR Verisi\n(gürültülü)", fontsize=11)
        axes[0].set_rmax(MAX_RANGE)
        axes[1].scatter(angles, lidar_filt,
                        s=8, c="#27AE60", alpha=0.7)
        axes[1].set_title("Filtrelenmiş LiDAR\n(medyan filtre)", fontsize=11)
        axes[1].set_rmax(MAX_RANGE)
        plt.suptitle("LiDAR Sensör Görselleştirmesi — Ham vs Filtrelenmiş",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT+"lidar_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("LiDAR → lidar_comparison.png")

    # ── 5. Engel Kümeleme Grafiği ─────────────
    if cluster_log:
        fig, ax = plt.subplots(figsize=(10, 4))
        steps_cl = np.arange(len(cluster_log)) * LIDAR_FREQ
        ax.plot(steps_cl, cluster_log, color="#E67E22",
                lw=1.5, label="Algılanan küme sayısı")
        for s in AFTERSHOCK_STEPS:
            if s < steps_cl[-1]:
                ax.axvline(s, color="red", ls="--",
                           lw=1.2, alpha=0.7, label=f"Artçı @{s}")
        ax.set_xlabel("Adım"); ax.set_ylabel("Küme Sayısı")
        ax.set_title("LiDAR Engel Kümeleme — Zaman İçinde Algılanan Küme Sayısı")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT+"clustering.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Kümeleme → clustering.png")

    print("\n✅ Tüm grafikler kaydedildi.")
    if errors:
        print(f"   EKF RMSE : {errors['rmse']:.4f} m")
        print(f"   EKF MAE  : {errors['mae']:.4f} m")
        if err_dr:
            dr_rmse = float(np.sqrt(np.mean(np.array(err_dr)**2)))
            print(f"   DR  RMSE : {dr_rmse:.4f} m  (karşılaştırma)")


if __name__ == "__main__":
    run_simulation()