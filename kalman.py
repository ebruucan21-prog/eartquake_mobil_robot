import numpy as np
from robot import DT


# Kalman Filtresi Gürültü Parametreleri

# Süreç gürültüsü kovaryans matrisi Q (model belirsizliği)
Q = np.diag([0.05**2,   # x (m)
             0.05**2,   # y (m)
             0.02**2])  # theta (rad)

# Enkoder ölçüm gürültüsü kovaryans matrisi R_enc
R_ENC = np.diag([0.03**2,   # x
                 0.03**2,   # y
                 0.015**2]) # theta

# IMU ölçüm gürültüsü (sadece theta)
R_IMU = np.array([[0.01**2]])

# LiDAR pozisyon gürültüsü (x, y)
R_LIDAR = np.diag([0.20**2,   # x  (toz/duman → yüksek)
                   0.20**2])  # y


class ExtendedKalmanFilter:
    """
    Genişletilmiş Kalman Filtresi (EKF)
    Durum: [x, y, theta]  (3×1)

    Üç sensörü birleştirir:
      1. Tekerlek enkoderi  → (x, y, theta) tahmini
      2. IMU               → theta ölçümü
      3. LiDAR             → (x, y) düzeltmesi

    Her simülasyon adımında:
      predict(v, omega)          — enkoder odometri ile ön tahmin
      update_imu(theta_meas)     — IMU ile theta düzeltmesi
      update_lidar(x_m, y_m)    — LiDAR ile konum düzeltmesi
    """

    def __init__(self, init_pose: np.ndarray):
        """
        init_pose : [x0, y0, theta0]
        """
        self.mu  = init_pose.astype(float).copy()   # durum tahmini 
        self.P   = np.eye(3) * 0.1                  # kovaryans matrisi (3×3)

        # Hata kaydı (RMSE/MAE hesabı için)
        self.history_mu  : list[np.ndarray] = [self.mu.copy()]
        self.history_true: list[np.ndarray] = []

    # 1. Tahmin Adımı (Predict)

    def predict(self, v: float, omega: float) -> np.ndarray:
        """
        Enkoder / kontrol girdisiyle durum tahmini.
        Non-holonomic unicycle modeli:
            x_new = x + v*cos(θ)*DT
            y_new = y + v*sin(θ)*DT
            θ_new = θ + ω*DT

        Jakobian F = ∂f/∂x linearizasyonu uygulanır (EKF).
        """
        x, y, th = self.mu

        # Durum geçiş tahmini
        mu_pred = np.array([
            x  + v * np.cos(th) * DT,
            y  + v * np.sin(th) * DT,
            th + omega * DT,
        ])
        mu_pred[2] = self._normalize(mu_pred[2])

        # Jakobian F (∂f/∂x)
        F = np.array([
            [1, 0, -v * np.sin(th) * DT],
            [0, 1,  v * np.cos(th) * DT],
            [0, 0,  1               ],
        ])

        # Kovaryans güncellemesi: P = F P Fᵀ + Q
        self.P  = F @ self.P @ F.T + Q
        self.mu = mu_pred

        self.history_mu.append(self.mu.copy())
        return self.mu.copy()

  
    # 2. IMU Güncelleme Adımı
  
    def update_imu(self, theta_meas: float) -> np.ndarray:
        """
        IMU'dan gelen theta (yaw) ölçümüyle durum düzeltmesi.
        Ölçüm modeli: z = theta  (H = [0, 0, 1])
        """
        H  = np.array([[0.0, 0.0, 1.0]])         # 1×3
        z  = np.array([theta_meas])               # 1×1

        # Yenilik (innovation)
        y_inn = z - H @ self.mu
        y_inn[0] = self._normalize(y_inn[0])

        # İnovasyon kovaryansı: S = H P Hᵀ + R
        S = H @ self.P @ H.T + R_IMU              # 1×1

        # Kalman kazancı: K = P Hᵀ S⁻¹
        K = self.P @ H.T @ np.linalg.inv(S)      # 3×1

        # Durum güncelleme
        self.mu = self.mu + (K @ y_inn).flatten()
        self.mu[2] = self._normalize(self.mu[2])

        # Kovaryans güncelleme: P = (I - KH) P
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P

        self.history_mu[-1] = self.mu.copy()
        return self.mu.copy()


    # 3. LiDAR Güncelleme Adımı
  
    def update_lidar(self, x_meas: float, y_meas: float) -> np.ndarray:
        """
        LiDAR'dan gelen (x, y) ölçümüyle konum düzeltmesi.
        Ölçüm modeli: z = [x, y]  (H = [[1,0,0],[0,1,0]])
        """
        H  = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])          # 2×3
        z  = np.array([x_meas, y_meas])           # 2×1

        # Yenilik
        y_inn = z - H @ self.mu                   # 2×1

        # İnovasyon kovaryansı
        S = H @ self.P @ H.T + R_LIDAR            # 2×2

        # Kalman kazancı
        K = self.P @ H.T @ np.linalg.inv(S)      # 3×2

        # Güncelleme
        self.mu = self.mu + K @ y_inn
        self.mu[2] = self._normalize(self.mu[2])

        I = np.eye(3)
        self.P = (I - K @ H) @ self.P

        self.history_mu[-1] = self.mu.copy()
        return self.mu.copy()

    # 4. Hata analizi (RMSE / MAE)
 
    def record_true(self, true_pose: np.ndarray):
        """Her adımda gerçek konumu kaydet (hata analizi için)."""
        self.history_true.append(true_pose.copy())

    def compute_errors(self) -> dict:
        """
        RMSE ve MAE hesapla.
        history_mu ve history_true eşit uzunlukta olmalı.
        """
        n = min(len(self.history_mu), len(self.history_true))
        if n == 0:
            return {}

        est   = np.array(self.history_mu[:n])
        truth = np.array(self.history_true[:n])

        # Sadece x, y üzerinden hesapla
        pos_err = np.linalg.norm(est[:, :2] - truth[:, :2], axis=1)

        rmse = float(np.sqrt(np.mean(pos_err**2)))
        mae  = float(np.mean(np.abs(pos_err)))
        max_e = float(np.max(pos_err))

        print(f"\n── Hata Analizi ({'n='+str(n)} adım) ──")
        print(f"  RMSE (konum) : {rmse:.4f} m")
        print(f"  MAE  (konum) : {mae:.4f} m")
        print(f"  Maks hata    : {max_e:.4f} m")

        return {"rmse": rmse, "mae": mae, "max": max_e,
                "pos_err": pos_err}

 
    # Yardımcılar
 
    @staticmethod
    def _normalize(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_estimate(self) -> np.ndarray:
        return self.mu.copy()

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()

    def __repr__(self):
        return (f"EKF(x={self.mu[0]:.2f}, y={self.mu[1]:.2f}, "
                f"θ={np.degrees(self.mu[2]):.1f}°, "
                f"P_trace={np.trace(self.P):.4f})")


# Hızlı test

if __name__ == "__main__":
    from robot import DifferentialRobot
    import matplotlib.pyplot as plt

    robot = DifferentialRobot(start_pos=(25, 25), start_theta=0.0, seed=7)
    ekf   = ExtendedKalmanFilter(init_pose=np.array([25.0, 25.0, 0.0]))

    true_xs, est_xs = [], []
    true_ys, est_ys = [], []

    v, omega = 0.6, 0.05

    for step in range(100):
        # Robot hareket eder
        true_pose = robot.step(v, omega)

        # Sensör ölçümleri
        enc = robot.read_encoder()
        imu = robot.read_imu()

        # EKF adımları
        ekf.predict(enc["v"], enc["omega"])
        ekf.update_imu(imu["theta_est"])
        # LiDAR güncellemesi: gerçek konuma gürültü ekleyerek simüle et
        lx = true_pose[0] + np.random.normal(0, 0.20)
        ly = true_pose[1] + np.random.normal(0, 0.20)
        ekf.update_lidar(lx, ly)

        ekf.record_true(true_pose)

        true_xs.append(true_pose[0])
        true_ys.append(true_pose[1])
        est = ekf.get_estimate()
        est_xs.append(est[0])
        est_ys.append(est[1])

    errors = ekf.compute_errors()

    # Görselleştirme
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Sol: yörüngeler
    ax = axes[0]
    ax.plot(true_xs, true_ys, "g-",  lw=2,   label="Gerçek yol")
    ax.plot(est_xs,  est_ys,  "b--", lw=1.5, label="EKF tahmini")
    ax.plot(true_xs[0], true_ys[0], "go", ms=10, label="Başlangıç")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Gerçek vs EKF Tahmini Yörünge")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Sağ: zaman içinde konum hatası
    ax2 = axes[1]
    ax2.plot(errors["pos_err"], color="#E74C3C", lw=1.5)
    ax2.axhline(errors["rmse"], color="#3498DB", ls="--", label=f"RMSE={errors['rmse']:.3f} m")
    ax2.axhline(errors["mae"],  color="#2ECC71", ls=":",  label=f"MAE={errors['mae']:.3f} m")
    ax2.set_xlabel("Adım"); ax2.set_ylabel("Konum hatası (m)")
    ax2.set_title("EKF Konum Hatası (Zaman İçinde)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/earthquake_robot/kalman_test.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Grafik kaydedildi → kalman_test.png")
    print(ekf)
