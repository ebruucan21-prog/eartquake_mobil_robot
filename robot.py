import numpy as np

# Robot Sabitleri

WHEEL_RADIUS   = 0.05   # metre (5 cm)
WHEEL_BASE     = 0.30   # metre (tekerlekler arası mesafe, 30 cm)
MAX_SPEED      = 1.0    # m/s (maksimum doğrusal hız)
MAX_OMEGA      = np.pi  # rad/s (maksimum açısal hız)
DT             = 0.1    # saniye (simülasyon adım süresi)

# Enkoder gürültüsü (standart sapma)
ENCODER_NOISE_V     = 0.02   # m/s
ENCODER_NOISE_OMEGA = 0.01   # rad/s

# IMU gürültüsü
IMU_NOISE_OMEGA = 0.008  # rad/s


class DifferentialRobot:
    """
    Non-holonomic diferansiyel sürüş robotu.

    Durum vektörü: [x, y, theta]
      x, y   : pozisyon (metre)
      theta  : yön açısı (radyan, 0 = +x yönü)

    Kontrol girdisi: [v, omega]
      v      : doğrusal hız (m/s)
      omega  : açısal hız (rad/s)
    """

    def __init__(self, start_pos: tuple[float, float], start_theta: float = 0.0,
                 seed: int = 0):
        self.x     = float(start_pos[0])
        self.y     = float(start_pos[1])
        self.theta = float(start_theta)

        self.rng   = np.random.default_rng(seed)

        # Geçmiş (görselleştirme için)
        self.path: list[tuple[float, float]] = [(self.x, self.y)]

        # Enkoder birikimli ölçümler
        self._enc_dist  = 0.0   # toplam kat edilen mesafe (enkoder)
        self._enc_angle = 0.0   # toplam dönüş açısı (enkoder)

   
    # Gerçek kinematik model (ground truth)
   
    def step(self, v: float, omega: float) -> np.ndarray:
        """
        Bir DT adımı ilerler.
        Unicycle (tek tekerlek) kinematik modeli:
            x_new     = x     + v * cos(theta) * DT
            y_new     = y     + v * sin(theta) * DT
            theta_new = theta + omega * DT

        Döndürür: gerçek durum [x, y, theta]
        """
        v     = np.clip(v,     -MAX_SPEED, MAX_SPEED)
        omega = np.clip(omega, -MAX_OMEGA, MAX_OMEGA)

        self.x     += v * np.cos(self.theta) * DT
        self.y     += v * np.sin(self.theta) * DT
        self.theta += omega * DT
        self.theta  = self._normalize_angle(self.theta)

        self.path.append((self.x, self.y))

        # Enkoder birikimi (gürültüsüz gerçek değer üzerinden)
        self._enc_dist  += v * DT
        self._enc_angle += omega * DT

        return np.array([self.x, self.y, self.theta])

  
    # Sensör simülasyonları
   
    def read_encoder(self) -> dict:
        """
        Tekerlek enkoderi ölçümü.
        Gerçek hıza Gaussian gürültü eklenerek simüle edilir.
        Döndürür: {'v': ..., 'omega': ..., 'dist': ..., 'angle': ...}
        """
        v_meas     = (self._enc_dist  / (len(self.path) * DT + 1e-9)
                      + self.rng.normal(0, ENCODER_NOISE_V))
        omega_meas = (self._enc_angle / (len(self.path) * DT + 1e-9)
                      + self.rng.normal(0, ENCODER_NOISE_OMEGA))

        # Anlık gürültülü hız (Kalman için kullanılan asıl ölçüm)
        v_noisy     = self.get_true_state_velocity()[0] \
                      + self.rng.normal(0, ENCODER_NOISE_V)
        omega_noisy = self.get_true_state_velocity()[1] \
                      + self.rng.normal(0, ENCODER_NOISE_OMEGA)

        return {
            "v"    : float(v_noisy),
            "omega": float(omega_noisy),
            "dist" : float(self._enc_dist  + self.rng.normal(0, ENCODER_NOISE_V * DT)),
            "angle": float(self._enc_angle + self.rng.normal(0, ENCODER_NOISE_OMEGA * DT)),
        }

    def read_imu(self) -> dict:
        """
        IMU ölçümü — açısal hız (yaw rate).
        Döndürür: {'omega': ..., 'theta_est': ...}
        """
        omega_true = self.get_true_state_velocity()[1]
        omega_meas = omega_true + self.rng.normal(0, IMU_NOISE_OMEGA)
        theta_meas = self.theta + self.rng.normal(0, IMU_NOISE_OMEGA * DT * 5)

        return {
            "omega"    : float(omega_meas),
            "theta_est": float(self._normalize_angle(theta_meas)),
        }

   
    # Durum erişimi
  
    def get_pose(self) -> np.ndarray:
        """Gerçek konum: [x, y, theta]"""
        return np.array([self.x, self.y, self.theta])

    def get_grid_pos(self) -> tuple[int, int]:
        """Izgara hücresi olarak konum (satır, sütun)."""
        return (int(round(self.x)), int(round(self.y)))

    def get_true_state_velocity(self) -> tuple[float, float]:
        """
        Son adımdaki gerçek hızı döndürür.
        (path uzunluğundan türetilir — basit yaklaşım)
        """
        if len(self.path) < 2:
            return (0.0, 0.0)
        dx = self.path[-1][0] - self.path[-2][0]
        dy = self.path[-1][1] - self.path[-2][1]
        v  = np.hypot(dx, dy) / DT
        return (v, 0.0)  # omega enkoder/IMU'dan okunur

   
    # Tekerlek hızlarına dönüşüm
   
    def unicycle_to_wheels(self, v: float, omega: float) \
            -> tuple[float, float]:
        """
        (v, omega) → (v_left, v_right) tekerlek hızları.
        v_r = v + (omega * L) / 2
        v_l = v - (omega * L) / 2
        """
        v_r = v + (omega * WHEEL_BASE) / 2.0
        v_l = v - (omega * WHEEL_BASE) / 2.0
        return (v_l, v_r)

    def wheels_to_unicycle(self, v_l: float, v_r: float) \
            -> tuple[float, float]:
        """(v_left, v_right) → (v, omega)"""
        v     = (v_r + v_l) / 2.0
        omega = (v_r - v_l) / WHEEL_BASE
        return (v, omega)

 
    # Yardımcılar
   
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Açıyı [-π, π] aralığına çek."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def reset(self, pos: tuple[float, float], theta: float = 0.0):
        """Robotu başlangıç durumuna sıfırla."""
        self.x     = float(pos[0])
        self.y     = float(pos[1])
        self.theta = float(theta)
        self.path  = [(self.x, self.y)]
        self._enc_dist  = 0.0
        self._enc_angle = 0.0

    def __repr__(self):
        return (f"DifferentialRobot(x={self.x:.2f}, y={self.y:.2f}, "
                f"θ={np.degrees(self.theta):.1f}°)")



# Hızlı test

if __name__ == "__main__":
    robot = DifferentialRobot(start_pos=(25, 25), start_theta=0.0, seed=7)

    print("Başlangıç:", robot)

    # 10 adım düz git, sonra sola dön
    for i in range(10):
        state = robot.step(v=0.5, omega=0.0)

    for i in range(5):
        state = robot.step(v=0.3, omega=np.pi / 6)

    print("Son durum :", robot)
    print("Izgara pos:", robot.get_grid_pos())

    enc = robot.read_encoder()
    imu = robot.read_imu()
    print(f"\nEnkoder  → v={enc['v']:.3f} m/s  omega={enc['omega']:.3f} rad/s")
    print(f"           toplam mesafe={enc['dist']:.3f} m  açı={enc['angle']:.3f} rad")
    print(f"IMU      → omega={imu['omega']:.3f} rad/s  theta_est={np.degrees(imu['theta_est']):.1f}°")

    v_l, v_r = robot.unicycle_to_wheels(0.5, 0.2)
    print(f"\nTekerlek hızları: sol={v_l:.3f} m/s  sağ={v_r:.3f} m/s")

    print(f"\nToplam adım sayısı: {len(robot.path) - 1}")
