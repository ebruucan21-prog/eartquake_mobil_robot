#Deprem Sonrası Bina İçi Otonom Navigasyon

**Sensör Füzyonu ve Lokalizasyon Kullanarak LiDAR Tabanlı Otonom Navigasyon**
(2B Simülasyon Ortamı — Mobil Robotlar Dersi Projesi)*


#Proje Özeti

Bu proje, deprem sonrası kısmen çökmüş bir bina içinde arama-kurtarma görevi üstlenen otonom bir mobil robotun Python simülasyonunu içermektedir. Robot; LiDAR, IMU ve tekerlek enkoderi verilerini Genişletilmiş Kalman Filtresi (EKF) ile birleştirerek konumunu tahmin eder, A* algoritmasıyla yol planlar ve Yapay Potansiyel Alan (APF) yöntemiyle dinamik engellerden kaçınır.

| Özellik | Değer |
|---|---|
| Ortam | 30×30 m, 2B ızgara |
| Engel sayısı | 10 sabit + artçı sarsıntılarla eklenen dinamik |
| Robot modeli | Non-holonomic diferansiyel sürüş (unicycle) |
| Lokalizasyon | EKF (LiDAR + IMU + Enkoder) |
| Yol planlama | A* + Yapay Potansiyel Alan (APF) |
| Dil | Python 3 |


#Proje Yapısı

```
earthquake_robot/
├── main.py            # Ana simülasyon — çalıştırılacak dosya
├── animate.py         # Canlı animasyon (isteğe bağlı)
├── environment.py     # 30×30 m bina ortamı, engeller, artçı sarsıntı
├── robot.py           # Non-holonomic robot kinematiği, enkoder, IMU
├── lidar.py           # LiDAR simülatörü, ışın döküm, engel kümeleme
├── kalman.py          # Genişletilmiş Kalman Filtresi (EKF)
├── navigation.py      # A* yol planlama + APF yerel kaçınma
└── outputs/           # Üretilen grafikler (otomatik oluşturulur)
    ├── environment_map.png
    ├── map_and_path.png
    ├── localization.png
    ├── lidar_comparison.png
    ├── error_analysis.png
    └── clustering.png
```



#Kurulum

#Gereksinimler

- Python 3.9 veya üzeri
- pip

#Kütüphanelerin Yüklenmesi

```bash
pip install numpy matplotlib
```

Standart kütüphaneler (`heapq`, `math`) ek kurulum gerektirmez.



#Çalıştırma

#Ana Simülasyon

```bash
python main.py
```

Simülasyon tamamlandığında `outputs/` klasörüne 6 grafik kaydedilir ve terminal çıktısında hata metrikleri (RMSE, MAE) görüntülenir.


#Canlı Animasyon

```bash
python animate.py
```

Robotun adım adım hareketini, LiDAR tarama noktalarını ve küme merkezlerini gerçek zamanlı olarak görselleştirir.


#Modüllerin Ayrı Ayrı Test Edilmesi

```bash
python robot.py        # Robot kinematik testi
python kalman.py       # EKF lokalizasyon testi + grafik
python lidar.py        # LiDAR tarama testi
python environment.py  # Ortam haritası oluşturma
```


#Senaryo

**Robot Görevi:** Deprem sonrası hasarlı bina içinde hayatta kalanları arayarak güvenli çıkış noktasına ulaşmak.

```
Başlangıç → (2, 2)   [Sol alt köşe — giriş noktası]
Hedef     → (27, 27) [Sağ üst köşe — güvenli çıkış]
```

**Ortam özellikleri:**
- Dış çevre ve iç bölme duvarları
- 10 adet sabit enkaz (bina hasarı)
- 3 artçı sarsıntı olayı (adım 500, 1000, 1500) — her biri 3 yeni dinamik engel ekler
- Sensör gürültüsü: LiDAR (σ=0.15 m), enkoder (σ=0.02 m/s), IMU (σ=0.008 rad/s)



#Teknik Detaylar

#Robot Modeli

Non-holonomic unicycle kinematiği:

```
x(t+1) = x(t) + v·cos(θ)·DT
y(t+1) = y(t) + v·sin(θ)·DT
θ(t+1) = θ(t) + ω·DT          (DT = 0.1 s)
```

#Sensör Füzyonu — EKF

Her adımda üç aşama:
1. `predict(v, ω)` — Enkoder ile durum tahmini, Jakobian linearizasyonu
2. `update_imu(θ)` — IMU ile yaw açısı düzeltmesi
3. `update_lidar(x, y)` — Her 3 adımda bir LiDAR konum düzeltmesi

#Navigasyon

| Katman | Yöntem | Açıklama |
|---|---|---|
| Global | A* | 8 yönlü ızgara arama, Öklid heuristic |
| Yerel | APF | LiDAR tabanlı itici kuvvet, engelden kaçınma |
| Yeniden planlama | Reaktif | Artçı sarsıntı sonrası + her 50 adımda |

---

#Çıktı Grafikleri

| Dosya | İçerik |
|---|---|
| `environment_map.png` | Ortam haritası — duvarlar, enkazlar, başlangıç/hedef |
| `map_and_path.png` | Planlanan yol + gerçek yol + EKF + dead reckoning |
| `localization.png` | 2B yörünge ve x(t)/y(t) zaman serileri karşılaştırması |
| `lidar_comparison.png` | Ham LiDAR vs medyan filtreli LiDAR (polar grafik) |
| `error_analysis.png` | EKF & dead reckoning hata zaman serileri, RMSE/MAE |
| `clustering.png` | Zaman içinde algılanan LiDAR engel kümesi sayısı |



#Yapay Zeka Kullanım Beyanı

Bu projede aşağıdaki yapay zeka araçları kullanılmıştır:

**Claude Sonnet 4.6** (Anthropic)

Kullanılan bölümler:
- EKF kod iskeletinin oluşturulması
- LiDAR ışın döküm algoritmasının geliştirilmesi
- A* modülünün hata ayıklama sürecine destek
- README ve rapor metninin düzenlenmesi

Öğrencinin kendi katkıları:
- Proje senaryosu ve sistem mimarisinin tasarlanması
- Tüm modüllerin test edilmesi, çalıştırılması ve parametre ayarları
- Simülasyon sonuçlarının değerlendirilmesi ve hata analizi


#Kaynaklar

1. V. Ušinskis et al., "Sensor-fusion based navigation for autonomous mobile robot," *Sensors*, 2025. doi: [10.3390/s25041248](https://doi.org/10.3390/s25041248)
2. Y. Ou et al., "Autonomous navigation by mobile robot with sensor fusion based on deep reinforcement learning," *Sensors*, 2024. doi: [10.3390/s24123895](https://doi.org/10.3390/s24123895)
3. B. Zhang ve C. Li, "The optimization and application research of the RRT-APF-based path planning algorithm," *Electronics*, 2024. doi: [10.3390/electronics13244963](https://doi.org/10.3390/electronics13244963)
4. S. Thrun, W. Burgard ve D. Fox, *Probabilistic Robotics*. MIT Press, 2005.
5. P. E. Hart et al., "A formal basis for the heuristic determination of minimum cost paths," *IEEE Trans. Systems Science and Cybernetics*, 1968. doi: [10.1109/TSSC.1968.300136](https://doi.org/10.1109/TSSC.1968.300136)

