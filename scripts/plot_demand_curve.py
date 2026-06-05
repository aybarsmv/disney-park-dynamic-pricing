import matplotlib.pyplot as plt
import numpy as np

# --- 1. Sentetik Veri Üretimi ---
np.random.seed(42)

# Panel B için Ham Veriler (Observed Raw Data)
# Off-Peak
q_off = np.random.normal(40000, 8000, 200)
p_off = 180 - 0.0012 * q_off + np.random.normal(0, 10, 200)

# Peak (Talep eğrisinin sağa kaydığı durumlar)
q_peak = np.random.normal(65000, 9000, 100)
p_peak = 220 - 0.001 * q_peak + np.random.normal(0, 15, 100)

# Panel A için Kontrollerden Arındırılmış Veriler (Partial Regression / Residuals)
# Sadece fiyatın miktar üzerindeki saf, inelastik etkisini izole ediyoruz
res_q = np.random.normal(0, 1500, 100)
res_p = -0.02 * res_q + np.random.normal(0, 5, 100)


# --- 2. Grafik Kurulumu ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('★ Disney Parks — The Demand Curve ★\nStructural Causal Effect vs. Observed Market Data', 
             fontsize=18, color='navy', fontweight='bold')

# --- PANEL A: Partial Regression Plot (Sol Taraf) ---
ax1.scatter(res_q, res_p, color='khaki', edgecolor='black', marker='*', s=120, alpha=0.8)

# Regresyon doğrusu (Inelastik talep)
m, b = np.polyfit(res_q, res_p, 1)
x_line = np.linspace(min(res_q), max(res_q), 100)
ax1.plot(x_line, m*x_line + b, color='red', linewidth=3, label='Controlled Price Effect (ε = -0.17)')

ax1.set_title('Panel A: Partial Regression Plot\n(Dışsal Şoklardan Arındırılmış Saf Fiyat Etkisi)', fontsize=13, fontweight='bold', color='darkred')
ax1.set_xlabel('Residualized Expected Visitor Count', fontsize=12, fontweight='bold', color='navy')
ax1.set_ylabel('Residualized Ticket Price ($)', fontsize=12, fontweight='bold', color='navy')
ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax1.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')


# --- PANEL B: Observed Market Raw Data (Sağ Taraf) ---
ax2.scatter(q_off, p_off, color='royalblue', alpha=0.4, label='Off-Peak days (observed)')
ax2.scatter(q_peak, p_peak, color='khaki', edgecolor='black', marker='*', s=90, alpha=0.7, label='Peak / Holiday days (observed)')

# Gözlemlenen Esnek Eğri (Off-Peak)
ax2.plot([25000, 55000], [150, 114], color='mediumblue', linewidth=4, label='Off-Peak demand curve (ε = -1.47, elastic)')

# Teorik İnelastik Eğri (Peak) ve Ekstrapolasyon
ax2.plot([61000, 65000], [180, 125], color='red', linewidth=4, label='Peak demand curve (ε = -0.17, inelastic)')
ax2.plot([56000, 61000], [249, 180], color='red', linewidth=2, linestyle='--') # Kesikli çizgi
ax2.scatter([56000], [249], color='red', marker='X', s=120, zorder=5) # Tepe noktası

# Kritik Etiket (Brand Ceiling)
bbox_props = dict(boxstyle="round,pad=0.4", fc="snow", ec="lightcoral", alpha=0.9)
ax2.annotate('Brand ceiling (untested,\nout-of-sample interpretation)\n$249',
             xy=(56000, 249), xytext=(30000, 230),
             arrowprops=dict(facecolor='red', edgecolor='red', width=2, headwidth=8),
             bbox=bbox_props, color='firebrick', fontweight='bold', ha='center', fontsize=11)

ax2.set_title('Panel B: Observed Market Raw Data\n(Dışsal Faktörlerin Talebi Sağa Kaydırdığı Ham Dağılım)', fontsize=13, fontweight='bold', color='darkred')
ax2.set_xlabel('Expected Visitor Count (Quantity Demanded)', fontsize=12, fontweight='bold', color='navy')
ax2.set_ylabel('Disney Ticket Price ($)', fontsize=12, fontweight='bold', color='navy')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

# --- 3. Düzenleme ve Gösterme ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.figtext(0.99, 0.01, 'Synthetic Data Simulation • Methodologically Adjusted', horizontalalignment='right', fontsize=9, color='gray')
plt.show()