# Sequence Modelling Without a Vocabulary: uçtan uca uygulama ve doğrulama

Tek makine: 4 CPU çekirdeği, 15 GB RAM, GPU yok. Makaledeki bütün parçalar (Bölüm 2–5) kodlandı, eğitildi ve ölçüldü. Ölçek makalenin ~1/400'ü (2.1M parametre, makalede 797M). Sonuçlar bu ölçek için geçerli. Bütün sayılar `runs/*.json` dosyalarında.

## Özet: iddia → sonuç

| # | İddia (makale) | Sonuç | Kanıt |
|---|---|---|---|
| P1 | b(a)⊙b(c) = A⊙b(a+c) | **Doğrulandı**: maks. sapma 1.4e-10 | E1 |
| P2 | ‖b(x)‖² sabit = Σ1/p | **Doğrulandı**: 2.5436161509, yayılım 8.9e-16 | E1 |
| P3 | Gram matrisi Toeplitz | **Doğrulandı**: 0.41791083 ± 2.7e-12 (kapalı form 0.41791083) | E1 |
| P4 | Sürekli; 1−cos ∝ δ² | **Doğrulandı**: δ=1e-4 → 1.589e-4, δ=1e-7 → 1.589e-10 | E1 |
| — | Tipli sözlükler ayrık (makalede 1.8e-7) | **Doğrulanamadı**: aksiyon–byte maks. \|cos\| 0.52, byte alfabesi içi ort. \|cos\| 0.24. Nedeni: p^(−1/2) genliğiyle etkin boyut (ΣA²)²/ΣA⁴ = **14.3** (R=2048'e rağmen) | E1 |
| 5.4 | log p varyantı, zeta sıfırları | **Doğrulandı (taban özelliği olarak)**: log p bankasının ilk 10 tepe noktası, ilk 10 zeta sıfırına ≤0.025 hatayla denk geliyor. Çarpımsal kanal: mn → faz toplamı, sapma 6e-16 | E1 |
| §3 | Model: attention, FFN, gömme tablosu ve pozisyon kodlaması yok | **Uygulandı**: bu bileşenlerde 0 parametre | E2 |
| §4 | Tersinirlik | **Doğrulandı**: fp64'te 0 hata, fp32'de 8.9e-3 (makalede 5.2e-3) | E2 |
| §4 | Unitary faz filtresi | **Doğrulandı**: norm oranı 1.0000007–1.0000010 | E2 |
| §4 | Sabit bellekli üretim = tam geçiş | **Doğrulandı**: göreli fark 1.2e-6, argmax %100 aynı. **Ama** "durum" aslında son K girdinin kayan penceresi; alıcı alan 4081 byte ile sınırlı (aşağıda) | E2 |
| §4 | Dil modeli öğreniyor | **Doğrulandı**: 3000 adımda held-out 3 alan ortalaması 1.78 bpb (JSON 0.97, kod 1.97, düzyazı 2.40) | main |
| §4 | Ağırlık ortalamasıyla birleştirme | **Doğrulandı**: ortalama 1.792, daha iyi tekil dal 1.827. Not: taban model zaten 1.788; ortalama tabana geri dönüyor | E7 |
| §4 | Faz adaptörü çok az parametreyle çok kazanç sağlıyor | **Kısmen**: θ,σ,α (10,240 param., %0.49) → vimdoc −2.28 bpb; yoğun V (1,048,576, %49.7) → −2.73. 100× daha az parametreyle kazancın %84'ü. Makalenin "faz > yoğun" sonucu tekrarlanmadı | E7 |
| 5.6 | Adaptörler toplanabilir | **Çürütüldü (bu ölçekte)**: Δθ_vim + Δθ_ch birlikte: vimdoc 1.54 → 2.96, cheader 2.90 → 3.96. Çıkarma tam (1.2e-7) ama toplam iyi davranmıyor | E7 |
| §4 | "Holografik" dayanıklılık | **Çürütüldü**: ağırlıkların yarısını sıfırlamak +10.4 bpb; 1/8 katmanı silmek +1.17 bpb | E2 |
| §4 | Bilgi θ'da yoğunlaşıyor | **Kısmen**: %25 gürültü: θ +1.32, V +0.57, κ +0.11 bpb (makalede 58× fark; burada 2.3×) | E2 |
| §4 | Kalibrasyon kontrolü (uniform byte) | **Sorun doğrulandı**: uniform rastgele byte'ta 13.8 bpb (öğrenmemiş model 8.0 verir). Model gürültüde aşırı emin; makaledeki 8.7 de aynı belirtiydi | main |
| 6.2 | **Taban ablation'ı** (eşit mimari, veri ve bütçe; 3 seed) | **Negatif**: öğrenilmiş tablo **2.156 ± 0.022** < rastgele frekans 2.340 ± 0.030 ≈ log p 2.343 ± 0.018 < geometrik 2.370 ± 0.016 < düz genlikli √p 2.412 ± 0.041 < **√p 2.449 ± 0.025**. √p donuk aileler arasında en kötüsü (log p'ye göre ~4.7σ) | E4 |
| 5.1 | Sürekli çıktı q = \|s\|²/Z | **Çalışıyor, ön koşulla**: Z'nin kapalı formu sayısal integrale 1.0000000 tutuyor. Ama √p birim ölçekte lokalize yoğunluğu **temsil edemiyor** (kütlenin %74'ü dışarı kaçıyor). Bölüm 7'deki 1/8 ölçekle düzeliyor | E5a |
| 5.1 | Faz-yoğunluk head'i, Gauss ve MDN'e karşı | **Olumlu**: 5 görevin 4'ünde hem Gauss'tan hem MDN'den iyi. Bimodal görevde kütle ±adım tepelerinde (0.35 / 0.33), ortada 0.05 | E5 |
| 5.2 | Sürekli zaman (Δt ile durum ilerletme) | **Olumlu (tek seed)**: düzensiz örneklemde NLL: CT 0.648 < Δt girdi olarak 0.705 < Δt yok 0.799 | E6 |
| 5.3 | Grup yasasıyla bağlama | **Ön koşulla**: σ=½ ile kapasite ~4 öğede çöküyor (R=2048'de bile). σ=0 ve ayrık anahtarlarla R=2048 128 bağlamayı %99.7 doğrulukla tutuyor. Yoğun tamsayı anahtarlar k_i+v = k_j+v_j çakışmasıyla bozuluyor | E9 |
| 6.5 | Uzun menzilli recall (MQAR) | **Negatif**: prime model 5000 adımda şans düzeyinde (%1.5–3.5). Benzer boyuttaki transformer n=4'te %41, n=8'de %26 | E8 |

**Kısa sonuç.** Matematiksel yapı (Önerme 1–4, tersinirlik, unitary operatörler, akışlı üretim, kapalı form Z, sürekli zaman) tam olarak çalışıyor ve kanıtlandı. Sistem eğitiliyor ve öğreniyor. Ama makalenin ampirik iddialarının çoğu bu ölçekte tutmadı:
- Eşit bütçede √p tabanı öğrenilmiş tablodan ~0.29 bpb kötü ve diğer donuk frekans ailelerinden de kötü.
- İçerik-adresli recall yapılamıyor.
- Adaptör toplamı bozuluyor.
- "Holografik" dayanıklılık yok.

En güçlü pozitif sonuçlar sürekli taraftan geldi: faz-yoğunluk head'i ve sürekli zaman. Bu, ilk yanıttaki "Seviye B en gerçekçi değer" tahminiyle uyumlu.

## Neler kuruldu

```
pfm/basis.py        donuk taban b(x)_k = p^-σ e^{2πi x f_k}; aileler: sqrt_p, log_p, random, geometric; fp64 faz
pfm/model.py        tersinir osilatör modeli: FFT nedensel konvolüsyon (zarf (j+1)^-1/2), dairesel faz filtresi e^{iθ},
                    W = U diag(e^σ) Vᵀ (U donuk ortogonal, V=U ile başlıyor), α=0 → her blok doğduğunda identity;
                    faz eşleme readout'u; akışlı adım (step) ve ters çevirme
pfm/continuous.py   5.1 faz-yoğunluk head'i (kapalı form Z), Gauss/MDN baseline head'leri,
                    5.2 CTConv: karmaşık üstel çekirdek, keyfi Δt, parçalı (chunked) tam scan
train_lm.py         byte-seviyesi eğitim (tam / yalnız faz / yalnız V)
exp_basis.py        E1  önermeler, desimal paylar, tipli sözlük, zeta varyantı, sıfır geri kazanımı
exp_props.py        E2  eğitilmiş modelin özellikleri
exp_head_capacity.py E5a sürekli head'in temsil kapasitesi
exp_continuous.py   E5 head karşılaştırması, E6 sürekli zaman
exp_adapt.py        E7 faz adaptörü, kompozisyon, birleştirme
exp_recall.py       E8 MQAR: prime model ve transformer
exp_binding.py      E9 süperpozisyonlu bağlama kapasitesi
```

**Makalenin belirlemediği uygulama tercihleri:**
1. F/G içinde tek doğrusal olmayan bileşen çarpımsal bir kapı (gate): `α ⊙ W(Phase(Conv(n)) ⊙ silu(n))`. Blok, F ve G ne olursa olsun tersinir kalır.
2. Readout'ta öğrenilebilir bir sıcaklık parametresi ve 2R boyutlu bir çıktı kazancı var (513 parametre).
3. Sürekli görevlerde taban frekansları 1/8 ile ölçeklendi (Bölüm 7'deki sabit afin ölçekleme). Bu olmadan √p head'i çalışmıyor (E5a).

## Ayrıntılar

### Ana dil modeli (`runs/main*.json`)
- **Yapı:** R=256 (d=512), L=8, K=256; 2,107,905 parametre (V %49.7, κ %49.7, θ/σ/α %0.49).
- **Veri:** Shakespeare (düzyazı), Python stdlib (kod) ve yerel JSON dosyaları; her alanın %5'i held-out.
- **Eğitim:** 1500 adım (lr 6e-3), container yeniden başladığı için 1500. adım checkpoint'inden 1500 adım daha (lr 3e-3). B=16, T=512, toplam ~24.6M byte.
- **Held-out ortalama:** 2.53 (adım 500) → 2.00 (1000) → 1.85 (1500) → 1.82 → **1.78** (3000).
- **Uzunluk ekstrapolasyonu** (T=512'de eğitildi; kod alanında bpb): 256: 1.92, 512: 1.97, 1024: 1.99, 2048: 2.16, 4096: 1.96. Patlama yok, ama her uzunlukta farklı pencere örneği kullanıldığı için gürültülü.
- **Bağlam gerçekte sınırlı.** Alıcı alan L·2·(K−1)+1 = 4081 byte. "Sabit bellek" iddiası doğru (4 MiB), ama bu bellek kayan bir pencere. Makaledeki 304 MiB da tam olarak 76·512·2048·4 byte ediyor, yani orada da durum bir pencere önbelleği.

### E5: sürekli head (NLL, x′ birimlerinde nat; düşük daha iyi)
| görev | faz-yoğunluk (parametresiz head) | Gauss | MDN-5 |
|---|---|---|---|
| sinüs karışımı | **0.80** | 1.19 | 1.17 |
| sarkaç | **−0.31** | 0.26 | −0.23 |
| Lotka–Volterra | −0.53 | −1.18 | **−1.50** |
| Lorenz | **0.01** | 0.58 | 0.19 |
| bimodal (±adım) | **0.07** | 2.07 | 1.85 |

### E6: düzensiz örneklenmiş sönümlü osilatör
CT (gerçek Δt ile durum ilerletme) **0.648**, sabit adım + Δt girdi olarak 0.705, sabit adım ve Δt yok 0.799. Tek seed; farkın büyüklüğü (0.06–0.15 nat) seed varyansı ölçülmeden kesin sayılmamalı.

### E8: recall (64 sınıf, şans 0.016)
| n çift | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| prime model (165k param., 5000 adım) | 0.016 | 0.035 | 0.022 | 0.019 | 0.015 |
| transformer (266k param., 5000 adım) | **0.406** | **0.258** | **0.133** | **0.075** | 0.028 |

## Yeniden üretme
```
pip install torch numpy
python3 data/build.py                      # data/raw/shakespeare.txt gerekir (karpathy/char-rnn tinyshakespeare)
python3 exp_basis.py; python3 exp_binding.py; python3 exp_head_capacity.py
python3 train_lm.py --lr 6e-3 --steps 3000 --B 16 --T 512 --out runs/main
./run_after_main.sh; ./run_ablation_seeds.sh
python3 exp_continuous.py --part heads; python3 exp_continuous.py --part time
```
Not: vimdoc/cheader adaptasyon verisi `data/build.py`'de değil; `/usr/share/vim/vim91/doc/*.txt` ve `/usr/include/**/*.h` dosyalarından 1.2 MB alınıp 1.0M/0.2M bölündü.

## Sınırlar
- Ölçek makalenin ~1/400'ü; CPU'da toplam eğitim ~2 saat. Negatif sonuçlar (ablation, recall) ölçekle değişebilir. Ama aynı bütçede ve aynı mimaride karşılaştırıldıkları için taban karşılaştırması adil.
- Ana modelin LR planı 1500. adımda yeniden başlatıldı (container yeniden başladı).
- E5 ve E6 tek seed.
