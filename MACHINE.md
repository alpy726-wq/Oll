# Makine: sürekli girdi, sürekli zaman, sürekli çıktı

Tek bir model. Girdisi gerçek bir sayı (tokenizer ya da gömme tablosu yok), zamanı keyfi aralıklı, çıktısı bir sonraki değerin tam olasılık yoğunluğu. Makalenin Bölüm 2, 5.1 ve 5.2'si birleşik olarak kuruldu.

```
x (gerçek sayı) ──► b(x) = p^-1/2 · e^{2πi x √p / 8}      donuk asal-frekans tabanı, 0 parametre
dt (herhangi)  ──► s ← e^{(−λ+iω)·dt} s + u               sürekli-zaman osilatör durumu, tersinir bloklar
q (sorgu anı)  ──► okuma e^{(−λ+iω)·q} ile ilerletiliyor
çıktı          ──► q(x) = |⟨b(x), h⟩|² / Z,  Z kapalı form  tam yoğunluk; softmax ve sözlük yok
```

Kod: `pfm/continuous.py` (makine), `pfm/world.py` (veri), `machine.py` (eğitim, değerlendirme, rapor), `tests/test_machine.py`, `plot_machine.py`.

## Görev
Tek model, aynı anda altı sinyal ailesi, her dizi yeni rastgele başlangıç koşullarıyla:

| aile | dinamik | örnekleme |
|---|---|---|
| sines | 3 rastgele sinüs + gürültü | düzensiz, ort. aralık 1.0 |
| pendulum | doğrusal olmayan sarkaç | düzensiz, ort. aralık 0.25 |
| lotka | Lotka–Volterra av–avcı | düzensiz, ort. aralık 0.35 |
| lorenz | kaotik Lorenz | düzensiz, ort. aralık 0.04 |
| oscillator | rastgele sönümlü osilatör | düzensiz, ort. aralık 0.35 |
| bimodal | x → x ± 0.25 (sınırda yansıma) | düzenli |

Farklı zaman birimleri ve örnekleme hızları bilerek karıştırıldı. Model her örneğin değerini, önceki örnekten bu yana geçen süreyi ve tahmin edilecek anın uzaklığını görüyor.

## 1. Çalıştığının kanıtı: tam özellikler (`python3 -m pytest -q tests` → 6/6 geçti)
| test | ne kanıtlıyor |
|---|---|
| `test_group_law` | b(x)⊙b(y) = A⊙b(x+y): aritmetik temsilin içinde |
| `test_density_normalised_closed_form` | kapalı form Z ile yoğunluğun integrali = 1 (1e-6) ve yoğunluk her yerde ≥ 0 |
| `test_streaming_equals_full_pass` | tek tek örnekle akan model = tüm diziyi bir kerede işleyen model (göreli hata < 1e-5) |
| `test_state_size_independent_of_length` | 10 ve 1000 örnekte bellek aynı (O(1) durum) |
| `test_continuous_time_semigroup` | a + b süre ilerletmek = a sonra b ilerletmek; zaman adım indeksi değil, gerçek sayı |
| `test_irregular_sampling_changes_prediction` | aynı değerler farklı zamanlarda gelince tahmin değişiyor; model zamanı kullanıyor |

## 2. Öğrendiğinin kanıtı
- **Eğitim:** 2000 adım × 24 dizi × 64 örnek, 3 seed. Eğitim NLL'i +0.0'dan −1.36…−1.47'ye iniyor (3 seed tutarlı). Grafik: `runs/machine/machine.png`.
- **Held-out:** her model için aynı, eğitimde görülmemiş 96 dizi × 6 aile.
- **Öğrenmeyen referansa karşı** (persistence: "sonraki = şimdiki + gürültü", σ veriden kestirildi): makine 6 ailenin **6'sında da** daha iyi. Ortalama NLL −1.35 ± 0.03'e karşı −0.40; CRPS 0.065'e karşı 0.091.
- **Çok adımlı olasılıksal tahmin:** 48 örneklik geçmişten 16 adım ileri, 32 örnek yol, akış modunda. Grafikteki %50 ve %90 bantları gerçek gidişatı kapsıyor.
- **Çok modlu yoğunluk:** bimodal süreçte iki tepe x±0.25'te çıkıyor. Değer sınıra yakınken (x=0.82) tek tepe veriyor; bu doğru, çünkü sınırda yansıma iki sonucu aynı noktaya düşürüyor. Model yansıma kuralını öğrenmiş.

## 3. Baseline'larla karşılaştırma (held-out; ortalama ± std; makine ve GRU 3 seed, diğerleri 1 seed)

| metrik (düşük daha iyi) | **makine** (faz head'i) | aynı gövde + Gauss | aynı gövde + MDN-5 | GRU + MDN-5 | persistence |
|---|---|---|---|---|---|
| ortalama NLL | −1.347 ± 0.028 | −0.753 | −1.416 | **−1.878 ± 0.305** | −0.397 |
| ortalama CRPS | 0.0646 ± 0.0010 | 0.1064 | 0.0709 | **0.0591 ± 0.0059** | 0.0913 |
| 16 adım CRPS | 0.168 ± 0.006 | 0.167 | **0.149** | 0.155 ± 0.014 | – |
| bimodal NLL | **−1.837 ± 0.016** | +0.441 | −1.039 | −1.221 ± **1.197** | +0.034 |
| bimodal CRPS | **0.116 ± 0.001** | 0.226 | 0.144 | 0.136 ± 0.014 | 0.151 |
| parametre | 112,769 | 113,282 | 116,623 | 81,615 | 1/aile |
| eğitim süresi (1 CPU çekirdeği) | 36 dk | 36 dk | 36 dk | 7 dk | – |

**Okuma:**
- **Kazandığı yerler:**
  - Çok modlu (bimodal) süreç: en iyi NLL ve CRPS, büyük farkla.
  - Kararlılık: seed'ler arası std 0.03. GRU'da 0.31; GRU bimodalde bir seed'de çöküyor (±1.20).
  - Aynı gövdede parametresiz faz head'i öğrenilmiş Gauss head'ini 6 ailenin 5'inde yeniyor (ortalama NLL −1.35'e karşı −0.75; Lotka'da Gauss önde).
- **Kaybettiği yerler:** düzgün, neredeyse deterministik sistemlerde (sarkaç, Lotka, Lorenz) tek adım keskinliği. Standart GRU+MDN burada daha iyi; aynı gövdede öğrenilmiş MDN head'i de ortalamada biraz daha iyi.
- **Nedeni ölçülmüş bir sınır:** çıktı tabanının en yüksek frekansı √p_R / 8 ≈ 3.35 döngü/birim. Bu, yoğunluğun ~0.075 (x biriminde) altına keskinleşemeyeceği anlamına geliyor. Gerçek dağılımın bundan dar olduğu ailelerde NLL tavana çarpıyor. Makalenin kendi Bölüm 7'si bu çözünürlük sınırını ("~1/√p_R") öngörüyor.

## Sonuç
Makine çalışıyor ve bu kanıtlandı:
- Tam matematiksel özellikleri testlerle doğrulandı.
- Altı farklı dinamik sistemi düzensiz zamanlarla tek modelde öğreniyor.
- Kalibre edilmiş çok modlu yoğunluklar ve çok adımlı olasılıksal tahminler üretiyor.

Standart bir GRU+MDN'den ortalamada üstün değil. Üstün olduğu yer çok modlu dağılımlar ve seed kararlılığı. Bir sonraki iyileştirme yeri belli: çıktı çözünürlüğü. Daha büyük R ya da hedefe göre ölçeklenmiş bir pencere önce denenecek şeyler.

## Yeniden üretme
```
python3 -m pytest -q tests
./run_machine.sh            # 8 eğitim + değerlendirme, 4 CPU'da ~75 dk
python3 machine.py report   # runs/machine/summary.json, summary.txt
python3 plot_machine.py     # runs/machine/machine.png
```
