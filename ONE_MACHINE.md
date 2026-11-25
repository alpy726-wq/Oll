# Tek Makine (OneMachine): makaledeki bütün yetenekler tek bir modelde

`pfm/one.py` · eğitim `one_train.py` · ölçüm `one_eval.py` · sonuçlar `runs/one/one_v2_capabilities.json`

Tek model. Aynı anda dört şey öğreniyor ve hepsi aynı eğitilmiş model üzerinde, görülmemiş verilerle ölçüldü:
- düzensiz zamanlı 6 fizik sistemi
- iki sensörün birleştirilmesi
- anahtar–değer hatırlama
- metin

Eğitim 4 CPU çekirdeğinde yapıldı, GPU yok.

| | v1 | **v2 (şimdiki)** |
|---|---|---|
| parametre | 118.915 | 158.480 |
| eğitim | 3000 adım, 38 dk | 6000 adım, 80 dk |
| hafıza | eleman bazlı faz bağlama, 128 boyut | geniş asal bankası (1024) üzerinde yönlü bağla/çöz |

## Yetenekler: v1 → v2

| # | Yetenek (makale bölümü) | v1 | **v2** |
|---|---|---|---|
| 1 | Sürekli girdi, sözlük yok; 6 fizik sistemi (§2, §5) | NLL −1.02 | **−1.28** |
| 2 | Tam olasılık yoğunluğu, iki olasılıklı gelecek (§4.1) | x±0.25'te 0.40 / 0.34, ortada 0.02 | **0.49 / 0.35, ortada 0.02**; integral 1.0000002 |
| 3 | Sürekli zaman (§3) | yarı-grup hatası 3×10⁻⁷ | 3×10⁻⁷; aynı değerler 2× zamanla gelince çıktı %64 değişiyor |
| 4 | Sembol / metin, aynı çıkıştan | 4.19 bit/harf | **2.89 bit/harf** |
| 5 | Tipli bölgeler, modaliteler tek uzayda (§2.3, §5.3) | rol hatası 1.8×10⁻¹⁵ | 1.8×10⁻¹⁵ |
| 6 | Sensör birleştirme, füzyon katmanı olmadan | B: −1.30 → −1.43 (A ile) | **B: −1.45 → −1.79 (A ile)** |
| 7 | Hafıza: bağla / çöz ile hatırlama (Önerme 1) | 2 / 4 / 8 / 16 çift: %59 / %30 / %13 / %6 | **%100 / %100 / %99.4 / %99.3**; hafıza kapatılınca %1–3 (tesadüf %1.6) |
| 8 | Senkroni ile bağlama (Kuramoto, §5.5) | kapatınca NLL 0.29 → 0.90 | kapatınca **0.00 → 2.66** (makine buna yoğun şekilde dayanıyor) |
| 9 | Faz adaptörüyle yeni beceri (§5.6), parametrelerin %6.8'i | Van der Pol −0.07 → −1.31; Duffing −0.50 → −1.03 | **Van der Pol −0.39 → −1.55; Duffing −0.45 → −1.34** |
| 10 | Adaptörü çıkar, taban birebir geri gelsin | 1×10⁻⁷ | 9×10⁻⁸ |
| 11 | İki adaptörü birlikte takma (§5.6) | ikisi de bozuluyor (+1.0) | tam adaptörler hâlâ bozuluyor (+1.04 / +1.08). **Frekans bandına ayrılmış adaptörlerle artık bozulmuyor** (−0.36 / −0.21, tabana yakın), ama beceriler korunmuyor |
| 12 | Sabit hafıza, akış = toplu işlem | fark 8×10⁻⁷ | fark 6×10⁻⁷; 64 ve 564 olaydan sonra durum boyutu aynı |
| 13 | Tersinirlik: bilgi kaybı yok | 4.8×10⁻¹³ | **4.2×10⁻¹⁵** |
| 14 | Toplama temsilin içinde (Önerme 1) | 2.7×10⁻¹³ | 3.6×10⁻¹³ |
| 15 | Zeta varyantı: çarpımsal tamsayı kanalı (§2.6) | sapma 0 | sapma 0 |

## Makineye eklenen geliştirmeler (makalenin ötesinde)
1. **Yönlü faz hafızası, geniş asal bankası üzerinde.** Bir olay çifti (önceki → şimdiki), 1024 asal frekanslık anahtar kodu ile değer kodunun bağlanmasıyla saklanıyor. Şimdiki olayla çözme, ondan sonra gelmiş olanı geri veriyor. Hafıza parametresiz ve zamana göre sürekli sönümleniyor. Durum boyutu sabit; akış modunda toplu hesapla aynı sonucu veriyor. Eğitimden önce bile 16 çiftte %98.6 doğru.
2. **Düz genlikli okuma.** Girdi tarafı p^(−1/2) genliğini koruyor. Çıkış tarafında düz genlik kullanılıyor, çünkü p^(−1/2) ayırt etme gücünü ~14 boyuta düşürüyor (§2.2).
3. **Akışa özel hafıza kapısı ve sembol sıcaklığı.** Toplam 14 parametre. Makine hafızayı yalnızca anahtar ve sembol akışlarında açmayı kendisi öğrendi: anahtar 3.37, sembol 1.96, diğerleri ≈ 0.
4. **Frekans bandına ayrılmış adaptörler** (§5.3'teki bant paylaştırma fikri). Toplanınca birbirini bozmuyorlar. Becerilerin birlikte korunması henüz çözülmedi.

## Açık kalanlar
- Adaptör kompozisyonu (madde 11): bozulma çözüldü, becerilerin birlikte korunması çözülmedi.
- Metin 2.89 bit/harf. Daha büyük model ve daha uzun eğitimle düşer.
