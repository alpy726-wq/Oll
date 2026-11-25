# Tek Makine: makaledeki bütün yetenekler tek bir modelde

`pfm/one.py` · eğitim `one_train.py` · ölçüm `one_eval.py` · sonuçlar `runs/one/one_s0_capabilities.json`

Tek model (118.915 parametre). 4 CPU çekirdeğinde 38 dakika eğitildi, GPU kullanılmadı. Aynı anda dört iş öğrendi:
- 6 sürekli fizik sistemi (düzensiz zamanlı)
- iki sensörün birleştirilmesi
- anahtar–değer hatırlama
- metin

Aşağıdaki bütün ölçümler **aynı eğitilmiş model** üzerinde, görülmemiş verilerle yapıldı.

| # | Yetenek (makale bölümü) | Ölçüm | Durum |
|---|---|---|---|
| 1 | Sürekli girdi, sözlük yok (§2) | gerçek sayılar doğrudan; 6 sistemde ortalama NLL −1.02 | ✅ |
| 2 | Sürekli çıktı: tam olasılık yoğunluğu (§4.1) | integral = 1.0000004; iki olasılıklı süreçte kütle x+0.25'te 0.40, x−0.25'te 0.34, ortada 0.02 | ✅ |
| 3 | Sürekli zaman (§3, denklem 3) | a+b süre = a sonra b (fark 3×10⁻⁷); aynı değerler 2× zamanla gelince çıktı %55 değişiyor | ✅ |
| 4 | Sembol / byte çıktısı aynı başlıktan (§4.1) | metin 4.19 bit/byte (rastgele 8.0) | ✅ zayıf ama öğreniyor |
| 5 | Tipli bölgeler, modaliteler tek uzayda (§2.3, §5.3) | rol bağlama hatası 1.8×10⁻¹⁵; füzyon katmanı yok | ✅ |
| 6 | Sensör birleştirme, füzyon katmanı olmadan | B sensörü tahmini, A sensörü de görülünce daha iyi: NLL −1.30 → −1.43 | ✅ |
| 7 | Faz hafızası: tam bağla / çöz (Önerme 1) | hatırlama doğruluğu 2 çiftte %59, 4 çiftte %30. Hafıza kapatılınca %2–3 (tesadüf %1.6) | ✅ çalışıyor, kapasite sınırlı |
| 8 | Senkroni ile bağlama (Kuramoto, §5.5) | öğrenilen bağlaşım ‖K‖ ≈ 4; kapatılınca NLL 0.29 → 0.90 kötüleşiyor, yani makine bunu kullanıyor | ✅ |
| 9 | Faz adaptörleri: yeni beceri (§5.6) | parametrelerin %6.8'i. Van der Pol: −0.07 → **−1.31**; Duffing: −0.50 → **−1.03** (her ikisi de eğitimde hiç görülmemiş sistemler) | ✅ |
| 10 | Adaptörü çıkar, taban birebir geri gelsin | fark 1×10⁻⁷ | ✅ |
| 11 | İki adaptörü toplama (§5.6) | ikisi birlikte takılınca iki beceri de bozuluyor (NLL ≈ +1.0) | ❌ makalenin açık sorusu: bu ölçekte hayır |
| 12 | Sabit hafıza, akış = toplu işlem | 64 ve 564 olaydan sonra durum boyutu aynı (13.312); fark 8×10⁻⁷ | ✅ |
| 13 | Tersinirlik: bilgi kaybı yok | ters hesap hatası 4.8×10⁻¹³ | ✅ |
| 14 | Toplama temsilin içinde (Önerme 1) | faz sapması 2.7×10⁻¹³ | ✅ |
| 15 | Zeta varyantı: çarpımsal tamsayı kanalı (§2.6) | n·m → faz toplamı, sapma 0 | ✅ |

**Notlar (dürüstçe):**
- Adaptör öğrenme oranı ilk denemede 1×10⁻² idi. Duffing adaptörü bu oranda kötüleşti (`runs/one/one_s0_capabilities_adapterlr1e-2.json`). Tablodaki sonuçlar 3×10⁻³ ile alındı.
- Hafızanın çalışması için iki düzeltme gerekti:
  - Bağlama yönlü hale getirildi (sabit bir permütasyonla). Simetrik bağlama, anahtarın hem öncesini hem sonrasını çağırıyordu.
  - Okuma tarafı düz genliğe alındı. p^(−1/2) genliği ayırt etme gücünü ~14 boyuta düşürüyordu (makalede §2.2).
- Metin ve hatırlama, 4 görev aynı küçük modeli paylaştığı için zayıf kaldı; ölçek ve eğitim süresiyle artar.
