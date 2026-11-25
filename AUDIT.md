# Başlangıç denetimi — 25 Kasım 2026, 20:30–21:30 UTC

Bu raporda yazan her sayı, bu denetim sırasında çalıştırılmış bir koddan geliyor. Kod ve çıktıları:
- `audit/verify.py` → `audit/verify_results.json`
- `runs/real/*.json`
- `runs/gpu/bench.json`

Durum etiketleri:
- **VERIFIED:** gerçek bir testle doğrulandı.
- **IMPLEMENTED, NOT VERIFIED:** kodu var, ama gerçek veride doğrulanmadı.
- **FAILED:** test veya deney başarısız oldu.
- **BLOCKED:** ilerlemek için gereken kaynak veya bilgi yok.
- **PROPOSED:** öneri; henüz uygulanmadı.

---

## A. SADE RAPOR

### 1. Şu anda gerçekte ne inşa ettik?
- **Tek bir model kodu: OneMachine** (`pfm/one.py`). Makaledeki bütün parçalar içinde: asal frekans girdisi, sürekli zaman, olasılık dağılımı olarak çıktı, tersinir katmanlar, faz hafızası, Kuramoto senkronisi, faz adaptörleri, zeta tamsayı kanalı. Bunlara benim eklediklerim:
  - yönlü ve iki bantlı hafıza
  - faz aritmetiği (PhaseLinear)
  - düz okuma
  - akış başına hafıza kapısı
- **Gerçek veri seti:** hava, şebeke, piyasa, konuşma, kitap, ülke bilgileri, ARC; toplam 46 MB. Parmak izleriyle dondurulmuş.
- **GPU hattı:** GitHub → Modal. Çalıştığı ölçüldü.
- **Hazır ama henüz kullanılmamış kod:** büyük veri hattı (Chronos/Monash zaman serileri, FineWeb metni) ve standart ETT değerlendirme protokolü.
- **Eski deneyler** (sentetik dünya, sentetik mantık, simüle ajan, NAB) hâlâ repoda. Bunlar artık kanıt sayılmaz.

### 2. Gerçekten çalıştığı doğrulanan özellikler (VERIFIED)
- **Matematiksel temel:** asal tabanın dört özelliği; dağılımın integralinin tam 1 olması; sürekli zaman işleminin üç ayrı hesap yolunun birbirini tutması (uzun boşluklu düzensiz veri dahil). Hepsi makine hassasiyetinde.
- **Öğrenme hesabı doğru:** geri yayılım, sonlu farklarla karşılaştırıldı; 6 parametre grubunda en kötü sapma %2.
- **Mekanik özellikler:** sabit bellek, akış = toplu hesap, tersinirlik. Otomatik testler: 10/10.
- **Hafızadan silme kusursuz:** bir bilgi silindiğinde yalnızca o gidiyor; diğer 19 cevabın hiçbiri değişmiyor.
- **Değerlendirme hijyeni:** veri parmak izleri tutuyor; ETT sınırları standartla aynı; metin testinde eğitimden sızıntı %0.03.
- **GPU hattı doğru sonuç veriyor:** aynı deney CPU'da ve B200'de aynı sonucu verdi.
- **Gerçek veride öğreniyor, ama zayıf.** 3 tekrarlı ortalama, düşük daha iyi:

| alan | makine | öğrenmesiz referans | durum |
|---|---|---|---|
| piyasa | −0.66 ± 0.06 | −0.23 | daha iyi |
| ses | −1.21 ± 0.00 | −0.46 | daha iyi |
| hava | −1.39 ± 0.01 | −1.57 | **daha kötü** |
| şebeke | −0.15 ± 0.01 | −0.49 | **daha kötü** |
| metin (bit/harf) | 4.90 ± 0.05 | 4.49 | **daha kötü** |

### 3. Kodda var ama gerçek veride doğrulanmamış olanlar (IMPLEMENTED, NOT VERIFIED)
- Kuramoto, faz aritmetiği ve faz adaptörlerinin gerçek veride katkısı (ablasyon kuyruğu container yeniden başlayınca öldü; koşmadı).
- Transformer ile aynı bütçede karşılaştırma (kod hazır; GPU işi iptal edildi).
- Unutmadan sürekli öğrenme (kod hazır; hiç koşmadı).
- Standart ETT tahmin sonuçları (protokol hazır ve doğrulandı; eğitilmiş bir modelle koşmadı).
- Büyük veriyle eğitim (yalnızca yerel küçük denemede çalıştı).
- Ses tanıma ("hangi rakam söylendi"), ARC mantık testi, gerçek veride ajan: hiçbiri yapılmadı.

### 4. Kanıtlanmamış iddialar (dürüst düzeltme)
- **Daha önce rapor ettiğim yetenek sonuçlarının çoğu sentetik veriye dayanıyor:**
  - tek gösterişte %100 hatırlama
  - %97 kural çıkarma
  - sanal sensör
  - iki tepeli tahmin
  - "Kuramoto şart"

  Gerçek dünyada kanıt değiller. **Makale PDF'i de bunları sonuç olarak sunuyor ve düzeltilmesi gerekiyor.**
- **Hafıza:** şimdiki kodda 20 bilgide %90 (FAILED, eşik %95). "%100" sonucu hafızanın eski sürümüne aitti.
- **"Makine transformer'dan harf başına daha hızlı öğreniyor":** tek tekrar, iki ölçüm noktası. Kanıt değil.

### 5. En büyük riskler
1. **Verimlilik (en kritik):** 17 milyon parametrede makine aynı boyuttaki transformer'dan **17 kat yavaş** ve **30 kat daha fazla bellek** kullanıyor (B200'de 120 GB'a karşı 4 GB). Bu yüzden L4, A10, L40S ve H100'e sığmadı. 30 dolarla büyük model eğitmenin önündeki ana engel bu.
2. **Kalite:** gerçek veride 5 alanın 3'ünde, hiçbir şey öğrenmeyen tahmincinin gerisinde. Daha uzun eğitimin bunu düzeltip düzeltmeyeceği bilinmiyor; bunu ölçecek GPU işi iptal edildi.
3. **Hafızanın yapısal sınırı:** hafızanın anahtarı yalnızca bir önceki olay. "Fransa:" yazınca "Paris"i değil, bütün başkentlerin baş harflerinin karışımını çağırıyor (ölçüldü: P, R, M). Gerçek bilgileri saklamak için anahtarın bağlam olması gerekiyor.
4. **Çıktı keskinliği:** dağılım belli bir genişlikten daha dar çizilemiyor; bu daha önce ölçülmüştü.
5. **Altyapı:** bu bulut ortamı yeniden başlayınca arka plandaki işler ölüyor; 18 CPU deneyinin 13'ü bu yüzden kayboldu. GPU yalnızca GitHub üzerinden kullanılabiliyor.
6. **Değerlendirme açıkları:** konuşma testinde aynı kişiler hem eğitimde hem testte var (kişiye genelleme ölçülmüyor). Eski NAB çalışmasında ayarlar test verisi üzerinde yapıldı.

### 6. Mimari hedefe ulaşmak için yeterli mi?
**Henüz değil.** Matematik doğru ve parçalar çalışıyor. Ama bir yapay zekâ sistemi olarak gerçek veride basit referansları bile geçtiği gösterilmedi. Şu haliyle ölçeklemek de pahalı. Yeterli olup olmadığına, aşağıdaki "karar deneyi" sonuç verdiğinde karar verebiliriz.

### 7. Eksik önemli bileşenler
- Verimli paralel hesap: sürekli zaman ve hafıza için.
- Bağlam anahtarlı hafıza.
- Gerçek veride aynı bütçeli transformer ve Mamba karşılaştırması.
- Kalibrasyon ölçümleri (CRPS/PIT) gerçek veride.
- Kişiye genelleme ve ses tanıma testi.
- Sürekli öğrenme sonuçları.
- Gerçek veride ajan.
- ARC hattı.

### 8. Potansiyeli engelleyen darboğazlar
- Hesap verimliliği (madde 5.1).
- Eğitim süresi: şimdiye kadar alan başına birkaç yüz adım.
- Bütçe: 30 dolar; tahminen ~3 dolar harcandı.
- CPU deneylerinin kırılganlığı.

### 9. Profesyonel standart için gerekenler
- Her deney öncesi yazılı hipotez ve başarı ölçütü (GPU_PLAN.md'de başladı).
- En az 3 tekrar.
- Aynı bütçeli rakip modeller.
- Bağımsız değerlendirme kodu (başlandı: `audit/verify.py`).
- Her kod değişikliğinde otomatik test (GitHub Actions'ta CI).
- Sentetik sonuçların makaleden ayrılması.
- Deneylerin, ortam yeniden başlasa bile kaybolmayacak şekilde (GitHub/Modal üzerinde) koşması.

---

## B. YOL HARİTASI (öncelik sırası)

| # | İş | Neden | Maliyet | Başarı ölçütü (önceden) | Durum |
|---|---|---|---|---|---|
| P0.1 | Verimlilik mühendisliği: sürekli zaman için ilişkisel paralel tarama / bölümlü FFT; hafıza okumasını parçalı ve float32 yapmak; akış döngülerini kaldırmak | Ölçeklemenin önündeki ana engel | 0 $ (CPU + kısa ölçüm) | 17M boyutta adım süresi ≤ 3× ve bellek ≤ 3× transformer; bütün kesinlik testleri geçmeye devam | PROPOSED |
| P0.2 | CI: her değişiklikte testler ve `audit/verify.py` GitHub'da otomatik koşsun | Regresyonları yakalamak | 0 $ | her commit yeşil | PROPOSED |
| P0.3 | Makaleyi düzeltmek: sentetik sonuçları ayrı ve açıkça etiketlemek | Dürüstlük | 0 $ | — | PROPOSED |
| P0.4 | Hafıza testlerini mevcut kodla yeniden yapıp raporları güncellemek | Eski iddiaları düzeltmek | 0 $ | — | PROPOSED |
| P1 | **Karar deneyi:** aynı gerçek veri, aynı bütçe, 3 tekrar. OneMachine ve EventGPT, 20 kat uzun eğitimle | Uzun eğitim makineyi referansların üstüne çıkarıyor mu? Transformer'a göre nerede? | ~3–5 $ (ölçülen en ucuz yeterli GPU) | **Geçer:** makine 5 alanın en az 4'ünde öğrenmesiz referansı geçer ve transformer'dan en fazla %10 kötüdür. **Geçmezse:** ana eğitime para harcanmaz; neden araştırılır | PROPOSED, **onayını bekliyor** |
| P2 | Ana eğitim: büyük gerçek veri + standart ETT protokolü + metin + yetenek ölçümleri | Literatürle kıyaslanabilir sonuç | ~15–20 $ | önceden yazılacak (P1 sonucuna göre) | BLOCKED (P1'e bağlı ve onayına) |
| P3 | Bağlam anahtarlı hafıza; gerçek bilgilerle tek seferde öğrenme ve silme testi | Hafızanın gerçek dünyada işe yaraması | 0–2 $ | 243 ülke bilgisinde ≥%95 hatırlama, silmede bozulma yok | PROPOSED |
| P4 | Sürekli öğrenme, kişiye genelleme (ses), gerçek piyasa verisinde ajan, ARC | Makalenin vaatleri | duruma göre | ayrı ayrı yazılacak | PROPOSED |

**Senin onayını gerektirenler:** P1 ve P2 (GPU harcaması). P0 işleri ücretsiz ve geri alınabilir; onları beklemeden başlıyorum.

---

## C. TEKNİK EKLER
- `audit/verify_results.json`: 18 kontrol, 17 PASS, 1 FAIL (hafıza hatırlama %90 < %95).
- `runs/gpu/bench.json`: B200 ölçümü:

| iş yükü | adım süresi | 1000 adım maliyeti | bellek |
|---|---|---|---|
| machine_small (81K) | 34.5 ms | 0.06 $ | 2.4 GB |
| gpt_small | 4.1 ms | 0.007 $ | — |
| machine_large (17.5M) | 1.18 s | 2.05 $ | 120.6 GB |
| gpt_large (25.8M) | 0.070 s | 0.12 $ | 4.1 GB |

  Diğer dört GPU'da büyük makine çalışırken uzak hata oluştu. Hata, istemcide torch olmadığı için çözümlenemedi; bellek taşması muhtemel ama doğrulanmadı.
- Gerçek veri, CPU: birlikte eğitim 3 tekrar, aşamalı eğitim 2 tekrar. Aşamalı eğitim metinde belirgin kötü (5.44'e karşı 4.90); diğer alanlarda fark tekrarlar arası oynamanın içinde.
- GPU tekrarı: `runs/gpu/001_long_training/machine_1k_s0.json` ve `runs/real/joint_s0.json` aynı sonucu veriyor (hava −1.391 / −1.387, piyasa −0.629 / −0.606).
- Denetim sırasında yapılan tek kod değişikliği: sürekli zaman taramasının uzun boşluklarda taşması düzeltildi (complex128 parça hesabı + DT_MAX). Hatayı küçük yerel deneme yakaladı (NaN). Düzeltmeden sonra 1526 adım sorunsuz; testler 10/10.
