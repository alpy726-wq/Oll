# GPU planı: önceden kayıt

Bu belge GPU'ya geçmeden **önce** yazıldı. Ne kurulacağı, ne ölçüleceği ve neyin başarı, neyin başarısızlık sayılacağı burada sabit. Sonuçlar geldikten sonra hedefler değiştirilmeyecek. Her madde bir komutla yeniden üretilebilir olacak.

## 1. Makine: makaleyle birebir, tek model

`pfm/one.py` içindeki OneMachine. Makaledeki her öğe kodda bir yere ve bir teste bağlı:

| Makale | Kodda | Test |
|---|---|---|
| §2 asal taban b(x), sözlük yok | `OneMachine.phases/code` | grup yasası, sabit norm |
| §2.3 tipli bölgeler / §5.3 tek uzayda modaliteler | rol fazları `psi` | rol bağlama hatası |
| §2.6 zeta varyantı, çarpımsal kanal | `integer_phase`, `family="log_p"` | n·m → faz toplamı |
| §3 tersinir bloklar, faz filtresi, W = U diag(e^σ) Vᵀ | `OneBranch`, `PhaseFilter`, `Spectral` | ters çevirme hatası |
| §4.1 sürekli çıktı, kapalı form Z | `density`, `G{i}` | integral = 1 |
| §5.2 sürekli zaman | `CTConv` | yarı-grup, Δt duyarlılığı |
| §5.5 senkroni (Kuramoto) | `Kuramoto` | kapatma ablasyonu |
| §5.6 faz adaptörleri | `adapter / apply_adapters` | yeni beceri, birebir geri alma |
| Önerme 1 ile bağla / çöz hafızası | `memory_read` | tek gösterişte öğrenme, tam silme, düzeltme |
| Faz aritmetiği (gece eklendi) | `PhaseLinear` | kural çıkarma |

**Kural:** GPU'da eğitilen model bu sınıfın kendisi olacak. Ayrı bir "dil modeli" yazılmayacak. Dil, makinenin akışlarından yalnızca biri.

## 2. Ölçülecekler ve başarı ölçütleri (önceden sabit)

| # | Soru | Nasıl ölçülür | Başarı |
|---|---|---|---|
| 1 | Büyüdükçe akıllanıyor mu? | 1M / 10M / 100M parametre; aynı veri ve aynı hesapla transformer ve Mamba ile birlikte; kayıp–hesap eğrisi | eğim transformer'ınkine eşit ya da daha iyi |
| 2 | Bilgiyi kesin silip düzeltebiliyor mu? | standart bilgi düzenleme testleri (CounterFact / zsRE tipi) | düzenlenen bilgide ≥%95, dokunulmayan bilgilerde bozulma <%1 (bugünkü yöntemler bunu tutturamıyor) |
| 3 | Düzensiz zamanlı gerçek veride ne yapıyor? | açık tıbbi veya sensör verisi, alanın bilinen yöntemleriyle aynı protokol | en iyi bilinen yöntemle aynı seviye ya da daha iyi |
| 4 | Mantık: hiç görmediği kuralları çıkarabiliyor mu? | eğitimde olmayan kural aileleri ve bileşik kurallar | bileşik kurallarda ≥%80 (şu an %14) |
| 5 | Deneme-yanılma ile ustalaşıyor mu? | sıfırdan, yalnızca ödülle kontrol görevleri | elle yazılmış kontrolcüye yakın (şu an rastgele −0.84, makine −0.46, kontrolcü −0.09) |
| 6 | Konuşabiliyor mu? | basit dilde hikâye verisi; aynı boyutta transformer ile | tutarlı cümleler; aynı hesapta transformer'la aynı ya da daha iyi kayıp |

**Başarısızlıklar da raporlanır.** Bir madde tutmazsa "tutmadı" yazılır ve nedeni araştırılır. Hiçbir sonuç seçilerek gösterilmez; her koşunun kaydı `runs/` altında kalır.

## 3. Bu geceye kadar yapılmış ve doğrulanabilir olanlar
- `python3 -m pytest -q tests`: kesinlik testleri.
- `python3 one_eval.py runs/one/one_v2`: 15 maddelik yetenek ölçümü.
- `python3 explore_one.py`: tek gösterişte öğrenme (%100), tam silme, düzeltme, sanal sensör.
- `runs/mind_logic_phaselinear_20k.json`: kural çıkarma %97.
- Hatalarım da kayıtlı: mantık testindeki maske hatası, ilk ajan denemeleri, NAB'da basit kuralın makineyle eşit çıkması.

## 4. Eleştirel gözden geçirme: GPU'dan önce düzeltilecekler

| # | Sorun | Düzeltme | Durum |
|---|---|---|---|
| 1 | Veri oyuncak (sentetik fizik, 12 MB metin) | gerçek sürekli veri: ses, sensörler, finans, tıbbi ölçümler, zaman damgalı metin | yapılacak |
| 2 | Bütün görevler aynı anda öğretiliyor ve birbirini etkiliyor (v2'de hafıza kapısı bazı akışlarda kapandı) | aşamalı eğitim: algı ve dünya modeli → dil → mantık ve hareket. v3 ile tek beyin ve ayrı modeller karşılaştırılacak | v3 sürüyor |
| 3 | Bazı testler kandırmaya açıktı (NAB'da test üzerinde ayar; kural testinde eğitimdeki aileler) | test setleri baştan ayrılır ve dokunulmaz; genelleme yalnızca hiç görülmemiş ailelerle ölçülür | yapılacak |
| 4 | Çoğu sonuç tek deneme | önemli her sonuç en az 3 tekrar | yapılacak |
| 5 | Gece eklenen parçalar (yönlü hafıza, düz okuma, PhaseLinear, hafıza kapısı) birlikte ölçülmedi | aç/kapa ablasyonu | yapılacak |
| 6 | Bilinen zayıflıklar: çıktı keskinliği, ajanın geçmişi kullanamaması, beceri kompozisyonu, uzun hayalde hata birikmesi | her biri için ayrı deney | açık |
| 7 | Kod GPU'ya hazır değil (hafıza ve akış döngüleri Python'da) | vektörleştirme ve gerekirse özel çekirdek | yapılacak, ilk iş |
| 8 | "Veri başına transformer'dan hızlı öğreniyor" sonucu çok küçük ölçekte ve tek denemeyle alındı | para harcamadan önce ucuz büyüme testi; eğri kötüyse dur | GPU aşama 2 |
| 9 | Unutmadan sürekli öğrenme hiç test edilmedi | ana ölçüt: yeni beceriler sırayla öğrenilirken eskiler korunuyor mu (hafıza + adaptör + taban) | plana eklendi |
