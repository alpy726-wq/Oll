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
