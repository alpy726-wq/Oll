# TÜBİTAK BİGG (1512) — başvuru metni taslağı

Not: BİGG başvuruları TÜBİTAK'ın yetkilendirdiği "uygulayıcı kuruluşlar" (teknoparklar, üniversite teknoloji transfer ofisleri vb.)
üzerinden yapılır. Güncel çağrıyı ve uygulayıcı kuruluş listesini TÜBİTAK sayfasından kontrol et.

## Proje adı
Oll: Kelime listesiz, sürekli zamanlı ve yaşarken öğrenen sinyal modeli

## Proje özeti (kısa)
Bugünkü yapay zeka modelleri veriyi önce sembollere (token) böler, sabit adımlarla ilerler ve eğitildikten sonra donar.
Oll, bunun yerine her ölçümü asal sayı frekanslarından oluşan sabit bir matematiksel tabanla temsil eder, gerçek zaman
aralıklarıyla çalışır ve sonucu güven aralığıyla birlikte olasılık dağılımı olarak verir. Model, tek geçişte bilgi yazılabilen
ve istenen bilginin diğerlerine dokunmadan tam silinebildiği bir faz hafızasına sahiptir.

## Problem
Sanayi, enerji, ulaşım ve sağlık sensörleri düzensiz aralıklarla, eksik ölçümlerle ve farklı hızlarda veri üretir. Mevcut
yöntemler bu veriyi önce sabit adımlara ve sembollere dönüştürmek zorundadır; belirsizliği güvenilir biçimde vermez ve yeni
veriyle güncellenmek için yeniden eğitim ister.

## Çözüm ve yenilik
- Ham gerçek sayıyı doğrudan girdi olarak alan, eğitilmeyen asal frekans tabanı (matematiksel özellikleri kanıtlı ve ölçülmüş).
- Herhangi bir zaman aralığıyla ilerleyen, sabit bellekle sonsuz akış işleyebilen sürekli zamanlı çekirdek.
- Kapalı formda normalize olan olasılık çıktısı: belirsizlik doğrudan modelin çıktısıdır.
- Tek geçişte yazılan ve tam silinebilen hafıza (veri gizliliği / "unutulma hakkı" açısından önemli).

## Ulaşılan sonuçlar (gerçek, daha önce görülmemiş veride ölçüldü)
- 17,5 milyon parametreli model, tek GPU'da 200 dakikada eğitildi.
- Belirsizlik tahmini (CRPS) öğrenmesiz referanstan hava verisinde 5,5 kat, elektrik verisinde 2,4 kat, ses verisinde 1,7 kat daha iyi.
- Standart ETTh2 uzun vadeli tahmin testinde tüm ufuklarda basit referans yöntemlerden daha düşük hata.
- Verinin %75'i silindiğinde bile çökmeden çalışıyor.
- 243 gerçek bilgiyi hiç eğitim yapmadan tek okumada %80 doğrulukla hatırlıyor; bir bilgi silindiğinde diğer doğru cevapların hiçbiri değişmiyor.
- Çalışan web uygulaması: [web adresi]. Makale: [arXiv numarası]. Kod ve model: [GitHub / Hugging Face adresi].

## Ticari uygulama ve hedef pazar
Endüstriyel sensör izleme ve kestirimci bakım, enerji yük ve üretim tahmini, IoT/uç cihazlarda düşük kaynakla çalışan tahmin,
kişisel veriyi silinebilir biçimde tutan kişiselleştirilmiş modeller. İlk ürün: belirsizlikli tahmin servisi (çalışır durumda).

## Rekabet avantajı
Kelime listesi gerektirmemesi, gerçek zaman aralıklarıyla çalışması, sabit bellekle akış, güvenilir belirsizlik ve tam silinebilir
hafızanın tek modelde birleşmesi. Sınırlar açıkça ölçülmüştür (makalede "negatif sonuçlar" bölümü).

## İş planı (12 ay, taslak)
1. Ay 1–3: Modelin büyütülmesi ve ölçekleme ölçümü; günlük ritim ve anomali tespiti zayıflıklarının giderilmesi.
2. Ay 4–6: Pilot müşteri verisiyle (ör. bir fabrika veya enerji şirketi) gerçek ortam denemesi.
3. Ay 7–9: Ürünleşme: API, uç cihaz sürümü, kullanıcı arayüzü.
4. Ay 10–12: İlk ücretli müşteriler, yatırım görüşmeleri.

## Bütçe kalemleri (taslak)
Bulut GPU kullanımı, veri ve bulut altyapısı, yazılım geliştirme, pilot uygulama giderleri.

## Girişimci
[Ad Soyad], 21 yaş. Modelin matematiksel temelini oluşturan makalenin yazarı. [Eğitim, iletişim bilgileri.]
