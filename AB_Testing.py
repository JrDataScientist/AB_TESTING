########################### İŞ PROBLEMİ ##############################

# Facebook kısa süre önce mevcut "maximum bidding" adı verilen teklif verme türüne
# alternatif olarak yeni bir teklif türü olan "average bidding"’i tanıttı.

# Müşterilerimizden biri olan bombabomba.com, bu yeni özelliği test etmeye karar verdi ve
# average bidding'in maximum bidding'den daha fazla dönüşüm getirip getirmediğini anlamak
# için bir A/B testi yapmak istiyor.

# A/B testi 1 aydır devam ediyor ve bombabomba.com şimdi sizden bu A/B testinin sonuçlarını
# analiz etmenizi bekliyor. Bombabomba.com için nihai başarı ölçütü Purchase'dır. Bu nedenle,
# istatistiksel testler için Purchase metriğine odaklanılmalıdır.

########################## Veri Seti Hikayesi ########################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve
# tıkladıkları reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri
# yer almaktadır. Kontrol ve Test grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleri
# ab_testing.xlsx excel’inin ayrı sayfalarında yer almaktadır. Kontrol grubuna Maximum Bidding,
# test grubuna Average Bidding uygulanmıştır.


# 4 Değişken - 40 Gözlem

# DEĞİŞKENLER
# Impression : Reklam görüntüleme sayısı
# Click      : Görüntülenen reklama tıklama sayısı
# Purchase   : Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning    : Satın alınan ürünler sonrası elde edilen kazanç

# Veriyi Hazırlama ve Analiz Etme

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

control_g = pd.read_excel("DATASETS/ab_testing.xlsx", sheet_name="Control Group")
test_g = pd.read_excel("DATASETS/ab_testing.xlsx", sheet_name="Test Group")

control_g.head(10)
#    Impression    Click  Purchase  Earning
# 0   82529.459 6090.077   665.211 2311.277
# 1   98050.452 3382.862   315.085 1742.807
# 2   82696.024 4167.966   458.084 1797.827
# 3  109914.400 4910.882   487.091 1696.229
# 4  108457.763 5987.656   441.034 1543.720
# 5   77773.634 4462.207   519.670 2081.852
# 6   95110.586 3555.581   512.929 1815.007
# 7  106649.183 4358.027   747.020 1965.100
# 8  122709.717 5091.559   745.986 1651.663
# 9   79498.249 6653.846   470.501 2456.304

test_g.head(10)
#   Impression    Click  Purchase  Earning
# 0  120103.504 3216.548   702.160 1939.611
# 1  134775.943 3635.082   834.054 2929.406
# 2  107806.621 3057.144   422.934 2526.245
# 3  116445.276 4650.474   429.034 2281.429
# 4  145082.517 5201.388   749.860 2781.698
# 5  115923.007 4213.869   778.373 2157.409
# 6  106116.437 3279.473   491.615 2560.411
# 7  125957.116 4690.570   855.720 2563.580
# 8  117442.865 3907.939   660.478 2242.233
# 9  131271.716 4721.188   532.279 2368.109

control_g.describe()
#       Impression    Click  Purchase  Earning
# count      40.000   40.000    40.000   40.000
# mean   101711.449 5100.657   550.894 1908.568
# std     20302.158 1329.985   134.108  302.918
# min     45475.943 2189.753   267.029 1253.990
# 25%     85726.690 4124.304   470.096 1685.847
# 50%     99790.701 5001.221   531.206 1975.161
# 75%    115212.817 5923.804   637.957 2119.803
# max    147539.336 7959.125   801.795 2497.295

test_g.describe()
#       Impression    Click  Purchase  Earning
# count      40.000   40.000    40.000   40.000
# mean   120512.412 3967.550   582.106 2514.891
# std     18807.449  923.095   161.153  282.731
# min     79033.835 1836.630   311.630 1939.611
# 25%    112691.971 3376.819   444.627 2280.537
# 50%    119291.301 3931.360   551.356 2544.666
# 75%    132050.579 4660.498   699.862 2761.545
# max    158605.920 6019.695   889.910 3171.490

control_g.isnull().values.any()
# False

test_g.isnull().values.any()
# False


# Kontrol ve Test grubu verilerini birleştirelim.
control_g["Host"] = "Kontrol"
test_g["Host"] = "Test"

main_g = pd.concat([control_g, test_g]).reset_index()
main_g.head()
#   index  Impression    Click  Purchase  Earning     Host
# 0      0   82529.459 6090.077   665.211 2311.277  Kontrol
# 1      1   98050.452 3382.862   315.085 1742.807  Kontrol
# 2      2   82696.024 4167.966   458.084 1797.827  Kontrol
# 3      3  109914.400 4910.882   487.091 1696.229  Kontrol
# 4      4  108457.763 5987.656   441.034 1543.720  Kontrol

main_g.tail()
#     index  Impression    Click  Purchase  Earning  Host
# 75     35   79234.912 6002.214   382.047 2277.864  Test
# 76     36  130702.239 3626.320   449.825 2530.841  Test
# 77     37  116481.873 4702.782   472.454 2597.918  Test
# 78     38   79033.835 4495.428   425.359 2595.858  Test
# 79     39  102257.454 4800.068   521.311 2967.518  Test

# Kontrol ve Test Gruplarına göre değişkenlerin ortalamalarına bir bakalım.
main_g.groupby("Host").agg({"Click": "mean",
                          "Purchase": "mean",
                          "Earning": "mean",
                           "Impression": "mean"})

#            Click  Purchase  Earning  Impression
# Host
# Kontrol 5100.657   550.894 1908.568  101711.449
# Test    3967.550   582.106 2514.891  120512.412

# Yorum olarak ;
# Yeni yöntemde Tıklama sayısı azalırken :
# Satınalma, Kazanç ve Görüntüleme sayıları artmıştır.
# Ancak bu farkların şans eserimi, yoksa gerçekten anlamlı mı
# olup olduğunu hipotez testleri yaparak öğrenmeliyiz.

# Hipotezimi oluşturalım.

# H0 : M1 = M2 ("Maximum Bidding" kampanyası sunulan Kontrol grubu ile "Average Bidding" kampanyası sunulan
# Test grubunun satın alma sayılarının ortalaması arasında istatistiksel olarak anlamlı bir fark yoktur.)

# H1 : M1!= M2 ("Maximum Bidding" kampanyası sunulan Kontrol grubu ile "Average Bidding" kampanyası sunulan
# Test grubunun satın alma sayılarının ortalaması arasında istatistiksel olarak anlamlı bir fark vardır.)

# # Normallik Varsayımı :
# # H0: Normal dağılım varsayımı sağlanmaktadır.
# # H1: Normal dağılım varsayımı sağlanmamaktadır.
# # p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
# # Test sonucuna göre normallik varsayımı kontrol ve test grupları için sağlanıyor mu ?
# Elde edilen p-value değerlerini göre yorumlamamızı yapalım.

test_stat, pvalue = shapiro(control_g.Purchase)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.9773, p-value = 0.5891
# p-value > 0.05 H0 reddedilemez.
# Normal dağılım sağlanmaktaır.

test_stat, pvalue = shapiro(test_g.Purchase)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.9589, p-value = 0.1541
# p-value > 0.05 H0 reddedilemez.
# Normal dağılım sağlanmaktaır.

# Varyans Homojenliği :
# H0: Varyanslar homojendir.
# H1: Varyanslar homojen Değildir.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
# Kontrol ve test grubu için varyans homojenliğinin sağlanıp sağlanmadığını Purchase değişkeni
# üzerinden test edelim. Test sonucuna göre normallik varsayımı sağlanıyor mu? Elde edilen
# p-value değerlerini yorumlayalım.

test_stat, pvalue = levene(control_g.Purchase, test_g.Purchase)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 2.6393, p-value = 0.1083
# p-value > 0.05 H0 reddedilemez.
# Varyanslar homojendir.


# Normallik dağılımı ve Varsayımlar(Varyans Homojenliği) sağlandığı için bağımsız iki örneklem t testi (parametrik test) yapılır.

test_stat, pvalue = ttest_ind(control_g.Purchase, test_g.Purchase, equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = -0.9416, p-value = 0.3493
# P değeri 0,05'ten büyük olduğundan H0 reddedilmez. Dolayısıyla, “maksimum teklif” kampanyası sunulan Kontrol grubu
# ile “ortalama teklif” kampanyası sunulan Test grubu arasında istatistiksel olarak anlamlı bir fark yoktur.

# ÇIKAN SONUÇLARA GÖRE YORUM:

# Satınalma sonuçları bakımından yöntemler arasında anlamlı bir fark olmadığından müşteri her iki yöntemi de seçebilir.
# Ancak burada Tıklanma, Etkileşim, Kazanç gibi diğer istatistiklerdeki farklılıklar da önem arz edecektir. Bu sebeple
# öneri olarak; diğer farklılıklar değerlendirilip, hangi yöntemin daha kazançlı olduğu tespit edilebilir. Tabi burada
# Test'in süresini uzatmamız gerekecektir.