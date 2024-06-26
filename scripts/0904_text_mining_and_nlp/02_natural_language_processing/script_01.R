
library(sentimentr)
library(syuzhet)


data <- readLines("./data/0904_text_mining_and_nlp/02_natural_language_processing/HR Appraisal process.txt")
sentiment(data)
# Key: <element_id, sentence_id>
#     element_id sentence_id word_count   sentiment
#          <int>       <int>      <int>       <num>
#  1:          1           1          4 -0.12500000
#  2:          2           1         15  0.19364917
#  3:          3           1         16  0.52500000
#  4:          4           1          5  0.00000000
#  5:          4           2          6 -0.07348469
#  6:          5           1          7 -0.39686270
#  7:          5           2          3  0.41569219
#  8:          6           1         12  0.23094011
#  9:          7           1          6  0.55113519
# 10:          7           2          3  0.28867513
# 11:          8           1         10  0.18973666
# 12:          9           1          5 -0.33541020
# 13:          9           2          3  0.00000000
# 14:         10           1          5  0.44721360
# 15:         10           2          3  0.77942286
# 16:         11           1          4  0.37500000
# 17:         11           2          3  0.77942286
# 18:         12           1          6  0.30618622
# 19:         13           1          7  0.27213442
# 20:         13           2          9 -0.03333333
# 21:         14           1          3  0.00000000
# 22:         14           2          4  0.00000000
# 23:         15           1          4  0.62500000
# 24:         15           2          1  0.75000000
# 25:         16           1          4  0.37500000
# 26:         16           2          4  0.37500000
# 27:         17           1          4  0.37500000
# 28:         18           1          4  0.00000000
# 29:         19           1          5  0.26832816
# 30:         19           2          2 -0.14142136
# 31:         20           1          9 -0.06000000
# 32:         21           1          5 -0.33541020
# 33:         21           2          2 -1.27279221
# 34:         22           1          5  0.35777088
# 35:         23           1          6  0.30618622
# 36:         24           1          5 -0.26832816
# 37:         25           1          8 -0.12374369
# 38:         26           1          5  0.60373835
# 39:         27           1          3  0.43301270
# 40:         27           2          2  0.70710678
# 41:         28           1          6  0.00000000
# 42:         29           1          9 -0.03333333
# 43:         30           1          3  0.00000000
# 44:         31           1          4  0.62500000
# 45:         31           2          3  0.51961524
# 46:         32           1          4  0.37500000
# 47:         32           2          6  0.32659863
# 48:         33           1         10  0.44271887
# 49:         34           1          8  0.28284271
# 50:         35           1          8  0.00000000
# 51:         36           1          7 -0.68033605
# 52:         37           1          5  0.00000000
# 53:         37           2          4 -0.05000000
# 54:         38           1         10  0.14230249
# 55:         39           1         12  0.50518149
# 56:         40           1         13 -0.18027756
# 57:         41           1          9  0.33333333
# 58:         42           1          4  0.37500000
# 59:         42           2          5  0.33541020
# 60:         43           1          4  0.50000000
# 61:         43           2          3  0.00000000
# 62:         44           1          7  0.37796447
# 63:         44           2          2  0.95459415
# 64:         45           1          9  0.45000000
# 65:         46           1          7  0.00000000
# 66:         47           1          5  0.33541020
# 67:         47           2          3  0.62353829
# 68:         48           1          3  0.43301270
# 69:         48           2          4  0.25000000
#     element_id sentence_id word_count   sentiment


sentiment_by(data)
# Key: <element_id>
#     element_id word_count         sd ave_sentiment
#          <int>      <int>      <num>         <num>
#  1:          1          4         NA  -0.125000000
#  2:          2         15         NA   0.193649167
#  3:          3         16         NA   0.525000000
#  4:          4         11 0.05196152  -0.040099592
#  5:          5         10 0.57456307   0.009414749
#  6:          6         12         NA   0.230940108
#  7:          7          9 0.18558729   0.419905163
#  8:          8         10         NA   0.189736660
#  9:          9          8 0.23717082  -0.183028759
# 10:         10          8 0.23490743   0.613318229
# 11:         11          7 0.28597015   0.577211432
# 12:         12          6         NA   0.306186218
# 13:         13         16 0.21599832   0.119400544
# 14:         14          7 0.00000000   0.000000000
# 15:         15          5 0.08838835   0.687500000
# 16:         16          8 0.00000000   0.375000000
# 17:         17          4         NA   0.375000000
# 18:         18          4         NA   0.000000000
# 19:         19          7 0.28973666   0.063453401
# 20:         20          9         NA  -0.060000000
# 21:         21          7 0.66282918  -0.804101201
# 22:         22          5         NA   0.357770876
# 23:         23          6         NA   0.306186218
# 24:         24          5         NA  -0.268328157
# 25:         25          8         NA  -0.123743687
# 26:         26          5         NA   0.603738354
# 27:         27          5 0.19381378   0.570059742
# 28:         28          6         NA   0.000000000
# 29:         29          9         NA  -0.033333333
# 30:         30          3         NA   0.000000000
# 31:         31          7 0.07451828   0.572307621
# 32:         32         10 0.03422494   0.350799316
# 33:         33         10         NA   0.442718872
# 34:         34          8         NA   0.282842712
# 35:         35          8         NA   0.000000000
# 36:         36          7         NA  -0.680336051
# 37:         37          9 0.03535534  -0.027284316
# 38:         38         10         NA   0.142302495
# 39:         39         12         NA   0.505181486
# 40:         40         13         NA  -0.180277564
# 41:         41          9         NA   0.333333333
# 42:         42          9 0.02799422   0.355205098
# 43:         43          7 0.35355339   0.272843165
# 44:         44          9 0.40773876   0.666279314
# 45:         45          9         NA   0.450000000
# 46:         46          7         NA   0.000000000
# 47:         47          8 0.20373733   0.479474244
# 48:         48          7 0.12940952   0.341506351
#     element_id word_count         sd ave_sentiment


t <- extract_sentiment_terms(data)
head(t)
# Key: <element_id, sentence_id>
#    element_id sentence_id    negative     positive
#         <int>       <int>      <list>       <list>
# 1:          1           1 transparent             
# 2:          2           1                  improve
# 3:          3           1             happy,salary
# 4:          4           1                         
# 5:          4           2   difficult  performance
# 6:          5           1  could have 


attributes(t)$count
#             words polarity     n
#            <char>    <num> <int>
#   1:    excellent     1.00     4
#   2:    satisfied     1.00     1
#   3:       better     0.80     4
#   4:      clearer     0.80     1
#   5:        happy     0.75    10
#  ---                            
# 133:    difficult    -0.50     5
# 134:       biased    -1.00     2
# 135: disappointed    -1.00     1
# 136:   could have    -1.05     1
# 137:     would be    -1.05     1


get_sentiment(data)
#  [1] -0.25  0.75  1.35 -0.10  0.40  0.80  1.25  0.60  0.75  1.75  1.50  0.75  0.30  0.00  2.00  1.50  0.75  0.00 -0.40 -0.10 -0.25  0.80  0.75  0.60  1.15  0.75  1.75  0.00 -0.10  0.00  1.75  1.55  1.40  0.80  0.00 -1.00 -0.10  0.25  1.15
# [40]  1.00  1.00  1.50  1.00  1.75  0.75  0.00  1.35  1.25


nrcsentiment <- get_nrc_sentiment(data)
head(nrcsentiment)
#   anger anticipation disgust fear joy sadness surprise trust negative positive
# 1     0            0       0    0   0       0        0     0        0        0
# 2     0            1       0    0   1       0        0     1        0        1
# 3     0            2       0    0   2       0        0     2        0        2
# 4     0            0       0    1   0       0        0     1        0        0
# 5     0            0       0    0   0       0        0     1        0        1
# 6     0            1       0    0   0       0        0     0        0        0


read_sentence           <- get_sentences(data)
read_sentence_sentiment <- get_sentiment(read_sentence)


plot(read_sentence_sentiment, type = "l", main = "Example Plot Trajectory", xlab = "Narrative Time", ylab = "Emotional Valence")
