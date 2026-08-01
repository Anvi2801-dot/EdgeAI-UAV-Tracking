import re
import csv

# Your log data
log_data = """[uav0000009_03358_v] frames=219 TP=4248 FP=104 FN=8283
[uav0000009_03358_v] Precision=0.976 Recall=0.339 F1=0.503

[uav0000073_00600_v] frames=328 TP=1092 FP=177 FN=13220
[uav0000073_00600_v] Precision=0.861 Recall=0.076 F1=0.140

[uav0000073_04464_v] frames=312 TP=2747 FP=63 FN=31597
[uav0000073_04464_v] Precision=0.978 Recall=0.080 F1=0.148

[uav0000077_00720_v] frames=780 TP=5643 FP=359 FN=13334
[uav0000077_00720_v] Precision=0.940 Recall=0.297 F1=0.452

[uav0000088_00290_v] frames=296 TP=5740 FP=123 FN=16188
[uav0000088_00290_v] Precision=0.979 Recall=0.262 F1=0.413

[uav0000119_02301_v] frames=179 TP=1436 FP=101 FN=4479
[uav0000119_02301_v] Precision=0.934 Recall=0.243 F1=0.385

[uav0000120_04775_v] frames=1000 TP=10636 FP=989 FN=31489
[uav0000120_04775_v] Precision=0.915 Recall=0.252 F1=0.396

[uav0000161_00000_v] frames=308 TP=4504 FP=418 FN=23314
[uav0000161_00000_v] Precision=0.915 Recall=0.162 F1=0.275

[uav0000188_00000_v] frames=260 TP=2825 FP=247 FN=14914
[uav0000188_00000_v] Precision=0.920 Recall=0.159 F1=0.271

[uav0000201_00000_v] frames=677 TP=3077 FP=1764 FN=15278
[uav0000201_00000_v] Precision=0.636 Recall=0.168 F1=0.265

[uav0000249_00001_v] frames=360 TP=4000 FP=538 FN=4003
[uav0000249_00001_v] Precision=0.881 Recall=0.500 F1=0.638

[uav0000249_02688_v] frames=244 TP=3659 FP=868 FN=3018
[uav0000249_02688_v] Precision=0.808 Recall=0.548 F1=0.653

[uav0000297_00000_v] frames=146 TP=3770 FP=443 FN=5133
[uav0000297_00000_v] Precision=0.895 Recall=0.423 F1=0.575

[uav0000297_02761_v] frames=373 TP=6804 FP=544 FN=22568
[uav0000297_02761_v] Precision=0.926 Recall=0.232 F1=0.371

[uav0000306_00230_v] frames=420 TP=6661 FP=697 FN=8005
[uav0000306_00230_v] Precision=0.905 Recall=0.454 F1=0.605

[uav0000355_00001_v] frames=468 TP=3153 FP=672 FN=9431
[uav0000355_00001_v] Precision=0.824 Recall=0.251 F1=0.384

[uav0000370_00001_v] frames=265 TP=315 FP=138 FN=2160
[uav0000370_00001_v] Precision=0.695 Recall=0.127 F1=0.215
""" # ... add full log here

# Define the regex to find the sequence name and the three metrics
pattern = r"\[(.*?)\] Precision=([\d.]+) Recall=([\d.]+) F1=([\d.]+)"

with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Sequence", "Precision", "Recall", "F1"])
    
    for match in re.finditer(pattern, log_data):
        writer.writerow(match.groups())