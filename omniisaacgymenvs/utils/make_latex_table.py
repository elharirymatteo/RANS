import pandas as pd
import numpy as np
import os

RL_root = "RL"
DC_root = "DC-real"
exp_keys = ["AN","VN","UF","TD","RTF"]
ordered_metrics_keys = ["PA1","PA2","PSA","OA1","OA2","OSA","ALV","AAV","AAC","PT5","PT2","PT1","OT5","OT2","OT1"]
metrics_keys = ["PT5","PT2","PT1","OT5","OT2","OT1","ALV","AAV","AAC"]
colored_metrics = ["PT5","PT2","PT1","OT5","OT2","OT1"]
exp_names = os.listdir(RL_root)
exp_names.sort()

#columns = ["Controller","AN","ON","UF","TD","RTK","PA2","PA1","PSA","OA2","OA1","OSA","ALV","AAV","AAC","PT5","PT2","PT1","OT5","OT2","OT1"]
columns = ["Controller","AN","VN","UF","TD","RTF","PT5","PT2","PT1","OT5","OT2","OT1","ALV","AAV","AAC"]

table = pd.DataFrame(0, columns=columns, index=range(len(exp_names)*2))
ctable = pd.DataFrame("none", columns=columns, index=range(len(exp_names)*2))

colors_name = np.array(['ForestGreen',
                        'LimeGreen',
                        'Goldenrod',
                        'Orange',
                        'OrangeRed'])
cv1 = np.array([0,20,40,60,80])
cv2 = np.array([20,40,60,80,100])

i = 0
exp_name = "ideal"
exp_keys_names = []
exp_keys_values = []
table["Controller"][i] = "RL"
table["Controller"][i+len(exp_names)] = "LQR"
RL_baseline = np.load(os.path.join(RL_root,exp_name,"aggregated_results.npy"))
DC_baseline = np.load(os.path.join(DC_root,exp_name,"aggregated_results.npy"))
for j, metric in enumerate(ordered_metrics_keys):
    if metric in metrics_keys:
        if metric in colored_metrics:
            table[metric][i] = int(RL_baseline[j]*100)
            table[metric][i+len(exp_names)] = int(DC_baseline[j]*100)
            ctable[metric][i] = 'black'
            ctable[metric][i+len(exp_names)] = 'black'
        else:
            table[metric][i] = RL_baseline[j]
            table[metric][i+len(exp_names)] = DC_baseline[j]


i = 1
for exp_name in exp_names:
    if exp_name == "ideal":
        continue
    exp_keys_names = []
    exp_keys_values = []
    table["Controller"][i] = "RL"
    table["Controller"][i+len(exp_names)] = "LQR"
    for tmp in exp_name.split("-"):
        exp_keys_name = tmp.split("_")[0]
        exp_keys_values = float(tmp.split("_")[1])
        table[exp_keys_name][i] = exp_keys_values
        table[exp_keys_name][i+len(exp_names)] = exp_keys_values
    RL_results = np.load(os.path.join(RL_root,exp_name,"aggregated_results.npy"))
    DC_results = np.load(os.path.join(DC_root,exp_name,"aggregated_results.npy"))
    RL_deltas = (RL_baseline - RL_results) / RL_baseline
    DC_deltas = (DC_baseline - DC_results) / DC_baseline
    for j, metric in enumerate(ordered_metrics_keys):
        if metric in metrics_keys:
            if metric in colored_metrics:
                table[metric][i] = int(RL_results[j]*100)
                table[metric][i+len(exp_names)] = int(DC_results[j]*100)

                if RL_deltas[j] < 0:
                    RL_deltas[j] = 0
                b1 = RL_deltas[j]*100 <= cv2
                b2 = RL_deltas[j]*100 >= cv1
                b = b1*b2
                ctable[metric][i] = colors_name[b][0]

                if DC_deltas[j] < 0:
                    DC_deltas[j] = 0
                b1 = DC_deltas[j]*100 <= cv2
                b2 = DC_deltas[j]*100 >= cv1
                b = b1*b2
                ctable[metric][i+len(exp_names)] = colors_name[b][0]
            else:
                table[metric][i] = RL_results[j]
                table[metric][i+len(exp_names)] = DC_results[j]

    i+=1



print(table)
print(ctable)
latex1 = table.to_latex(float_format="%.2f")
latex2 = ctable.to_latex(float_format="%.2f")
print(latex1)
print(latex2)

l1s = latex1.split("\n")
l2s = latex2.split("\n")
l3s = []
for i in range(4,40):
    ll1s = l1s[i].split('&')
    ll2s = l2s[i].split('&')
    ll1s = [lll1s.strip() for lll1s in ll1s]
    print(ll1s)
    for j in range(2,7):
        if (ll1s[j] == "0.00") or (ll1s[j] == "0"):
            ll1s[j] = "-"

    for j in range(7,13):
        ll1s[j] = '\\textcolor{'+ll2s[j].strip()+'}{'+ll1s[j]+'}'
    ll1s = ll1s[1:]
    l3s.append('&'.join(ll1s))
l3s = l1s[:4]+l3s+l1s[40:]

header =["\\begin{tabular}{|l|l|ccccc|ccccccccc|}",
"\\toprule",
"\multirow{2}{*}{Conditions}& \multirow{2}{*}{Controllers} & \multicolumn{5}{c|}{Disturbances} & \multicolumn{9}{c|}{Metrics} \\",
"&  & AN & VN & UF & TD & RTF & PT5 & PT2 & PT1 & OT5 & OT2 & OT1 & ALV & AAV & AAC \\",
"\midrule\hline",
]

data = [
'\multirow{2}{*}{Ideal} &'+l3s[4],
'&'+l3s[22]+'\hline\hline',
'\multirow{6}{*}{Velocity Noise} &'+l3s[19],
'&'+l3s[20],
'&'+l3s[21]+"\cline{2-16}",
'&'+l3s[37],
'&'+l3s[38],
'&'+l3s[39]+'\hline\hline',
'\multirow{8}{*}{Action Noise}&'+l3s[7],
'&'+l3s[8]+"\cline{2-16}",
'&'+l3s[25],
'&'+l3s[26]+'\hline\hline',
'\multirow{4}{*}{Constant Torque}&'+l3s[12],
'&'+l3s[13]+"\cline{2-16}",
'&'+l3s[30],
'&'+l3s[31]+'\hline\hline',
'\multirow{6}{*}{Constant Force}&'+l3s[14],
'&'+l3s[16],
'&'+l3s[18]+"\cline{2-16}",
'&'+l3s[32],
'&'+l3s[34],
'&'+l3s[36]+'\hline\hline',
'\multirow{4}{*}{Constant Force \& Torque}&'+l3s[15],
'&'+l3s[17]+"\cline{2-16}",
'&'+l3s[33],
'&'+l3s[35]+'\hline\hline',
'&'+l3s[9],
'&'+l3s[10],
'\multirow{6}{*}{Thruster Failures}&'+l3s[11]+"\cline{2-16}",
'&'+l3s[27],
'&'+l3s[28],
'&'+l3s[29]+'\hline\hline']

footer = ["\\bottomrule",
"\end{tabular}",
"}",
"\caption{",
"Description TBD. PT, OT higher",
"}",
"\label{tab:my_label}",
"\end{table*}",
]

latex3 = "\n".join(header+data+footer)
latexs = "\n".join(data)

print(latex3)
    #print([ll1s[k] for k in range(7,13)])
    #print([ll2s[k] for k in range(7,13)])
    #print(['{\color{'+ll2s[k]+'}'+ll1s[k]+'}' for k in range(7,13)])
print(latexs)
