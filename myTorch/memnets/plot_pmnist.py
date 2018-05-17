import matplotlib.pyplot as plt
import seaborn as sns



fname = "pmnist.csv"
file = open(fname, "r")

line = file.readline().strip().split(",")

total_models = int(len(line)/2)

mname = list()
vals = list()
for i in range(0, total_models):
    mname.append(line[i*2+1])
    a = dict()
    vals.append(a)


while True:
    line = file.readline()
    if not line:
        break
    line = line.strip().split(",")
    for i in range(0, total_models):
        if len(line[i*2+1]) > 0:
            vals[i][int(line[i*2])] = float(line[i*2+1])

cbest = list()
cbest_iter = list()

for i in range(0, total_models):
    best = 0.0
    best_iter = 0
    for j in range(0, 100):
        if j in vals[i]:
            if vals[i][j] > best:
                best = vals[i][j]
                best_iter = j
    cbest.append(best)
    cbest_iter.append(best_iter)


colors = {}
colors["LSTM-chrono"] = "C0"
colors["NRU"] = "C1"
colors["JANET"] = "C2"
colors["GRU"] = "C3"
colors["RNN-orth"] = "C4"
colors["RNN-id"] = "C5"
colors["LSTM"] = "C6"
colors["SRU"] = "C7"
colors["EURNN"] = "C8"
colors["GORU"] = "C9"


for i in range(0, total_models):
    pvals = list()
    for j in range(1,100):
        pvals.append(vals[i][j]*100)
    plt.plot(pvals, color=colors[mname[i]], label=mname[i])

plt.ylabel('Validation Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.grid()
plt.show()



print(mname)

print(cbest)
print(cbest_iter)