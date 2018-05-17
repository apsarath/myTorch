
file = open("eurnn.txt", "r")
ofile = open("eurnn_valid.csv", "w")

count = 0

while True:
    line = file.readline()
    if not line:
        break
    line = line.strip().split(" ")
    acc = line[1].split("=")[1]
    ofile.write("{},{}\n".format(count+1, acc))
    count+=1

ofile.close()