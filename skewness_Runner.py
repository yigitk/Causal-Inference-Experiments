import os

for i in range(1, 2):
    for prob in [25, 50, 75]:
        # os.system("python skewnessESP.py " +
        #           str(prob) + " " + str(i))
        # os.system("python skewnessESP.1.py " +
        #           str(prob) + " " + str(i))

        os.system("python skewnessFL.py " +
                  str(prob) + " " + str(i))
        os.system("python skewnessFL.1.py " +
                  str(prob) + " " + str(i))
