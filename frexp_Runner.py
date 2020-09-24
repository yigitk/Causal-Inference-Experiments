import os

for i in range(1, 2):
    for prob in [25, 50, 75]:
        # os.system("python frexpESP.py " + str(prob) + " " + str(i))
        # os.system("python frexpESP.1.py " + str(prob) + " " + str(i))

        os.system("python frexpFL.py " + str(prob) + " " + str(i))
        os.system("python frexpFL.1.py " + str(prob) + " " + str(i))
