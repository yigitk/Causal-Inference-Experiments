import os

for i in range(1, 2):
    for prob in [25, 50, 75]:
        # os.system("python clausenLongESP.py " + str(prob) + " " + str(i))
        # os.system("python clausenLongESP.1.py " + str(prob) + " " + str(i))

        os.system("python clausenLongFL.py " + str(prob) + " " + str(i))
        os.system("python clausenLongFL.1.py " + str(prob) + " " + str(i))
