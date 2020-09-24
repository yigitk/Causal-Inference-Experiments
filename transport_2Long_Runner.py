import os

for i in range(1, 2):
    for prob in [25, 50, 75]:
        # os.system("python transport_2LongESP.py " + str(prob) + " " + str(i))
        # os.system("python transport_2LongESP.1.py " + str(prob) + " " + str(i))

        os.system("python transport_2LongFL.py " + str(prob) + " " + str(i))
        os.system("python transport_2LongFL.1.py " + str(prob) + " " + str(i))
