import os

for i in range(1, 2):
    for prob in [25, 50, 75]:
        # os.system("python synchrotron_1LongESP.py " + str(prob) + " " + str(i))
        # os.system("python synchrotron_1LongESP.1.py " +
        #           str(prob) + " " + str(i))
        # os.system("python synchrotron_1LongESP.2.py " +
        #           str(prob) + " " + str(i))
        # os.system("python synchrotron_1LongESP.3.py " +
        #           str(prob) + " " + str(i))
        # os.system("python synchrotron_1LongESP.4.py " +
        #           str(prob) + " " + str(i))

        os.system("python synchrotron_1LongFL.py " + str(prob) + " " + str(i))
        os.system("python synchrotron_1LongFL.1.py " +
                  str(prob) + " " + str(i))
        os.system("python synchrotron_1LongFL.2.py " +
                  str(prob) + " " + str(i))
        os.system("python synchrotron_1LongFL.3.py " +
                  str(prob) + " " + str(i))
        os.system("python synchrotron_1LongFL.4.py " +
                  str(prob) + " " + str(i))
