import os

for i in range(1, 2):
    for prob in [25, 50, 75]:
        # os.system("python matric_transposeLongESP.py " +
        #           str(prob) + " " + str(i))
        # os.system("python matric_transposeLongESP.1.py " +
        #           str(prob) + " " + str(i))

        os.system("python matric_transposeLongFL.py " +
                  str(prob) + " " + str(i))
        os.system("python matric_transposeLongFL.1.py " +
                  str(prob) + " " + str(i))
