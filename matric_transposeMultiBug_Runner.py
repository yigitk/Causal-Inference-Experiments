import os

for i in range(1, 11):
    for prob in [25, 50, 75]:
        os.system("python matric_transposeLongESP2Bug.py " +
                  str(prob) + " " + str(i))
        os.system("python matric_transposeLongESP3Bug.py " +
                  str(prob) + " " + str(i))
        os.system("python matric_transposeLongFL2Bug.py " +
                  str(prob) + " " + str(i))
        os.system("python matric_transposeLongFL3Bug.py " +
                  str(prob) + " " + str(i))
