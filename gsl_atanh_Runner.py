import os

for i in range(1, 2):
    for prob in [25, 50, 75]:
        # os.system("python gsl_atanhESP.py " + str(prob) + " " + str(i))
        # os.system("python gsl_atanhESP.1.py " + str(prob) + " " + str(i))

        os.system("python gsl_atanhFL.py " + str(prob) + " " + str(i))
        os.system("python gsl_atanhFL.1.py " + str(prob) + " " + str(i))
