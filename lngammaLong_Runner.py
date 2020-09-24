import os

for i in range(1, 2):
    for prob in [25, 50, 75]:
        # os.system("python lngammaLongESP.py " + str(prob) + " " + str(i))
        # os.system("python lngammaLongESP.1.py " + str(prob) + " " + str(i))
        # os.system("python lngammaLongESP.2.py " + str(prob) + " " + str(i))
        # os.system("python lngammaLongESP.3.py " + str(prob) + " " + str(i))
        # os.system("python lngammaLongESP.4.py " + str(prob) + " " + str(i))

        os.system("python lngammaLongFL.py " + str(prob) + " " + str(i))
        os.system("python lngammaLongFL.1.py " + str(prob) + " " + str(i))
        os.system("python lngammaLongFL.2.py " + str(prob) + " " + str(i))
        os.system("python lngammaLongFL.3.py " + str(prob) + " " + str(i))
        os.system("python lngammaLongFL.4.py " + str(prob) + " " + str(i))
