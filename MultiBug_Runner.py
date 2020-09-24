import os

for i in range(1, 11):
    for prob in [25, 50, 75]:
        os.system("python J0LongESP2Bug.py " +
                  str(prob) + " " + str(i))
        os.system("python J0LongESP3Bug.py " +
                  str(prob) + " " + str(i))
        os.system("python J0LongFL2Bug.py " +
                  str(prob) + " " + str(i))
        os.system("python J0LongFL3Bug.py " +
                  str(prob) + " " + str(i))

        os.system("python lngammaLongESP2Bug.py " +
                  str(prob) + " " + str(i))
        os.system("python lngammaLongESP3Bug.py " +
                  str(prob) + " " + str(i))
        os.system("python lngammaLongFL2Bug.py " +
                  str(prob) + " " + str(i))
        os.system("python lngammaLongFL3Bug.py " +
                  str(prob) + " " + str(i))
