import os

commands = []
for i in range(1, 2):
    for prob in [25, 50, 75]:
        # commands.append("python lngammaLongESP2Bug.py " +
        #                 str(prob) + " " + str(i))
        # commands.append("python lngammaLongESP3Bug.py " +
        #                 str(prob) + " " + str(i))
        commands.append("python lngammaLongFL2Bug.py " +
                        str(prob) + " " + str(i))
        commands.append("python lngammaLongFL3Bug.py " +
                        str(prob) + " " + str(i))

os.system(' & '.join(commands))
