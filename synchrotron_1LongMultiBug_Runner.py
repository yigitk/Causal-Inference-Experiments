import os

commands = []
for i in range(1, 2):
    for prob in [25, 50, 75]:
        # commands.append("python synchrotron_1LongESP2Bug.py " +
        #                 str(prob) + " " + str(i))
        # commands.append("python synchrotron_1LongESP3Bug.py " +
        #                 str(prob) + " " + str(i))
        commands.append("python synchrotron_1LongFL2Bug.py " +
                        str(prob) + " " + str(i))
        commands.append("python synchrotron_1LongFL3Bug.py " +
                        str(prob) + " " + str(i))
os.system(' & '.join(commands))
