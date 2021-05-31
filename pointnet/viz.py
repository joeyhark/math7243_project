import matplotlib.pyplot as plt

data = {'Ty Pitch': (0.9728533574648864, [[659646, 10428], [9611, 58491]]),
        'Ty Roll': (0.8854817183041923, [[357843, 52940], [1, 51509]]),
        'Avi NoDrape': (0.9586647752425437, [[404919, 18950], [2059, 82331]]),
        'Ty Yaw': (0.9857463233708543, [[418978, 2059], [4551, 38152]]),
        'Alina Mask': (0.9301863921437732, [[249094, 5492], [18352, 68600]]),
        'Alina Pitch': (0.9156276726796236, [[263749, 25646], [6613, 86333]]),
        'Alina NoDrape NoMask': (0.9702144868492002, [[233736, 9941], [2, 90141]]),
        'Alina NoDrape Mask': (0.9924961850352491, [[261758, 2975], [0, 131732]]),
        'Alina FullMask': (0.9283197790985174, [[270834, 11502], [14561, 66704]])}

cats = list(data.keys())

accs = []
cfs = []
for cat in cats:
    cfs.append(data[cat][1])
    accs.append(data[cat][0])

plt.bar(cats, accs)
plt.draw()
plt.ylim(0.75, 1.0)
plt.xticks(rotation=25)
plt.show()
