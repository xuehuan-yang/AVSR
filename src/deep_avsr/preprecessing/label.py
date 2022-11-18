def labelvo_func(label, labelvo, trig):
    with open(label, 'r') as f1, open(labelvo, 'w') as f2:
        lines = f1.readlines()
        temp0 = lines[0].split(' ')
        temp0[2] = trig
        lines[0] = ' '.join(temp0)
        if len(lines) >= 4:
            temp4 = lines[4].split(' ')
            temp4[0] = trig
            lines[4] = ' '.join(temp4)
        for i in range(len(lines)):
            f2.write(lines[i])


label= './00063.txt'
labelvo= './10063.txt'
labelvo_func(label, labelvo, 'Hello')
