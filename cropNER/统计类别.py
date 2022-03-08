d = {}
f = 1
with open('test.txt', 'r', encoding='utf8') as f:
    for line in f:
        line = line.replace('\n', ' ')
        if line != ' ':
            s = line.split(' ')
            if s[1] != 'O':
                label = s[1].split('-')[1]
                if f == 1:
                    if label not in d.keys():
                        d[label] = 1
                    else:
                        d[label] += 1
                f = 0
            else:
                f = 1
print(d)
