import fileinput

c=fileinput.input('valid.src.id',backup='.bak',inplace=1)
for line in c:
    d=fileinput.filelineno()
    print(d)