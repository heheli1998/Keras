wflag =False                #写标记
newline = []                #创建一个新的列表
with open("test.tgt","w",encoding="utf-8") as fi:
    with open('test.txt', 'r', encoding='utf-8') as fo:
        # while (line):
        #     line = fo.readline()
        #     print(line)
        # f.close()
            n=1
            for line in fo:            #按行读入文件，此时line的type是str
                if "<" in line:        #重置写标记
                    wflag =False
                if "<summary>" in line:     #检验是否到了要写入的内容
                    wflag = True
                    continue
                if wflag == True:
                    K = list(line)
                    if len(K)>1:           #去除文本中的空行
                        for i in K :
                            # newline.append(n) #写入需要内容
                            newline.append(i)
                        # n+=1
            strlist = "".join(newline)      #合并列表元素
            newlines = str(strlist)         #list转化成str

            for D in range(1,100):                       #删掉句中（）
                newlines = newlines.replace("<summary>".format(D),"")

            # for P in range(0,9):                               #删掉前面数值标题
            #     for O in  range(0,9):
            #         for U in range(0, 9):
            #            newlines = newlines.replace("{}.{}{}".format(P,O,U), "")

            fi.write(newlines)

            fo.close()
            fi.close()
# wflag =False                #写标记
# newline = []
# with open('train.src', 'w', encoding='utf-8') as fi:
#     with open('test.txt', 'r', encoding='utf-8') as f:
#         line = f.readline()
#         print(line)
#         # if(line):
#         #     if "<" in line:        #重置写标记
#         #         wflag =False
#         # if "<short_text>" in line:     #检验是否到了要写入的内容
#         #     wflag = True
#         # if wflag == True:
#         #     K = list(line)
#         #     if len(K)>1:           #去除文本中的空行
#         #         for i in K :       #写入需要内容
#         #             newline.append(i)
#         #
#         # strlist = "".join(newline)      #合并列表元素
#         # newlines = str(strlist)         #list转化成str
#         #
#         # for D in range(1,100):                       #删掉句中（）
#         #     newlines = newlines.replace("<short_text>".format(D),"")
#         # fi.write(newlines)
#         while(line):
#             line = f.readline()
#             print(line)
#             # if (line):
#             # if "<" in line:  # 重置写标记
#             #         wflag = False
#             if "<short_text>" in line:  # 检验是否到了要写入的内容
#                 wflag = True
#             if wflag == True:
#                 K = list(line)
#                 if len(K) > 1:  # 去除文本中的空行
#                     for i in K:  # 写入需要内容
#                         newline.append(i)
#
#             strlist = "".join(newline)  # 合并列表元素
#             newlines = str(strlist)  # list转化成str
#
#             for D in range(1, 100):  # 删掉句中（）
#                 newlines = newlines.replace("<short_text>".format(D), "")
#             fi.write(newlines)
#         f.close()