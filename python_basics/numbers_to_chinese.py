inputStr = input("请输入一串数字：")

# def processInput (str):
#     outputStr = ''
#     for i in str:
#         outputStr += transfer(i)
#     return outputStr

# def transfer (number):
#     if number == '1':
#         return '一'
#     elif number == '2':
#         return '二'
#     elif number == '3':
#         return '三'
#     elif number == '4':
#         return '四'
#     elif number == '5':
#         return '五'
#     elif number == '6':
#         return '六'
#     elif number == '7':
#         return '七'
#     elif number == '8':
#         return '八'
#     elif number == '9':
#         return '九'
#     elif number == '0':
#         return '零'

dictionary = {'1':'一','2':'二','3':'三','4':'四','5':'五','6':'六','7':'七','8':'八','9':'九','0':'零'}
def processInput(str):
    outputStr = ''
    for i in str:
        outputStr += dictionary[i]
    return outputStr
print(processInput(inputStr))