# 获得根路径
import os


def getRootPath():
    # 获取文件目录
    curPath = os.path.abspath(os.path.dirname(__file__))
    # 获取项目根路径，内容为当前项目的名字
    rootPath = curPath[:curPath.find('wzry_ai') + len('wzry_ai')]
    return rootPath