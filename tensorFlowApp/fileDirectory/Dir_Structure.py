import os

class RootDir:
    # obtain absolute path
    projectRootDir = os.path.dirname(os.path.abspath(__file__))
    # projectRootDir = os.getcwd()
    # print (projectRootDir)

    # def __init__(self):
    #     # self.rootDir = os.getcwd()
    def test(self):
        print ("done")

        # return RootDir.projectRootDir

    @staticmethod
    def getRootDir():
        cur_path = RootDir.projectRootDir
        tindex = cur_path.rindex("/")
        subStr = cur_path[0:tindex]
        RootDir.projectRootDir = subStr
        return RootDir.projectRootDir

# if __name__ == "__main__":
#     cur_path = '/home/tensor/PycharmProjects/tensorFlowApp/fileDirectory'
#     t = cur_path.rindex("/")
#     temp = cur_path[0:t]
#     print (RootDir.projectRootDir)






    # print ("obtain root directory:" + project_dir)