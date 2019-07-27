"""

"""


class MyFirstClass:
    """MyFirstClass class definition"""   # __doc__

    myVersion = '1.1'  # 静态数据

    def printFoo(self):  # 方法
        print("this is invoke Foo")


if __name__ == '__main__':
    myObj = MyFirstClass
    myObj.x = 4
    myObj.y = 5

    print(myObj.x + myObj.y)

    myObj.printFoo
    print(dir(MyFirstClass))  # 查看类的属性

    print(MyFirstClass.__dict__)  # 查看类的属性

    print(MyFirstClass.__name__)
    print(MyFirstClass.__doc__)
