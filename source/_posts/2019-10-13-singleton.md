---
author:
  nick: 小蛋子
  link: 'https://github.com/xv44586'
title: python中实现单例模式
date: 2019-10-13 11:33:59
tags: Python
categories: Programming
cover: /2019/10/13/singleton/cover.jpeg
---
<!-- toc -->

python中实现单例模式的方式大致有四种：
1.模块
2.改写类的__new__方法，控制实例生成
3.装饰器
4.元类

1.模块
python中的模块是天然的单例模式，并且是线程安全的，所有的模块只会在运行时加载一次。
所以，利用模块就可以实现一个线程安全的单例。如：
```python
# my singleton.py
class My_singleton(object):
    def foo(self):
        pass
my_singleton = My_singleton()
```

2. __new__方法
python中每个类在实例化对象时，首先会调用__new__方法创建一个实例，然后再调用__init__方法动态绑定其他属性.如：
```python
class Singleton(object):
    _instance = None
    def __new__(cls, *args, **kw):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kw)  
        return cls._instance  
class MyClass(Singleton):  
    a = 1
```

3.装饰器
装饰器可以动态的修改一个类或方法的表现，利用装饰器生成单例
```python
from functools import wraps


def singleton(cls):
    instances = {}

    @wraps(cls)
    def getinstance(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return getinstance


@singleton
class MyClass(object):
    a = 1
```

4.元类
元类可以控制其子类的类对象(__new__)及类实例对象(__call__)的创建。
```python
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# Python2
class MyClass(object):
    __metaclass__ = Singleton

    # Python3
    # class MyClass(metaclass=Singleton):
    #    pass
```
最著名的就是Django中的ORM，其model中就使用了元类来控制所有model的表现

**关于头图**

摄于内蒙古乌兰布统草原