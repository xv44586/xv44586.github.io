---
author:
  nick: 小蛋子
  link: 'https://github.com/xv44586'
title: 记一次npm环境问题
date: 2020-01-11 12:31:20
categories: Life
tags: npm
cover: /2020/01/11/npm-environment/cover.jpeg
---
<!-- toc -->


# 起因
上周写完博客，本地预览时，突然报错，看了一下是 sharp.js 的问题，以为是个小场面，然后就开始了一周的痛苦修环境经历，哭了😭

# Ddbug
1. 首先根据提示，重新安装  
```python
rm -rf node_modules/sharp
npm i sharp
```
此时报错：
```python
c++: error: unrecognized command line option '-stdlib=libc++'
make: *** [Release/.node] Error 1
gyp ERR! build error
gyp ERR! stack Error: `make` failed with exit code: 2
gyp ERR! stack     at ChildProcess.onExit (/usr/local/Cellar/node@8/8.16.2/lib/node_modules/npm/node_modules/node-gyp/lib/build.js:262:23)
gyp ERR! stack     at emitTwo (events.js:126:13)
gyp ERR! stack     at ChildProcess.emit (events.js:214:7)
gyp ERR! stack     at Process.ChildProcess._handle.onexit (internal/child_process.js:198:12)
```

这个问题的本质是当前的包需要通过源码编译，而当前用的是gcc（macOS），而gcc不支持当前命令，之前装环境没有遇到过这个问题，可能是我最近跟新了gcc？
查了一下解决这个问题最简单的方式是指定c++:
```python
CXX=clang++ npm i xxx
```
这次确实装成功了，走起！
此时又报新错：
```python
can not find sharp xxx
rm node_modules/sharp and rebuild
```
阿嘞？装上了又找不到？why？
又尝试了全局安装，依然找不到，此时，有点上头，我干脆把node_modules全部删掉，重新装吧。
```python
rm -rf node_modules
CXX=clang++ npm i
```
阿嘞？这次装也失败了，错误大致原因是node-gyp rebuild nodejieba失败。
开始以为是node-gyp的问题，后来查了一下，node-gyp是用来帮助丛源码编译的工具，所以本质上不是他的问题，还是别的问题。
又查了一下nodejieba, 有人说是在lunr.js中用的nodejieba在node高版本中会存在编译失败，建议用 node8.x ,python版本最好是2.7，那我上次没失败？先将node降级到node8, python切到2.7。
再装一次，依然失败。但是错误信息不够定位，查一下怎么看更全的日志。
```python
CXX=clang++ npm i --verbose
```  
发现两条重要信息，大概是无法找到lib下某个库，可能是本级的环境出了问题，整理一下吧。
 ```python
brew update
brew cleanup
brew doctor
```  
环境的一堆issue解决掉（主要是link无效），然后把之前无法找到的两个lib重新装了一次（很慢）。
此时还是同样问题，这次又查看了一下当前环境问题，两个包没link，一个是python(其实是python3，之前link的python2)， 一个是swig，后来一想，可能是swig这个工具在源码编译是缺失导致的，
```python
brew link swig
```
此时在装，搞定！走起，又报新错：
```python
FATAL Something's wrong. Maybe you can find the solution here: https://hexo.io/docs/troubleshooting.html
Error: spawn /Users/xuming/Project/blog/node_modules/optipng-bin/vendor/optipng ENOENT
    at Process.ChildProcess._handle.onexit (internal/child_process.js:190:19)
    at onErrorNT (internal/child_process.js:362:16)
    at _combinedTickCallback (internal/process/next_tick.js:139:11)
    at process._tickCallback (internal/process/next_tick.js:181:9)

```
这是缺module，但是我是npm i，为什么还缺？此时可能只是当前node_modules的问题，为了验证本机其他环境已经ok了，新建了一个博客，验证后发现本机确实OK了。
额，那就缺什么装什么吧。
```python
npm i optipng-bin
```
走起！一切正常！此时我的心情就好比火箭发射成功一样。折磨了一周的环境问题，终于搞定了，中间还有许多其他方向的试探，但都记不得了。- -！

# 结论

1. npm 失败后，可以加 --verbose 参数查看详细日志，定位问题。
2. 编译源码可能你还需要装{% raw %}xcode-select --install{% endraw %}
3. 对于 {%raw %}Error: spawn .../node_modules/xxx/vendor/.. ENOENT{% endraw %},单独安装一下对应缺失module即可。
4. node-gyp 和 libvips可能也会影响，建议重装一次
5. MacOS中，可能会gcc与clang并存，加上系统升级，可能导致相应版本不兼容问题。指定clang ：{% codeblock lang:python %}CXX=clang++ npm i{% endcodeblock %}
6. 本机环境问题，可以通过{% codeblock lang:python %}brew doctor{% endcodeblock %}

# 关于头
雪中奥森