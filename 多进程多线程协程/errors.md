## multiprocessing.Process 不能控制进程数

```python
    manager = multiprocessing.Manager()
    lock = multiprocessing.Lock()
    texts = manager.list()

    text_processes = []
    wiki = os.path.join(args.raw_path, 'wiki_zh_2019', 'wiki_zh')
    for first_dir in os.listdir(wiki):
        tmp_path = os.path.join(wiki, first_dir)
        if os.path.isdir(tmp_path):
            for file in os.listdir(tmp_path):
                path = os.path.join(tmp_path, file)
                if os.path.isfile(path):
                    text_processes.append(multiprocessing.Process(target=read_file, args=(path, lock, texts, ['text'])))

    for process in text_processes:
        process.start()
    for process in text_processes:
        process.join()
```

由于multiprocessing.Process不能控制进程数，这里由于文件里面有多个文件，导致报错：brokenpipeerror errno 32 broken pipe

网上查了说的是，因为信号处理问题报错。我感觉是因为进程数量开的太多的原因，所以做了如下的修改

```python
    manager = multiprocessing.Manager()
    # lock = multiprocessing.Lock()
    lock = manager.Lock() #pool进程池的Lock要用manager的
    texts = manager.list()
    pool = multiprocessing.Pool(processes=16) #限制进程数量
    
        wiki = os.path.join(args.raw_path, 'wiki_zh_2019', 'wiki_zh')
    for first_dir in os.listdir(wiki):
        tmp_path = os.path.join(wiki, first_dir)
        if os.path.isdir(tmp_path):
            for file in os.listdir(tmp_path):
                path = os.path.join(tmp_path, file)
                if os.path.isfile(path):
                    # text_processes.append(multiprocessing.Process(target=read_file, args=(path, lock, texts, ['text'])))
                     pool.apply(read_file, (path, lock, texts, ['text']))

    # for process in text_processes:
    #     process.start()
    # for process in text_processes:
    #     process.join()
    pool.close()
    pool.join() #join之前必须加close
    
```

