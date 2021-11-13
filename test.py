def test(cb):
    for i in range(100):
        print(1)
        cb(i)


def callback():
    image_num = 0

    def upload_storage(iteration):
        if iteration % 10 == 1:
            nonlocal image_num
            image_num += 1
        print(image_num)

    return upload_storage


def thread_main():
    cb_func = callback()
    test(cb_func)


if __name__ == "__main__":
    thread_main()