import qrcode


def main():
    # text='n=dfdfuer34eajtoetopi4po3i9\n349poggpop542upouo4uoiu6p9u4po3o'
    # ctrl = input()
    # text = ctrl
    text = 'data=222'
    url = '47.108.203.248:3389' + '/judge/?' + text

    # url = 'http://baidu.com/' + 's?wd=' + text



    func0(url)


def func0(data):
    image = qrcode.make(data=data)
    image.show()
    image.save("test11.png")


def func1(data1, data2):
    qr = qrcode.QRCode()
    qr.add_data(data1)
    qr.add_data(data2)
    other_img = qr.make_image()
    return other_img


if __name__ == "__main__":
    main()
