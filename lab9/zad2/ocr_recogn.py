import string
import numpy as np
from PIL import ImageDraw, Image, ImageFont
import cv2

asc_chars = list(string.ascii_lowercase)
marks = list('?!. ')


def generate_strings(str_len = 100, rows = 10):
    strings = []
    lastmark = False
    for i in range(0, rows):
        st = ''
        last_space = 0
        for j in range(0, np.random.randint(int(0.75*str_len), int(str_len))):
            last_space += 1
            if lastmark:
                s = np.random.choice(asc_chars)
                lastmark = False
            else:
                if last_space >np.random.randint(4,10):
                    s= ' '
                    last_space = 0
                    lastmark =True
                else:
                    s = np.random.choice(asc_chars + marks)
                    if s in marks:
                        lastmark = True

            st += s

        strings.append(st)
    return strings


def make_img_for_str(s, text_i, f_size, font):
    x,y =f_size
    image = Image.new('RGB',(x, (y+2)*len(s)))
    for i,st in enumerate(s):
        ImageDraw.Draw(image).text((0,i*(y+2)), st, (255,255,255), font = font)
    image.save('./res/img/'+'{}.png'.format(text_i))
def make_img_for_char(s, i, f_size, font):
    image = Image.new('RGB',f_size)
    ImageDraw.Draw(image).text((0,0), s, (255,255,255), font = font)
    image.save('./res/fonts/'+'{}.png'.format(i))

def save_results(image, imn, lnns):
    with open('./res/result/{}.txt'.format(image), 'w') as f:
        for st in lnns:
            f.write('\n' + st)
        import scipy.misc
        scipy.misc.imsave('./res/result/{}.png'.format(image),imn)

    pass


def build_resources():
    os.makedirs('./res/')
    os.makedirs('./res/fonts/')
    os.makedirs('./res/img/')
    os.makedirs('./res/result/')
    for ch in asc_chars:
        font_size = ImageDraw.Draw(Image.new('RGB', (1, 1))).textsize(ch, font)
        make_img_for_char(ch, ch, font_size, font)

    for chk, ch in [('dot', '.'), ('com', ','), ('exc', '!'), ('ask', '?')]:
        font_size = ImageDraw.Draw(Image.new('RGB', (1, 1))).textsize(ch, font)
        make_img_for_char(ch, chk, font_size, font)

    font_size = ImageDraw.Draw(Image.new('RGB', (1, 1))).textsize(''.join(asc_chars) * M_C, font)
    for texts in range(0, N):
        with open('./res/{}.txt'.format(texts), 'w') as f:
            sts = generate_strings(str_len, rows=10)
            for st in sts:
                f.write('\n' + st)
            make_img_for_str(sts, texts, font_size, font)


def get_text(images, patterns):
    from lab.lab9.zad2.deskew import deskew

    for image in images.keys():
        line_map = {}
        im,imn = images[image]
        im = deskew(im)
        imn = deskew(imn)

        for p_key in patterns.keys():
            pattern = patterns[p_key][0]
            fp = np.fft.fft2(rot90(pattern, 2), im.shape)
            fi = np.fft.fft2(im)
            m = np.multiply(fp, fi)
            corr = np.fft.ifft2(m)
            corr = np.abs(corr)
            corr = corr.astype(float)
            i_M, j_M = corr.shape
            it = 0
            corr[corr < 0.99 * np.amax(corr)] = 0


            def mark(i, j, c):
                x,y = (i-line_height//2)//line_height, j-line_height//2
                if x in line_map:
                    line_map[x][y] = c
                else:
                    line_map[x] = {}
                    line_map[x][y]=c

                for x in range(i - 10, i):
                    for y in range(j - 10, j):
                        imn[x, y][0] = 255
                        imn[x, y][1] = 255
                        imn[x, y][2] = 255

            for i in range(i_M):
                for j in range(j_M):
                    it += 1
                    if corr[i, j] > 0:
                        print(corr[i, j])
                        mark(i, j,patterns[p_key][1])

        lnns = []
        for line in sorted(line_map.keys()):
            line_st =''
            chars_in_line =[x for x in line_map[line].keys()]
            chars_in_line = sorted(chars_in_line)

            for i,lk in enumerate(sorted(line_map[line].keys())):
                line_st+=line_map[line][lk]
                if i+1 < len(chars_in_line):
                    if(abs(chars_in_line[i] - chars_in_line[i+1]) > line_height):
                        line_st+=' '
            lnns.append(line_st)

        save_results(image, imn, lnns)




if __name__ == '__main__':
    N=10
    str_len = 100
    font = ImageFont.truetype("LiberationSerif-Regular.ttf", 20)
    import sys
    if len(sys.argv) >1:
        im_path = sys.argv[1]
    else:
        im_path = None
    import os
    if not os.path.exists('./res/'):
        build_resources()

    else:
        print('found img')
    w,h = ImageDraw.Draw(Image.new('RGB', (1, 1))).textsize(''.join(asc_chars) * 5, font)
    line_height = h+2
    import numpy as np
    from scipy import ndimage, rot90
    patterns = {x: (ndimage.imread('./res/fonts/{}.png'.format(x),flatten=True),x) for x in asc_chars}
    for x in [('ask','?')]:
        patterns[x[0]] = (ndimage.imread('./res/fonts/{}.png'.format(x[0]), flatten=True), x[1])
    if im_path is not None:
        images = {'in': (ndimage.imread(im_path, flatten=True),ndimage.imread(im_path) )}
    else:
        images = {x: (ndimage.imread('./res/img/{}.png'.format(x), flatten=True),ndimage.imread('./res/img/{}.png'.format(x)) ) for x in range(0,10)}

    get_text(images, patterns)



