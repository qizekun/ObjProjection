from OpenGL.GL import *
from OpenGL.GLU import *
from objloader import *
import pandas as pd
from PIL import Image, ImageOps
import glfw
import glob
import clip
import torch
import os
import time
import multiprocessing

path = "/Users/qizekun/Downloads/ShapeNetCore/"
meta = pd.read_json(path + "shapenet55.json")
cls_list = meta['describe'].values.tolist()

acc = []
acc_detail = []
p_list = []
flag = []
pwd = os.getcwd()
complete_num = 1
length = len(cls_list)
for i in range(length):
    acc.append([])
    acc_detail.append([])
    flag.append(0)

def render(path="model_normalized.obj", angle=45, size=1024):
    viewport = (size, size)
    glfw.init()
    window = glfw.create_window(size, size,'3D Obj File Viewer', None,None)
    glfw.make_context_current(window)
    # color, color, color, color
    # glClearColor(1., 1., 1., 1.)

    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (1., 1., 1., 0.))

    glLightfv(GL_LIGHT0, GL_POSITION,  (1., 1., 1., 0.))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (.5, .5, .5, 0.))
    
    glLightfv(GL_LIGHT1, GL_POSITION,  (1., -1., 1., 0.))
    glLightfv(GL_LIGHT1, GL_SPECULAR, (.5, .5, .5, 0.))

    glLightfv(GL_LIGHT2, GL_AMBIENT, (.1, .1, .1, 0.))


    glShadeModel(GL_SMOOTH)           # most obj files expect to be smooth-shaded
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)
    glEnable(GL_LIGHT2)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_COLOR_MATERIAL)
    
    # LOAD OBJECT AFTER PYGAME INIT
    obj = OBJ(path)
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    width, height = viewport
    gluPerspective(20.0, 1, 1.0, 100.0)

    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_MODELVIEW)
    rx, ry = (5 - angle, 25)
    tx, ty = (-0.7, -0.6)
    zpos = 8

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslate(tx, ty, - zpos)
    glRotate(ry, 1, 0, 0)
    glRotate(rx, 0, 1, 0)
    glCallList(obj. gl_list)

    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    width , height = viewport
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), data)
    image = ImageOps.flip(image)
    image.save(f'angle{angle}.png', 'PNG')
    glfw.terminate()


def test_clip(model, preprocess, path, cls_list):
    image = preprocess(Image.open(path)).unsqueeze(0)
    text = clip.tokenize(cls_list)
    with torch.no_grad():
        # image_features = model.encode_image(image)
        # text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return(probs[0])
    # print("Label probs:", probs)

def main(cls_list, path, index, row):
    global acc, acc_detail, complete_num, flag
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    obj_list = glob.glob(path + "0" + str(row.catalogue) + "/*")
    num = len(obj_list)
    right_num = 0
    for obj_path in obj_list:
        prediction = []
        for angle in (45, 135, 225, 315):
            os.chdir(obj_path + "/models")
            render(angle=angle)
            prediction.append(test_clip(model, preprocess, f'angle{angle}.png', cls_list))
        prediction = np.max(np.array(prediction), axis=0)
        acc_detail[index].append(prediction[index])
        if np.argmax(prediction) == index:
            right_num += 1
            print('right!')
        else:
            print('wrong!')
            print(obj_path)
        complete_num += 1
        time.sleep(1)
        # this_time = time.time()
        # print(f"ETA: {(51300 - complete_num) / complete_num * (this_time - start_time) / 3600} h")
    acc[index] = right_num / num
    flag[index] = 1
    
def moniter(start_time):
    global acc, acc_detail, complete_num, flag, pwd
    while True:
        time.sleep(60)
        this_time = time.time()
        print(f"ETA: {(51300 - complete_num) / complete_num * (this_time - start_time) / 3600} h")
        if sum(flag) == len(flag):
            os.chdir(pwd)  
            np.save('acc.npy', np.array(acc))
            np.save('acc_detail.npy', np.array(acc_detail))
            break

if __name__ == "__main__":
    pool_length = 8

    start_time = time.time()
    moniter_process = multiprocessing.Process(target=moniter, args=(start_time, ))
    moniter_process.start()

    for index, row in meta.iterrows():
        if len(p_list) < pool_length:
            p = multiprocessing.Process(target=main, args=(cls_list, path, index, row))
            p_list.append(p)
            p.start()
        else:
            while True:
                flag = False
                for i in range(pool_length):
                    if not p_list[i].is_alive():
                        p_list[i] = multiprocessing.Process(target=main, args=(cls_list, path, index, row))
                        p_list[i].start()
                        flag = True
                        break
                if flag:
                    break
                else:
                    time.sleep(60)



    # model, preprocess = clip.load("ViT-B/32", device="cpu")
    # path = "/Users/qizekun/Downloads/ShapeNetCore/"
    # meta = pd.read_json(path + "shapenet55.json")
    # cls_list = meta['describe'].values.tolist()

    # t_1 = time.time()
    # for i in range(10):
    #     angle = 45
    #     main(angle=angle, size=1024)
    #     test_clip(model, preprocess, f'angle{angle}.png', cls_list)
    # print((time.time() - t_1)/10)
    # exit()

