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


def main(path="model_normalized.obj", angle=45, size=224):
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

if __name__ == "__main__":

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

    path = "/Users/qizekun/Downloads/ShapeNetCore/"
    meta = pd.read_json(path + "shapenet55.json")
    cls_list = meta['describe'].values.tolist()
    model, preprocess = clip.load("ViT-B/32", device="cpu")

    acc = []
    acc_detail = []
    complete_num = 0
    start_time = time.time()

    pwd = os.getcwd()
    for index, row in meta.iterrows():
        obj_list = glob.glob(path + "0" + str(row.catalogue) + "/*")
        num = len(obj_list)
        right_num = 0
        acc_detail.append([])
        for obj_path in obj_list:
            prediction = []
            for angle in (45, 135, 225, 315):
                os.chdir(obj_path + "/models")
                main(angle=angle)
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
            this_time = time.time()
            print(f"ETA: {(51300 - complete_num) / complete_num * (this_time - start_time) / 3600} h")
        acc.append(right_num / num)
    
    os.chdir(pwd)
    np.save('acc.npy', np.array(acc))
    np.save('acc_detail.npy', np.array(acc_detail))

exit()
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from OpenGL.arrays import vbo
import ctypes
from PIL import Image
gCamAng = 45.
gCamHeight = 5.
vertices = None 
normals = None
faces = None
dropped = 0
modeFlag = 0
distanceFromOrigin = 20
def dropCallback(window, path):
    global vertices, normals, faces, dropped, gVertexArraySeparate
    numberOfFacesWith3Vertices = 0
    numberOfFacesWith4Vertices = 0
    numberOfFacesWithMoreThan4Vertices = 0
    dropped = 1
    if(path.split('.')[1].lower() != "obj"):
        print("Invalid File\nPlease provide an .obj file")
        return
    with open(path) as f:
        lines = f.readlines()
        vStrings = [x.strip('v') for x in lines if x.startswith('v ')]
        vertices = convertVertices(vStrings)
        if np.amax(vertices) <= 1.2:
            vertices /= np.amax(vertices)
        else:
            vertices /= np.amax(vertices)/2
        vnStrings = [x.strip('vn') for x in lines if x.startswith('vn')]
        if not vnStrings: #if There is no normal vectors in the obj file then compute them
            normals = fillNormalsArray(len(vStrings))
        else:
            normals = convertVertices(vnStrings)
        faces = [x.strip('f') for x in lines if x.startswith('f')]
    for face in faces: 
        if len(face.split()) == 3:
            numberOfFacesWith3Vertices +=1
        elif len(face.split()) == 4:
            numberOfFacesWith4Vertices +=1
        else:
            numberOfFacesWithMoreThan4Vertices +=1
    print("File name:",path,"\nTotal number of faces:", len(faces),
        "\nNumber of faces with 3 vertices:",numberOfFacesWith3Vertices, 
        "\nNumber of faces with 4 vertices:",numberOfFacesWith4Vertices,
        "\nNumber of faces with more than 4 vertices:",numberOfFacesWithMoreThan4Vertices)
    if(numberOfFacesWith4Vertices > 0 or numberOfFacesWithMoreThan4Vertices > 0):
        faces = triangulate()
    gVertexArraySeparate = createVertexArraySeparate()
    ##########EMPTYING USELESS VARIABLES FOR MEMORY MANAGEMENT##########
    faces = []
    normals = []
    vertices = []
    ####################################################################
def fillNormalsArray(numberOfVertices):
    normals = np.zeros((numberOfVertices, 3))
    i = 0
    for vertice in vertices:
        normals[i] = normalized(vertice)
        i +=1
    return normals
def convertVertices(verticesStrings):
    v = np.zeros((len(verticesStrings), 3))
    i = 0
    for vertice in verticesStrings:
        j = 0
        for t in vertice.split():
            try:
                v[i][j] = (float(t))
            except ValueError:
                pass
            j+=1
        i+=1
    return v
def triangulate():
    facesList = []
    nPolygons = []
    for face in faces:
        if(len(face.split())>=4):
            nPolygons.append(face)
        else:
            facesList.append(face)
    for face in nPolygons:
        for i in range(1, len(face.split())-1):
            seq = [str(face.split()[0]), str(face.split()[i]), str(face.split()[i+1])]
            string = ' '.join(seq)
            facesList.append(string)
    return facesList
def createVertexArraySeparate():
    varr = np.zeros((len(faces)*6,3), 'float32')
    i=0
    normalsIndex = 0
    verticeIndex = 0
    for face in faces:
        for f in face.split():
            if '//' in f: # f v//vn
                verticeIndex = int(f.split('//')[0])-1 
                normalsIndex = int(f.split('//')[1])-1
            elif '/' in f: 
                if len(f.split('/')) == 2: # f v/vt
                    verticeIndex = int(f.split('/')[0])-1 
                    normalsIndex = int(f.split('/')[0])-1
                else: # f v/vt/vn
                    verticeIndex = int(f.split('/')[0])-1 
                    normalsIndex = int(f.split('/')[2])-1
            else: # f v v v
                verticeIndex = int(f.split()[0])-1 
                normalsIndex = int(f.split()[0])-1
            varr[i] = normals[normalsIndex]
            varr[i+1] = vertices[verticeIndex]
            i+=2
    return varr

def render(ang):
    global gCamAng, gCamHeight, distanceFromOrigin, dropped
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION) # use projection matrix stack for projection transformation for correct lighting
    glLoadIdentity()
    gluPerspective(distanceFromOrigin, 1, 1,10)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(5*np.sin(gCamAng),gCamHeight,5*np.cos(gCamAng), 0,0,0, 0,1,0)

    drawFrame()
    glEnable(GL_LIGHTING)   #comment: no lighting
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)
    glEnable(GL_LIGHT2)
    # light position
    glPushMatrix()
    lightPos0 = (1.,2.,3.,1.)    # try to change 4th element to 0. or 1.
    lightPos1 = (3.,2.,1.,1.)
    lightPos2 = (2.,3.,1.,1.)
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos0)
    glLightfv(GL_LIGHT1, GL_POSITION, lightPos1)
    glLightfv(GL_LIGHT2, GL_POSITION, lightPos2)
    glPopMatrix()

    # light intensity for each color channel
    ambientLightColor0 = (.1,.1,.1,1.)
    diffuseLightColor0 = (1.,1.,1.,1.)
    specularLightColor0 = (1.,1.,1.,1.)
    ambientLightColor1 = (.075,.075,.075,1.)
    diffuseLightColor1 = (0.75,0.75,0.75,0.75)
    specularLightColor1 = (0.75,0.75,0.75,0.75)
    ambientLightColor2 = (.05,.05,.05,1.)
    diffuseLightColor2 = (0.5,0.5,0.,0.5)
    specularLightColor2 = (0.5,0.5,0.,0.5)
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLightColor0)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLightColor0)
    glLightfv(GL_LIGHT0, GL_SPECULAR, specularLightColor0)
    glLightfv(GL_LIGHT1, GL_AMBIENT, ambientLightColor1)
    glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuseLightColor1)
    glLightfv(GL_LIGHT1, GL_SPECULAR, specularLightColor1)
    glLightfv(GL_LIGHT2, GL_AMBIENT, ambientLightColor2)
    glLightfv(GL_LIGHT2, GL_DIFFUSE, diffuseLightColor2)
    glLightfv(GL_LIGHT2, GL_SPECULAR, specularLightColor2)
    # material reflectance for each color channel
    diffuseObjectColor = (0.4,0.6,0.5,1.)
    specularObjectColor = (0.6,0.3,0.3,.5)
    # glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, diffuseObjectColor)
    glMaterialfv(GL_FRONT, GL_SPECULAR, specularObjectColor)

    glPushMatrix()
    if dropped == 1:
        draw_glDrawArray()
    glPopMatrix()

    glDisable(GL_LIGHTING)
def draw_glDrawArray():
    global gVertexArraySeparate
    varr = gVertexArraySeparate
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glNormalPointer(GL_FLOAT, 6*varr.itemsize, varr)
    glVertexPointer(3, GL_FLOAT, 6*varr.itemsize, ctypes.c_void_p(varr.ctypes.data + 3*varr.itemsize))
    glDrawArrays(GL_TRIANGLES, 0, int(varr.size/6))

def drawFrame():
    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex3fv(np.array([0.,0.,0.]))
    glVertex3fv(np.array([1.,0.,0.]))
    glColor3ub(0, 255, 0)
    glVertex3fv(np.array([0.,0.,0.]))
    glVertex3fv(np.array([0.,1.,0.]))
    glColor3ub(0, 0, 255)
    glVertex3fv(np.array([0.,0.,0]))
    glVertex3fv(np.array([0.,0.,1.]))
    glEnd()

gVertexArraySeparate = np.zeros((3, 3))
def main():
    global gVertexArraySeparate
    if not glfw.init():
        return
    window = glfw.create_window(640,640,'3D Obj File Viewer', None,None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    dropCallback(window, 'model_normalized.obj')
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glfw.swap_buffers

    glfw.swap_interval(1)
    count = 0
    while not glfw.window_should_close(window):
        glfw.poll_events()
        count+=1
        ang = count % 360
        render(ang)
        count += 1
        glfw.swap_buffers(window)
    glfw.terminate()

    # glPixelStorei(GL_PACK_ALIGNMENT, 1)
    # width = 640
    # height = 640
    # data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    # image = Image.frombytes("RGBA", (width, height), data)
    # image.save('glutout.png', 'PNG')
    # import time
    # time.sleep(2)
    # image = ImageOps.flip(image) # in my case image is flipped top-bottom for some reason image.save('glutout.png', 'PNG')
def l2norm(v):
    return np.sqrt(np.dot(v, v))
def normalized(v):
    l = l2norm(v)
    return 1/l * np.array(v)
def framebuffer_size_callback(window, width, height):
    glViewport(0, 0, width, height)
if __name__ == "__main__":
    main()
