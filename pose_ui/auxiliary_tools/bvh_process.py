import os, struct
import numpy as np
from numpy import array, dot
from math import radians, cos, sin

class Node(object):
    def __init__(self, root=False):
        self.name = None
        self.channels = []
        self.offset = (0, 0, 0)
        self.children = []
        self._is_root = root

    @property
    def is_root(self):
        return self._is_root

    @property
    def is_end_site(self):
        return len(self.children) == 0
    
class BvhReader(object):
    def __init__(self, filename):
        self.filename = filename
        self._token_list = []
        self._line_num = 0
        self.root = None
        self._node_stack = []
        self.num_channels = 0
    
    def read(self):
        with open(self.filename, 'r') as self._file_handle:
            self.read_hierarchy()
            self.on_hierarchy(self.root)

    def on_hierarchy(self, root):
        self.root = root  # Save root for later use
        self.keyframes = []  # Used later in onFrame

    def read_hierarchy(self):
        tok = self.token()
        if tok != "HIERARCHY":
            raise SyntaxError("Syntax error in line %d: 'HIERARCHY' expected, "
                              "got '%s' instead" % (self._line_num, tok))
        tok = self.token()
        if tok != "ROOT":
            raise SyntaxError("Syntax error in line %d: 'ROOT' expected, "
                              "got '%s' instead" % (self._line_num, tok))

        self.root = Node(root=True)
        self._node_stack.append(self.root)
        self.read_node()

    def read_node(self):
        name = self.token()
        self._node_stack[-1].name = name
        tok = self.token()
        if tok != "{":
            raise SyntaxError("Syntax error in line %d: '{' expected, "
                              "got '%s' instead" % (self._line_num, tok))
        while 1:
            tok = self.token()
            if tok == "OFFSET":
                x = self.float_token()
                y = self.float_token()
                z = self.float_token()
                self._node_stack[-1].offset = (x, y, z)
            elif tok == "CHANNELS":
                n = self.int_token()
                channels = []
                for i in range(n):
                    tok = self.token()
                    if tok not in ["Xposition", "Yposition", "Zposition",
                                   "Xrotation", "Yrotation", "Zrotation"]:
                        raise SyntaxError("Syntax error in line %d: Invalid "
                                          "channel name: '%s'"
                                          % (self._line_num, tok))
                    channels.append(tok)
                self.num_channels += len(channels)
                self._node_stack[-1].channels = channels
            elif tok == "JOINT":
                node = Node()
                self._node_stack[-1].children.append(node)
                self._node_stack.append(node)
                self.read_node()
            elif tok == "End":
                node = Node()
                self._node_stack[-1].children.append(node)
                self._node_stack.append(node)
                self.read_node()
            elif tok == "}":
                if self._node_stack[-1].is_end_site:
                    self._node_stack[-1].name = "End Site"
                self._node_stack.pop()
                break
            else:
                raise SyntaxError("Syntax error in line %d: Unknown "
                                  "keyword '%s'" % (self._line_num, tok))

    def int_token(self):
        """Return the next token which must be an int. """
        tok = self.token()
        try:
            return int(tok)
        except ValueError:
            raise SyntaxError("Syntax error in line %d: Integer expected, "
                              "got '%s' instead" % (self._line_num, tok))

    def float_token(self):
        tok = self.token()
        try:
            return float(tok)
        except ValueError:
            raise SyntaxError("Syntax error in line %d: Float expected, "
                              "got '%s' instead" % (self._line_num, tok))

    def token(self):
        if self._token_list:
            tok = self._token_list[0]
            self._token_list = self._token_list[1:]
            return tok

        # Read a new line
        s = self.read_line()
        self.create_tokens(s)
        return self.token()

    def read_line(self):
        self._token_list = []
        while 1:
            s = self._file_handle.readline()
            self._line_num += 1
            if s == "":
                raise StopIteration
            return s

    def create_tokens(self, s):
        s = s.strip()
        a = s.split()
        self._token_list = a

class Joint:

    def __init__(self, name):
        self.name = name
        self.children = []
        # list entry is one of [XYZ]position, [XYZ]rotation
        self.hasparent = 0  # flag
        self.parent = 0  # joint.addchild() sets this
        self.strans = array([0., 0., 0.])  # I think I could just use regular Python arrays.
      
        self.trtr = {}  # self.trtr[time]  A premultiplied series of translation and rotation matrices.
        self.worldpos = {}  # Time-based worldspace xyz position of the joint's endpoint.  A list of vec4's

    def addchild(self, childjoint):
        self.children.append(childjoint)
        childjoint.hasparent = 1
        childjoint.parent = self

class Skeleton:

    def __init__(self, hips, ignore_root_offset=True):
        self.root = hips


        if ignore_root_offset:
            self.root.strans[0] = 0.0
            self.root.strans[1] = 0.0
            self.root.strans[2] = 0.0
            self.root.stransmat = array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                  [0., 0., 1., 0.], [0., 0., 0., 1.]])

    @staticmethod
    def joint_dfs(root):
        nodes = []
        stack = [root]
        while stack:
            cur_node = stack.pop(0)
            nodes.append(cur_node)
            for child in cur_node.children:
                if 'End' in child.name:
                    continue
                stack.insert(0, child)
        return nodes
    
    def get_frames_worldpos(self):
        joints = self.joint_dfs(self.root)

        frame_data = []
        for j in joints:
            frame_data.extend(j.worldpos[0][:3])

        header = ["{}.{}".format(j.name, thing) for j in joints
                  for thing in ("X", "Y", "Z")]
        return header, frame_data
       
    # def change_offsetfromdata(self, displacement):
    # 使用displacement更换节点中的offset值

    #     skeleton_dict, index_dict = get_skdict()
    #     stack = [self.root]
    #     while(stack):
    #         cur_node = stack[0]
    #         stack.pop(0)
    #         if('End' in cur_node.name):
    #             continue
    #         index = skeleton_dict[cur_node.name]      
    #         cur_node.strans += array(displacement[index*3:index*3+3])
    #         cur_node.stransmat[0,3] = cur_node.strans[0]
    #         cur_node.stransmat[1,3] = cur_node.strans[1]
    #         cur_node.stransmat[2,3] = cur_node.strans[2]
    #         for child in cur_node.children:
    #             stack.append(child)

def get_skdict(path=None):
    #获取骨架序列号字典
    skeleton_dict ={}
    index_dict = {}
    with open(path,'r') as f:
        for i in range(59):
            name = f.readline().strip().split(' ')[0]
            skeleton_dict[name] = i
            index_dict[i] = name
    return skeleton_dict, index_dict

def process_bvhnode(node, parentname='hips'):
    # node2joint
    name = node.name
    if (name == "End Site") or (name == "end site"):
        name = parentname + "End"
    
    b1 = Joint(name)
    b1.strans[0] = node.offset[0]
    b1.strans[1] = node.offset[1]
    b1.strans[2] = node.offset[2]
    b1.stransmat = array([[1., 0., 0., 0.], 
                          [0., 1., 0., 0.],
                          [0., 0., 1., 0.], 
                          [0., 0., 0., 1.]])
    b1.stransmat[0, 3] = b1.strans[0]
    b1.stransmat[1, 3] = b1.strans[1]
    b1.stransmat[2, 3] = b1.strans[2]

    for child in node.children:
        b2 = process_bvhnode(child, name)
        b1.addchild(b2)
    return b1


def process_bvhkeyframe(keyframe, joint, t):
    if 'End' in joint.name:
        return
    
    drotmat = array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                     [0., 0., 1., 0.], [0., 0., 0., 1.]])
    skeleton_dict, index_dict = get_skdict(path='data/bvh/a.txt')
    index = skeleton_dict[joint.name]

    yrot = keyframe[index*3]
    xrot = keyframe[index*3+1]
    zrot = keyframe[index*3+2]

    # Xrotation
    theta = radians(xrot)
    mycos = cos(theta)
    mysin = sin(theta)
    drotmat2 = array([[1.,    0.,     0., 0.], 
                      [0., mycos, -mysin, 0.],
                      [0., mysin,  mycos, 0.], 
                      [0.,    0.,     0., 1.]])
    drotmat = dot(drotmat, drotmat2)

    # Yrotation
    theta = radians(yrot)
    mycos = cos(theta)
    mysin = sin(theta)
    drotmat2 = array([[mycos,  0., mysin, 0.], 
                      [0.,     1.,    0., 0.], 
                      [-mysin, 0., mycos, 0.], 
                      [0.,     0.,    0., 1.]])
    drotmat = dot(drotmat, drotmat2)

    # Zrotation
    theta = radians(zrot)
    mycos = cos(theta)
    mysin = sin(theta)
    drotmat2 = array([[mycos, -mysin, 0., 0.], 
                      [mysin, mycos,  0., 0.], 
                      [0.,       0.,  1., 0.], 
                      [0.,       0.,  0., 1.]])
    drotmat = dot(drotmat, drotmat2)

   
    if joint.hasparent:  # Not hips
        parent_trtr = joint.parent.trtr[t]  # Dictionary-based rewrite
        localtoworld = dot(parent_trtr, joint.stransmat)
    else:  # Hips
        localtoworld = joint.stransmat

    joint.trtr[t] = dot(localtoworld, drotmat)

    worldpos = array([localtoworld[0, 3], localtoworld[1, 3],
                      localtoworld[2, 3], localtoworld[3, 3]])
    joint.worldpos[t] = worldpos  # Dictionary-based approach

    for child in joint.children:
        process_bvhkeyframe(keyframe, child, t)

def read_data(path=r'C:\Users\user\Desktop\https\data_time\exp3\1200.npy'):
    data = bytes(np.load(path))[1:]
    skeleton_data = data[64:]
    displacement = []
    rotation = []   
    for i in range(59):
        each_skeleton =  skeleton_data[i*24:(i+1)*24]
        temp_data = []
        for j in range(6):
            temp_data.append(struct.unpack('<f', each_skeleton[j*4:(j+1)*4]))
        displacement+= [temp_data[0][0],temp_data[1][0],temp_data[2][0]]
        rotation+=[temp_data[3][0],temp_data[4][0],temp_data[5][0]]
    return displacement, rotation

def process_bvhfile(filename):
    my_bvh = BvhReader(filename)
    my_bvh.read()
    hips = process_bvhnode(my_bvh.root)
    # path = os.path.join(r'C:\Users\user\Desktop\https\data_time\exp3','{}.npy'.format(10+2))
    # displacement, rotation = read_data() 
    myskeleton = Skeleton(hips)
    return myskeleton
    # process_bvhkeyframe(rotation, myskeleton.root, 0)
    # header, frames = myskeleton.get_frames_worldpos()
    # skeleton_dict, index_dict = get_skdict()
    # xyz_dict = {'X':0, 'Y':1, 'Z':2}
    # skeleton = np.zeros((59,3))
    # for j in range(1, len(frames)):
    #     name = header[j].split('.')
    #     k = skeleton_dict[name[0]]
    #     m = xyz_dict[name[1]]
    #     skeleton[k][m] = frames[j] 

    # return skeleton
if __name__ == '__main__':
    a = process_bvhfile('./真实.bvh')