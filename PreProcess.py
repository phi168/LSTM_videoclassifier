
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentinel DeepFake Challenge
@author: Thore
"""
import glob
import cv2
import numpy as np
import pickle
import os

from collections.abc import Sequence
from itertools import compress
from scipy import signal

from pathlib import Path
import csv
#%% classes
class squares(Sequence):
    def __init__(self, coords, idx):
        self.frame_idx = [idx]
        self.faces = [coords]
        super().__init__()      
    def __len__(self):
        return len(self.faces)
    def __str__(self):
        return str(self.faces)
    def __getitem__(self, i):
        try:
            return np.array(self.faces)[np.array(self.frame_idx) == i].tolist()[0]
        except IndexError:
            return []
    def __iter__(self):
        self.n = 0
        return self
    def __next__(self):
        if self.n < len(self.faces):
            self.n += 1
            return self.faces[self.n-1]
        else:
            raise StopIteration
    def get_start(self):
        return self.frame_idx[0]
    def get_end(self):
        return self.frame_idx[-1]
    def append(self, coords, idx):
        #check that each coordinate has a frame number
        assert len(self.frame_idx) == len(self.faces)
        #number of frames since last frame
        frame_step = idx - self.frame_idx[-1]
        assert(frame_step > 0) # time goes forwared    
        c0 = np.array(self.faces[-1])
        cN = np.array(coords)
        d = (cN - c0)/ frame_step
        for i in range(frame_step):
            c = c0 + d*(i+1)
            self.faces.append(c.tolist())
            self.frame_idx.append(self.frame_idx[-1]+1)
    def get_mean_size(self):
        shp = list(np.mean(np.array(self.faces), axis = 0))
        return np.sqrt(shp[2]*shp[3])
    def grow(self, frac = 0.5):
        faces = []
        npx = self.get_mean_size()*frac
        for l in self.faces: 
            faces.append([l[0]-npx/2, l[1]-npx/2, l[2]+npx, l[3]+npx])
        s = squares([],[])
        s.faces = faces
        s.frame_idx = self.frame_idx    
        return s
    def uniform_size(self):
        #set all squares to have the same
        shp = list(np.max(np.array(self.faces), axis = 0))
        width_max = shp[2]
        height_max  = shp[3]
        faces = []
        for l in self.faces:
            dx = width_max - l[2] #diff in width
            dy = height_max - l[3] #diff in height
            faces.append([l[0]-dx/2, l[1]-dy/2, l[2]+dx, l[3]+dy])
        s = squares([],[])
        s.faces = faces
        s.frame_idx = self.frame_idx
        return s
    def split(self, n_frames):
        N = len(self)
        nsquares = int(np.floor(N/n_frames))
        squareslist = []
        for i in range(nsquares):
            s = squares([],[])
            idx_start = i*n_frames
            s.frame_idx = self.frame_idx[idx_start:idx_start+n_frames]
            s.faces = self.faces[idx_start:idx_start+n_frames]
            squareslist.append(s)        
        return squareslist
    def lazy_squares(self, dt = 5):
         b, a = signal.butter(4, 1/dt/2, fs = 1)
         faces = list(signal.filtfilt(b, a, np.array(self.faces), 
                                           axis=0, method="gust"))
         s = squares([],[])
         s.faces = faces
         s.frame_idx = self.frame_idx
         return s
    def copy(self, s):
        self.faces = s.faces
        self.frame_idx = s.frame_idx
            
class list_of_squares():
    def __init__(self, frame_num = 0):
        self.list_of_squares = []
        self.A_thresh = 0.15 #threshold overlap fraction 
        self.layer_search_depth = 2 
        self.start_frame = frame_num #which frame is first frame
        self.frame_num = frame_num #which frame are we on
    def __str__(self):
        return str([s.__str__() for s in self.list_of_squares])
    def __getitem__(self,i):
        return(self.list_of_squares[i])
    def append(self, x, y, w, h, *args):
        if len(args) == 1:
            idx = args[0] #we are on this frame 
            frame_step = idx - self.frame_num #skipped this many frames
            assert(frame_step >= 0) #time goes forward 
            self.frame_num = idx
        else:
            self.frame_num += 1
            idx = self.frame_num
            frame_step = 1
        ff = False #found match flag
        i = 0
        while not ff and i < self.layer_search_depth: 
            i += 1
            for squares in self.list_of_squares:
                #get the square below      
                if self.get_overlap(*squares[idx-i], x,y,w,h) > self.A_thresh: 
                    squares.append([x,y,w,h], idx)
                    ff = True                     
        if not ff:
            #did not overlap, create new sequence
            self.add_sequence([x,y,w,h], idx)  
    def get_num_sequences(self):
        return len(self.list_of_squares)   
    def shape(self):
        return [(len(s), s.get_start(), s.get_end()) for s in self.list_of_squares]    
    def add_sequence(self, seed, idx):
        #Generate new squares object
        s = squares(seed, idx)
        self.list_of_squares.append(s)                              
    def get_overlap(self, *args):
        if len(args) == 8:
            x, y, w, h, x2, y2, w2, h2 = args
            Lx = max(0, min([x+w, x2+w2]) - max([x,x2]))
            Ly = max(0, min([y+h, y2+h2]) - max([y,y2]))
            A_union = w*h + w2*h2 - Lx*Ly
            return Lx*Ly/(A_union)
        else: # this happens if one square didn't exist
            return 0
    def clean_up_by_length(self, minframes = 20):
        self.list_of_squares = list(compress(self.list_of_squares, 
                 [len(s) >= minframes for s in self.list_of_squares]))
    def clean_up_by_size(self, minsize = 50):
        self.list_of_squares = list(compress(self.list_of_squares, 
                 [s.get_mean_size() >= minsize for s in self.list_of_squares]))
    def filter_video(self):
        for f in self.list_of_squares:
            t = f.grow()
            t = t.uniform_size()
            t = t.lazy_squares()
            f.copy(t)

        
#%% Functions    
def GetFaces(file):
    import mtcnn  
    j = 0
    cap = cv2.VideoCapture(file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    faces = []
    i = 0
    detector = mtcnn.MTCNN()
    while cap.isOpened():
        ret, frame = cap.read()
        i += 1
        if ret==True:   
          print('file %d frame %1.2f' % (j, i/length))
          f = detector.detect_faces(frame)
          faces.append([x['box'] for x in f])
        else:
          break
    print('writing to ' + file + 'faces.p')
    pickle.dump(faces, open(file+'faces.p', 'wb'))
    cap.release()
    cv2.destroyAllWindows()

def ProcessFilesToGetFaces(root = ''):
    files = glob.glob(os.path.join(root + 'videos', 'fake','*.mp4'))  
    for j in tqdm(range(len(files))):
        file = files[j] 
        if os.path.isfile(file+'faces.p'):
            print('file ' + file + 'faces.p already exists')
            continue
        else:
            print('file ' + file + 'faces.p not found. Calculating...')
            GetFaces(file)
            
def setup_data(test_fraction = 0.3, root = '', isolate_faces = True):   
    """
    Sorts videos in train and test, chops them into chunks with isolated faces
    and creates a table to refer back to their location

    Parameters
    ----------
    test_fraction : 0-1, optional
        what fraction of videos will be used for testing. The default is 0.3.
    root : TYPE, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    None.
    
        """
    cases = ['real', 'fake']
    files = []
    files.append( os.listdir(root + 'videos/' + cases[0]))
    files.append( os.listdir(root + 'videos/' + cases[1]))
    data = []
    #For each case
    for i in range(2):
        # split data set into train and test
        num_files_test = round(len(files[i]) * test_fraction)
        j = -1
        for filename in files[i]:
            j+=1
            if j < num_files_test:
               # it is a test file
               train_or_test = 'test'
            else:
                # it is a train file
                train_or_test = 'train'
                
            src = os.path.join(root, 'videos', cases[i], filename)
            filename_, ext = filename.split('.')
            folderdir = os.path.join(root, 'data', train_or_test, cases[i],
                                     filename_)
            # Path to chop video into sections
            Path(folderdir).mkdir(parents=True, exist_ok=True)
            if isolate_faces:
                facesfile = src + 'faces.p'
                faces = pickle.load(open(facesfile, 'rb'))
                ret = split_video_facedetect(src, folderdir, faces, data)
            else: 
                ret = split_video_raw(src, folderdir, data)
            assert ret

    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data)
    print("Extracted and wrote %d video files." % (len(data)))

def split_video_raw(src, folderdir, data):
    from subprocess import call
    dest = folderdir + '/chunk%03d.mp4'
    # Split video into sections
    call(["ffmpeg", "-i", src, "-c", "copy", "-map", "0", 
         "-segment_time", "00:00:10",  "-f", "segment",  
         "-reset_timestamps", "1", dest])
    chunks = os.listdir(folderdir)
    #Get size of video 
    cap = cv2.VideoCapture(folderdir + '/chunk000.mp4')
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()
    #create folder of jpgs for each chunk
    for chunk in chunks:
        chunkname, ext = chunk.split('.')
        dest = folderdir + '/' + chunkname + '/frame%04d.jpg'
        Path(folderdir + '/' + chunkname).mkdir(parents=True, exist_ok=True)
        call(["ffmpeg", "-i", folderdir + '/' + chunk, dest])
        # Remove chunk video
        os.remove(folderdir + '/' + chunk)
        # Count number of frames 
        num_frames = len(os.listdir(folderdir + '/' + chunkname))
        (_,train_or_test, case, fillename_) = folderdir.split('/')
        data.append([train_or_test, case, filename_, 
                     chunkname, num_frames, (height, width)])
        
    return True

def split_video_facedetect(src, folderdir, faces, data):
    #first cleanup the faces list
    cap = cv2.VideoCapture(src)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    faceList = list_of_squares() 
    for i in range(N): # for every frame
        for face in faces[i]: #for all faces identified in frame
            #add face to mask
            faceList.append(*face, i)
            
        if i % 500 == 0: # remove short sequences every 500 frames
            faceList.clean_up_by_length()
    faceList.clean_up_by_length()
    faceList.clean_up_by_size()
    faceList.filter_video()

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(src)
    i = 0
    for faceSequence in faceList:
        i += 1
        cap = cv2.VideoCapture(src)
        subsequences = faceSequence.split(int(fps*2)) #2 second chunks
        chunkify(subsequences, cap, folderdir, i, data)
        cap.release()
    return True
        
def chunkify(subsequences, cap, folderdir, i, data):
    start_frame = subsequences[0].frame_idx[0]
    # set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    j = 0
    fnum = start_frame
    (_, train_or_test, case, fillename_) = folderdir.split('/')
    for seq in subsequences:
        j += 1
        chunkname = 'chunk%03d_%04d' % (i, j)
        dest = folderdir + '/' + chunkname
        Path(dest).mkdir(parents=True, exist_ok=True)
        k = 0
        for x, y, w, h in seq:
            k += 1
            ret, frame = cap.read()
            fnum+= 1
            assert ret,  'frame does not extst f = %1.0f' % fnum #should only operate within range of clip frames
            im = frame[int(y):int(y)+int(h), int(x):int(x)+int(w), :]
            cv2.imwrite(dest+'/frame%04d.png' % k, im)
            
        data.append([train_or_test, cases[i], filename_, 
                     chunkname, k, (int(h), int(w))])
        
    
#%%




