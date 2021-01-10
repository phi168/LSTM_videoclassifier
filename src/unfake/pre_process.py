
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
import mtcnn 
from tqdm import tqdm

from collections.abc import Sequence
from itertools import compress
from scipy import signal

from pathlib import Path
import csv
#%% classes
class Squares(Sequence):
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
        s = Squares([],[])
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
        s = Squares([],[])
        s.faces = faces
        s.frame_idx = self.frame_idx
        return s
    def split(self, n_frames):
        N = len(self)
        nsquares = int(np.floor(N/n_frames))
        squareslist = []
        for i in range(nsquares):
            s = Squares([],[])
            idx_start = i*n_frames
            s.frame_idx = self.frame_idx[idx_start:idx_start+n_frames]
            s.faces = self.faces[idx_start:idx_start+n_frames]
            squareslist.append(s)        
        return squareslist
    def lazy_squares(self, dt = 5):
         b, a = signal.butter(4, 1/dt/2, fs = 1)
         faces = list(signal.filtfilt(b, a, np.array(self.faces), 
                                           axis=0, method="gust"))
         s = Squares([],[])
         s.faces = faces
         s.frame_idx = self.frame_idx
         return s
    def copy(self, s):
        self.faces = s.faces
        self.frame_idx = s.frame_idx
            
class ListOfSquares():
    """List Of Squares class"""
    
    def __init__(self, frame_num = 0):
        self.list_of_squares = []
        self.A_thresh = 0.15 #threshold overlap fraction 
        self.layer_search_depth = 2 
        self.start_frame = frame_num #which frame is first frame
        self.frame_num = frame_num #which frame are we on
        
    def __str__(self):
        """give string representation"""
        return str([s.__str__() for s in self.list_of_squares])
    
    def __getitem__(self,i):
        """return indexed Squares object"""
        return(self.list_of_squares[i])
    
    def append(self, x, y, w, h, *args):
        """checks if new coordinates belong to previous Squares or if new 
        Squares should be made. Addes new coordinates accordingly."""
        
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
                if self._get_overlap(*squares[idx-i], x,y,w,h) > self.A_thresh: 
                    squares.append([x,y,w,h], idx)
                    ff = True                     
        if not ff:
            #did not overlap, create new sequence
            s = Squares([x,y,w,h], idx)
            self.list_of_squares.append(s)  
            

    def get_num_sequences(self):
        """get number of squares in list"""
        return len(self.list_of_squares)  
    
    def shape(self):
        """get shape of square_list (len, start, end)"""
    
        return [(len(s), s.get_start(), s.get_end()) for s in self.list_of_squares]    

    def _get_overlap(self, *args):
        """Compute overlap between squares. Used internally."""
        
        if len(args) == 8:
            x, y, w, h, x2, y2, w2, h2 = args
            Lx = max(0, min([x+w, x2+w2]) - max([x,x2]))
            Ly = max(0, min([y+h, y2+h2]) - max([y,y2]))
            A_union = w*h + w2*h2 - Lx*Ly
            return Lx*Ly/(A_union)
        else: # this happens if one square didn't exist
            return 0
        
    def clean_up_by_length(self, minframes = 20):
        """removes Squares which are less then minframes in length"""
        
        self.list_of_squares = list(compress(self.list_of_squares, 
                 [len(s) >= minframes for s in self.list_of_squares]))
        
    def clean_up_by_size(self, minsize = 50):
        """removes Squares which are less then minsize pixels in extend"""
        
        self.list_of_squares = list(compress(self.list_of_squares, 
                 [s.get_mean_size() >= minsize for s in self.list_of_squares]))
        
    def filter_video(self):
        """Make each square constant size and less jittery in time"""
        
        for f in self.list_of_squares:
            t = f.grow()
            t = t.uniform_size()
            t = t.lazy_squares()
            f.copy(t)

        
#%% Functions    
def get_faces(file:str):
    """for file (path to file), open video and extract facial locations"""
    
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

def process_files_to_get_faces(root = '', suffix = 'mp4'):
    """for all videos in videos/fake and videos/real, apply mtcnn to find 
    location of face in each frame"""
    
    files = glob.glob(os.path.join(root + 'videos', 'fake','*.' + suffix))  
    files.extend(glob.glob(os.path.join(root + 'videos', 'real','*.' + suffix)))  
                 
    for j in tqdm(range(len(files))):
        file = files[j] 
        if os.path.isfile(file+'faces.p'):
            print('file ' + file + 'faces.p already exists')
            continue
        else:
            print('file ' + file + 'faces.p not found. Calculating...')
            get_faces(file)
            
def setup_data(test_fraction = 0.3, root = '', suffix = 'mp4'):   
    """
    Sorts videos in train and test, chops them into chunks with isolated faces
    and creates a table to refer back to their location. Videos should be in 
    root/videos/ and processed data will be saved in root/data/

    Parameters
    ----------
    test_fraction : 0-1, optional
        what fraction of videos will be used for testing. The default is 0.3.
    root : TYPE, optional
        directory pointing to where the videos are. The default is ''.
    Returns
    -------
    None.
    
    """
    
    cases = ['real', 'fake']
    files = []
    files.append(glob.glob(os.path.join(root + 'videos', cases[0],'*.' + suffix)))
    files.append((glob.glob(os.path.join(root + 'videos', cases[1],'*.' + suffix))))

    data = []
    #For each case
    for i in range(2):
        # split data set into train and test
        num_files_test = round(len(files[i]) * test_fraction)
        j = -1
        for file_path in files[i]:
            j+=1
            if j < num_files_test:
               # it is a test file
               train_or_test = 'test'
            else:
                # it is a train file
                train_or_test = 'train'

            filename = os.path.splitext(os.path.basename(file_path))[0]
            
            folderdir = os.path.join(root, 'data', train_or_test, cases[i],
                                     filename)
            # Path to chop video into sections
            Path(folderdir).mkdir(parents=True, exist_ok=True)
            #load list of face coordinates            
            facesfile = file_path + 'faces.p'
            faces = pickle.load(open(facesfile, 'rb'))
            #process video
            split_video_facedetect(file_path, folderdir, faces, data)

    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data)
    print("Extracted and wrote %d video files." % (len(data)))

def split_video_facedetect(videopath:str, folderdir:str, faces:list, data:list):
    """take video input an create subvideos with just faces of interest"
    

    Parameters
    ----------
    videopath : str
        path to video.
    folderdir : str
        destination folder (where to put processed chunks)
    faces : list
        list with coordinates for faces for each frame (output from get_faces()).
    data : list
        list keeping track of filenames and locaitons

    """
    #first cleanup the faces list
    cap = cv2.VideoCapture(videopath) #get video
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #num frames
    face_list = ListOfSquares()
    for i in range(N): # for every frame
        for face in faces[i]: #for all faces identified in frame
            #add face to mask
            face_list.append(*face, i)

        if i % 500 == 0: # remove short sequences every 500 frames
            face_list.clean_up_by_length()
    #filter face      
    face_list.clean_up_by_length()
    face_list.clean_up_by_size()
    face_list.filter_video()

    #get frames per second
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(videopath)

    #for each consecutive sequenc of face frames
    for i, face_sequence in enumerate(face_list):
        cap = cv2.VideoCapture(videopath)
        subsequences = faceSequence.split(int(fps*2)) #2 second chunks
        chunkify(subsequences, cap, folderdir, i, data)
        cap.release()

        
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
            
        data.append([train_or_test, case[i], filename_, 
                     chunkname, k, (int(h), int(w))])
        
    
#%%




