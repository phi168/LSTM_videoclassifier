
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains tools for pre-processing videos
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
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#%% classes
class Squares(Sequence):
    """a Squares is a list of coordinates, each corresponding to a frame. 
    The series is continuous."""
    
    def __init__(self, coords, idx):
        """initiate a square. this is just a list of coordinate"""
        
        self.frame_idx = [idx] #each coordindate corresponds to a frame
        self.faces = [coords] #list of coordinates x,y,w,h
        super().__init__()      
        
    def __len__(self):
        """get the length of list of corrdinates"""
        return len(self.faces)
    
    def __str__(self):
        """how to print Squares"""
        return str(self.faces)
    
    def __getitem__(self, i):
        """return indexed item. Don't crash if outside of index"""
        try:
            return np.array(self.faces)[np.array(self.frame_idx) == i].tolist()[0]
        except IndexError:
            return []
        
    def __iter__(self):
        """we can iterate through the list"""
        self.n = 0
        return self
    
    def __next__(self):
        """on iteration return next set of coordinates"""
        if self.n < len(self.faces):
            self.n += 1
            return self.faces[self.n-1]
        else:
            raise StopIteration
            
    def get_start(self):
        """this is the first frame of the video"""
        return self.frame_idx[0]
    
    def get_end(self):
        """this is the last frame of the video"""
        return self.frame_idx[-1]
    
    def append(self, coords, idx):
        """got new coordinates. Append them to list"""
        #check that each coordinate has a frame number
        assert len(self.frame_idx) == len(self.faces)
        #number of frames since last frame
        frame_step = idx - self.frame_idx[-1]
        assert(frame_step >= 0) # time goes forward 
        if frame_step == 0:
            #recorded two overlapping faces in the same frame
            #choose the bigger one
            A_new = coords[2] * coords[3]
            A_old = self.faces[-1][2] * self.faces[-1][3]
            if A_new < A_old:
                #ignore this one
                return
            else:
                #remove last element and add new one(below)
                self.faces = self.faces[:-1]
                self.frame_idx = self.frame_idx[:-1]
                frame_step = 1
                
        
        #if frames have been skipped, we stitch with linear interpolation
        c0 = np.array(self.faces[-1]) 
        cN = np.array(coords)
        d = (cN - c0)/ frame_step
        for i in range(frame_step):
            c = c0 + d*(i+1)
            self.faces.append(c.tolist())
            self.frame_idx.append(self.frame_idx[-1]+1)
            
    def get_mean_size(self):
        """get mean size of all squares"""
        shp = list(np.mean(np.array(self.faces), axis = 0))
        return np.sqrt(shp[2]*shp[3])
    
    def grow(self, frac = 0.5):
        """grow all squares by a fraction"""
        faces = []
        npx = self.get_mean_size()*frac
        for l in self.faces: 
            faces.append([l[0]-npx/2, l[1]-npx/2, l[2]+npx, l[3]+npx])
        s = Squares([],[])
        s.faces = faces
        s.frame_idx = self.frame_idx    
        return s
    
    def uniform_size(self):
        """set all squares to have the same size"""
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
        """return a list of Squares, each containing a subset of n_frames"""
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
        """filter movement with a low-path filder"""
        b, a = signal.butter(4, 1/dt/2, fs = 1)
        faces = list(signal.filtfilt(b, a, np.array(self.faces), 
                                          axis=0, method="gust"))
        s = Squares([],[])
        s.faces = faces
        s.frame_idx = self.frame_idx
        return s
     
    def copy(self, s):
        """set values of this instance to be the same of the argument"""
        self.faces = s.faces
        self.frame_idx = s.frame_idx
            
class ListOfSquares():
    """List Of Squares class"""
    
    def __init__(self, frame_num = 0):
        self.list_of_squares = []
        self.A_thresh = 0.15 #threshold overlap fraction 
        self.layer_search_depth = 2 #how many frames to insist on frame continuity for
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
        
        if len(args) == 1: #specied frame index
            idx = args[0] #we are on this frame 
            frame_step = idx - self.frame_num #skipped this many frames
            assert(frame_step >= 0) #time goes forward (or same frame)
            self.frame_num = idx
        else: #assume we went on by one frame
            self.frame_num += 1
            idx = self.frame_num
            frame_step = 1
            
        ff = False #found match flag
        i = 0
        #check for previouse frames if we recorded squares
        while not ff and i < self.layer_search_depth: 
            i += 1
            for squares in self.list_of_squares:
                #get the square below, check for overlap  
                if self._get_overlap(*squares[idx-i], x,y,w,h) > self.A_thresh: 
                    #significant overlap. we assign this to the same square
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

            videoname = os.path.splitext(os.path.basename(file_path))[0]
            
            folderdir = os.path.join(root, 'data', train_or_test, cases[i],
                                     videoname)
            # Path to chop video into sections
            Path(folderdir).mkdir(parents=True, exist_ok=True)
  
            #process video
            split_video_facedetect(file_path, folderdir, data)

    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data)
    print("Extracted and wrote %d video files." % (len(data)))

def split_video_facedetect(videopath:str, folderdir:str, data:list, seq_length = 2):
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
    seq_length : TYPE, optional
        duration of each video chunk in seconds. The default is 2.

    """
    
    #first cleanup the faces list
    #load list of face coordinates            
    facesfile = videopath + 'faces.p'
    faces = pickle.load(open(facesfile, 'rb'))
    #get video
    cap = cv2.VideoCapture(videopath) 
    # N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #num frames
    face_list = ListOfSquares()
    for i,faces_in_frame in enumerate(faces): # for every frame
        for face in faces_in_frame: #for all faces identified in frame
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

    #for each consecutive sequence of face frames
    for sequence_id, face_sequence in enumerate(face_list):
        cap = cv2.VideoCapture(videopath)
        #turn sequence into list of 2s long sequences
        subsequences = face_sequence.split(int(fps * seq_length)) #2 second chunks
        
        if subsequences: #might be empty if sequency is short than seq_length
            chunkify(subsequences, cap, folderdir, sequence_id, data)
            
        cap.release()

        
def chunkify(subsequences, cap, folderdir, sequence_id, data):
    """loops over each sub-sequence and creates a separate chunk for it"""
    
    start_frame = subsequences[0].frame_idx[0]
    # set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fnum = start_frame
    pathnameslist = folderdir.split('/')
    pathnameslist.reverse()
    (video_name, case, train_or_test, *_) = pathnameslist

    
    #loop over each sub-sequence
    for subsequence_id, seq in enumerate(subsequences):
        #generate name for sub-sequence
        chunkname = 'chunk%03d_%04d' % (sequence_id, subsequence_id)
        #create new folder for this chunk
        dest = folderdir + '/' + chunkname
        #make folder
        Path(dest).mkdir(parents=True, exist_ok=True)
        
        #loop over each frame in sub-sequence
        for frame_num, (x, y, w, h) in enumerate(seq):

            ret, frame = cap.read()
            fnum+= 1
            #should only operate within range of clip frames
            assert ret,  'frame does not extst f = %1.0f' % fnum 
            #get face
            # print((x,y,w,h))
            y = abs(y)
            x = abs(x)
            h = abs(h)
            w = abs(w)
            im = frame[int(y):int(y)+int(h), int(x):int(x)+int(w), :]
            #save to file
            cv2.imwrite(dest+'/frame%04d.png' % frame_num, im)
        
        #make record of sub-sequence    
        data.append([train_or_test, case, video_name, 
                     chunkname, frame_num + 1, (int(h), int(w))])
        
    
#%%
# def show_list_of_squares(videopath, face_list):
#     videopath = 'resources/videos/real/302012592572313393031555569823.mp4'
#     cap = cv2.VideoCapture(videopath) 
    
#     while(True):
#         ret, frame = cap.read()    
#         if not ret:
#             break
#         # Display the resulting frame
#         cv2.imshow('frame',frame)
#         if cv2.waitKey(20) & 0xFF == ord('q'):
#             break
    
#     # When everything done, release the capture
#     cap.release()
#     cv2.destroyAllWindows()
    



