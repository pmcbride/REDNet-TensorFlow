import subprocess as sp
import os
import multiprocessing as mp

def process_images_mp(image_path):
    sp.Popen(['python', 'sr.py', '--file', image_path]).wait()   

def mp_test(image_list):
    #print(image_list)
    sp.Popen(['echo', image_list], shell=True).wait()
    #return None
    
def process_images(image_list):
    for im in image_list:
        sp.Popen(['python', 'sr.py', '--file', im]).wait()
        
def process_images_small(image_list):
    for im in image_list:
        sp.Popen(['python', 'sr.py', '--file', im, '--layers=8', '--filters=96']).wait()

def process_images_compact(image_list):
    for im in image_list:
        sp.Popen(['python', 'sr.py', 
                  '--file', im, 
                  '--scale=2', 
                  '--layers=7', 
                  '--filters=32', 
                  '--min_filters=8', 
                  '--filters_decay_gamma=1.2', 
                  '--nin_filters=24', 
                  '--nin_filters2=8', 
                  '--reconstruct_layers=0', 
                  '--self_ensemble=1', 
                  '--batch_image_size=32', 
                  '--pixel_shuffler_filters=1']
                ).wait()  
#p.map(process_image, image_list)
# for im in os.listdir(VIDEO_SAVE_DIR):
#     sp.Popen(['python', 'sr.py', '--file', os.path.join(VIDEO_SAVE_DIR, im)])
    #%run sr.py --file os.path.join(VIDEO_SAVE_DIR, im)

if __name__ == "__main__":
    save_dir = "../data/images/images_test_srgan-tf"
    video_dir = "../data/videos"
    video_file = "../data/videos/video.mp4"

    # Define absolute directory paths
    ROOT_DIR = os.getcwd()
    VIDEO_DIR = os.path.join(video_dir)
    VIDEO_SAVE_DIR = os.path.join(save_dir)
    VIDEO_FILE_PATH = os.path.join(video_file)
    cpus = mp.cpu_count()
    p = mp.Pool(cpus)
    image_list = os.listdir(save_dir)
    image_path = [os.path.join(save_dir, name) for name in image_list]
    #p.map(process_image, image_path)
    process_images_compact(image_path)
