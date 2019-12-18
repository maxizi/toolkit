from PIL import Image

from got10k.trackers import Tracker, IdentityTrackerRGBD
from got10k.experiments import ExperimentVOT
from got10k.datasets import VOT
from got10k.utils.viz import show_frame

ROOT_DIR = '/home/maxi/datasets/VOT2019RGBD'

def example_track_val_set():
    # setup tracker
    tracker = IdentityTrackerRGBD()

    # run experiment on validation set
    experiment = ExperimentVOT(
        root_dir=ROOT_DIR,
        version='RGBD2019',
        read_image=True,
        #list_file=None,
        experiments='unsupervised', 
        result_dir='results',
        report_dir='reports'
        )
    
    experiment.run(tracker, visualize=False)

    # report performance
    experiment.report([tracker.name])

    print('Finished')

if __name__ == '__main__':
    example_track_val_set()